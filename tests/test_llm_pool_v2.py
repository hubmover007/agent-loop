"""Tests for the new LLM Pool (security + protocol adapters + circuit breaker)."""

from __future__ import annotations

import asyncio
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# ── helpers ──────────────────────────────────────────────────────────────────

MINIMAL_YAML = """
version: "1.0"
providers:
  - id: cheap-model
    type: openai_compatible
    protocol: openai-completions
    auth:
      method: env_var
      key_env: TEST_API_KEY
    endpoint:
      base_url: http://localhost/v1
      model: cheap
      timeout_s: 10
    capabilities: [general, coding]
    context_window: 8000
    max_output_tokens: 1024
    cost:
      input: 0.10
      output: 0.20
    limits:
      max_concurrent: 5
      requests_per_minute: 100
    circuit_breaker:
      failure_threshold: 3
      recovery_timeout_s: 30
    enabled: true
    tags: [cheap]

  - id: powerful-model
    type: openai_compatible
    protocol: openai-completions
    auth:
      method: env_var
      key_env: TEST_API_KEY
    endpoint:
      base_url: http://localhost/v1
      model: powerful
      timeout_s: 60
    capabilities: [general, coding, reasoning, vision]
    context_window: 200000
    max_output_tokens: 8192
    cost:
      input: 3.0
      output: 15.0
    limits:
      max_concurrent: 3
      requests_per_minute: 20
    circuit_breaker:
      failure_threshold: 3
      recovery_timeout_s: 60
    enabled: true
    tags: [powerful]

  - id: disabled-model
    type: openai_compatible
    protocol: openai-completions
    auth:
      method: none
    endpoint:
      model: disabled
    capabilities: [general]
    cost:
      input: 0.0
      output: 0.0
    limits:
      max_concurrent: 1
    circuit_breaker:
      failure_threshold: 5
      recovery_timeout_s: 60
    enabled: false
    tags: []

selection_strategies:
  cheapest:
    sort_by: cost.input
    ascending: true
  most_capable:
    sort_by: cost.input
    ascending: false
  fastest:
    sort_by: limits.requests_per_minute
    ascending: false
  balanced:
    weights:
      cost: 0.35
      speed: 0.30
      capability: 0.35

task_strategies:
  default: balanced
  coding: cheapest
  reasoning: most_capable
"""


@pytest.fixture
def config_path(tmp_path):
    """Write YAML to a temp file, set TEST_API_KEY env var."""
    p = tmp_path / "llm_pool.yaml"
    p.write_text(MINIMAL_YAML)
    os.environ["TEST_API_KEY"] = "test-key-12345"
    yield p
    os.environ.pop("TEST_API_KEY", None)


@pytest.fixture
def pool(config_path, tmp_path):
    from src.llm_pool import LLMPool
    p = LLMPool(config_path=config_path, state_dir=tmp_path / "state")
    p.initialize()
    return p


# ── config loading ─────────────────────────────────────────────────────────

def test_pool_config_load(config_path):
    from src.llm_pool.config import PoolConfig
    cfg = PoolConfig.from_yaml(config_path)
    assert cfg.version == "1.0"
    assert len(cfg.providers) == 3
    enabled = cfg.get_enabled()
    assert len(enabled) == 2
    ids = {p.id for p in enabled}
    assert "cheap-model" in ids
    assert "powerful-model" in ids
    assert "disabled-model" not in ids


def test_pool_config_cost_fields(config_path):
    from src.llm_pool.config import PoolConfig
    cfg = PoolConfig.from_yaml(config_path)
    cheap = next(p for p in cfg.providers if p.id == "cheap-model")
    assert cheap.cost.input == 0.10
    assert cheap.cost.output == 0.20
    assert cheap.limits.max_concurrent == 5
    assert cheap.circuit_breaker.failure_threshold == 3


def test_pool_config_auth_fields(config_path):
    from src.llm_pool.config import PoolConfig
    cfg = PoolConfig.from_yaml(config_path)
    cheap = next(p for p in cfg.providers if p.id == "cheap-model")
    assert cheap.auth.method == "env_var"
    assert cheap.auth.key_env == "TEST_API_KEY"
    # No secrets in config
    assert not hasattr(cheap.auth, "api_key")


# ── auth resolver ──────────────────────────────────────────────────────────

def test_auth_resolver_env_var():
    from src.llm_pool.auth import AuthResolver
    from src.llm_pool.config import AuthConfig
    os.environ["_TEST_KEY_ABC"] = "secret-value"
    try:
        resolver = AuthResolver()
        cfg = AuthConfig(method="env_var", key_env="_TEST_KEY_ABC")
        result = resolver.resolve(cfg)
        assert result["api_key"] == "secret-value"
    finally:
        os.environ.pop("_TEST_KEY_ABC", None)


def test_auth_resolver_none():
    from src.llm_pool.auth import AuthResolver
    from src.llm_pool.config import AuthConfig
    resolver = AuthResolver()
    result = resolver.resolve(AuthConfig(method="none"))
    assert result == {}


def test_auth_resolver_missing_key_returns_none():
    from src.llm_pool.auth import AuthResolver
    from src.llm_pool.config import AuthConfig
    os.environ.pop("_NONEXISTENT_KEY_XYZ", None)
    resolver = AuthResolver()
    cfg = AuthConfig(method="env_var", key_env="_NONEXISTENT_KEY_XYZ")
    result = resolver.resolve(cfg)
    assert result.get("api_key") is None


def test_auth_validator_ok():
    from src.llm_pool.auth import AuthResolver
    from src.llm_pool.config import AuthConfig
    os.environ["_VALIDATE_TEST_KEY"] = "val"
    try:
        resolver = AuthResolver()
        ok, reason = resolver.validate_required(AuthConfig(method="env_var", key_env="_VALIDATE_TEST_KEY"))
        assert ok
    finally:
        os.environ.pop("_VALIDATE_TEST_KEY", None)


# ── circuit breaker ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures():
    from src.llm_pool.circuit_breaker import CircuitBreaker, CBState
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout_s=999)
    assert cb.state == CBState.CLOSED
    for _ in range(3):
        await cb.record_failure()
    assert cb.state == CBState.OPEN
    assert not cb.is_available()


@pytest.mark.asyncio
async def test_circuit_breaker_resets_on_success():
    from src.llm_pool.circuit_breaker import CircuitBreaker, CBState
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout_s=999)
    await cb.record_failure()
    await cb.record_failure()
    await cb.record_success()
    assert cb.state == CBState.CLOSED
    assert cb._consecutive_failures == 0


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_recovery():
    from src.llm_pool.circuit_breaker import CircuitBreaker, CBState
    import time
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout_s=0.01)
    await cb.record_failure()
    await cb.record_failure()
    assert cb.state == CBState.OPEN

    await asyncio.sleep(0.05)  # wait for recovery timeout
    allowed = await cb.acquire()
    assert allowed
    assert cb.state == CBState.HALF_OPEN

    await cb.record_success()
    assert cb.state == CBState.CLOSED


# ── pool selection ─────────────────────────────────────────────────────────

def test_pool_selects_cheapest(pool):
    from src.llm_pool import LLMPool
    cfg = pool.select(capabilities=["general"], strategy="cheapest")
    assert cfg is not None
    assert cfg.id == "cheap-model"


def test_pool_selects_most_capable(pool):
    cfg = pool.select(capabilities=["general"], strategy="most_capable")
    assert cfg is not None
    assert cfg.id == "powerful-model"


def test_pool_filters_by_capability(pool):
    # vision only available in powerful-model
    cfg = pool.select(capabilities=["vision"], strategy="cheapest")
    assert cfg is not None
    assert cfg.id == "powerful-model"


def test_pool_returns_none_when_no_match(pool):
    cfg = pool.select(capabilities=["nonexistent_capability_xyz"])
    assert cfg is None


def test_pool_task_type_routing(pool):
    cfg_coding = pool.select(task_type="coding")
    assert cfg_coding is not None
    assert cfg_coding.id == "cheap-model"   # coding → cheapest

    cfg_reasoning = pool.select(task_type="reasoning")
    assert cfg_reasoning is not None
    assert cfg_reasoning.id == "powerful-model"  # reasoning → most_capable


# ── pool acquire with mock adapter ────────────────────────────────────────

@pytest.mark.asyncio
async def test_pool_acquire_and_chat(pool):
    from src.llm_pool.pool import PoolManagedProvider
    from src.loop_engine import LLMResponse

    provider = await pool.acquire(capabilities=["coding"], strategy="cheapest")
    assert isinstance(provider, PoolManagedProvider)
    assert provider.provider_id == "cheap-model"

    # Mock the inner adapter
    provider._adapter.chat = AsyncMock(return_value=LLMResponse(
        content="def hello(): pass",
        model="cheap",
        usage={"input_tokens": 10, "output_tokens": 5},
    ))

    result = await provider.chat([{"role": "user", "content": "write hello world"}])
    assert "hello" in result.content


@pytest.mark.asyncio
async def test_pool_circuit_breaker_blocks_after_failures(pool):
    from src.llm_pool.circuit_breaker import CBState

    cb = pool._circuit_breakers["cheap-model"]
    # Force open
    for _ in range(3):
        await cb.record_failure()
    assert cb.state == CBState.OPEN

    provider = pool._providers["cheap-model"]
    with pytest.raises(RuntimeError, match="circuit breaker is OPEN"):
        await provider.chat([{"role": "user", "content": "test"}])


# ── usage tracker ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tracker_records_usage(tmp_path):
    from src.llm_pool.tracker import PoolTracker, UsageRecord
    log_path = tmp_path / "usage.jsonl"
    tracker = PoolTracker(usage_log_path=log_path)

    record = UsageRecord(
        timestamp="2026-01-01T00:00:00+00:00",
        provider_id="test-provider",
        model="test-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=123.4,
        success=True,
    )
    await tracker.record(record)

    # Check in-memory stats
    stats = tracker.get_stats("test-provider")
    assert stats["call_count"] == 1
    assert stats["success_rate"] == 1.0
    assert stats["total_input_tokens"] == 100

    # Check JSONL file
    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    import json
    data = json.loads(lines[0])
    assert data["provider_id"] == "test-provider"
    assert data["input_tokens"] == 100


@pytest.mark.asyncio
async def test_tracker_failure_rate(tmp_path):
    from src.llm_pool.tracker import PoolTracker, UsageRecord
    tracker = PoolTracker()

    for i in range(3):
        await tracker.record(UsageRecord(
            timestamp="2026-01-01T00:00:00+00:00",
            provider_id="p1", model="m",
            input_tokens=10, output_tokens=5,
            latency_ms=100, success=(i < 2),
        ))

    stats = tracker.get_stats("p1")
    assert stats["call_count"] == 3
    assert round(stats["success_rate"], 2) == 0.67


# ── provider_status introspection ─────────────────────────────────────────

def test_pool_provider_status(pool):
    statuses = pool.provider_status()
    assert len(statuses) == 3  # all providers (incl. disabled)

    in_pool = [s for s in statuses if s["in_pool"]]
    assert len(in_pool) == 2

    ids = {s["id"] for s in statuses}
    assert "cheap-model" in ids
    assert "disabled-model" in ids

    for s in statuses:
        # Auth method should be visible but no secrets
        assert "auth_method" in s
        assert "api_key" not in s
        assert "secret" not in str(s).lower()
