"""Tests for LLM Pool v3 — JSON-configured, single-file module."""

from __future__ import annotations

import asyncio
import json
import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock

# ── helpers ──────────────────────────────────────────────────────────────────

MINIMAL_JSON = {
    "providers": [
        {
            "id": "cheap-model",
            "type": "openai",
            "endpoint": "http://localhost/v1",
            "model": "cheap",
            "api_key_source": "env:TEST_API_KEY",
            "capabilities": ["general", "coding"],
            "cost_per_1m_input": 0.10,
            "cost_per_1m_output": 0.20,
            "max_concurrent": 5,
            "verified": False,
            "verified_at": None,
            "enabled": True,
            "tags": ["cheap"]
        },
        {
            "id": "powerful-model",
            "type": "openai",
            "endpoint": "http://localhost/v1",
            "model": "powerful",
            "api_key_source": "env:TEST_API_KEY",
            "capabilities": ["general", "coding", "reasoning", "vision"],
            "cost_per_1m_input": 3.0,
            "cost_per_1m_output": 15.0,
            "max_concurrent": 3,
            "verified": False,
            "verified_at": None,
            "enabled": True,
            "tags": ["powerful"]
        },
        {
            "id": "disabled-model",
            "type": "openai",
            "endpoint": "http://localhost/v1",
            "model": "disabled",
            "api_key_source": "none",
            "capabilities": ["general"],
            "cost_per_1m_input": 0.0,
            "cost_per_1m_output": 0.0,
            "max_concurrent": 1,
            "verified": False,
            "verified_at": None,
            "enabled": False,
            "tags": []
        }
    ],
    "selection": {
        "default_strategy": "balanced",
        "task_mapping": {
            "coding": "cheapest",
            "reasoning": "most_capable",
            "quick": "cheapest",
            "general": "balanced"
        },
        "strategies": {
            "cheapest": {"sort_by": "cost_per_1m_input", "ascending": True},
            "most_capable": {"sort_by": "cost_per_1m_input", "ascending": False},
            "balanced": {"weights": {"cost": 0.5, "capability_match": 0.5}}
        }
    }
}


@pytest.fixture
def config_path(tmp_path):
    """Write JSON config to a temp file."""
    p = tmp_path / "llm_pool.json"
    p.write_text(json.dumps(MINIMAL_JSON, indent=2))
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
    """JSON config loads correctly, disabled providers excluded."""
    from src.llm_pool import PoolConfigJSON
    cfg = PoolConfigJSON.from_json(config_path)
    assert len(cfg.providers) == 3
    enabled = cfg.get_enabled()
    assert len(enabled) == 2
    ids = {p.id for p in enabled}
    assert "cheap-model" in ids
    assert "powerful-model" in ids
    assert "disabled-model" not in ids


def test_pool_config_cost_fields(config_path):
    """Cost fields parsed correctly from JSON."""
    from src.llm_pool import PoolConfigJSON
    cfg = PoolConfigJSON.from_json(config_path)
    cheap = next(p for p in cfg.providers if p.id == "cheap-model")
    assert cheap.cost_per_1m_input == 0.10
    assert cheap.cost_per_1m_output == 0.20
    assert cheap.max_concurrent == 5


def test_pool_config_api_key_source(config_path):
    """API key source parsed correctly."""
    from src.llm_pool import PoolConfigJSON
    cfg = PoolConfigJSON.from_json(config_path)
    cheap = next(p for p in cfg.providers if p.id == "cheap-model")
    assert cheap.api_key_source == "env:TEST_API_KEY"
    # No plaintext keys in config
    assert not hasattr(cheap, "api_key")


# ── auth resolver ──────────────────────────────────────────────────────────

def test_auth_resolver_env_var():
    from src.llm_pool import AuthResolver
    os.environ["_TEST_KEY_ABC"] = "secret-value"
    try:
        resolver = AuthResolver()
        result = resolver.resolve("env:_TEST_KEY_ABC")
        assert result["api_key"] == "secret-value"
    finally:
        os.environ.pop("_TEST_KEY_ABC", None)


def test_auth_resolver_none():
    from src.llm_pool import AuthResolver
    resolver = AuthResolver()
    result = resolver.resolve("none")
    assert result == {}


def test_auth_resolver_missing_key():
    from src.llm_pool import AuthResolver
    resolver = AuthResolver()
    result = resolver.resolve("env:_NONEXISTENT_KEY_XYZ")
    # Returns None api_key, not an error (checked at validation time)
    assert result.get("api_key") is None


def test_auth_validator_ok():
    from src.llm_pool import AuthResolver
    os.environ["_VALIDATE_TEST_KEY"] = "val"
    try:
        resolver = AuthResolver()
        ok, reason = resolver.validate("env:_VALIDATE_TEST_KEY")
        assert ok
    finally:
        os.environ.pop("_VALIDATE_TEST_KEY", None)


def test_auth_validator_fail():
    from src.llm_pool import AuthResolver
    resolver = AuthResolver()
    ok, reason = resolver.validate("env:_NONEXISTENT_KEY_XYZ_123")
    assert not ok


# ── circuit breaker ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_failures():
    from src.llm_pool import CircuitBreaker
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout_s=999)
    assert cb.state == "closed"
    for _ in range(3):
        await cb.record_failure()
    assert cb.state == "open"
    assert not cb.is_available()


@pytest.mark.asyncio
async def test_circuit_breaker_resets_on_success():
    from src.llm_pool import CircuitBreaker
    cb = CircuitBreaker("test", failure_threshold=3, recovery_timeout_s=999)
    await cb.record_failure()
    await cb.record_failure()
    await cb.record_success()
    assert cb.state == "closed"


@pytest.mark.asyncio
async def test_circuit_breaker_half_open_recovery():
    from src.llm_pool import CircuitBreaker
    import time
    cb = CircuitBreaker("test", failure_threshold=2, recovery_timeout_s=0.01)
    await cb.record_failure()
    await cb.record_failure()
    assert cb.state == "open"
    await asyncio.sleep(0.05)
    allowed = await cb.acquire()
    assert allowed
    assert cb.state == "half_open"
    await cb.record_success()
    assert cb.state == "closed"


# ── pool selection ─────────────────────────────────────────────────────────

def test_pool_selects_cheapest(pool):
    cfg = pool.select(capabilities=["general"], strategy="cheapest")
    assert cfg is not None
    assert cfg.id == "cheap-model"


def test_pool_selects_most_capable(pool):
    cfg = pool.select(capabilities=["general"], strategy="most_capable")
    assert cfg is not None
    assert cfg.id == "powerful-model"


def test_pool_filters_by_capability(pool):
    cfg = pool.select(capabilities=["vision"], strategy="cheapest")
    assert cfg is not None
    assert cfg.id == "powerful-model"


def test_pool_returns_none_when_no_match(pool):
    # No provider has "nonexistent" capability
    cfg = pool.select(capabilities=["nonexistent_capability_xyz"])
    # Falls back to any available provider
    assert cfg is not None  # Fallback selects any
    assert cfg.id in ("cheap-model", "powerful-model")


def test_pool_task_type_routing(pool):
    cfg_coding = pool.select(task_type="coding")
    assert cfg_coding is not None
    assert cfg_coding.id == "cheap-model"  # coding → cheapest

    cfg_reasoning = pool.select(task_type="reasoning")
    assert cfg_reasoning is not None
    assert cfg_reasoning.id == "powerful-model"  # reasoning → most_capable


# ── pool acquire with mock ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pool_acquire_and_chat(pool):
    from src.llm_pool import PoolManagedProvider
    from src.loop_engine import LLMResponse

    provider = await pool.acquire(capabilities=["coding"], strategy="cheapest")
    assert isinstance(provider, PoolManagedProvider)
    assert provider.provider_id == "cheap-model"

    # Mock the internal client
    provider._client = AsyncMock()
    provider._client.chat.completions.create = AsyncMock()
    mock_choice = AsyncMock()
    mock_choice.message.content = "def hello(): pass"
    mock_usage = AsyncMock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 5
    provider._client.chat.completions.create.return_value = AsyncMock(
        choices=[mock_choice],
        usage=mock_usage,
    )

    result = await provider.chat([{"role": "user", "content": "write hello world"}])
    assert "hello" in result.content


@pytest.mark.asyncio
async def test_pool_circuit_breaker_blocks_after_failures(pool):
    cb = pool._circuit_breakers["cheap-model"]
    for _ in range(5):
        await cb.record_failure()
    assert cb.state == "open"

    provider = pool._providers["cheap-model"]
    with pytest.raises(RuntimeError, match="circuit breaker is OPEN"):
        await provider.chat([{"role": "user", "content": "test"}])


# ── usage tracker ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tracker_records_usage(tmp_path):
    from src.llm_pool import PoolTracker, UsageRecord
    log_path = tmp_path / "usage.jsonl"
    tracker = PoolTracker(usage_log_path=log_path)

    await tracker.record(UsageRecord(
        timestamp="2026-01-01T00:00:00+00:00",
        provider_id="test-provider",
        model="test-model",
        input_tokens=100,
        output_tokens=50,
        latency_ms=123.4,
        success=True,
    ))

    stats = tracker.get_stats("test-provider")
    assert stats["call_count"] == 1
    assert stats["success_rate"] == 1.0
    assert stats["total_input_tokens"] == 100

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["provider_id"] == "test-provider"


@pytest.mark.asyncio
async def test_tracker_failure_rate(tmp_path):
    from src.llm_pool import PoolTracker, UsageRecord
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
        assert "id" in s
        # No secrets in status output
        assert "api_key" not in s
        assert "secret" not in str(s).lower()


# ── JSON CRUD ──────────────────────────────────────────────────────────────

def test_pool_add_provider(config_path, tmp_path):
    from src.llm_pool import LLMPool, ProviderConfigJSON
    pool = LLMPool(config_path=config_path, state_dir=tmp_path / "state")
    pool.initialize()

    new_provider = ProviderConfigJSON(
        id="new-model",
        type="openai",
        endpoint="http://localhost:8080/v1",
        model="new-model-v1",
        capabilities=["general"],
        cost_per_1m_input=0.05,
        max_concurrent=2,
    )
    pool.add(new_provider)

    # Verify it was saved
    cfg2 = pool._pool_config
    ids = [p.id for p in cfg2.providers]
    assert "new-model" in ids


def test_pool_remove_provider(pool, config_path):
    pool.remove("cheap-model")
    ids = [p.id for p in pool._pool_config.providers]
    assert "cheap-model" not in ids
    assert "cheap-model" not in pool._providers


def test_pool_update_provider(pool):
    result = pool.update("cheap-model", verified=True, cost_per_1m_input=0.08)
    assert result is True
    cheap = next(p for p in pool._pool_config.providers if p.id == "cheap-model")
    assert cheap.verified is True
    assert cheap.cost_per_1m_input == 0.08


def test_pool_list_providers(pool):
    providers = pool.list_providers()
    assert len(providers) == 3
    assert isinstance(providers, list)
    for p in providers:
        assert "id" in p
        assert "capabilities" in p


def test_pool_get_available(pool):
    # None are verified yet
    available = pool.get_available()
    assert len(available) == 0

    # Verify one
    pool.update("cheap-model", verified=True)
    available = pool.get_available()
    assert len(available) == 1
    assert available[0]["id"] == "cheap-model"
