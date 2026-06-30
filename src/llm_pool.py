"""LLM Pool v3 — JSON-configured multi-provider management (single-file, no subpackage).

Architecture:
  LLMPool
    ├── JSON config loading (config/llm_pool.json)
    ├── AuthResolver (inline, env_var / none)
    ├── CircuitBreaker (inline, simplified)
    ├── PoolTracker (inline, usage.jsonl append)
    └── ProviderAdapter (inline, OpenAI-compatible only)

Usage:
    pool = LLMPool("config/llm_pool.json")
    provider = await pool.acquire(capabilities=["coding"], strategy="cheapest")
    result = await provider.chat([{"role": "user", "content": "..."}])

Or:
    async with pool.use(capabilities=["reasoning"]) as provider:
        result = await provider.chat([...])
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

from .loop_engine import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)

# ============================================================
# JSON Config models
# ============================================================


@dataclass
class ProviderConfigJSON:
    """A single provider entry from llm_pool.json."""
    id: str
    type: str = "openai"
    endpoint: str = ""
    model: str = ""
    api_key_source: str = "none"  # "none" | "env:VAR_NAME"
    capabilities: list[str] = field(default_factory=list)
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0
    max_concurrent: int = 1
    verified: bool = False
    verified_at: str | None = None
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "ProviderConfigJSON":
        return cls(
            id=data["id"],
            type=data.get("type", "openai"),
            endpoint=data.get("endpoint", ""),
            model=data.get("model", ""),
            api_key_source=data.get("api_key_source", "none"),
            capabilities=data.get("capabilities", []),
            cost_per_1m_input=data.get("cost_per_1m_input", 0.0),
            cost_per_1m_output=data.get("cost_per_1m_output", 0.0),
            max_concurrent=data.get("max_concurrent", 1),
            verified=data.get("verified", False),
            verified_at=data.get("verified_at"),
            enabled=data.get("enabled", True),
            tags=data.get("tags", []),
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "endpoint": self.endpoint,
            "model": self.model,
            "api_key_source": self.api_key_source,
            "capabilities": self.capabilities,
            "cost_per_1m_input": self.cost_per_1m_input,
            "cost_per_1m_output": self.cost_per_1m_output,
            "max_concurrent": self.max_concurrent,
            "verified": self.verified,
            "verified_at": self.verified_at,
            "enabled": self.enabled,
            "tags": self.tags,
        }


@dataclass
class SelectionConfig:
    """Selection configuration from llm_pool.json."""
    default_strategy: str = "balanced"
    task_mapping: dict[str, str] = field(default_factory=lambda: {
        "reasoning": "most_capable",
        "coding": "cheapest",
        "quick": "cheapest",
        "general": "balanced",
    })
    strategies: dict[str, dict] = field(default_factory=lambda: {
        "cheapest": {"sort_by": "cost_per_1m_input", "ascending": True},
        "most_capable": {"sort_by": "cost_per_1m_input", "ascending": False},
        "balanced": {"weights": {"cost": 0.5, "capability_match": 0.5}},
    })


@dataclass
class PoolConfigJSON:
    """Full pool configuration loaded from llm_pool.json."""
    providers: list[ProviderConfigJSON] = field(default_factory=list)
    selection: SelectionConfig = field(default_factory=SelectionConfig)

    @classmethod
    def from_json(cls, path: str | Path) -> "PoolConfigJSON":
        with open(path) as f:
            data = json.load(f)
        providers = [ProviderConfigJSON.from_dict(p) for p in data.get("providers", [])]
        sel_data = data.get("selection", {})
        selection = SelectionConfig(
            default_strategy=sel_data.get("default_strategy", "balanced"),
            task_mapping=sel_data.get("task_mapping", {}),
            strategies=sel_data.get("strategies", {}),
        )
        return cls(providers=providers, selection=selection)

    def get_enabled(self) -> list[ProviderConfigJSON]:
        return [p for p in self.providers if p.enabled]

    def get_available(self) -> list[ProviderConfigJSON]:
        """Return verified AND enabled providers (for AgentManager queries)."""
        return [p for p in self.providers if p.enabled and p.verified]

    def to_dict(self) -> dict:
        return {
            "providers": [p.to_dict() for p in self.providers],
            "selection": {
                "default_strategy": self.selection.default_strategy,
                "task_mapping": self.selection.task_mapping,
                "strategies": self.selection.strategies,
            },
        }

    def save(self, path: str | Path) -> None:
        """Write config back to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
            f.write("\n")


# ============================================================
# AuthResolver (inline, simplified)
# ============================================================


class AuthResolver:
    """Resolve API keys from environment variables."""

    def resolve(self, api_key_source: str) -> dict[str, Any]:
        """Resolve API key from source specification.

        "none" → no auth
        "env:VAR_NAME" → read from environment variable
        """
        if api_key_source == "none" or not api_key_source:
            return {}

        if api_key_source.startswith("env:"):
            var_name = api_key_source[4:]
            api_key = os.environ.get(var_name)
            if api_key is None:
                # Try dotenv (from workspace root .env)
                dotenv_path = Path.home() / ".openclaw" / ".env"
                if dotenv_path.exists():
                    for line in dotenv_path.read_text().splitlines():
                        line = line.strip()
                        if line.startswith("#") or "=" not in line:
                            continue
                        key, _, val = line.partition("=")
                        val = val.strip().strip('"').strip("'")
                        if key.strip() == var_name:
                            api_key = val
                            break
            return {"api_key": api_key}

        return {}

    def validate(self, api_key_source: str) -> tuple[bool, str]:
        """Check if required credentials are available."""
        if api_key_source == "none" or not api_key_source:
            return True, "no auth required"

        if api_key_source.startswith("env:"):
            var_name = api_key_source[4:]
            if var_name in os.environ:
                return True, "ok"
            dotenv_path = Path.home() / ".openclaw" / ".env"
            if dotenv_path.exists():
                content = dotenv_path.read_text()
                if var_name + "=" in content:
                    return True, "ok"
            return False, f"${var_name} not set"

        return False, f"unknown source: {api_key_source}"


# ============================================================
# CircuitBreaker (inline, simplified)
# ============================================================


class CircuitBreaker:
    """Simplified per-provider circuit breaker.

    States: CLOSED → OPEN (after N failures) → HALF_OPEN → CLOSED
    """

    OPEN = "open"
    CLOSED = "closed"
    HALF_OPEN = "half_open"

    def __init__(self, provider_id: str, failure_threshold: int = 5, recovery_timeout_s: float = 60.0):
        self.provider_id = provider_id
        self.failure_threshold = failure_threshold
        self.recovery_timeout_s = recovery_timeout_s
        self._state = self.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._half_open_in_flight = 0
        self._lock = asyncio.Lock()

    @property
    def state(self) -> str:
        return self._state

    def is_available(self) -> bool:
        if self._state == self.CLOSED:
            return True
        if self._state == self.OPEN and self._opened_at:
            if time.monotonic() - self._opened_at >= self.recovery_timeout_s:
                return True
        if self._state == self.HALF_OPEN:
            return self._half_open_in_flight < 2
        return False

    async def acquire(self) -> bool:
        async with self._lock:
            if self._state == self.CLOSED:
                return True
            if self._state == self.OPEN and self._opened_at:
                elapsed = time.monotonic() - self._opened_at
                if elapsed >= self.recovery_timeout_s:
                    self._state = self.HALF_OPEN
                    self._half_open_in_flight = 0
                    self._half_open_in_flight += 1
                    return True
            if self._state == self.HALF_OPEN:
                if self._half_open_in_flight < 2:
                    self._half_open_in_flight += 1
                    return True
            return False

    async def record_success(self) -> None:
        async with self._lock:
            self._consecutive_failures = 0
            self._state = self.CLOSED
            self._half_open_in_flight = max(0, self._half_open_in_flight - 1)

    async def record_failure(self) -> None:
        async with self._lock:
            self._consecutive_failures += 1
            if self._state == self.HALF_OPEN:
                self._state = self.OPEN
                self._opened_at = time.monotonic()
                self._half_open_in_flight = max(0, self._half_open_in_flight - 1)
            elif self._consecutive_failures >= self.failure_threshold:
                self._state = self.OPEN
                self._opened_at = time.monotonic()

    def status(self) -> dict:
        return {
            "provider_id": self.provider_id,
            "state": self._state,
            "consecutive_failures": self._consecutive_failures,
        }


# ============================================================
# PoolTracker (inline, simplified)
# ============================================================


@dataclass
class UsageRecord:
    """A single LLM call record."""
    timestamp: str
    provider_id: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "provider_id": self.provider_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": round(self.latency_ms, 1),
            "success": self.success,
            "error": self.error,
        }


class PoolTracker:
    """In-memory stats tracker with JSONL persistence."""

    def __init__(self, usage_log_path: Path | None = None):
        self._usage_log_path = usage_log_path
        self._lock = asyncio.Lock()
        self._stats: dict[str, dict] = {}

    def _ensure(self, provider_id: str) -> dict:
        if provider_id not in self._stats:
            self._stats[provider_id] = {
                "call_count": 0, "success_count": 0,
                "total_latency_ms": 0.0,
                "total_input_tokens": 0, "total_output_tokens": 0,
            }
        return self._stats[provider_id]

    async def record(self, record: UsageRecord) -> None:
        async with self._lock:
            s = self._ensure(record.provider_id)
            s["call_count"] += 1
            if record.success:
                s["success_count"] += 1
            s["total_latency_ms"] += record.latency_ms
            s["total_input_tokens"] += record.input_tokens
            s["total_output_tokens"] += record.output_tokens

            if self._usage_log_path:
                try:
                    self._usage_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self._usage_log_path, "a") as f:
                        f.write(json.dumps(record.to_dict()) + "\n")
                except Exception as e:
                    logger.warning("Failed to write usage log: %s", e)

    def get_stats(self, provider_id: str | None = None) -> dict:
        if provider_id:
            s = self._stats.get(provider_id, {})
            return self._format(provider_id, s)
        return {pid: self._format(pid, s) for pid, s in self._stats.items()}

    def _format(self, pid: str, s: dict) -> dict:
        calls = s.get("call_count", 0)
        successes = s.get("success_count", 0)
        return {
            "provider_id": pid,
            "call_count": calls,
            "success_rate": round(successes / calls, 3) if calls > 0 else 1.0,
            "avg_latency_ms": round(s.get("total_latency_ms", 0) / calls, 1) if calls > 0 else 0,
            "total_input_tokens": s.get("total_input_tokens", 0),
            "total_output_tokens": s.get("total_output_tokens", 0),
        }


# ============================================================
# PoolManagedProvider
# ============================================================


class PoolManagedProvider(LLMProvider):
    """Wraps an OpenAI-compatible client with pool bookkeeping."""

    def __init__(self, cfg: ProviderConfigJSON, api_key: str | None,
                 semaphore: asyncio.Semaphore,
                 circuit_breaker: CircuitBreaker,
                 tracker: PoolTracker,
                 cost_controller: Any | None = None):
        self.provider_id = cfg.id
        self._cfg = cfg
        self._api_key = api_key or "not-needed"
        self._endpoint = cfg.endpoint
        self._model = cfg.model
        self._max_output = 8192
        self._timeout = 120.0
        self._semaphore = semaphore
        self._cb = circuit_breaker
        self._tracker = tracker
        self._cost_controller = cost_controller
        self._client: Any = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise RuntimeError("openai package required: pip install openai")
        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._endpoint,
            timeout=self._timeout,
            max_retries=0,
        )

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        if not await self._cb.acquire():
            raise RuntimeError(
                f"Provider '{self.provider_id}' circuit breaker is OPEN. Try another provider."
            )

        start = time.monotonic()
        success = False
        input_tokens = 0
        output_tokens = 0
        error_msg = None
        model = kwargs.pop("model", None) or self._model

        # ── Cost check (before call, optional) ────────────────
        if self._cost_controller:
            # Estimate cost from config rates + estimated tokens
            estimated = (
                self._cfg.cost_per_1m_input * 0.001 +
                self._cfg.cost_per_1m_output * 0.002
            )  # rough estimate ~1k input + 2k output
            if not self._cost_controller.check(estimated):
                remaining = self._cost_controller.get_remaining()
                raise RuntimeError(
                    f"CostController: budget exceeded. "
                    f"Daily: ${remaining['daily_spent']:.2f}/${remaining['daily_limit']:.2f}, "
                    f"Monthly: ${remaining['monthly_spent']:.2f}/${remaining['monthly_limit']:.2f}"
                )

        try:
            async with self._semaphore:
                self._ensure_client()
                response = await asyncio.wait_for(
                    self._client.chat.completions.create(
                        model=model,
                        messages=messages,
                        max_tokens=kwargs.pop("max_tokens", self._max_output),
                        temperature=kwargs.pop("temperature", 0.7),
                        **kwargs,
                    ),
                    timeout=self._timeout,
                )
            content = response.choices[0].message.content or ""
            usage = response.usage
            input_tokens = usage.prompt_tokens if usage else 0
            output_tokens = usage.completion_tokens if usage else 0
            success = True
            await self._cb.record_success()
            return LLMResponse(
                content=content,
                model=model,
                usage={"input_tokens": input_tokens, "output_tokens": output_tokens},
            )
        except Exception as e:
            error_msg = str(e)
            await self._cb.record_failure()
            raise
        finally:
            latency_ms = (time.monotonic() - start) * 1000
            # ── Cost record (after call, optional) ────────────
            if self._cost_controller and success:
                actual_cost = (
                    self._cfg.cost_per_1m_input * input_tokens / 1_000_000 +
                    self._cfg.cost_per_1m_output * output_tokens / 1_000_000
                )
                self._cost_controller.record(
                    actual_cost=actual_cost,
                    provider_id=self.provider_id,
                    tokens_in=input_tokens,
                    tokens_out=output_tokens,
                )
            await self._tracker.record(UsageRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                provider_id=self.provider_id,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                success=success,
                error=error_msg,
            ))

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        self._ensure_client()
        texts = [text] if isinstance(text, str) else text
        response = await self._client.embeddings.create(
            model=self._model, input=texts,
        )
        return [item.embedding for item in response.data]


# ============================================================
# LLMPool — main entry point
# ============================================================


class LLMPool:
    """Multi-provider LLM pool with JSON config + strategy-based selection."""

    def __init__(self, config_path: str | Path = "config/llm_pool.json",
                 state_dir: str | Path = "state",
                 cost_controller: Any | None = None):
        self._config_path = Path(config_path)
        self._state_dir = Path(state_dir)
        self._pool_config: PoolConfigJSON | None = None
        self._providers: dict[str, PoolManagedProvider] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._auth_resolver = AuthResolver()
        self._tracker = PoolTracker(
            usage_log_path=self._state_dir / "llm_pool" / "usage.jsonl"
        )
        self._cost_controller = cost_controller
        self._initialized = False

    def initialize(self) -> None:
        """Load config and instantiate all enabled providers (sync)."""
        if self._initialized:
            return
        self._pool_config = PoolConfigJSON.from_json(self._config_path)
        self._build_providers()
        self._initialized = True
        logger.info("LLMPool initialized: %d providers", len(self._providers))

    def _build_providers(self) -> None:
        if not self._pool_config:
            return
        for cfg in self._pool_config.get_enabled():
            # Validate auth
            ok, reason = self._auth_resolver.validate(cfg.api_key_source)
            if not ok:
                logger.warning("Provider '%s' skipped: auth failed (%s)", cfg.id, reason)
                continue
            auth_kwargs = self._auth_resolver.resolve(cfg.api_key_source)
            api_key = auth_kwargs.get("api_key") if auth_kwargs else None

            sem = asyncio.Semaphore(cfg.max_concurrent)
            cb = CircuitBreaker(provider_id=cfg.id)

            self._semaphores[cfg.id] = sem
            self._circuit_breakers[cfg.id] = cb
            self._providers[cfg.id] = PoolManagedProvider(
                cfg=cfg, api_key=api_key,
                semaphore=sem, circuit_breaker=cb, tracker=self._tracker,
                cost_controller=self._cost_controller,
            )

    # ── Selection ──────────────────────────────────────────────────

    def _resolve_strategy(self, strategy: str | None, task_type: str | None) -> str:
        if strategy:
            return strategy
        if task_type and self._pool_config:
            return self._pool_config.selection.task_mapping.get(
                task_type, self._pool_config.selection.default_strategy
            )
        if self._pool_config:
            return self._pool_config.selection.default_strategy
        return "balanced"

    def select(self, capabilities: list[str] | None = None,
               strategy: str | None = None,
               task_type: str | None = None) -> ProviderConfigJSON | None:
        """Select the best provider config without acquiring it."""
        if not self._initialized:
            self.initialize()
        if not self._pool_config:
            return None

        strat_name = self._resolve_strategy(strategy, task_type)
        strat_cfg = self._pool_config.selection.strategies.get(strat_name, {})

        # Filter candidates
        candidates: list[ProviderConfigJSON] = []
        for cfg in self._pool_config.get_enabled():
            if cfg.id not in self._providers:
                continue
            if capabilities:
                if not all(c in cfg.capabilities for c in capabilities):
                    continue
            cb = self._circuit_breakers.get(cfg.id)
            if cb and not cb.is_available():
                continue
            candidates.append(cfg)

        if not candidates:
            # Fallback: try without capability filter
            for cfg in self._pool_config.get_enabled():
                if cfg.id in self._providers:
                    cb = self._circuit_breakers.get(cfg.id)
                    if not cb or cb.is_available():
                        candidates.append(cfg)
            if not candidates:
                return None

        # Sort by strategy
        sort_by = strat_cfg.get("sort_by")
        ascending = strat_cfg.get("ascending", True)
        weights = strat_cfg.get("weights")

        if weights:
            all_enabled = self._pool_config.get_enabled()
            max_cost = max((p.cost_per_1m_input for p in all_enabled), default=1) or 1

            def _balanced_score(c: ProviderConfigJSON) -> float:
                cost_score = c.cost_per_1m_input / max_cost
                cap_score = 0.5
                w = weights
                return w.get("cost", 0.5) * cost_score + w.get("capability_match", 0.5) * cap_score

            candidates.sort(key=_balanced_score)
        elif sort_by:
            reverse = not ascending
            candidates.sort(
                key=lambda c: getattr(c, sort_by, 0) or 0,
                reverse=reverse,
            )

        return candidates[0] if candidates else None

    async def acquire(self, capabilities: list[str] | None = None,
                      strategy: str | None = None,
                      task_type: str | None = None,
                      cost_controller: Any | None = None) -> PoolManagedProvider:
        """Acquire the best matching provider."""
        if not self._initialized:
            self.initialize()
        cfg = self.select(capabilities=capabilities, strategy=strategy, task_type=task_type)
        if not cfg:
            avail = list(self._providers.keys())
            raise RuntimeError(
                f"No provider available for capabilities={capabilities} "
                f"strategy={strategy}. Available: {avail}"
            )
        return self._providers[cfg.id]

    @asynccontextmanager
    async def use(self, capabilities: list[str] | None = None,
                  strategy: str | None = None,
                  task_type: str | None = None) -> AsyncIterator[PoolManagedProvider]:
        """Context manager: acquire + auto-release."""
        provider = await self.acquire(capabilities=capabilities, strategy=strategy, task_type=task_type)
        try:
            yield provider
        finally:
            pass  # Semaphore released by chat() context

    # ── CRUD ───────────────────────────────────────────────────────

    def add(self, provider: ProviderConfigJSON) -> None:
        """Add a new provider and persist to JSON."""
        if not self._pool_config:
            self.initialize()
        self._pool_config.providers.append(provider)
        self._pool_config.save(self._config_path)
        logger.info("LLMPool: added provider '%s'", provider.id)

    def remove(self, provider_id: str) -> None:
        """Remove a provider and persist to JSON."""
        if not self._pool_config:
            self.initialize()
        self._pool_config.providers = [
            p for p in self._pool_config.providers if p.id != provider_id
        ]
        self._pool_config.save(self._config_path)
        if provider_id in self._providers:
            del self._providers[provider_id]
        logger.info("LLMPool: removed provider '%s'", provider_id)

    def update(self, provider_id: str, **kwargs) -> bool:
        """Update provider fields and persist to JSON."""
        if not self._pool_config:
            self.initialize()
        for p in self._pool_config.providers:
            if p.id == provider_id:
                for key, value in kwargs.items():
                    if hasattr(p, key):
                        setattr(p, key, value)
                self._pool_config.save(self._config_path)
                logger.info("LLMPool: updated provider '%s'", provider_id)
                return True
        return False

    def list_providers(self) -> list[dict]:
        """List all configured providers."""
        if not self._pool_config:
            self.initialize()
        return [p.to_dict() for p in (self._pool_config.providers if self._pool_config else [])]

    async def verify(self, provider_id: str) -> bool:
        """Test a provider with a real LLM call, mark verified on success.

        Uses the provider's chat() method (OpenAI client) as primary path.
        Falls back to direct HTTP (aiohttp/httpx) if the provider's chat() is unavailable.

        Steps:
          1. Get provider config from pool
          2. Resolve API Key from environment variable
          3. Send a simple chat request ("Say hello")
          4. Success → verified=true, verified_at=now
          5. Failure → verified=false, log error
          6. Write results back to config JSON
        """
        if not self._pool_config:
            self.initialize()

        cfg = None
        for p in (self._pool_config.providers if self._pool_config else []):
            if p.id == provider_id:
                cfg = p
                break
        if not cfg:
            raise ValueError(f"Provider '{provider_id}' not found")

        # Try primary: use the provider's chat() method (OpenAI client)
        provider = self._providers.get(provider_id)
        if not provider:
            self._build_providers()
            provider = self._providers.get(provider_id)

        if provider:
            try:
                response = await provider.chat([
                    {"role": "user", "content": "Reply with only the word: OK"}
                ])
                if response and response.content.strip():
                    cfg.verified = True
                    cfg.verified_at = datetime.now(timezone.utc).isoformat()
                    self._pool_config.save(self._config_path)
                    logger.info("LLMPool: verified provider '%s'", provider_id)
                    return True
            except Exception as e:
                logger.debug("LLMPool: primary verify '%s' failed: %s", provider_id, e)

        # Fallback: direct HTTP verification
        logger.debug("LLMPool: trying direct HTTP verify for '%s'", provider_id)
        success, error = await self._verify_via_http(cfg)

        cfg.verified = success
        cfg.verified_at = datetime.now(timezone.utc).isoformat() if success else None
        self._pool_config.save(self._config_path)

        if success:
            logger.info("LLMPool: verified provider '%s' via HTTP", provider_id)
        else:
            logger.warning("LLMPool: verify '%s' failed via HTTP: %s", provider_id, error)

        return success

    async def _verify_via_http(self, cfg: ProviderConfigJSON) -> tuple[bool, str | None]:
        """Verify a provider by making a direct HTTP call to its endpoint.

        Uses httpx if available, aiohttp as fallback.

        Returns (success, error_message).
        """
        endpoint = cfg.endpoint.rstrip("/")
        api_key = self._auth_resolver.resolve(cfg.api_key_source).get("api_key")

        payload = {
            "model": cfg.model,
            "messages": [{"role": "user", "content": "Say hello"}],
            "max_tokens": 10,
        }
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Try httpx first (project dependency)
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    f"{endpoint}/chat/completions",
                    json=payload,
                    headers=headers,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content.strip():
                        return True, None
                    return False, f"Empty response from {cfg.id}"
                return False, f"HTTP {resp.status_code}: {resp.text[:200]}"
        except ImportError:
            pass
        except Exception as e:
            logger.debug("LLMPool: httpx verify '%s' failed: %s", cfg.id, e)

        # Try aiohttp as fallback
        try:
            import aiohttp
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=15)
            ) as session:
                async with session.post(
                    f"{endpoint}/chat/completions",
                    json=payload,
                    headers=headers,
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                        if content.strip():
                            return True, None
                        return False, f"Empty response from {cfg.id}"
                    text = await resp.text()
                    return False, f"HTTP {resp.status}: {text[:200]}"
        except ImportError:
            pass
        except Exception as e:
            logger.debug("LLMPool: aiohttp verify '%s' failed: %s", cfg.id, e)

        return False, "No HTTP client available (install httpx or aiohttp)"

    def get_available(self) -> list[dict]:
        """Return verified + enabled providers list (for AgentManager queries)."""
        if not self._pool_config:
            self.initialize()
        return [
            p.to_dict() for p in (self._pool_config.providers if self._pool_config else [])
            if p.enabled and p.verified
        ]

    # ── Introspection ──────────────────────────────────────────────

    def provider_status(self) -> list[dict]:
        """Return status of all configured providers."""
        if not self._pool_config:
            return []
        result = []
        for cfg in self._pool_config.providers:
            cb = self._circuit_breakers.get(cfg.id)
            in_pool = cfg.id in self._providers
            stats = self._tracker.get_stats(cfg.id) if in_pool else {}
            result.append({
                "id": cfg.id,
                "type": cfg.type,
                "capabilities": cfg.capabilities,
                "enabled": cfg.enabled,
                "verified": cfg.verified,
                "verified_at": cfg.verified_at,
                "cost_per_1m_input": cfg.cost_per_1m_input,
                "max_concurrent": cfg.max_concurrent,
                "tags": cfg.tags,
                "in_pool": in_pool,
                "circuit_breaker": cb.status() if cb else None,
                "stats": stats,
            })
        return result

    def usage_stats(self) -> dict:
        """Return usage statistics for all providers."""
        return self._tracker.get_stats()
