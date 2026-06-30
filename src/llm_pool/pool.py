"""LLMPool — the main entry point for provider selection and lifecycle.

Usage:
    pool = LLMPool("config/llm_pool.yaml")
    provider = await pool.acquire(capabilities=["coding"], strategy="cheapest")
    try:
        result = await provider.chat([...])
    finally:
        await pool.release(provider.provider_id)

Or as context manager:
    async with pool.use(capabilities=["reasoning"], strategy="most_capable") as provider:
        result = await provider.chat([...])
"""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from ..loop_engine import LLMProvider, LLMResponse
from .auth import AuthResolver
from .circuit_breaker import CircuitBreaker, CBState
from .config import PoolConfig, ProviderConfig, StrategyConfig
from .protocols import create_adapter, ProtocolAdapter
from .tracker import PoolTracker, UsageRecord

logger = logging.getLogger(__name__)


class PoolManagedProvider(LLMProvider):
    """Wraps a protocol adapter with pool bookkeeping:
    - Concurrency semaphore
    - Circuit breaker integration
    - Usage tracking
    """

    def __init__(self, config: ProviderConfig, adapter: ProtocolAdapter,
                 semaphore: asyncio.Semaphore, circuit_breaker: CircuitBreaker,
                 tracker: PoolTracker):
        self.provider_id = config.id
        self._config = config
        self._adapter = adapter
        self._semaphore = semaphore
        self._cb = circuit_breaker
        self._tracker = tracker

    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        if not await self._cb.acquire():
            raise RuntimeError(
                f"Provider '{self.provider_id}' circuit breaker is OPEN "
                f"(state={self._cb.state.value}). Try another provider."
            )

        start = time.monotonic()
        success = False
        input_tokens = 0
        output_tokens = 0
        error_msg = None

        try:
            async with self._semaphore:
                result = await self._adapter.chat(messages, **kwargs)
            success = True
            input_tokens = result.usage.get("input_tokens", 0)
            output_tokens = result.usage.get("output_tokens", 0)
            await self._cb.record_success()
            return result

        except Exception as e:
            error_msg = str(e)
            await self._cb.record_failure()
            raise

        finally:
            latency_ms = (time.monotonic() - start) * 1000
            await self._tracker.record(UsageRecord(
                timestamp=datetime.now(timezone.utc).isoformat(),
                provider_id=self.provider_id,
                model=self._config.endpoint.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                success=success,
                error=error_msg,
            ))

    async def embed(self, text) -> list[list[float]]:
        return await self._adapter.embed(text)


class LLMPool:
    """Multi-provider LLM pool with strategy-based selection.

    Lifecycle:
      1. __init__: load config
      2. acquire(capabilities, strategy) → PoolManagedProvider
      3. use the provider
      4. release(provider_id)  OR  use async context manager
    """

    def __init__(self, config_path: str | Path = "config/llm_pool.yaml",
                 state_dir: str | Path = "state"):
        self._config_path = Path(config_path)
        self._state_dir = Path(state_dir)
        self._pool_config: PoolConfig | None = None
        self._providers: dict[str, PoolManagedProvider] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._circuit_breakers: dict[str, CircuitBreaker] = {}
        self._auth_resolver = AuthResolver()
        self._tracker = PoolTracker(
            usage_log_path=self._state_dir / "llm_pool" / "usage.jsonl"
        )
        self._initialized = False

    def initialize(self) -> None:
        """Load config and instantiate all enabled providers (sync)."""
        if self._initialized:
            return

        self._pool_config = PoolConfig.from_yaml(self._config_path)
        self._build_providers()
        self._initialized = True
        logger.info(
            "LLMPool initialized: %d providers enabled",
            len(self._providers)
        )

    def _build_providers(self) -> None:
        for cfg in self._pool_config.get_enabled():
            # Validate auth
            ok, reason = self._auth_resolver.validate_required(cfg.auth)
            if not ok:
                logger.warning(
                    "Provider '%s' skipped: auth validation failed (%s)",
                    cfg.id, reason
                )
                continue

            # Resolve credentials
            auth_kwargs = self._auth_resolver.resolve(cfg.auth)

            # Create protocol adapter
            try:
                adapter = create_adapter(cfg, auth_kwargs)
            except Exception as e:
                logger.warning("Provider '%s' adapter creation failed: %s", cfg.id, e)
                continue

            # Semaphore + Circuit breaker
            sem = asyncio.Semaphore(cfg.limits.max_concurrent)
            cb = CircuitBreaker(
                provider_id=cfg.id,
                failure_threshold=cfg.circuit_breaker.failure_threshold,
                recovery_timeout_s=cfg.circuit_breaker.recovery_timeout_s,
                half_open_requests=cfg.circuit_breaker.half_open_requests,
            )

            self._semaphores[cfg.id] = sem
            self._circuit_breakers[cfg.id] = cb
            self._providers[cfg.id] = PoolManagedProvider(
                config=cfg, adapter=adapter,
                semaphore=sem, circuit_breaker=cb, tracker=self._tracker
            )

            logger.debug("Provider '%s' registered (protocol=%s)", cfg.id, cfg.protocol)

    # --------------------------------------------------------
    # Provider selection
    # --------------------------------------------------------

    def _score_provider(self, cfg: ProviderConfig, strategy: StrategyConfig) -> float:
        """Compute a selection score (lower = better for sort ascending)."""
        if strategy.weights:
            # Balanced: compute composite score
            all_enabled = self._pool_config.get_enabled()
            max_cost = max((p.cost.input for p in all_enabled), default=1) or 1
            max_rpm = max((p.limits.requests_per_minute for p in all_enabled), default=1) or 1

            cost_score = cfg.cost.input / max_cost            # 0-1, lower=cheaper
            speed_score = 1 - cfg.limits.requests_per_minute / max_rpm  # lower=faster
            cap_score = 0.5  # neutral when not filtered by capability

            w = strategy.weights
            return (
                w.get("cost", 0.35) * cost_score +
                w.get("speed", 0.30) * speed_score +
                w.get("capability", 0.35) * cap_score
            )

        if strategy.sort_by:
            # Dot-path resolution: "cost.input" → cfg.cost.input
            parts = strategy.sort_by.split(".")
            val: object = cfg
            for p in parts:
                val = getattr(val, p, 0)
            numeric = float(val) if val is not None else 0.0
            return numeric if strategy.ascending else -numeric

        return 0.0

    def select(self, capabilities: list[str] | None = None,
               strategy: str | None = None,
               task_type: str | None = None) -> ProviderConfig | None:
        """Select the best provider config without acquiring it.

        Returns the config of the best matching provider, or None.
        """
        if not self._initialized:
            self.initialize()

        # Resolve strategy name
        strat_name = strategy
        if not strat_name and task_type and self._pool_config:
            strat_name = self._pool_config.task_strategies.get(task_type)
        if not strat_name:
            strat_name = self._pool_config.task_strategies.get("default", "balanced") if self._pool_config else "balanced"

        strat_cfg = self._pool_config.selection_strategies.get(strat_name) if self._pool_config else None

        # Filter: enabled + in providers dict (auth passed) + capability match + CB available
        candidates: list[ProviderConfig] = []
        for cfg in (self._pool_config.get_enabled() if self._pool_config else []):
            if cfg.id not in self._providers:
                continue  # auth failed at init
            if capabilities:
                if not all(c in cfg.capabilities for c in capabilities):
                    continue
            cb = self._circuit_breakers.get(cfg.id)
            if cb and not cb.is_available():
                logger.debug("Provider '%s' skipped: circuit breaker OPEN", cfg.id)
                continue
            # Prefer tags if strategy has prefer_tags
            if strat_cfg and strat_cfg.prefer_tags:
                if not any(t in cfg.tags for t in strat_cfg.prefer_tags):
                    # still include but deprioritize — handled by scoring
                    pass
            candidates.append(cfg)

        if not candidates:
            # Try fallback strategy
            if strat_cfg and strat_cfg.fallback:
                return self.select(capabilities=capabilities, strategy=strat_cfg.fallback)
            return None

        # Sort by strategy score
        if strat_cfg:
            # Prefer-tags first
            if strat_cfg.prefer_tags:
                preferred = [c for c in candidates if any(t in c.tags for t in strat_cfg.prefer_tags)]
                rest = [c for c in candidates if c not in preferred]
                candidates = preferred + rest

            candidates.sort(key=lambda c: self._score_provider(c, strat_cfg))

        return candidates[0] if candidates else None

    async def acquire(self, capabilities: list[str] | None = None,
                      strategy: str | None = None,
                      task_type: str | None = None) -> PoolManagedProvider:
        """Acquire the best matching provider.

        Raises RuntimeError if no suitable provider is available.
        """
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

    async def release(self, provider_id: str) -> None:
        """Release a provider (no-op for now; semaphore released by context manager)."""
        pass  # semaphore is managed by asynccontextmanager in PoolManagedProvider.chat

    @asynccontextmanager
    async def use(self, capabilities: list[str] | None = None,
                  strategy: str | None = None,
                  task_type: str | None = None) -> AsyncIterator[PoolManagedProvider]:
        """Context manager: acquire + auto-release."""
        provider = await self.acquire(capabilities=capabilities, strategy=strategy, task_type=task_type)
        try:
            yield provider
        finally:
            await self.release(provider.provider_id)

    # --------------------------------------------------------
    # Introspection
    # --------------------------------------------------------

    def provider_status(self) -> list[dict]:
        """Return status of all configured providers (for platform UI)."""
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
                "protocol": cfg.protocol,
                "capabilities": cfg.capabilities,
                "enabled": cfg.enabled,
                "auth_method": cfg.auth.method,
                "model": cfg.endpoint.model,
                "cost_per_1m_input": cfg.cost.input,
                "max_concurrent": cfg.limits.max_concurrent,
                "tags": cfg.tags,
                "in_pool": in_pool,
                "circuit_breaker": cb.status() if cb else None,
                "stats": stats,
            })
        return result

    def usage_stats(self) -> dict:
        """Return usage statistics for all providers."""
        return self._tracker.get_stats()
