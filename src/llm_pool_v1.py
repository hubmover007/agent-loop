"""LLM Pool — multi-provider management with capability routing and concurrency control.

Usage:
    from agent_loop.llm_pool import LLMPool

    pool = LLMPool("config/llm_pool.yaml")
    provider = await pool.acquire(["coding"], "cheapest")
    # ... use provider.chat() ...
    await pool.release(provider.provider_id)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .loop_engine import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


# ============================================================
# Provider Config
# ============================================================

@dataclass
class ProviderConfig:
    """Configuration for a single LLM provider."""
    id: str
    type: str
    capabilities: list[str] = field(default_factory=list)
    model: str = ""
    base_url: str = ""
    api_key_env: str | None = None
    cost_per_1k_tokens: float = 0.0
    avg_latency_ms: float = 0.0
    max_concurrent: int = 1
    enabled: bool = True


# ============================================================
# Usage Record
# ============================================================

@dataclass
class UsageRecord:
    """Record of a single LLM call."""
    timestamp: str
    provider_id: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    task_id: str | None = None
    agent_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "provider_id": self.provider_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "task_id": self.task_id,
            "agent_id": self.agent_id,
        }


# ============================================================
# PoolTrackedLLMProvider (wrapper)
# ============================================================

class PoolTrackedLLMProvider(LLMProvider):
    """Wraps an LLMProvider with usage tracking and semaphore control."""

    def __init__(self, provider_id: str, inner: LLMProvider,
                 semaphore: asyncio.Semaphore, config: ProviderConfig,
                 usage_log_path: Path | None = None):
        self.provider_id = provider_id
        self._inner = inner
        self._semaphore = semaphore
        self._config = config
        self._usage_log_path = usage_log_path

        # Stats
        self.call_count: int = 0
        self.success_count: int = 0
        self.total_latency_ms: float = 0.0

    async def chat(self, messages: list[dict], thinking: bool = False,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   model: str | None = None, **kwargs) -> LLMResponse:
        """Chat with concurrency limiting and usage tracking."""
        start_time = time.monotonic()
        success = False
        input_tokens = 0
        output_tokens = 0

        try:
            async with self._semaphore:
                result = await self._inner.chat(
                    messages=messages,
                    thinking=thinking,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    model=model or self._config.model,
                    **kwargs,
                )

            elapsed = (time.monotonic() - start_time) * 1000.0
            success = True
            usage = result.usage or {}
            input_tokens = usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0)
            output_tokens = usage.get("output_tokens", 0) or usage.get("completion_tokens", 0)

            self._update_stats(success, elapsed)
            self._log_usage(success, input_tokens, output_tokens, elapsed)

            return result

        except Exception:
            elapsed = (time.monotonic() - start_time) * 1000.0
            self._update_stats(False, elapsed)
            self._log_usage(False, input_tokens, output_tokens, elapsed)
            raise

    async def embed(self, text: str | list[str], model: str | None = None) -> list[list[float]]:
        """Embed with concurrency limiting."""
        async with self._semaphore:
            return await self._inner.embed(text, model=model)

    def _update_stats(self, success: bool, latency_ms: float) -> None:
        self.call_count += 1
        if success:
            self.success_count += 1
        self.total_latency_ms += latency_ms

    def _log_usage(self, success: bool, input_tokens: int,
                   output_tokens: int, latency_ms: float) -> None:
        if not self._usage_log_path:
            return
        record = UsageRecord(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            provider_id=self.provider_id,
            model=self._config.model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 1),
            success=success,
        )
        try:
            self._usage_log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._usage_log_path, "a") as f:
                f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.warning("LLMPool: failed to write usage log: %s", e)

    @property
    def success_rate(self) -> float:
        return self.success_count / max(self.call_count, 1)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.call_count, 1)

    @property
    def usage_stats(self) -> dict:
        return {
            "call_count": self.call_count,
            "success_count": self.success_count,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
        }


# ============================================================
# LLMPool
# ============================================================

class LLMPool:
    """Multi-provider LLM pool with capability routing and concurrency control.

    Capabilities:
      - Load providers from YAML config
      - Filter providers by required capabilities
      - Sort providers by selection strategy (cheapest, fastest, most_capable, balanced)
      - Concurrency limiting per provider (asyncio.Semaphore)
      - Usage tracking with JSONL logs
    """

    def __init__(self, config_path: str | Path = "config/llm_pool.yaml",
                 state_dir: str | Path = "state"):
        self.config_path = Path(config_path)
        self.state_dir = Path(state_dir)

        # Load config
        self._config: dict = {}
        self._providers: list[ProviderConfig] = []
        self._strategies: dict[str, str] = {}
        self._strategy_definitions: dict[str, dict] = {}

        # Runtime state
        self._provider_wrappers: dict[str, PoolTrackedLLMProvider] = {}
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._provider_configs: dict[str, ProviderConfig] = {}

        self._load_config()

    # ----- Config Loading -----

    def _load_config(self) -> None:
        """Load provider config from YAML file."""
        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

        self._providers = []
        for p in self._config.get("providers", []):
            if p.get("enabled", True):
                self._providers.append(ProviderConfig(
                    id=p["id"],
                    type=p.get("type", "openai_compatible"),
                    capabilities=p.get("capabilities", []),
                    model=p.get("model", ""),
                    base_url=p.get("base_url", ""),
                    api_key_env=p.get("api_key_env"),
                    cost_per_1k_tokens=p.get("cost_per_1k_tokens", 0.0),
                    avg_latency_ms=p.get("avg_latency_ms", 0.0),
                    max_concurrent=p.get("max_concurrent", 1),
                    enabled=p.get("enabled", True),
                ))

        self._strategies = self._config.get("strategies", {})
        self._strategy_definitions = self._config.get("strategy_definitions", {})

        # Initialize semaphores and config maps
        for p in self._providers:
            self._semaphores[p.id] = asyncio.Semaphore(p.max_concurrent)
            self._provider_configs[p.id] = p

        logger.info("LLMPool: loaded %d providers", len(self._providers))

    # ----- Provider Acquisition -----

    async def acquire(self, capabilities: list[str] | None = None,
                      strategy: str = "balanced") -> PoolTrackedLLMProvider:
        """Acquire a provider matching capabilities with the given strategy.

        Args:
            capabilities: Required capabilities (e.g., ["coding", "reasoning"]).
                          Empty or None means all providers are candidates.
            strategy: Selection strategy name (e.g., "cheapest", "fastest", "balanced").

        Returns:
            A PoolTrackedLLMProvider instance ready to use.

        Raises:
            ValueError: If no provider matches the required capabilities.
        """
        # Resolve strategy name via strategy map
        strategy_name = self._strategies.get(strategy, strategy)

        # Filter by capabilities
        candidates = self._filter_by_capabilities(capabilities or [])

        if not candidates:
            caps_str = ", ".join(capabilities or ["any"])
            raise ValueError(f"No enabled provider with capabilities: [{caps_str}]")

        # Sort by strategy
        sorted_candidates = self._sort_by_strategy(candidates, strategy_name)

        # Acquire the best available (semaphore + lazy init)
        for pc in sorted_candidates:
            wrapper = await self._get_or_create_wrapper(pc)
            return wrapper

        raise RuntimeError("All matching providers are at capacity")

    def _filter_by_capabilities(self, capabilities: list[str]) -> list[ProviderConfig]:
        """Filter providers that have all required capabilities."""
        if not capabilities:
            return list(self._providers)

        return [
            p for p in self._providers
            if all(cap in p.capabilities for cap in capabilities)
        ]

    def _sort_by_strategy(self, candidates: list[ProviderConfig],
                          strategy: str) -> list[ProviderConfig]:
        """Sort candidates by the given strategy."""
        definition = self._strategy_definitions.get(strategy, {})

        # Balanced strategy: use formula
        if "formula" in definition:
            return self._sort_balanced(candidates)

        # Simple sort strategies
        sort_by = definition.get("sort_by", "cost_per_1k_tokens")
        ascending = definition.get("ascending", True)

        return sorted(
            candidates,
            key=lambda p: getattr(p, sort_by, 0),
            reverse=not ascending,
        )

    def _sort_balanced(self, candidates: list[ProviderConfig]) -> list[ProviderConfig]:
        """Sort by balanced formula: cost * 0.4 + latency_normalized * 0.3 + capability_score * 0.3."""
        # Normalize values
        costs = [p.cost_per_1k_tokens for p in candidates]
        latencies = [p.avg_latency_ms for p in candidates]
        cap_counts = [len(p.capabilities) for p in candidates]

        max_cost = max(costs) if costs else 1
        max_latency = max(latencies) if latencies else 1
        max_cap = max(cap_counts) if cap_counts else 1

        def score(p: ProviderConfig) -> float:
            cost_norm = p.cost_per_1k_tokens / max(max_cost, 0.001)
            lat_norm = p.avg_latency_ms / max(max_latency, 1)
            cap_norm = len(p.capabilities) / max(max_cap, 1)
            # Lower is better
            return cost_norm * 0.4 + lat_norm * 0.3 - cap_norm * 0.3

        return sorted(candidates, key=score)

    async def _get_or_create_wrapper(self, pc: ProviderConfig) -> PoolTrackedLLMProvider:
        """Lazily create or return cached provider wrapper."""
        if pc.id in self._provider_wrappers:
            return self._provider_wrappers[pc.id]

        # Build the inner LLM provider
        inner = self._build_inner_provider(pc)

        # Create wrapper
        usage_log_path = self.state_dir / "llm_pool" / "usage.jsonl"
        wrapper = PoolTrackedLLMProvider(
            provider_id=pc.id,
            inner=inner,
            semaphore=self._semaphores[pc.id],
            config=pc,
            usage_log_path=usage_log_path,
        )

        self._provider_wrappers[pc.id] = wrapper
        return wrapper

    def _build_inner_provider(self, pc: ProviderConfig) -> LLMProvider:
        """Build the actual LLM provider instance from config."""
        if pc.type == "openai_compatible":
            from .llm import OpenAICompatibleProvider
            api_key = os.environ.get(pc.api_key_env, "not-needed") if pc.api_key_env else "not-needed"
            return OpenAICompatibleProvider(
                base_url=pc.base_url,
                default_model=pc.model,
                api_key=api_key,
            )
        elif pc.type == "deepseek":
            from .llm import DeepSeekProvider
            api_key = os.environ.get(pc.api_key_env, "") if pc.api_key_env else ""
            return DeepSeekProvider(
                api_key=api_key,
                base_url=pc.base_url,
                default_model=pc.model,
            )
        elif pc.type == "anthropic_bedrock":
            from .llm import AnthropicBedrockProvider
            return AnthropicBedrockProvider(
                aws_access_key=os.environ.get("AWS_ACCESS_KEY_ID", ""),
                aws_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY", ""),
                default_model=pc.model,
            )
        else:
            raise ValueError(f"Unknown provider type: {pc.type}")

    async def release(self, provider_id: str) -> None:
        """Release a provider back to the pool.

        The semaphore is released automatically when the PoolTrackedLLMProvider's
        async context manager exits. This method exists for explicit API symmetry.
        """
        # Semaphore release happens via the context manager in PoolTrackedLLMProvider.chat().
        # This is a no-op for explicit release — the wrapper is reusable.
        pass

    # ----- Stats -----

    def usage_stats(self) -> dict:
        """Return usage statistics for all providers."""
        stats = {}
        for pid, wrapper in self._provider_wrappers.items():
            stats[pid] = wrapper.usage_stats
        return stats

    @property
    def provider_count(self) -> int:
        return len(self._providers)

    @property
    def providers(self) -> list[ProviderConfig]:
        return list(self._providers)

    @property
    def strategy_names(self) -> list[str]:
        return list(self._strategies.keys())
