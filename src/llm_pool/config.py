"""LLM Pool configuration models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class AuthConfig:
    """Authentication configuration (no secrets stored here)."""
    method: str = "none"       # env_var | aws_sdk | gcp_adc | none
    key_env: str | None = None # env var name (not the value)
    profile: str | None = None # AWS CLI profile name
    region: str | None = None  # AWS region


@dataclass
class EndpointConfig:
    """Endpoint/connection configuration."""
    model: str = ""
    base_url: str = ""
    timeout_s: float = 60.0
    max_retries: int = 3


@dataclass
class CostConfig:
    """Cost in $/1M tokens (aligned with OpenClaw format)."""
    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0


@dataclass
class LimitsConfig:
    """Rate and concurrency limits."""
    max_concurrent: int = 5
    requests_per_minute: int = 60
    tokens_per_minute: int = 0       # 0 = unlimited


@dataclass
class CircuitBreakerConfig:
    """Per-provider circuit breaker settings."""
    failure_threshold: int = 5       # consecutive failures to open
    recovery_timeout_s: float = 60.0 # seconds before half-open retry
    half_open_requests: int = 2      # requests allowed in half-open state


@dataclass
class ProviderConfig:
    """Full configuration for a single LLM provider."""
    id: str
    type: str                              # openai_compatible | anthropic_bedrock | google_gemini
    protocol: str = "openai-completions"  # openai-completions | bedrock-converse-stream | google-gemini
    auth: AuthConfig = field(default_factory=AuthConfig)
    endpoint: EndpointConfig = field(default_factory=EndpointConfig)
    capabilities: list[str] = field(default_factory=list)
    context_window: int = 32000
    max_output_tokens: int = 4096
    cost: CostConfig = field(default_factory=CostConfig)
    limits: LimitsConfig = field(default_factory=LimitsConfig)
    circuit_breaker: CircuitBreakerConfig = field(default_factory=CircuitBreakerConfig)
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> "ProviderConfig":
        auth_d = data.get("auth", {})
        ep_d = data.get("endpoint", {})
        cost_d = data.get("cost", {})
        lim_d = data.get("limits", {})
        cb_d = data.get("circuit_breaker", {})

        return cls(
            id=data["id"],
            type=data.get("type", "openai_compatible"),
            protocol=data.get("protocol", "openai-completions"),
            auth=AuthConfig(
                method=auth_d.get("method", "none"),
                key_env=auth_d.get("key_env"),
                profile=auth_d.get("profile"),
                region=auth_d.get("region"),
            ),
            endpoint=EndpointConfig(
                model=ep_d.get("model", ""),
                base_url=ep_d.get("base_url", ""),
                timeout_s=float(ep_d.get("timeout_s", 60)),
                max_retries=int(ep_d.get("max_retries", 3)),
            ),
            capabilities=data.get("capabilities", []),
            context_window=data.get("context_window", 32000),
            max_output_tokens=data.get("max_output_tokens", 4096),
            cost=CostConfig(
                input=float(cost_d.get("input", 0)),
                output=float(cost_d.get("output", 0)),
                cache_read=float(cost_d.get("cache_read", 0)),
                cache_write=float(cost_d.get("cache_write", 0)),
            ),
            limits=LimitsConfig(
                max_concurrent=int(lim_d.get("max_concurrent", 5)),
                requests_per_minute=int(lim_d.get("requests_per_minute", 60)),
                tokens_per_minute=int(lim_d.get("tokens_per_minute", 0)),
            ),
            circuit_breaker=CircuitBreakerConfig(
                failure_threshold=int(cb_d.get("failure_threshold", 5)),
                recovery_timeout_s=float(cb_d.get("recovery_timeout_s", 60)),
                half_open_requests=int(cb_d.get("half_open_requests", 2)),
            ),
            enabled=data.get("enabled", True),
            tags=data.get("tags", []),
        )


@dataclass
class StrategyConfig:
    """A selection strategy definition."""
    name: str
    description: str = ""
    sort_by: str | None = None        # dot-path like "cost.input"
    ascending: bool = True
    prefer_tags: list[str] = field(default_factory=list)
    fallback: str | None = None
    weights: dict[str, float] = field(default_factory=dict)  # for balanced


@dataclass
class PoolConfig:
    """Full pool configuration loaded from YAML."""
    version: str = "1.0"
    providers: list[ProviderConfig] = field(default_factory=list)
    selection_strategies: dict[str, StrategyConfig] = field(default_factory=dict)
    task_strategies: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PoolConfig":
        with open(path) as f:
            data = yaml.safe_load(f)

        providers = [ProviderConfig.from_dict(p) for p in data.get("providers", [])]

        strats: dict[str, StrategyConfig] = {}
        for name, sd in data.get("selection_strategies", {}).items():
            strats[name] = StrategyConfig(
                name=name,
                description=sd.get("description", ""),
                sort_by=sd.get("sort_by"),
                ascending=sd.get("ascending", True),
                prefer_tags=sd.get("prefer_tags", []),
                fallback=sd.get("fallback"),
                weights=sd.get("weights", {}),
            )

        return cls(
            version=data.get("version", "1.0"),
            providers=providers,
            selection_strategies=strats,
            task_strategies=data.get("task_strategies", {}),
        )

    def get_enabled(self) -> list[ProviderConfig]:
        return [p for p in self.providers if p.enabled]
