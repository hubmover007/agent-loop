"""LLM Pool — multi-provider management with security, protocol adapters, and circuit breaker.

Design references:
  - OpenClaw: provider registry with auth/protocol separation
  - Claude Code: resolvedModel pattern (requested ≠ actual)

Architecture:
    LLMPool
      ├── ProviderRegistry    — loads config/llm_pool.yaml, resolves auth
      ├── AuthResolver        — env_var / aws_sdk / gcp_adc / none
      ├── ProtocolAdapter     — openai-completions / bedrock-converse-stream / google-gemini
      ├── CircuitBreaker      — per-provider failure tracking
      └── PoolTracker         — concurrency semaphores + usage stats
"""

from .pool import LLMPool
from .config import ProviderConfig, PoolConfig
from .tracker import UsageRecord

__all__ = ["LLMPool", "ProviderConfig", "PoolConfig", "UsageRecord"]
