"""Usage tracker — records LLM calls to state/llm_pool/usage.jsonl."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class UsageRecord:
    """A single LLM call record."""
    timestamp: str
    provider_id: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    task_id: str | None = None
    agent_id: str | None = None
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
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "error": self.error,
        }


class PoolTracker:
    """Per-provider stats tracker with optional JSONL persistence."""

    def __init__(self, usage_log_path: Path | None = None):
        self._usage_log_path = usage_log_path
        self._lock = asyncio.Lock()

        # In-memory stats: provider_id → stats dict
        self._stats: dict[str, dict] = {}

    def _ensure_provider(self, provider_id: str) -> dict:
        if provider_id not in self._stats:
            self._stats[provider_id] = {
                "call_count": 0,
                "success_count": 0,
                "total_latency_ms": 0.0,
                "total_input_tokens": 0,
                "total_output_tokens": 0,
            }
        return self._stats[provider_id]

    async def record(self, record: UsageRecord) -> None:
        """Record a usage event: update in-memory stats + append to JSONL."""
        async with self._lock:
            stats = self._ensure_provider(record.provider_id)
            stats["call_count"] += 1
            if record.success:
                stats["success_count"] += 1
            stats["total_latency_ms"] += record.latency_ms
            stats["total_input_tokens"] += record.input_tokens
            stats["total_output_tokens"] += record.output_tokens

            # Persist to JSONL
            if self._usage_log_path:
                try:
                    self._usage_log_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self._usage_log_path, "a") as f:
                        f.write(json.dumps(record.to_dict()) + "\n")
                except Exception as e:
                    logger.warning("Failed to write usage log: %s", e)

    def get_stats(self, provider_id: str | None = None) -> dict:
        """Return usage statistics, optionally for a single provider."""
        if provider_id:
            s = self._stats.get(provider_id, {})
            return self._format_stats(provider_id, s)

        return {pid: self._format_stats(pid, s) for pid, s in self._stats.items()}

    def _format_stats(self, provider_id: str, s: dict) -> dict:
        calls = s.get("call_count", 0)
        successes = s.get("success_count", 0)
        return {
            "provider_id": provider_id,
            "call_count": calls,
            "success_rate": round(successes / calls, 3) if calls > 0 else 1.0,
            "avg_latency_ms": round(s.get("total_latency_ms", 0) / calls, 1) if calls > 0 else 0,
            "total_input_tokens": s.get("total_input_tokens", 0),
            "total_output_tokens": s.get("total_output_tokens", 0),
        }
