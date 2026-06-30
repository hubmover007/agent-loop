"""Streaming progress output for AgentLoop.

Emits ProgressEvents during loop execution for UI, logging, and debugging.
Supports three consumption modes:
  1. Synchronous callbacks (simple scenarios)
  2. asyncio.Queue subscription (streaming scenarios)
  3. JSONL file logging (audit trail)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class ProgressEvent:
    """A progress event emitted during AgentLoop execution.

    Types:
      - "phase_start" / "phase_done" — stage entry/exit
      - "tool_call" — tool invocation
      - "llm_call" — LLM provider call
      - "error" — exception
      - "approval_needed" — requires human confirmation
    """

    type: str
    agent_id: str
    phase: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "type": self.type,
            "agent_id": self.agent_id,
            "phase": self.phase,
            "message": self.message,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ProgressEvent":
        return cls(**d)


class ProgressEmitter:
    """Emits ProgressEvents to registered subscribers.

    All emit() calls are synchronous (non-blocking) — they push events
    into subscriber queues without waiting for consumption.

    Usage:
        emitter = ProgressEmitter("agent-1")
        emitter.on_event(lambda ev: print(ev.message))
        emitter.emit("phase_start", "PLAN", "Generating plan")
    """

    def __init__(self, agent_id: str, log_path: str | None = None):
        self.agent_id = agent_id
        self._subscribers: list[asyncio.Queue] = []
        self._callbacks: list[Callable] = []
        self._log_path = Path(log_path) if log_path else None

        # Ensure log directory exists
        if self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, type: str, phase: str, message: str, **data) -> None:
        """Emit a progress event synchronously (non-blocking).

        Pushes into all subscriber queues and calls all callbacks.
        Subscriber queue puts are fire-and-forget — if a queue is full,
        the event is dropped with a warning.
        """
        event = ProgressEvent(
            type=type,
            agent_id=self.agent_id,
            phase=phase,
            message=message,
            data=data,
        )

        # Notify callbacks (synchronous)
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.warning("ProgressEmitter[%s]: callback error: %s", self.agent_id, e)

        # Push to subscriber queues (non-blocking)
        for q in self._subscribers:
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("ProgressEmitter[%s]: subscriber queue full, dropping event",
                               self.agent_id)

        # Append to log file
        if self._log_path:
            try:
                line = json.dumps(event.to_dict(), ensure_ascii=False)
                with open(self._log_path, "a") as f:
                    f.write(line + "\n")
            except Exception as e:
                logger.warning("ProgressEmitter[%s]: log write error: %s", self.agent_id, e)

    async def subscribe(self) -> asyncio.Queue:
        """Subscribe to the progress event stream.

        Returns an asyncio.Queue that receives ProgressEvent objects.
        The queue is unbounded initially; events are dropped if not consumed.

        Usage:
            q = await emitter.subscribe()
            event = await q.get()  # blocks until next event
        """
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers.append(q)
        return q

    def on_event(self, callback: Callable) -> None:
        """Register a synchronous callback for each progress event.

        The callback receives a ProgressEvent. Exceptions are caught
        and logged — they will not disrupt the emitter.

        Usage:
            emitter.on_event(lambda ev: print(f"[{ev.phase}] {ev.message}"))
        """
        self._callbacks.append(callback)

    async def stream_to_sink(self, sink: Callable) -> None:
        """Continuously consume events from a subscriber queue and forward to sink.

        This is an async generator-style consumer that creates a dedicated
        queue, subscribes, and forwards every event to the sink callable.

        The sink should be an async callable accepting a ProgressEvent.

        Usage:
            async def printer(ev):
                print(f"[{ev.phase}] {ev.message}")

            await emitter.stream_to_sink(printer)
        """
        q = await self.subscribe()
        while True:
            try:
                event = await q.get()
                if asyncio.iscoroutinefunction(sink):
                    await sink(event)
                else:
                    sink(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("ProgressEmitter[%s]: sink error: %s", self.agent_id, e)

    def unsubscribe(self, q: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        if q in self._subscribers:
            self._subscribers.remove(q)

    def detach(self) -> None:
        """Remove all subscribers and callbacks (cleanup)."""
        self._subscribers.clear()
        self._callbacks.clear()

    # ── P0: Streaming tokens & tool call events ───────────────────

    def emit_token(self, token: str) -> None:
        """Emit a single token during streaming."""
        self.emit("token", "EXECUTE", token)

    def emit_tool_call(self, tool_name: str, args: dict) -> None:
        """Emit a tool call event."""
        self.emit("tool_call", "EXECUTE", f"Calling {tool_name}",
                  tool=tool_name, args=args)

    def emit_tool_result(self, tool_name: str, result: str) -> None:
        """Emit a tool result event."""
        self.emit("tool_result", "EXECUTE", result,
                  tool=tool_name, result=result)
