"""LoopMetrics — per-phase duration, token, and LLM call tracking.

Tracks wall-clock time, LLM token consumption, and call counts for each
phase of the MainLoop cycle. Provides a summary() for logging/display.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PhaseMetric:
    """Metrics for a single MainLoop phase."""
    name: str
    start: float = 0.0
    duration: float = 0.0
    llm_calls: int = 0
    tokens_used: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    errors: int = 0
    cost: float = 0.0


@dataclass
class LoopMetrics:
    """Tracks consumption across all phases of a single MainLoop run.

    Usage:
        m = LoopMetrics()
        m.start()
        m.start_phase("retrieve")
        # ... do retrieval ...
        m.record_llm_call(prompt_tokens=100, completion_tokens=50, cost=0.001)
        m.end_phase("retrieve")
        m.finish()
        print(m.summary())
    """
    phases: dict[str, PhaseMetric] = field(default_factory=dict)
    total_duration: float = 0.0
    total_tokens: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_llm_calls: int = 0
    total_cost: float = 0.0
    total_errors: int = 0
    _current_phase: str | None = None
    _start_time: float = 0.0

    def start(self) -> None:
        """Mark the start of the entire loop."""
        self._start_time = time.time()

    def start_phase(self, name: str) -> None:
        """Mark the start of a phase."""
        self._current_phase = name
        self.phases[name] = PhaseMetric(name=name, start=time.time())

    def end_phase(self, name: str) -> None:
        """Mark the end of a phase."""
        if name in self.phases:
            self.phases[name].duration = time.time() - self.phases[name].start

    def record_llm_call(
        self,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
    ) -> None:
        """Record an LLM call within the current phase."""
        total = prompt_tokens + completion_tokens

        if self._current_phase and self._current_phase in self.phases:
            p = self.phases[self._current_phase]
            p.llm_calls += 1
            p.tokens_used += total
            p.prompt_tokens += prompt_tokens
            p.completion_tokens += completion_tokens
            p.cost += cost

        self.total_llm_calls += 1
        self.total_tokens += total
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_cost += cost

    def record_error(self) -> None:
        """Record an error in the current phase."""
        if self._current_phase and self._current_phase in self.phases:
            self.phases[self._current_phase].errors += 1
        self.total_errors += 1

    def finish(self) -> None:
        """Mark the end of the entire loop."""
        if self._start_time > 0:
            self.total_duration = time.time() - self._start_time

    def summary(self) -> str:
        """Return a human-readable summary."""
        lines = [
            f"Loop completed in {self.total_duration:.2f}s",
            f"  Total: {self.total_tokens} tokens "
            f"({self.total_prompt_tokens} in / {self.total_completion_tokens} out), "
            f"{self.total_llm_calls} LLM calls, ${self.total_cost:.4f}",
        ]
        if self.total_errors:
            lines.append(f"  Errors: {self.total_errors}")
        for p in self.phases.values():
            tok = f"{p.tokens_used} tokens" if p.tokens_used else "0 tokens"
            calls = f"{p.llm_calls} calls" if p.llm_calls else "no LLM"
            errs = f", {p.errors} errors" if p.errors else ""
            lines.append(
                f"  {p.name}: {p.duration:.2f}s, {tok}, {calls}{errs}"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Serialize for API/CLI consumption."""
        return {
            "total_duration": self.total_duration,
            "total_tokens": self.total_tokens,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_llm_calls": self.total_llm_calls,
            "total_cost": self.total_cost,
            "total_errors": self.total_errors,
            "phases": {
                name: {
                    "duration": p.duration,
                    "tokens_used": p.tokens_used,
                    "llm_calls": p.llm_calls,
                    "errors": p.errors,
                    "cost": p.cost,
                }
                for name, p in self.phases.items()
            },
        }
