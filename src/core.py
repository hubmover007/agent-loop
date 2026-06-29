"""Core types for Agent-Loop system."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DISCARDED = "discarded"
    CANCELLED = "cancelled"


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    DESTROYED = "destroyed"


class ToolResultStatus(str, Enum):
    SUCCESS = "success"
    TRANSIENT_ERROR = "transient_error"
    FATAL_ERROR = "fatal_error"


class LoopPhase(str, Enum):
    """Phases of the Loop Engine."""
    INPUT = "input"
    RETRIEVE = "retrieve"
    REASON = "reason"
    DECOMPOSE = "decompose"
    DISPATCH = "dispatch"
    EXECUTE = "execute"
    COLLECT = "collect"
    OUTPUT = "output"


class MemoryLayer(str, Enum):
    """M-FLOW inspired four-layer memory topology."""
    FACT = "fact"          # Layer 0: Entities & FacetPoints (锥尖)
    FACET = "facet"        # Layer 1: Semantic dimensions
    EPISODE = "episode"    # Layer 2: Event contexts
    PROJECT = "project"    # Layer 3: Project overviews (锥底)


@dataclass
class ToolResult:
    """Result from a tool execution."""
    status: ToolResultStatus
    data: Any = None
    error: str | None = None
    execution_time_ms: float = 0.0
    retry_count: int = 0


@dataclass
class StepLog:
    """Single execution step log within an Agent's BranchSpace."""
    step: int
    action: str
    tool_name: str | None
    input: dict[str, Any] = field(default_factory=dict)
    output: Any = None
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskResult:
    """Result submitted by an Agent for a Task."""
    task_id: str
    agent_id: str
    status: TaskStatus
    summary: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    steps: list[StepLog] = field(default_factory=list)
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class EvaluationResult:
    """Quality evaluation of an Agent's output."""
    scores: dict[str, float]  # e.g. {"completeness": 0.9, "correctness": 0.85, ...}
    overall: float
    action: str  # "accept" or "discard"
    reason: str


@dataclass
class DiscardRecord:
    """Record in the discard pool for audit."""
    agent_id: str
    task_id: str
    result: TaskResult | None
    reason: str
    timestamp: datetime
    agent_log: list[StepLog] = field(default_factory=list)


@dataclass
class ExpertProfile:
    """Profile for MoE-style Agent routing."""
    expert_id: str
    embedding: list[float]
    success_count: int = 0
    total_count: int = 0
    current_load: int = 0
    specialties: list[str] = field(default_factory=list)


__all__ = [
    "TaskStatus",
    "AgentStatus",
    "ToolResultStatus",
    "LoopPhase",
    "MemoryLayer",
    "ToolResult",
    "StepLog",
    "TaskResult",
    "EvaluationResult",
    "DiscardRecord",
    "ExpertProfile",
]
