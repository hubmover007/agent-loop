"""Loop Engine - the heart of Agent-Loop.

Three levels of loops:
  MainLoop:  INPUT → RETRIEVE → REASON → DECOMPOSE → DISPATCH → COLLECT → OUTPUT
  AgentLoop: INIT → PLAN → EXECUTE → SELF_EVAL → SUBMIT → DESTROY
  ToolLoop:  CALL → VERIFY → RETRY(3x) → RETURN/FAIL
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..core import (
    AgentStatus, LoopPhase, TaskStatus, ToolResult, ToolResultStatus,
    StepLog, TaskResult, EvaluationResult,
)
from ..tools.base import ToolInterface, ToolRegistry

logger = logging.getLogger(__name__)


# ============================================================
# Configuration
# ============================================================

@dataclass
class LoopConfig:
    """Configuration for the Loop Engine."""
    # MainLoop
    max_reason_loops: int = 5           # Max internal reasoning iterations
    reason_confidence_threshold: float = 0.85  # ACT: stop when confidence > threshold
    max_agent_concurrent: int = 50       # Max concurrent agents

    # AgentLoop
    max_agent_steps: int = 20           # Max steps per agent
    agent_step_timeout: float = 300.0   # Timeout per step (seconds)
    agent_ttl: float = 1800.0           # Max agent lifetime (seconds)

    # ToolLoop
    tool_max_retries: int = 3           # Max tool retry attempts
    tool_retry_backoff_base: float = 2.0  # Exponential backoff base

    # Evaluator
    accept_threshold: float = 0.7       # Accept agent result if score >= this
    evaluation_weights: dict[str, float] = field(default_factory=lambda: {
        "completeness": 0.30,
        "correctness": 0.30,
        "relevance": 0.25,
        "efficiency": 0.15,
    })


# ============================================================
# LLM Provider Interface
# ============================================================

@dataclass
class LLMResponse:
    """Response from LLM provider."""
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    thinking: str | None = None  # Extended thinking content


class LLMProvider(ABC):
    """Abstract LLM provider."""

    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Send a chat completion request."""
        ...

    @abstractmethod
    async def embed(self, text: str | list[str]) -> list[list[float]]:
        """Generate embeddings for text."""
        ...


# ============================================================
# Loop Context (shared state across loops)
# ============================================================

@dataclass
class LoopContext:
    """Shared context flowing through all loop phases."""
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_input: str = ""
    current_phase: LoopPhase = LoopPhase.INPUT

    # Retrieval
    retrieved_context: list[dict] = field(default_factory=list)

    # Reasoning
    reason_iterations: int = 0
    reason_confidence: float = 0.0
    reason_output: str = ""
    thought_chain: list[str] = field(default_factory=list)

    # Task decomposition
    task_ids: list[str] = field(default_factory=list)

    # Collection
    agent_results: list[TaskResult] = field(default_factory=list)
    discarded_results: list[str] = field(default_factory=list)

    # Output
    final_output: str = ""
    errors: list[str] = field(default_factory=list)


# ============================================================
# ToolLoop
# ============================================================

class ToolLoop:
    """Level 3: Tool execution with retry logic."""

    def __init__(self, registry: ToolRegistry, config: LoopConfig):
        self.registry = registry
        self.config = config

    async def execute(self, tool_name: str, **params) -> ToolResult:
        """Execute a tool with retry logic."""
        tool = self.registry.get(tool_name)
        if tool is None:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=f"Tool not found: {tool_name}",
            )

        last_error = None
        for attempt in range(1, self.config.tool_max_retries + 1):
            start = datetime.now()

            try:
                result = await tool.execute(**params)
            except Exception as e:
                result = ToolResult(
                    status=ToolResultStatus.TRANSIENT_ERROR,
                    error=str(e),
                )

            elapsed = (datetime.now() - start).total_seconds() * 1000
            result.execution_time_ms = elapsed
            result.retry_count = attempt - 1

            if result.status == ToolResultStatus.SUCCESS:
                return result
            if result.status == ToolResultStatus.FATAL_ERROR:
                return result

            # Transient error → retry with backoff
            last_error = result.error
            backoff = self.config.tool_retry_backoff_base ** attempt
            logger.warning(
                "Tool %s attempt %d failed: %s, retrying in %.1fs",
                tool_name, attempt, result.error, backoff
            )
            await asyncio.sleep(backoff)

        return ToolResult(
            status=ToolResultStatus.FATAL_ERROR,
            error=f"Max retries ({self.config.tool_max_retries}) exceeded: {last_error}",
            retry_count=self.config.tool_max_retries,
        )


# ============================================================
# AgentLoop
# ============================================================

class AgentLoop:
    """Level 2: Individual agent execution loop."""

    def __init__(self, tool_loop: ToolLoop, llm: LLMProvider, config: LoopConfig):
        self.tool_loop = tool_loop
        self.llm = llm
        self.config = config

    async def run(self, agent_id: str, task_scope: str, context: dict,
                  allowed_tools: list[str]) -> TaskResult:
        """Execute AgentLoop: PLAN → EXECUTE → SELF_EVAL → SUBMIT.

        This runs within an isolated BranchSpace.
        """
        steps: list[StepLog] = []
        artifacts: dict = {}
        started_at = datetime.now()

        logger.info("AgentLoop[%s] started: %s", agent_id, task_scope[:50])

        try:
            # Phase 1: PLAN
            plan = await self._plan(task_scope, context)
            steps.append(StepLog(step=0, action="plan", tool_name=None, output=plan))

            # Phase 2: EXECUTE
            step_num = 1
            for action in plan:
                if step_num > self.config.max_agent_steps:
                    steps.append(StepLog(
                        step=step_num, action="abort",
                        tool_name=None, error="Max steps exceeded"
                    ))
                    break

                # Check if this step requires a tool call
                tool_name = action.get("tool")
                if tool_name and tool_name in allowed_tools:
                    result = await self.tool_loop.execute(tool_name, **action.get("params", {}))
                    steps.append(StepLog(
                        step=step_num,
                        action=action.get("description", f"execute {tool_name}"),
                        tool_name=tool_name,
                        input=action.get("params", {}),
                        output=result.data if result.status == ToolResultStatus.SUCCESS else None,
                        error=result.error,
                    ))

                    if result.status == ToolResultStatus.FATAL_ERROR:
                        break

                    if result.data:
                        artifacts[action.get("output_key", f"step_{step_num}")] = result.data
                else:
                    # Pure reasoning step
                    steps.append(StepLog(
                        step=step_num,
                        action=action.get("description", "reason"),
                        tool_name=None,
                    ))

                step_num += 1

                # Check TTL
                if (datetime.now() - started_at).total_seconds() > self.config.agent_ttl:
                    logger.warning("AgentLoop[%s]: TTL exceeded", agent_id)
                    steps.append(StepLog(
                        step=step_num, action="abort",
                        tool_name=None, error="Agent TTL exceeded"
                    ))
                    break

            # Phase 3: SELF_EVAL
            eval_result = await self._self_evaluate(task_scope, steps, artifacts)

            # Phase 4: SUBMIT (or self-destruct)
            if eval_result >= self.config.accept_threshold:
                status = TaskStatus.DONE
                summary = await self._generate_summary(task_scope, steps, artifacts)
                logger.info("AgentLoop[%s]: completed (score=%.2f)", agent_id, eval_result)
            else:
                status = TaskStatus.FAILED
                summary = f"Self-evaluation failed (score={eval_result:.2f})"
                logger.warning("AgentLoop[%s]: self-destruct (score=%.2f)", agent_id, eval_result)

        except Exception as e:
            status = TaskStatus.FAILED
            summary = f"Agent error: {e}"
            artifact = {}
            steps.append(StepLog(
                step=len(steps) + 1, action="error",
                tool_name=None, error=str(e)
            ))
            logger.error("AgentLoop[%s]: error: %s", agent_id, e)

        return TaskResult(
            task_id=context.get("task_id", ""),
            agent_id=agent_id,
            status=status,
            summary=summary,
            artifacts=artifacts,
            steps=steps,
        )

    async def _plan(self, task_scope: str, context: dict) -> list[dict]:
        """Generate an execution plan for the task."""
        prompt = f"""You are an execution planner. Given the task and context, produce a step-by-step plan.

Task: {task_scope}

Context: {context}

Output a JSON array of steps, each with:
- "description": what this step does
- "tool": tool name to use (or null for pure reasoning)
- "params": tool parameters (or empty object)
- "output_key": key to store result under (or null)

Keep it concise, maximum {self.config.max_agent_steps} steps.
Respond with ONLY the JSON array, no other text."""

        try:
            response = await self.llm.chat([{"role": "user", "content": prompt}])
            import json
            plan = json.loads(response.content)
            return plan if isinstance(plan, list) else []
        except Exception as e:
            logger.error("Plan generation failed: %s", e)
            return [{"description": task_scope, "tool": None, "params": {}, "output_key": None}]

    async def _self_evaluate(self, task_scope: str, steps: list[StepLog],
                             artifacts: dict) -> float:
        """Agent self-evaluates its own output quality using LLM.

        Constructs an evaluation prompt, parses LLM JSON response.
        Falls back to simple heuristics on failure.
        """
        if not steps:
            return 0.0

        # Gather errors
        errors = [s.error for s in steps if s.error]

        # Build evaluation prompt
        prompt = f"""You are a quality evaluator. Rate the agent output on a 0-1 scale.

Task: {task_scope}
Steps completed: {len(steps)}
Artifacts: {list(artifacts.keys())}
Errors: {errors if errors else 'none'}

Output JSON only: {{"score": <number between 0 and 1>, "reason": "<brief explanation>"}}"""

        try:
            import json
            response = await self.llm.chat([{"role": "user", "content": prompt}])
            # Extract JSON from response (may be wrapped in markdown code block)
            content = response.content.strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            result = json.loads(content)
            score = float(result.get("score", 0.5))
            logger.info("LLM self-eval score=%.2f reason=%s", score, result.get("reason", ""))
            return min(1.0, max(0.0, score))

        except Exception as e:
            logger.warning("LLM self-eval failed: %s, falling back to heuristics", e)
            return self._heuristic_evaluate(steps)

    def _heuristic_evaluate(self, steps: list[StepLog]) -> float:
        """Fallback heuristic evaluation."""
        completeness = 1.0 if any(s.tool_name for s in steps) else 0.3
        correctness = 1.0 if not any(s.error for s in steps) else max(0.0, 1.0 - sum(1 for s in steps if s.error) / len(steps) * 0.5)
        relevance = 1.0 if len(steps) >= 2 else 0.5
        efficiency = 1.0 if len(steps) <= self.config.max_agent_steps * 0.5 else 0.6

        score = (
            completeness * 0.30 +
            correctness * 0.30 +
            relevance * 0.25 +
            efficiency * 0.15
        )
        return min(1.0, max(0.0, score))

    async def _generate_summary(self, task_scope: str, steps: list[StepLog],
                                artifacts: dict) -> str:
        """Generate a human-readable summary using LLM.

        Falls back to string concatenation on failure.
        """
        # Gather errors
        errors = [s.error for s in steps if s.error]

        prompt = f"""Summarize the following agent execution in 1-3 natural language sentences.

Task: {task_scope}
Steps executed: {len(steps)}
Step descriptions: {[s.action for s in steps]}
Artifacts produced: {list(artifacts.keys())}
Errors encountered: {errors if errors else 'none'}

Provide a concise summary:"""

        try:
            response = await self.llm.chat([{"role": "user", "content": prompt}])
            summary = response.content.strip()
            if summary:
                logger.info("LLM summary generated: %s", summary[:80])
                return summary
        except Exception as e:
            logger.warning("LLM summary failed: %s, falling back to string concat", e)

        # Fallback: simple string concatenation
        parts = [f"Task: {task_scope}"]
        for s in steps:
            parts.append(f"  Step {s.step}: {s.action}" + (f" [ERROR: {s.error}]" if s.error else ""))
        return "\n".join(parts)
