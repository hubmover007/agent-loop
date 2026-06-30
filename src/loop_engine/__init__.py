"""Loop Engine - the heart of Agent-Loop.

Three levels of loops:
  MainLoop:  INPUT → RETRIEVE → REASON → DECOMPOSE → DISPATCH → COLLECT → OUTPUT
  AgentLoop: INIT → PLAN → EXECUTE → SELF_EVAL → SUBMIT → DESTROY
  ToolLoop:  CALL → VERIFY → RETRY(3x) → RETURN/FAIL
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from ..core import (
    AgentStatus, LoopPhase, TaskStatus, ToolResult, ToolResultStatus,
    StepLog, TaskResult, EvaluationResult,
)
from ..tools.base import ToolInterface, ToolRegistry

logger = logging.getLogger(__name__)

# ── P0: Default limits ───────────────────────────────────────────
DEFAULT_MAX_ITERATIONS = 20
DEFAULT_TIMEOUT_SECONDS = 300


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
    finish_reason: str = "stop"  # stop | tool_calls | length
    tool_calls: list[dict] | None = None  # OpenAI-format tool_calls


# ── P0: Tool Calling / Streaming types ───────────────────────────

@dataclass
class ToolFunction:
    """Function definition in a tool call."""
    name: str
    arguments: str  # JSON string


@dataclass
class ToolCall:
    """A single tool call from the LLM."""
    id: str
    function: ToolFunction


@dataclass
class ChatResponse:
    """High-level chat response with parsed tool calls."""
    content: str
    tool_calls: list[ToolCall] | None = None
    finish_reason: str = "stop"


@dataclass
class ChatStreamChunk:
    """A single streaming chunk from the LLM."""
    delta_content: str = ""
    delta_tool_calls: list[dict] | None = None
    finish_reason: str | None = None


class LLMProvider(ABC):
    """Abstract LLM provider."""

    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse:
        """Send a chat completion request.

        Providers SHOULD accept these optional kwargs:
            tools: OpenAI-format tool definitions list
            tool_choice: "auto" | "none" | "required" | specific dict
        """
        ...

    async def chat_stream(self, messages: list[dict],
                          tools: list[dict] | None = None) -> AsyncGenerator:
        """Stream chat response token-by-token.

        Default: falls back to chat() and yields a single chunk.
        Providers SHOULD override for true streaming.

        Yields:
            ChatStreamChunk
        """
        response = await self.chat(messages, tools=tools)
        yield ChatStreamChunk(
            delta_content=response.content,
            finish_reason=response.finish_reason,
        )

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

    # Unified memory context (set during RETRIVE phase)
    memory_context: Any | None = None

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
            start = datetime.now(timezone.utc)

            try:
                result = await tool.execute(**params)
            except Exception as e:
                result = ToolResult(
                    status=ToolResultStatus.TRANSIENT_ERROR,
                    error=str(e),
                )

            elapsed = (datetime.now(timezone.utc) - start).total_seconds() * 1000
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

    def __init__(self, tool_loop: ToolLoop, llm: LLMProvider, config: LoopConfig,
                 agent_soul: Any | None = None,
                 interaction_hub: Any | None = None,
                 mailbox: Any | None = None,
                 evolution: Any | None = None,
                 emitter: Any | None = None,
                 tool_registry: Any | None = None,
                 multimodal_processor: Any | None = None,
                 sandbox: Any | None = None,
                 agent_permissions: Any | None = None,
                 max_iterations: int = DEFAULT_MAX_ITERATIONS,
                 timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS):
        self.tool_loop = tool_loop
        self.llm = llm
        self.config = config
        self.agent_soul = agent_soul  # Optional AgentSoul for EVOLVE
        self.interaction_hub = interaction_hub  # Optional Human-in-the-Loop hub
        self.mailbox = mailbox  # Optional AgentMailbox for inter-agent comms
        self.evolution = evolution  # Optional EvolutionEngine for structured EVOLVE
        self.emitter = emitter  # Optional ProgressEmitter for streaming
        self.tool_registry = tool_registry  # Optional MCP ToolRegistry
        self.multimodal_processor = multimodal_processor  # Optional MultimodalProcessor
        self.sandbox = sandbox  # Optional SandboxManager for code execution
        self.agent_permissions = agent_permissions  # Optional AgentPermissions for access control
        self.max_iterations = max_iterations  # P0: Max agent loop iterations
        self.timeout_seconds = timeout_seconds  # P0: Agent timeout in seconds

    async def run(self, agent_id: str, task_scope: str, context: dict,
                  allowed_tools: list[str],
                  multimodal_inputs: list | None = None) -> TaskResult:
        """Execute AgentLoop: PLAN → EXECUTE → SELF_EVAL → EVOLVE → SUBMIT.

        Phases:
          1. PLAN — generate execution steps
          2. EXECUTE — run steps with tool calls (via tool_registry if available)
          3. SELF_EVAL — LLM self-evaluation
          4. EVOLVE — write JOURNAL.md + update profile.json (if score >= threshold)
          5. SUBMIT — return result (or self-destruct on failure)

        If multimodal_inputs is provided and multimodal_processor is set,
        media inputs are processed into content blocks attached to context.

        This runs within an isolated BranchSpace.
        """
        steps: list[StepLog] = []
        artifacts: dict = {}
        iteration = 0
        started_at = datetime.now(timezone.utc)
        loop_start_time = time.time()

        # ── P0: Iteration & timeout guard ──────────────────────────
        # AgentLoop.run() is a single-pass per invocation; the iteration
        # counter is checked once here and at timeout points.
        # Multi-pass retry logic lives at MainLoop level.
        iteration += 1
        if iteration > self.max_iterations:
            logger.warning("AgentLoop[%s]: max iterations (%d) reached before start",
                           agent_id, self.max_iterations)
            return TaskResult(
                task_id=context.get("task_id", ""),
                agent_id=agent_id,
                status=TaskStatus.FAILED,
                summary=f"Max iterations ({self.max_iterations}) reached",
            )
        if time.time() - loop_start_time > self.timeout_seconds:
            logger.warning("AgentLoop[%s]: timeout %ds exceeded before start",
                           agent_id, self.timeout_seconds)
            return TaskResult(
                task_id=context.get("task_id", ""),
                agent_id=agent_id,
                status=TaskStatus.FAILED,
                summary=f"Timeout after {self.timeout_seconds}s",
            )

        # ── Multimodal input processing ──────────────────────────
        if multimodal_inputs and self.multimodal_processor:
            try:
                blocks = await self.multimodal_processor.process(multimodal_inputs)
                context["multimodal_blocks"] = blocks
                logger.info("AgentLoop[%s]: processed %d multimodal blocks", agent_id, len(blocks))
            except Exception as e:
                logger.warning("AgentLoop[%s]: multimodal processing failed: %s", agent_id, e)

        logger.info("AgentLoop[%s] started: %s", agent_id, task_scope[:50])

        # Phase emission helper (called inline, non-blocking)
        def _emit(ev_type, phase, msg, **kw):
            if self.emitter:
                self.emitter.emit(ev_type, phase, msg, **kw)

        try:
            # Phase 1: PLAN
            _emit("phase_start", "PLAN", f"Starting plan for: {task_scope[:80]}")

            # ── P0: Timeout guard ──────────────────────────────
            if time.time() - loop_start_time > self.timeout_seconds:
                return TaskResult(
                    task_id=context.get("task_id", ""),
                    agent_id=agent_id,
                    status=TaskStatus.FAILED,
                    summary=f"Timeout after {self.timeout_seconds}s",
                )

            plan = await self._plan(task_scope, context)
            steps.append(StepLog(step=0, action="plan", tool_name=None, output=plan))
            _emit("phase_done", "PLAN", f"Plan generated: {len(plan)} steps")

            # ── P0: Timeout guard after plan ──────────────────
            if time.time() - loop_start_time > self.timeout_seconds:
                return TaskResult(
                    task_id=context.get("task_id", ""),
                    agent_id=agent_id,
                    status=TaskStatus.FAILED,
                    summary=f"Timeout after {self.timeout_seconds}s",
                )

            # Phase 2: EXECUTE
            _emit("phase_start", "EXECUTE", f"Executing {len(plan)} steps")
            step_num = 1
            for action in plan:
                # ── P0: Timeout guard during execution ─────────
                if time.time() - loop_start_time > self.timeout_seconds:
                    return TaskResult(
                        task_id=context.get("task_id", ""),
                        agent_id=agent_id,
                        status=TaskStatus.FAILED,
                        summary=f"Timeout after {self.timeout_seconds}s",
                    )

                if step_num > self.config.max_agent_steps:
                    steps.append(StepLog(
                        step=step_num, action="abort",
                        tool_name=None, error="Max steps exceeded"
                    ))
                    break

                # Check if this step requires a tool call
                tool_name = action.get("tool")
                # Allow tool if: (1) in allowed_tools list, OR (2) registered in tool_registry
                tool_allowed = (tool_name in allowed_tools) if allowed_tools else True
                if not tool_allowed and self.tool_registry is not None:
                    spec = self.tool_registry.get(tool_name)
                    tool_allowed = spec is not None and spec.enabled
                if tool_name and tool_allowed:
                    # ── Human-in-the-Loop: check high-risk operations ─────
                    if self.interaction_hub is not None:
                        from ..interaction import detect_risk_level, RISK_LEVEL_ORDER
                        action_desc = action.get("description", "") or str(action.get("params", {}))
                        tool_cmd = f"{tool_name} {action_desc}"
                        risk = detect_risk_level(tool_cmd)
                        if risk and RISK_LEVEL_ORDER.get(risk, 0) >= RISK_LEVEL_ORDER.get(self.interaction_hub._risk_threshold, 1):
                            _emit("approval_needed", "EXECUTE",
                                  f"Approval needed: {tool_name}", risk_level=risk)
                            try:
                                approval = await self.interaction_hub.request_approval(
                                    agent_id=agent_id,
                                    action=f"{tool_name}: {action_desc}",
                                    details=str(action.get("params", {})),
                                    risk_level=risk,
                                    task_scope=task_scope,
                                )
                                if approval.status != "approved":
                                    steps.append(StepLog(
                                        step=step_num,
                                        action=f"blocked: {tool_name}",
                                        tool_name=tool_name,
                                        error=f"User {approval.status}: {approval.reply}",
                                    ))
                                    break
                            except Exception as e:
                                logger.warning("AgentLoop[%s]: interaction hub error: %s", agent_id, e)

                    # ── Mailbox: check for incoming messages before step ────
                    if self.mailbox is not None:
                        try:
                            msg = await self.mailbox.receive(timeout=0.5)
                            if msg and msg.type == "handoff":
                                steps.append(StepLog(
                                    step=step_num,
                                    action=f"handoff from {msg.from_agent}",
                                    tool_name=None,
                                    output=msg.body,
                                ))
                                logger.info("AgentLoop[%s]: received handoff from %s", agent_id, msg.from_agent)
                        except Exception:
                            pass  # Mailbox receive timeout is fine

                    _emit("tool_call", "EXECUTE", f"Calling {tool_name}",
                          step=step_num, params=action.get("params", {}))

                    # ── Sandbox: code execution tools go through SandboxManager ──
                    if (self.sandbox is not None
                            and tool_name in ('code.execute', 'shell.run_command', 'run_code',
                                             'shell.exec', 'python.exec')):
                        tool_params = action.get("params", {})
                        command = tool_params.get('command', tool_params.get('code', ''))
                        language = tool_params.get('language',
                            'python' if tool_name in ('run_code', 'python.exec') else 'shell')
                        try:
                            sandbox_result = await self.sandbox.execute_command(
                                command,
                                permissions=self.agent_permissions,
                                interaction_hub=self.interaction_hub,
                            )
                            if sandbox_result.get('exit_code', -1) != 0:
                                result = ToolResult(
                                    status=ToolResultStatus.TRANSIENT_ERROR,
                                    error=sandbox_result.get('stderr', 'Sandbox execution failed'),
                                    data=sandbox_result,
                                )
                            else:
                                result = ToolResult(
                                    status=ToolResultStatus.SUCCESS,
                                    data=sandbox_result,
                                )
                        except Exception as se:
                            result = ToolResult(
                                status=ToolResultStatus.TRANSIENT_ERROR,
                                error=f"Sandbox error: {se}",
                            )
                    else:
                        result = await self.tool_loop.execute(tool_name, **action.get("params", {}))

                    _emit("tool_call", "EXECUTE", f"{tool_name} returned: {result.status.value}",
                          step=step_num, status=result.status.value)
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
                if (datetime.now(timezone.utc) - started_at).total_seconds() > self.config.agent_ttl:
                    logger.warning("AgentLoop[%s]: TTL exceeded", agent_id)
                    steps.append(StepLog(
                        step=step_num, action="abort",
                        tool_name=None, error="Agent TTL exceeded"
                    ))
                    break

            _emit("phase_done", "EXECUTE",
                  f"Execution complete: {len([s for s in steps if s.tool_name])} tool calls")

            # Phase 3: SELF_EVAL
            _emit("phase_start", "SELF_EVAL", "Running self-evaluation")
            eval_result = await self._self_evaluate(task_scope, steps, artifacts)
            _emit("phase_done", "SELF_EVAL", f"Self-eval score: {eval_result:.2f}")

            # Phase 4: EVOLVE — structured evolution + legacy journal
            _emit("phase_start", "EVOLVE", f"Starting evolution (score={eval_result:.2f})")

            # ── Structured evolution (EvolutionEngine) ──
            if self.evolution:
                try:
                    from ..evolution import JournalEntry as JEntry
                    import uuid as _uuid
                    from datetime import timezone as _tz, datetime as _dt

                    tools_used = list(set(
                        s.tool_name for s in steps if s.tool_name
                    ))
                    duration = (_dt.now(_tz.utc) - started_at).total_seconds()

                    task_type = "coding" if any(kw in task_scope.lower()
                        for kw in ["code", "debug", "implement", "refactor", "pull"]) \
                        else "reasoning" if any(kw in task_scope.lower()
                        for kw in ["reason", "analyze", "plan", "think"]) \
                        else "ops" if any(kw in task_scope.lower()
                        for kw in ["deploy", "ssh", "docker", "server"]) \
                        else "general"

                    outcome = "success" if eval_result >= self.config.accept_threshold else \
                        "partial" if eval_result >= self.config.accept_threshold * 0.7 else "failure"

                    entry = JEntry(
                        id=f"entry:{_uuid.uuid4().hex[:8]}",
                        timestamp=_dt.now(_tz.utc).isoformat(),
                        task_scope=task_scope,
                        task_type=task_type,
                        outcome=outcome,
                        score=round(eval_result, 2),
                        duration_seconds=duration,
                        tools_used=tools_used,
                        llm_provider=getattr(self.llm, "model", "unknown"),
                        cost_estimate=duration * 0.001,
                        lessons=[s.error for s in steps if s.error] if outcome == "failure" else [],
                        tags=[task_type, outcome],
                    )
                    await self.evolution.record_entry(entry)
                    await self.evolution.adjust_traits(entry)
                    await self.evolution.extract_knowledge()
                    logger.debug("AgentLoop[%s]: EVOLVE — structured evolution recorded", agent_id)
                except Exception as ev:
                    logger.warning("AgentLoop[%s]: EVOLVE (structured) failed: %s", agent_id, ev)

            # ── Legacy evolution (AgentSoul) for backward compat ──
            if eval_result >= self.config.accept_threshold and self.agent_soul:
                try:
                    await self.agent_soul.evolve(
                        f"Completed: {task_scope}. Score: {eval_result:.2f}"
                    )
                    # Reward: increase efficiency on success
                    await self.agent_soul.update_identity_trait("efficiency", +0.02)
                    await self.agent_soul.record_task(success=True)
                    logger.debug("AgentLoop[%s]: EVOLVE — soul updated", agent_id)
                except Exception as ev:
                    logger.warning("AgentLoop[%s]: EVOLVE failed: %s", agent_id, ev)
            elif self.agent_soul:
                # Record failure for learning
                try:
                    await self.agent_soul.evolve(
                        f"Failed: {task_scope}. Score: {eval_result:.2f}"
                    )
                    # Punish: decrease efficiency on failure
                    await self.agent_soul.update_identity_trait("efficiency", -0.05)
                    await self.agent_soul.record_task(success=False)
                except Exception as ev:
                    logger.warning("AgentLoop[%s]: EVOLVE failed: %s", agent_id, ev)

            # ── Self-modification: Agent autonomously adjusts IDENTITY on high scores ──
            if self.agent_soul and hasattr(self.agent_soul, 'self_modify'):
                if eval_result > 0.85:
                    try:
                        # Only self-modify if permissions allow (or no permissions check)
                        can_modify = (
                            self.agent_permissions is None
                            or self.agent_permissions.can_modify_file('identity')
                        )
                        if can_modify:
                            current = self.agent_soul.identity_content
                            if 'experience' not in current.lower():
                                new_content = current + '\n\n## 经验\n积累了成功的执行经验。'
                                await self.agent_soul.self_modify(
                                    'identity', new_content, self.agent_permissions
                                )
                                logger.debug(
                                    "AgentLoop[%s]: EVOLVE — self_modify IDENTITY enhanced",
                                    agent_id
                                )
                    except Exception:
                        pass  # self_modify failure doesn't affect main flow

            _emit("phase_done", "EVOLVE", f"Evolution complete")

            # Phase 5: SUBMIT (or self-destruct)
            if eval_result >= self.config.accept_threshold:
                status = TaskStatus.DONE
                summary = await self._generate_summary(task_scope, steps, artifacts)
                logger.info("AgentLoop[%s]: completed (score=%.2f)", agent_id, eval_result)
            else:
                status = TaskStatus.FAILED
                summary = f"Self-evaluation failed (score={eval_result:.2f})"
                logger.warning("AgentLoop[%s]: self-destruct (score=%.2f)", agent_id, eval_result)

        except Exception as e:
            _emit("error", "EXECUTE", f"Agent error: {e}", error=str(e))
            status = TaskStatus.FAILED
            summary = f"Agent error: {e}"
            artifacts = {}
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
            response = await asyncio.wait_for(
                self.llm.chat([{"role": "user", "content": prompt}]),
                timeout=60.0,
            )
            from ..utils import extract_json_from_llm_response
            plan = extract_json_from_llm_response(response.content, default=[])
            return plan if isinstance(plan, list) else []
        except asyncio.TimeoutError:
            logger.error("Plan generation timed out")
            return [{"description": task_scope, "tool": None, "params": {}, "output_key": None}]
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
            response = await asyncio.wait_for(
                self.llm.chat([{"role": "user", "content": prompt}]),
                timeout=30.0,
            )
            from ..utils import extract_json_from_llm_response
            result = extract_json_from_llm_response(response.content, default={})
            score = float(result.get("score", 0.5))
            logger.info("LLM self-eval score=%.2f reason=%s", score, result.get("reason", ""))
            return min(1.0, max(0.0, score))

        except asyncio.TimeoutError:
            logger.warning("LLM self-eval timed out, falling back to heuristics")
            return self._heuristic_evaluate(steps)
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
            response = await asyncio.wait_for(
                self.llm.chat([{"role": "user", "content": prompt}]),
                timeout=30.0,
            )
            summary = response.content.strip()
            if summary:
                logger.info("LLM summary generated: %s", summary[:80])
                return summary
        except asyncio.TimeoutError:
            logger.warning("LLM summary timed out, falling back to string concat")
        except Exception as e:
            logger.warning("LLM summary failed: %s, falling back to string concat", e)

        # Fallback: simple string concatenation
        parts = [f"Task: {task_scope}"]
        for s in steps:
            parts.append(f"  Step {s.step}: {s.action}" + (f" [ERROR: {s.error}]" if s.error else ""))
        return "\n".join(parts)
