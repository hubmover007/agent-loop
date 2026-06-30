"""System Agents — the two fixed management agents.

Architecture (per owner's explicit requirement):

  MainLoop (orchestrator, not a manager)
    ├── TaskAgent (fixed) — manages task lifecycle
    │     • Receives reasoning output from MainLoop
    │     • LLM decomposes into subtasks
    │     • Registers tasks in TaskRegistry
    │     • Dependency-aware scheduling
    │     • Tracks task status (pending→running→done/failed)
    │     • Handles retries and re-planning
    │
    ├── AgentManagerAgent (fixed) — manages agent lifecycle
    │     • Creates worker agents on demand
    │     • Routes tasks to best-fit agents (MoE)
    │     • Tracks agent progress
    │     • Evaluates agent output quality
    │     • Destroys underperforming agents
    │     • Sends discarded results to discard pool
    │
    └── Worker Agents (dynamic, created by AgentManagerAgent)
          • Execute individual subtasks
          • Each has isolated BranchSpace
          • Commit results to mainline on success

This separation ensures:
  - TaskAgent never touches agent creation/destruction
  - AgentManagerAgent never touches task decomposition
  - Worker agents are ephemeral and replaceable
  - MainLoop only coordinates the 7-phase cycle
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .core import (
    TaskStatus, AgentStatus, TaskResult, EvaluationResult,
    StepLog, AgentRole, LoopPhase,
)
from .agent import Agent, AgentPool, AgentEvaluator
from .loop_engine import LLMProvider, LLMResponse, LoopConfig, AgentLoop
from .task import BranchSpace
from .external_agents import ExternalAgentBridge, KNOWN_AGENTS

logger = logging.getLogger(__name__)


# ============================================================
# Shared Task Registry (used by TaskAgent)
# ============================================================

@dataclass
class ManagedTask:
    """A task in the system, managed by TaskAgent."""

    task_id: str
    scope: str
    priority: int = 3
    dependencies: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)
    parent_id: str | None = None

    # Lifecycle
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: str | None = None
    result: TaskResult | None = None
    evaluation: EvaluationResult | None = None

    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Retry
    retry_count: int = 0
    max_retries: int = 2
    error: str | None = None
    branch_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "scope": self.scope,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "required_tools": self.required_tools,
            "status": self.status.value,
            "assigned_agent_id": self.assigned_agent_id,
            "created_at": self.created_at.isoformat(),
            "error": self.error,
            "retry_count": self.retry_count,
        }

    @property
    def is_ready(self) -> bool:
        """Ready to dispatch: pending and deps could be satisfied."""
        return self.status == TaskStatus.PENDING

    @property
    def is_terminal(self) -> bool:
        """Task has reached a final state."""
        return self.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)


class TaskRegistry:
    """Central task registry — single source of truth for all tasks."""

    def __init__(self):
        self._tasks: dict[str, ManagedTask] = {}
        self._order: list[str] = []
        self._lock = asyncio.Lock()  # asyncio-safe: concurrent assign() calls

    async def register(self, scope: str, priority: int = 3,
                       dependencies: list[str] | None = None,
                       required_tools: list[str] | None = None,
                       parent_id: str | None = None) -> ManagedTask:
        task = ManagedTask(
            task_id=f"task:{uuid.uuid4().hex[:12]}",
            scope=scope,
            priority=priority,
            dependencies=dependencies or [],
            required_tools=required_tools or [],
            parent_id=parent_id,
        )
        async with self._lock:
            self._tasks[task.task_id] = task
            self._order.append(task.task_id)
        return task

    def get(self, task_id: str) -> ManagedTask | None:
        return self._tasks.get(task_id)

    def all_tasks(self) -> list[ManagedTask]:
        return [self._tasks[tid] for tid in self._order]

    def get_ready(self) -> list[ManagedTask]:
        """Get tasks whose dependencies are all done, sorted by priority."""
        done_ids = {
            tid for tid, t in self._tasks.items()
            if t.status == TaskStatus.DONE
        }
        ready = [
            t for t in self._tasks.values()
            if t.status == TaskStatus.PENDING
            and all(dep in done_ids for dep in t.dependencies)
        ]
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    def get_by_status(self, status: TaskStatus) -> list[ManagedTask]:
        return [t for t in self._tasks.values() if t.status == status]

    def stats(self) -> dict:
        tasks = list(self._tasks.values())
        return {
            "total": len(tasks),
            "pending": sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            "running": sum(1 for t in tasks if t.status == TaskStatus.RUNNING),
            "done": sum(1 for t in tasks if t.status == TaskStatus.DONE),
            "failed": sum(1 for t in tasks if t.status == TaskStatus.FAILED),
        }


# ============================================================
# TaskAgent — manages task lifecycle (decompose, schedule, track)
# ============================================================

class TaskAgent:
    """System Agent #1: Task Manager.

    Responsibilities (ONLY task-related):
      1. Receive reasoning output from MainLoop
      2. Use LLM to decompose into subtasks with dependencies
      3. Register subtasks in TaskRegistry
      4. Schedule tasks by dependency + priority
      5. Track task status throughout lifecycle
      6. Handle retries and re-planning on failure

    Does NOT:
      - Create or destroy agents (that's AgentManagerAgent's job)
      - Execute tasks (that's Worker Agents' job)
      - Evaluate agent quality (that's AgentManagerAgent's job)
    """

    role = AgentRole.MANAGER

    def __init__(self, llm: LLMProvider, registry: TaskRegistry):
        self.llm = llm
        self.registry = registry

    async def decompose(self, reasoning_output: str,
                        original_input: str) -> list[ManagedTask]:
        """Decompose reasoning into subtasks using LLM."""
        try:
            subtasks = await self._llm_decompose(reasoning_output, original_input)
        except Exception as e:
            logger.warning("TaskAgent: LLM decompose failed: %s", e)
            subtasks = [{"scope": original_input, "priority": 3,
                        "dependencies": [], "required_tools": []}]

        scope_to_id: dict[str, str] = {}
        registered: list[ManagedTask] = []

        for st in subtasks:
            task = await self.registry.register(
                scope=st["scope"],
                priority=st.get("priority", 3),
                dependencies=[],
                required_tools=st.get("required_tools", []),
            )
            scope_to_id[st["scope"]] = task.task_id
            registered.append(task)

        # Resolve dependency edges
        for st, task in zip(subtasks, registered):
            dep_ids = [scope_to_id[d] for d in st.get("dependencies", [])
                       if d in scope_to_id and d != task.scope]
            task.dependencies = dep_ids

        logger.info("TaskAgent: decomposed → %d tasks", len(registered))
        return registered

    async def _llm_decompose(self, reasoning: str, original: str) -> list[dict]:
        prompt = f"""You are a Task Agent. Decompose the task into subtasks.

Original: {original}
Reasoning: {reasoning}

Rules:
- SIMPLE task → return single item
- Complex → 2-5 subtasks
- Dependencies: reference other subtask's scope
- priority: 5=urgent, 1=low

JSON:
```json
[{{"scope": "...", "priority": 3, "required_tools": [], "dependencies": []}}]
```"""
        resp = await self.llm.chat([{"role": "user", "content": prompt}])
        from ..utils import extract_json_from_llm_response
        result = extract_json_from_llm_response(resp.content, default=[])
        return result if isinstance(result, list) else [result]

    def get_ready_tasks(self) -> list[ManagedTask]:
        """Get tasks ready for dispatch (deps satisfied)."""
        return self.registry.get_ready()

    async def update_status(self, task_id: str, status: TaskStatus,
                            result: TaskResult | None = None,
                            error: str | None = None) -> None:
        """Update task status."""
        task = self.registry.get(task_id)
        if not task:
            return

        task.status = status
        if result:
            task.result = result
        if error:
            task.error = error
        if status in (TaskStatus.RUNNING,):
            task.started_at = datetime.now(timezone.utc)
        if status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED):
            task.completed_at = datetime.now(timezone.utc)

    async def replan(self, failed_task: ManagedTask) -> list[ManagedTask] | None:
        """Re-plan a failed task using LLM."""
        try:
            prompt = f"""A task failed. Re-plan it.

Failed: {failed_task.scope}
Error: {failed_task.error}
Attempts: {failed_task.retry_count}

Return JSON array of alternative subtasks, or empty array if impossible."""
            resp = await self.llm.chat([{"role": "user", "content": prompt}])
            from ..utils import extract_json_from_llm_response
            subtasks = extract_json_from_llm_response(resp.content, default=[])
            if not subtasks:
                return None

            new_tasks = []
            for st in subtasks:
                t = await self.registry.register(
                    scope=st["scope"],
                    priority=st.get("priority", failed_task.priority),
                    parent_id=failed_task.parent_id,
                )
                new_tasks.append(t)
            logger.info("TaskAgent: re-planned %s → %d new tasks",
                       failed_task.task_id, len(new_tasks))
            return new_tasks
        except Exception as e:
            logger.error("TaskAgent: replan failed: %s", e)
            return None


# ============================================================
# AgentManagerAgent — manages agent lifecycle (create, route, track, destroy)
# ============================================================

class AgentManagerAgent:
    """System Agent #2: Agent Manager.

    Responsibilities (ONLY agent-related):
      1. Create worker agents on demand
      2. Route tasks to best-fit agents (MoE-style)
      3. Track agent progress
      4. Evaluate agent output quality (4-dimension scoring)
      5. Destroy underperforming agents
      6. Send discarded results to discard pool for audit

    Does NOT:
      - Decompose tasks (that's TaskAgent's job)
      - Manage task dependencies (that's TaskAgent's job)
      - Execute tasks (that's Worker Agents' job)
    """

    role = AgentRole.MANAGER

    def __init__(self, memory: Any, agent_loop: AgentLoop,
                 config: LoopConfig, registry: TaskRegistry,
                 external_bridge: ExternalAgentBridge | None = None,
                 llm_pool: Any | None = None,
                 state_store: Any | None = None):
        self.memory = memory
        self.agent_loop = agent_loop
        self.config = config
        self.registry = registry

        # Optional LLM Pool for provider selection
        self.llm_pool = llm_pool

        # Optional StateStore for structured persistence
        self.state_store = state_store

        # Internal agent pool
        self.pool = AgentPool(max_concurrent=config.max_agent_concurrent)
        self.evaluator = AgentEvaluator(
            accept_threshold=config.accept_threshold,
            weights=config.evaluation_weights,
        )

        # External agent bridge (Codex, Claude Code, Gemini, etc.)
        self.external_bridge = external_bridge or ExternalAgentBridge()

        # In-flight tracking: task_id → (agent, asyncio.Task)
        self._inflight: dict[str, tuple[Agent, asyncio.Task]] = {}

    async def assign(self, task: ManagedTask,
                     prefer_external: bool = False) -> bool:
        """Assign a task to a worker agent (internal or external).

        Decision logic:
          1. If task.required_tools contains external agent name → dispatch externally
          2. If prefer_external=True → try external first, fallback to internal
          3. Otherwise → use internal Worker Agent

        Args:
            task: The task to assign
            prefer_external: Prefer external agents (Codex/Claude) when available

        Returns True if successfully assigned.
        """
        if len(self._inflight) >= self.config.max_agent_concurrent:
            return False

        # Check if task should go to an external agent
        external_agent = self._should_use_external(task, prefer_external)
        if external_agent:
            return await self._assign_external(task, external_agent)

        # Use internal worker agent
        return await self._assign_internal(task)

    def _should_use_external(self, task: ManagedTask,
                             prefer_external: bool) -> str | None:
        """Determine if a task should be dispatched to an external agent.

        Returns external agent_type if external dispatch is recommended, None otherwise.
        """
        # Check if required_tools explicitly mentions an external agent
        for tool in task.required_tools:
            if tool in KNOWN_AGENTS and self.external_bridge.is_available(tool):
                return tool

        # If prefer_external, recommend best agent based on task scope
        if prefer_external:
            # Simple heuristic: coding tasks go to external coding agents
            scope_lower = task.scope.lower()
            if any(kw in scope_lower for kw in ["code", "debug", "refactor",
                                                 "implement", "fix", "test"]):
                recommended = self.external_bridge.recommend_agent(
                    task.scope, required_capabilities=["code"]
                )
                if recommended:
                    return recommended

        return None

    async def _assign_internal(self, task: ManagedTask) -> bool:
        """Assign task to an internal worker agent."""
        agent = await self._acquire_worker(task)
        if not agent:
            return False

        task.status = TaskStatus.RUNNING
        task.assigned_agent_id = agent.agent_id
        task.started_at = datetime.now(timezone.utc)

        asyncio_task = asyncio.create_task(self._execute(task, agent))
        self._inflight[task.task_id] = (agent, asyncio_task)

        logger.info("AgentManager: assigned %s → internal %s",
                    task.task_id, agent.agent_id)
        return True

    async def _assign_external(self, task: ManagedTask,
                               agent_type: str) -> bool:
        """Assign task to an external agent (Codex, Claude Code, etc.).

        Uses ExternalAgentBridge to dispatch via ACP or CLI.
        """
        task.status = TaskStatus.RUNNING
        task.assigned_agent_id = f"external:{agent_type}:{task.task_id}"
        task.started_at = datetime.now(timezone.utc)

        asyncio_task = asyncio.create_task(self._execute_external(task, agent_type))
        # Store with a placeholder Agent for compatibility
        placeholder = Agent(agent_id=task.assigned_agent_id)
        placeholder.role = AgentRole.EXPERT
        placeholder.expertise = [agent_type]
        self._inflight[task.task_id] = (placeholder, asyncio_task)

        logger.info("AgentManager: assigned %s → external %s",
                    task.task_id, agent_type)
        return True

    async def _acquire_worker(self, task: ManagedTask) -> Agent | None:
        """Find idle agent or create new one.

        If llm_pool is available, assigns an LLM provider based on
        capability requirements derived from the task scope.
        """
        # Try to reuse idle agent
        for agent in self.pool.agents.values():
            if agent.status == AgentStatus.IDLE:
                # Could do MoE matching here based on task.required_tools
                return agent

        # Create new worker
        agent = await self.pool.acquire()
        if agent:
            agent.role = AgentRole.WORKER
            agent.expertise = task.required_tools

            # Use llm_pool to select LLM provider for this agent if available
            if self.llm_pool:
                try:
                    capabilities = self._infer_capabilities(task)
                    provider = await self.llm_pool.acquire(
                        capabilities=capabilities,
                        strategy="balanced",
                    )
                    # Attach provider reference to agent (stored in external data)
                    agent._llm_provider = provider
                    if self.state_store:
                        agent._llm_provider_id = provider.provider_id
                except Exception as e:
                    logger.warning("AgentManager: llm_pool acquire failed: %s", e)

        return agent

    @staticmethod
    def _infer_capabilities(task: ManagedTask) -> list[str]:
        """Infer required LLM capabilities from task scope and required_tools."""
        capabilities: set = set()
        scope_lower = task.scope.lower()

        if any(kw in scope_lower for kw in ["code", "debug", "implement", "refactor"]):
            capabilities.add("coding")
        if any(kw in scope_lower for kw in ["reason", "analyze", "plan", "think"]):
            capabilities.add("reasoning")
        if any(kw in scope_lower for kw in ["math", "calculate", "compute"]):
            capabilities.add("math")
        if any(kw in scope_lower for kw in ["long", "document", "report"]):
            capabilities.add("long_context")

        if not capabilities:
            capabilities.add("general")

        return list(capabilities)

    async def _execute(self, task: ManagedTask, agent: Agent) -> None:
        """Execute a task with a worker agent.

        Full lifecycle:
          1. Create BranchSpace (isolated workspace)
          2. Agent runs task via AgentLoop
          3. Evaluate result (4-dimension scoring)
          4. Accept → commit to mainline → release agent
          5. Discard → retry or fail → destroy agent → discard pool
        """
        # Save agent state on start (if state_store available)
        if self.state_store:
            try:
                await self.state_store.save_agent(agent.agent_id, {
                    "agent_id": agent.agent_id,
                    "status": agent.status,
                    "role": agent.role,
                    "expertise": agent.expertise,
                    "task_count": agent.task_count,
                    "success_count": agent.success_count,
                    "created_at": agent.created_at,
                    "llm_provider_id": getattr(agent, '_llm_provider_id', None),
                })
            except Exception as e:
                logger.warning("AgentManager: state_store.save_agent failed: %s", e)

        # Create branch space
        branch = BranchSpace(task_id=task.task_id, agent_id=agent.agent_id)
        try:
            await branch.init(self.memory)
            task.branch_id = task.task_id
            branch.log("task_started", {"scope": task.scope})
        except Exception as e:
            logger.warning("AgentManager: branch init failed: %s", e)

        try:
            context = {
                "task_id": task.task_id,
                "scope": task.scope,
                "priority": task.priority,
                "required_tools": task.required_tools,
                "branch_dir": str(branch.base_dir),
            }

            result = await agent.run(
                self.agent_loop, task.scope, context,
                allowed_tools=task.required_tools,
            )

            branch.log("task_completed", {"summary": result.summary})

            # Evaluate quality
            eval_result = self.evaluator.evaluate(result, task.scope)
            task.evaluation = eval_result

            if eval_result.action == "accept":
                # Accept: commit to mainline
                task.status = TaskStatus.DONE
                task.result = result
                task.completed_at = datetime.now(timezone.utc)
                await branch.commit(self.memory)
                logger.info("AgentManager: %s DONE (score=%.2f)",
                           task.task_id, eval_result.overall)
            else:
                # Discard: retry or fail
                branch.log("task_discarded", {"reason": eval_result.reason})
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.assigned_agent_id = None
                    logger.info("AgentManager: %s retry %d/%d",
                               task.task_id, task.retry_count, task.max_retries)
                else:
                    # Permanently fail + destroy agent
                    await self._destroy_agent_on_failure(task, agent, eval_result.reason)
                    return

            # Save task state on completion (if state_store available)
            if self.state_store:
                try:
                    await self.state_store.save_task(task)
                except Exception as e:
                    logger.warning("AgentManager: state_store.save_task failed: %s", e)

        except Exception as e:
            branch.log("task_exception", {"error": str(e)})
            await self._destroy_agent_on_failure(task, agent, f"Exception: {e}")
            # Save failed task state
            if self.state_store:
                try:
                    await self.state_store.save_task(task)
                except Exception as se:
                    logger.warning("AgentManager: state_store.save_task failed: %s", se)
            logger.error("AgentManager: %s exception: %s", task.task_id, e)

        finally:
            await branch.cleanup()
            # Release agent back to pool if not destroyed
            if agent.agent_id in self.pool.agents:
                await self.pool.release(agent.agent_id)
            self._inflight.pop(task.task_id, None)

    async def _destroy_agent_on_failure(self, task: ManagedTask, agent: Agent,
                                        reason: str) -> None:
        """Destroy an agent that failed permanently and record to discard pool."""
        task.status = TaskStatus.FAILED
        task.error = reason
        task.completed_at = datetime.now(timezone.utc)

        # Send to discard pool for audit
        if self.memory:
            try:
                await self.memory.discard_result(
                    agent_id=agent.agent_id,
                    task_id=task.task_id,
                    reason=reason,
                )
            except Exception as e:
                logger.warning("AgentManager: discard pool write failed: %s", e)

        # Destroy the agent (only if internal)
        if not agent.agent_id.startswith("external:"):
            await self.pool.destroy(agent.agent_id)
        logger.warning("AgentManager: agent %s destroyed (task %s failed: %s)",
                      agent.agent_id, task.task_id, reason)

    async def _execute_external(self, task: ManagedTask, agent_type: str) -> None:
        """Execute a task via an external agent (Codex, Claude Code, etc.).

        Uses ExternalAgentBridge to dispatch and collect results.
        External agents have their own evaluation — we trust their output
        but still run through quality evaluation.
        """
        try:
            result_data = await self.external_bridge.dispatch(
                agent_type=agent_type,
                task=task.scope,
                workspace=task.branch_id,
                timeout=300,
            )

            # Build TaskResult from external output
            external_result = TaskResult(
                task_id=task.task_id,
                agent_id=task.assigned_agent_id or "",
                status=TaskStatus.DONE if result_data.get("status") == "done"
                       else TaskStatus.FAILED,
                summary=result_data.get("output", "")[:500],
                artifacts={"full_output": result_data.get("output", ""),
                          "dispatch_method": result_data.get("dispatch_method", "")},
                steps=[],
            )

            if external_result.status == TaskStatus.DONE:
                # Evaluate quality (simplified for external agents)
                eval_result = self.evaluator.evaluate(external_result, task.scope)
                task.evaluation = eval_result

                if eval_result.action == "accept":
                    task.status = TaskStatus.DONE
                    task.result = external_result
                    task.completed_at = datetime.now(timezone.utc)
                    logger.info("AgentManager: external %s → %s DONE",
                               agent_type, task.task_id)
                else:
                    # External result not good enough — retry or fail
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        task.status = TaskStatus.PENDING
                        task.assigned_agent_id = None
                        logger.info("AgentManager: external %s → %s retry",
                                   agent_type, task.task_id)
                    else:
                        task.status = TaskStatus.FAILED
                        task.error = eval_result.reason
                        task.completed_at = datetime.now(timezone.utc)
            else:
                # External agent failed
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.assigned_agent_id = None
                else:
                    task.status = TaskStatus.FAILED
                    task.error = result_data.get("error", "External agent failed")
                    task.completed_at = datetime.now(timezone.utc)

                    if self.memory:
                        await self.memory.discard_result(
                            agent_id=task.assigned_agent_id or agent_type,
                            task_id=task.task_id,
                            reason=task.error or "External agent failed",
                        )

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = f"External agent exception: {e}"
            task.completed_at = datetime.now(timezone.utc)
            logger.error("AgentManager: external %s exception: %s",
                        agent_type, e)
        finally:
            self._inflight.pop(task.task_id, None)

    async def collect_all(self, timeout: float = 300.0) -> dict[str, TaskResult]:
        """Wait for all in-flight tasks to complete."""
        if not self._inflight:
            return {}

        try:
            await asyncio.wait_for(
                asyncio.gather(*[t for _, t in self._inflight.values()],
                              return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("AgentManager: collect_all timed out")

        results = {}
        for task in self.registry.all_tasks():
            if task.result:
                results[task.task_id] = task.result
        return results

    async def cancel(self, task_id: str) -> None:
        """Cancel a task and destroy its agent."""
        pair = self._inflight.pop(task_id, None)
        if pair:
            agent, asyncio_task = pair
            asyncio_task.cancel()
            await self.pool.destroy(agent.agent_id)

        task = self.registry.get(task_id)
        if task:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now(timezone.utc)

    def stats(self) -> dict:
        pool_stats = self.pool.stats()
        return {
            **self.registry.stats(),
            "inflight": len(self._inflight),
            "agents_active": pool_stats["active"],
            "agents_idle": pool_stats["idle"],
        }
