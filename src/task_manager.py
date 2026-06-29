"""TaskManager Agent — the central orchestrator agent.

Inspired by:
  - AgentOrchestra (arxiv 2506.12508): Planning Agent decomposes → delegates → collects
  - AutoGen Orchestrator-Worker: central orchestrator dispatches to workers
  - CrewAI Hierarchical: manager agent dynamically creates tasks and delegates

Architecture:
  MainLoop (user input → reason)
    → TaskManager Agent (decompose → register → dispatch → track → collect)
      → Worker Agent 1 (execute subtask)
      → Worker Agent 2 (execute subtask)
      → ...

TaskManager is itself an Agent with LLM access. It:
  1. Receives the reasoning output from MainLoop
  2. Uses LLM to decompose into subtasks (with dependencies)
  3. Registers subtasks in the TaskRegistry (persisted to memory)
  4. Dispatches ready tasks to available Worker Agents
  5. Monitors progress, handles retries and failures
  6. Collects results and returns to MainLoop

This replaces the old TaskPipeline with an agent-driven approach where
the TaskManager has full autonomy over task lifecycle management.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .core import TaskStatus, AgentStatus, TaskResult, EvaluationResult, StepLog, AgentRole
from .agent import AgentPool, AgentRouter, AgentEvaluator, Agent
from .loop_engine import LLMProvider, LLMResponse, LoopConfig, AgentLoop

logger = logging.getLogger(__name__)


# ============================================================
# Task Registry — persisted task store
# ============================================================

@dataclass
class ManagedTask:
    """A task managed by the TaskManager Agent."""

    task_id: str
    parent_id: str | None = None
    scope: str = ""
    priority: int = 3
    dependencies: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)

    # Lifecycle
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent_id: str | None = None
    result: TaskResult | None = None
    evaluation: EvaluationResult | None = None

    # Timestamps
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    # Retry
    retry_count: int = 0
    max_retries: int = 2
    error: str | None = None

    # Branch space (for isolated agent work)
    branch_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "parent_id": self.parent_id,
            "scope": self.scope,
            "priority": self.priority,
            "dependencies": self.dependencies,
            "required_tools": self.required_tools,
            "status": self.status.value,
            "assigned_agent_id": self.assigned_agent_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "error": self.error,
        }

    @property
    def is_ready(self) -> bool:
        """Ready to dispatch: pending and all deps done."""
        return self.status == TaskStatus.PENDING

    @property
    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)


class TaskRegistry:
    """In-memory + persistent task registry.

    All tasks are tracked here and also persisted to the MemoryPool
    for cross-session continuity.
    """

    def __init__(self, memory: Any | None = None):
        self._tasks: dict[str, ManagedTask] = {}
        self._order: list[str] = []
        self._memory = memory

    async def register(self, scope: str, priority: int = 3,
                       dependencies: list[str] | None = None,
                       required_tools: list[str] | None = None,
                       parent_id: str | None = None) -> ManagedTask:
        """Register a new task."""
        task = ManagedTask(
            task_id=f"task:{uuid.uuid4().hex[:12]}",
            parent_id=parent_id,
            scope=scope,
            priority=priority,
            dependencies=dependencies or [],
            required_tools=required_tools or [],
        )
        self._tasks[task.task_id] = task
        self._order.append(task.task_id)

        # Persist to memory
        if self._memory:
            await self._memory.register_task(
                task_id=task.task_id,
                parent_id=parent_id,
                scope=scope,
                priority=priority,
            )

        return task

    def get(self, task_id: str) -> ManagedTask | None:
        return self._tasks.get(task_id)

    def all_tasks(self) -> list[ManagedTask]:
        return [self._tasks[tid] for tid in self._order]

    def get_ready(self, completed_ids: set[str] | None = None) -> list[ManagedTask]:
        """Get tasks whose dependencies are all done."""
        completed = completed_ids or {
            tid for tid, t in self._tasks.items()
            if t.status == TaskStatus.DONE
        }
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            if all(dep in completed for dep in task.dependencies):
                ready.append(task)
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
# TaskManager Agent — the central orchestrator
# ============================================================

class TaskManagerAgent:
    """The central TaskManager Agent.

    This is a special agent that manages the task lifecycle:
      1. DECOMPOSE: Uses LLM to break down reasoning output into subtasks
      2. REGISTER: Registers subtasks in TaskRegistry (persisted to memory)
      3. DISPATCH: Assigns ready tasks to Worker Agents (via AgentRouter)
      4. MONITOR: Tracks progress, handles retries and failures
      5. COLLECT: Gathers results and returns synthesized output

    Unlike a Worker Agent, the TaskManager:
      - Has access to the full TaskRegistry
      - Can create/destroy Worker Agents
      - Makes routing decisions using LLM
      - Handles dependency-aware scheduling
      - Can re-plan if tasks fail

    Inspired by:
      - AgentOrchestra's Planning Agent (central decomposition + delegation)
      - AutoGen's Orchestrator (dispatch + collect + re-dispatch)
      - CrewAI's Manager Agent (dynamic task creation + delegation)
    """

    def __init__(self, memory: Any, llm: LLMProvider,
                 agent_loop: AgentLoop, config: LoopConfig):
        self.memory = memory
        self.llm = llm
        self.agent_loop = agent_loop
        self.config = config

        # Task registry (single source of truth for all tasks)
        self.registry = TaskRegistry(memory)

        # Worker agent management
        self.worker_pool = AgentPool(max_concurrent=config.max_agent_concurrent)
        self.router = AgentRouter(memory, self.worker_pool)
        self.evaluator = AgentEvaluator(
            accept_threshold=config.accept_threshold,
            weights=config.evaluation_weights,
        )

        # In-flight tracking
        self._inflight: dict[str, asyncio.Task] = {}  # task_id → asyncio task

    # ============================================================
    # Phase 1: DECOMPOSE (LLM-driven task breakdown)
    # ============================================================

    async def decompose(self, reasoning_output: str,
                        original_input: str) -> list[ManagedTask]:
        """Use LLM to decompose reasoning into subtasks with dependencies.

        The TaskManager Agent uses its own LLM to:
        1. Analyze the reasoning output
        2. Determine if decomposition is needed
        3. Create subtasks with proper dependency edges
        4. Assign priority and required tools
        """
        try:
            subtasks = await self._llm_decompose(reasoning_output, original_input)
        except Exception as e:
            logger.warning("TaskManager LLM decompose failed: %s, single task", e)
            subtasks = [{
                "scope": original_input,
                "priority": 3,
                "dependencies": [],
                "required_tools": [],
            }]

        # Register all subtasks
        scope_to_id: dict[str, str] = {}
        registered: list[ManagedTask] = []

        for st in subtasks:
            task = await self.registry.register(
                scope=st["scope"],
                priority=st.get("priority", 3),
                dependencies=[],  # Resolve after all registered
                required_tools=st.get("required_tools", []),
                parent_id=None,
            )
            scope_to_id[st["scope"]] = task.task_id
            registered.append(task)

        # Resolve dependencies from scope text → task_id
        for st, task in zip(subtasks, registered):
            dep_ids = []
            for dep_scope in st.get("dependencies", []):
                dep_id = scope_to_id.get(dep_scope)
                if dep_id and dep_id != task.task_id:
                    dep_ids.append(dep_id)
            task.dependencies = dep_ids

        logger.info("TaskManager: decomposed into %d tasks", len(registered))
        return registered

    async def _llm_decompose(self, reasoning: str, original: str) -> list[dict]:
        """LLM prompt for task decomposition."""
        prompt = f"""You are a Task Manager Agent. Decompose the following task into subtasks.

Original request: {original}

Reasoning analysis: {reasoning}

Rules:
- If SIMPLE (single action), return one subtask
- If complex, break into 2-5 subtasks
- Each subtask must be concrete and independently executable
- Dependencies: reference other subtask's scope text
- priority: 5=urgent, 3=normal, 1=low
- required_tools: list tool names needed (ssh, web, code, etc.)

Respond as JSON:
```json
[
  {{
    "scope": "concrete subtask description",
    "priority": 3,
    "required_tools": [],
    "dependencies": []
  }}
]
```"""

        response = await self.llm.chat([{"role": "user", "content": prompt}])
        content = response.content

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        result = json.loads(content.strip())
        if not isinstance(result, list):
            result = [result]
        return result

    # ============================================================
    # Phase 2: DISPATCH (assign to worker agents)
    # ============================================================

    async def dispatch_ready(self) -> list[str]:
        """Dispatch all ready tasks to worker agents.

        For each ready task:
        1. Find best worker agent via AgentRouter (MoE routing)
        2. If no available agent, create a new one
        3. Launch async execution
        4. Track in-flight status
        """
        ready = self.registry.get_ready()
        dispatched = []

        for task in ready:
            if len(self._inflight) >= self.config.max_agent_concurrent:
                logger.debug("TaskManager: pool full, %s waiting", task.task_id)
                break

            agent = await self._acquire_worker(task)
            if agent:
                task.status = TaskStatus.RUNNING
                task.assigned_agent_id = agent.agent_id
                task.started_at = datetime.utcnow()

                # Launch async execution
                asyncio_task = asyncio.create_task(self._execute_worker(task, agent))
                self._inflight[task.task_id] = asyncio_task
                dispatched.append(task.task_id)

                logger.info("TaskManager: dispatched %s → agent %s",
                           task.task_id, agent.agent_id)
            else:
                logger.debug("TaskManager: no worker for %s", task.task_id)

        return dispatched

    async def _acquire_worker(self, task: ManagedTask) -> Agent | None:
        """Find or create a worker agent for a task."""
        # Route to best existing agent
        candidates = await self.router.route(task.scope, task.priority)
        if candidates:
            return candidates[0]

        # Create new worker
        agent = await self.worker_pool.acquire()
        if agent:
            # Configure agent's expert profile based on task
            agent.role = AgentRole.WORKER
            agent.expertise = task.required_tools

        return agent

    # ============================================================
    # Phase 3: EXECUTE + MONITOR (worker lifecycle)
    # ============================================================

    async def _execute_worker(self, task: ManagedTask, agent: Agent) -> None:
        """Execute a single task with a worker agent.

        Full lifecycle:
          1. Agent runs the task via AgentLoop
          2. Evaluate result quality
          3. Accept → mark done
          4. Discard → retry or fail
          5. Exception → fail + destroy agent
        """
        try:
            context = {
                "task_id": task.task_id,
                "scope": task.scope,
                "priority": task.priority,
                "required_tools": task.required_tools,
                "branch_id": task.branch_id,
            }

            result = await agent.run(
                self.agent_loop,
                task.scope,
                context,
                allowed_tools=task.required_tools,
            )

            # Evaluate quality
            eval_result = self.evaluator.evaluate(result, task.scope)
            task.evaluation = eval_result

            if eval_result.action == "accept":
                task.status = TaskStatus.DONE
                task.result = result
                task.completed_at = datetime.utcnow()

                if self.memory:
                    await self.memory.update_task_status(task.task_id, "done", {
                        "summary": result.summary,
                        "artifacts": result.artifacts,
                        "score": eval_result.overall,
                    })
                logger.info("TaskManager: task %s DONE (score=%.2f)",
                           task.task_id, eval_result.overall)
            else:
                # Discard — retry if possible
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.assigned_agent_id = None
                    logger.info("TaskManager: task %s retry %d/%d",
                               task.task_id, task.retry_count, task.max_retries)
                else:
                    await self._fail_task(task, eval_result.reason)
                    await self.worker_pool.destroy(agent.agent_id)
                    return

        except Exception as e:
            await self._fail_task(task, f"Agent exception: {e}")
            await self.worker_pool.destroy(agent.agent_id)
            logger.error("TaskManager: task %s exception: %s", task.task_id, e)

        finally:
            # Release agent back to pool (if not destroyed)
            if agent.agent_id in self.worker_pool._agents:
                await self.worker_pool.release(agent.agent_id)
            self._inflight.pop(task.task_id, None)

    async def _fail_task(self, task: ManagedTask, reason: str) -> None:
        """Mark a task as failed and record to discard pool."""
        task.status = TaskStatus.FAILED
        task.error = reason
        task.completed_at = datetime.utcnow()

        if self.memory:
            await self.memory.discard_result(
                agent_id=task.assigned_agent_id or "unknown",
                task_id=task.task_id,
                reason=reason,
            )
            await self.memory.update_task_status(task.task_id, "failed")

    # ============================================================
    # Phase 4: COLLECT (wait for results)
    # ============================================================

    async def collect_all(self, timeout: float = 300.0) -> dict[str, TaskResult]:
        """Wait for all in-flight tasks to complete."""
        if not self._inflight:
            return {}

        # Dispatch any remaining ready tasks while waiting
        await self.dispatch_ready()

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._inflight.values(), return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("TaskManager: collect_all timed out after %ss", timeout)

        # After completion, check if new tasks became ready (deps fulfilled)
        # and dispatch them
        remaining_ready = self.registry.get_ready()
        if remaining_ready:
            await self.dispatch_ready()
            if self._inflight:
                await asyncio.wait_for(
                    asyncio.gather(*self._inflight.values(), return_exceptions=True),
                    timeout=timeout,
                )

        results = {}
        for task in self.registry.all_tasks():
            if task.result:
                results[task.task_id] = task.result
        return results

    async def collect(self, task_id: str, timeout: float = 120.0) -> TaskResult | None:
        """Wait for a single task result."""
        if task_id not in self._inflight:
            task = self.registry.get(task_id)
            return task.result if task else None

        try:
            await asyncio.wait_for(self._inflight[task_id], timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("TaskManager: collect %s timed out", task_id)

        task = self.registry.get(task_id)
        return task.result if task else None

    # ============================================================
    # Phase 5: RE-PLAN (adaptive re-planning on failure)
    # ============================================================

    async def replan(self, failed_task: ManagedTask) -> list[ManagedTask] | None:
        """Re-plan a failed task using LLM.

        The TaskManager can:
        1. Break the failed task into smaller subtasks
        2. Try a different approach
        3. Or give up and return None
        """
        try:
            prompt = f"""A task has failed. Re-plan it into a different approach.

Failed task: {failed_task.scope}
Error: {failed_task.error}
Previous attempts: {failed_task.retry_count}

Options:
1. Break into smaller steps
2. Try a completely different approach
3. Return empty array if the task is truly impossible

Respond as JSON array of subtasks (same format as decompose)."""

            response = await self.llm.chat([{"role": "user", "content": prompt}])
            content = response.content

            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            subtasks = json.loads(content.strip())
            if not subtasks:
                return None

            # Register new subtasks
            new_tasks = []
            for st in subtasks:
                task = await self.registry.register(
                    scope=st["scope"],
                    priority=st.get("priority", failed_task.priority),
                    dependencies=[failed_task.parent_id] if failed_task.parent_id else [],
                    required_tools=st.get("required_tools", []),
                    parent_id=failed_task.parent_id,
                )
                new_tasks.append(task)

            logger.info("TaskManager: re-planned %s into %d new tasks",
                       failed_task.task_id, len(new_tasks))
            return new_tasks

        except Exception as e:
            logger.error("TaskManager: re-plan failed: %s", e)
            return None

    # ============================================================
    # Cancellation
    # ============================================================

    async def cancel(self, task_id: str) -> None:
        """Cancel a task and its worker agent."""
        task = self.registry.get(task_id)
        if not task:
            return

        inflight = self._inflight.pop(task_id, None)
        if inflight:
            inflight.cancel()

        if task.assigned_agent_id:
            await self.worker_pool.destroy(task.assigned_agent_id)

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()

        if self.memory:
            await self.memory.update_task_status(task_id, "cancelled")
        logger.info("TaskManager: cancelled %s", task_id)

    async def cancel_all(self) -> None:
        """Cancel all in-flight tasks."""
        for task_id in list(self._inflight.keys()):
            await self.cancel(task_id)

    # ============================================================
    # Status
    # ============================================================

    def status(self) -> dict:
        """Get full status overview."""
        registry_stats = self.registry.stats()
        pool_stats = self.worker_pool.stats()
        return {
            **registry_stats,
            "inflight": len(self._inflight),
            "agents_active": pool_stats["active"],
            "agents_idle": pool_stats["idle"],
        }
