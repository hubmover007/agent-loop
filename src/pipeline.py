"""Unified Task Pipeline — task decomposition + agent assignment + state tracking.

This module replaces the fragmented TaskScheduler/AgentOrchestrator/MainLoop._decompose
flow with a single coherent pipeline:

  LLM Decompose → TaskTree Build → Dependency Schedule → Agent Assign → State Track → Collect

The pipeline is the single entry point for going from a reasoning output
to dispatched, tracked, collected task results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime
from typing import Any

from .core import TaskStatus, AgentStatus, TaskResult, EvaluationResult, StepLog
from .agent import AgentPool, AgentRouter, AgentEvaluator, Agent
from .loop_engine import LLMProvider, LLMResponse, LoopConfig

logger = logging.getLogger(__name__)


# ============================================================
# Task Node (unified — replaces TaskNode in task/__init__.py)
# ============================================================

class PipelineTask:
    """A single task in the pipeline, with full lifecycle tracking."""

    def __init__(self, scope: str, priority: int = 3,
                 dependencies: list[str] | None = None,
                 required_tools: list[str] | None = None,
                 parent_id: str | None = None):
        self.task_id: str = f"task:{uuid.uuid4().hex[:12]}"
        self.parent_id: str | None = parent_id
        self.scope: str = scope
        self.priority: int = priority
        self.dependencies: list[str] = dependencies or []
        self.required_tools: list[str] = required_tools or []

        # Lifecycle
        self.status: TaskStatus = TaskStatus.PENDING
        self.assigned_agent_id: str | None = None
        self.result: TaskResult | None = None
        self.evaluation: EvaluationResult | None = None
        self.created_at: datetime = datetime.utcnow()
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None
        self.error: str | None = None

        # Retry
        self.retry_count: int = 0
        self.max_retries: int = 2

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
            "error": self.error,
            "retry_count": self.retry_count,
        }

    @property
    def is_ready(self) -> bool:
        """A task is ready if all dependencies are done."""
        return self.status == TaskStatus.PENDING

    @property
    def is_terminal(self) -> bool:
        return self.status in (TaskStatus.DONE, TaskStatus.FAILED, TaskStatus.CANCELLED)


# ============================================================
# Task Pipeline — the unified flow
# ============================================================

class TaskPipeline:
    """Unified pipeline: decompose → schedule → assign → track → collect.

    Replaces the fragmented TaskScheduler + AgentOrchestrator + MainLoop._decompose
    with a single coherent flow. The MainLoop calls:

        pipeline = TaskPipeline(memory, llm, agent_loop, config)
        tasks = await pipeline.decompose_and_dispatch(reason_output, context)
        results = await pipeline.collect_all()
    """

    def __init__(self, memory: Any, llm: LLMProvider,
                 agent_loop: Any, config: LoopConfig):
        self.memory = memory
        self.llm = llm
        self.agent_loop = agent_loop
        self.config = config

        # Agent management
        self.agent_pool = AgentPool(max_concurrent=config.max_agent_concurrent)
        self.agent_router = AgentRouter(memory, self.agent_pool)
        self.evaluator = AgentEvaluator(
            accept_threshold=config.accept_threshold,
            weights=config.evaluation_weights,
        )

        # Task registry (in-memory index, also persisted to memory)
        self._tasks: dict[str, PipelineTask] = {}
        self._task_order: list[str] = []  # Creation order

        # In-flight tracking
        self._inflight: dict[str, asyncio.Task] = {}  # task_id → asyncio task

    # ============================================================
    # Phase 1: Decompose (LLM-driven)
    # ============================================================

    async def decompose(self, reasoning_output: str,
                        original_input: str) -> list[PipelineTask]:
        """Decompose reasoning output into a task tree with dependencies.

        Uses LLM to break down complex tasks, then builds PipelineTask objects
        with proper dependency edges. Simple tasks get a single task.
        """
        try:
            subtasks = await self._llm_decompose(reasoning_output, original_input)
        except Exception as e:
            logger.warning("LLM decompose failed: %s, using single task", e)
            subtasks = [{
                "scope": original_input,
                "priority": 3,
                "dependencies": [],
                "required_tools": [],
            }]

        # Build PipelineTask objects
        # First pass: create all tasks (so dependencies can reference them)
        scope_to_id: dict[str, str] = {}  # scope text → task_id (for dependency resolution)

        for st in subtasks:
            task = PipelineTask(
                scope=st["scope"],
                priority=st.get("priority", 3),
                dependencies=[],  # Will resolve in second pass
                required_tools=st.get("required_tools", []),
            )
            self._tasks[task.task_id] = task
            self._task_order.append(task.task_id)
            scope_to_id[st["scope"]] = task.task_id

            # Register in memory for persistence
            await self.memory.register_task(
                task_id=task.task_id,
                parent_id=None,
                scope=task.scope,
                priority=task.priority,
            )

        # Second pass: resolve dependencies from scope text to task_id
        for st, task in zip(subtasks, [self._tasks[tid] for tid in self._task_order[-len(subtasks):]]):
            dep_ids = []
            for dep_scope in st.get("dependencies", []):
                dep_id = scope_to_id.get(dep_scope)
                if dep_id and dep_id != task.task_id:
                    dep_ids.append(dep_id)
            task.dependencies = dep_ids

        logger.info("Pipeline: decomposed into %d tasks", len(subtasks))
        return [self._tasks[tid] for tid in self._task_order[-len(subtasks):]]

    async def _llm_decompose(self, reasoning: str, original: str) -> list[dict]:
        """Call LLM to decompose task into subtasks."""
        prompt = f"""Based on the analysis below, decompose the task into subtasks.

Original request: {original}

Analysis: {reasoning}

Rules:
- If SIMPLE (single action), return array with one item
- If complex, break into 2-5 subtasks
- Each subtask: concrete, independently executable
- Mark dependencies by referencing other subtask's scope text
- priority: 5=urgent, 1=low

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
    # Phase 2: Schedule (dependency-aware)
    # ============================================================

    def get_ready_tasks(self) -> list[PipelineTask]:
        """Get tasks whose dependencies are all done, sorted by priority."""
        ready = []
        for task in self._tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            # Check all dependencies are done
            deps_done = all(
                self._tasks.get(dep_id) and
                self._tasks[dep_id].status == TaskStatus.DONE
                for dep_id in task.dependencies
            )
            if deps_done:
                ready.append(task)

        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    # ============================================================
    # Phase 3: Dispatch (assign to agents)
    # ============================================================

    async def dispatch_ready(self) -> list[str]:
        """Dispatch all ready tasks to agents. Returns dispatched task_ids."""
        ready = self.get_ready_tasks()
        dispatched = []

        for task in ready:
            # Check agent pool capacity
            if len(self._inflight) >= self.config.max_agent_concurrent:
                break

            agent_id = await self._assign_agent(task)
            if agent_id:
                task.status = TaskStatus.RUNNING
                task.assigned_agent_id = agent_id
                task.started_at = datetime.utcnow()

                # Launch async execution
                asyncio_task = asyncio.create_task(self._execute_task(task))
                self._inflight[task.task_id] = asyncio_task
                dispatched.append(task.task_id)

                logger.info("Pipeline: dispatched %s → agent %s", task.task_id, agent_id)
            else:
                logger.debug("Pipeline: no agent for %s, will retry", task.task_id)

        return dispatched

    async def _assign_agent(self, task: PipelineTask) -> str | None:
        """Find or create an agent for a task."""
        # Route to best agent
        candidates = await self.agent_router.route(task.scope, task.priority)
        if candidates:
            return candidates[0].agent_id

        # Create new agent
        agent = await self.agent_pool.acquire()
        return agent.agent_id if agent else None

    # ============================================================
    # Phase 4: Execute + Track (single task lifecycle)
    # ============================================================

    async def _execute_task(self, task: PipelineTask) -> None:
        """Execute a single task: run agent → evaluate → accept/discard → update state."""
        agent = self.agent_pool._agents.get(task.assigned_agent_id)
        if not agent:
            task.status = TaskStatus.FAILED
            task.error = f"Agent {task.assigned_agent_id} not found"
            self._inflight.pop(task.task_id, None)
            return

        try:
            # Run agent on this task
            context = {
                "task_id": task.task_id,
                "scope": task.scope,
                "priority": task.priority,
                "required_tools": task.required_tools,
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

                await self.memory.update_task_status(task.task_id, "done", {
                    "summary": result.summary,
                    "artifacts": result.artifacts,
                    "score": eval_result.overall,
                })
                logger.info("Pipeline: task %s DONE (score=%.2f)",
                           task.task_id, eval_result.overall)
            else:
                # Discard — retry if possible
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.assigned_agent_id = None
                    logger.info("Pipeline: task %s retry %d/%d",
                               task.task_id, task.retry_count, task.max_retries)
                else:
                    task.status = TaskStatus.FAILED
                    task.error = eval_result.reason
                    task.completed_at = datetime.utcnow()

                    await self.memory.discard_result(
                        agent_id=agent.agent_id,
                        task_id=task.task_id,
                        reason=eval_result.reason,
                        result={"summary": result.summary},
                        agent_log=[s.__dict__ for s in result.steps],
                    )
                    await self.memory.update_task_status(task.task_id, "failed")
                    await self.agent_pool.destroy(agent.agent_id)
                    logger.warning("Pipeline: task %s FAILED after %d retries",
                                  task.task_id, task.retry_count)

        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.utcnow()

            await self.memory.discard_result(
                agent_id=agent.agent_id,
                task_id=task.task_id,
                reason=f"Exception: {e}",
            )
            await self.memory.update_task_status(task.task_id, "failed")
            await self.agent_pool.destroy(agent.agent_id)
            logger.error("Pipeline: task %s exception: %s", task.task_id, e)

        finally:
            # Release agent back to pool (if not destroyed)
            if agent.agent_id in self.agent_pool._agents:
                await self.agent_pool.release(agent.agent_id)

            self._inflight.pop(task.task_id, None)

    # ============================================================
    # Phase 5: Collect (wait for results)
    # ============================================================

    async def collect_all(self, timeout: float = 300.0) -> dict[str, TaskResult]:
        """Wait for all in-flight tasks to complete. Returns task_id → result."""
        if not self._inflight:
            return {}

        try:
            await asyncio.wait_for(
                asyncio.gather(*self._inflight.values(), return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("Pipeline: collect_all timed out after %ss", timeout)

        # Gather results
        results = {}
        for task in self._tasks.values():
            if task.result:
                results[task.task_id] = task.result
        return results

    async def collect(self, task_id: str, timeout: float = 120.0) -> TaskResult | None:
        """Wait for and collect a single task result."""
        if task_id not in self._inflight:
            # Already completed
            task = self._tasks.get(task_id)
            return task.result if task else None

        try:
            await asyncio.wait_for(self._inflight[task_id], timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning("Pipeline: collect %s timed out", task_id)

        task = self._tasks.get(task_id)
        return task.result if task else None

    # ============================================================
    # Pipeline Status
    # ============================================================

    def status(self) -> dict:
        """Get pipeline status summary."""
        tasks = list(self._tasks.values())
        return {
            "total": len(tasks),
            "pending": sum(1 for t in tasks if t.status == TaskStatus.PENDING),
            "running": sum(1 for t in tasks if t.status == TaskStatus.RUNNING),
            "done": sum(1 for t in tasks if t.status == TaskStatus.DONE),
            "failed": sum(1 for t in tasks if t.status == TaskStatus.FAILED),
            "inflight": len(self._inflight),
            "agents_active": self.agent_pool.stats()["active"],
            "agents_idle": self.agent_pool.stats()["idle"],
        }

    def get_task(self, task_id: str) -> PipelineTask | None:
        return self._tasks.get(task_id)

    def all_tasks(self) -> list[PipelineTask]:
        return [self._tasks[tid] for tid in self._task_order]

    # ============================================================
    # Cancel
    # ============================================================

    async def cancel(self, task_id: str) -> None:
        """Cancel a task and its agent."""
        task = self._tasks.get(task_id)
        if not task:
            return

        # Cancel asyncio task if running
        inflight = self._inflight.pop(task_id, None)
        if inflight:
            inflight.cancel()

        # Destroy agent if assigned
        if task.assigned_agent_id:
            await self.agent_pool.destroy(task.assigned_agent_id)

        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        await self.memory.update_task_status(task_id, "cancelled")
        logger.info("Pipeline: cancelled %s", task_id)

    async def cancel_all(self) -> None:
        """Cancel all in-flight tasks."""
        task_ids = list(self._inflight.keys())
        for tid in task_ids:
            await self.cancel(tid)
