"""Agent management system - lifecycle, pool, routing, evaluation."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..core import (
    AgentStatus, TaskStatus, TaskResult, EvaluationResult,
    DiscardRecord, ExpertProfile, AgentRole,
)

logger = logging.getLogger(__name__)


# ============================================================
# Agent Wrapper
# ============================================================

@dataclass
class Agent:
    """An agent instance in the system."""
    agent_id: str
    status: AgentStatus = AgentStatus.IDLE
    role: AgentRole = AgentRole.WORKER
    expertise: list[str] = field(default_factory=list)  # tool names this agent is good at
    created_at: datetime = field(default_factory=datetime.now)
    task_count: int = 0
    success_count: int = 0
    expert_profile: ExpertProfile | None = None

    # Isolation
    branch_space_id: str | None = None
    process_id: int | None = None

    async def run(self, agent_loop, task_scope: str, context: dict,
                  allowed_tools: list[str]) -> TaskResult:
        """Execute agent loop for a task."""
        self.status = AgentStatus.RUNNING
        try:
            result = await agent_loop.run(
                self.agent_id, task_scope, context, allowed_tools
            )
            self.task_count += 1
            if result.status == TaskStatus.DONE:
                self.success_count += 1
                self.status = AgentStatus.DONE
            else:
                self.status = AgentStatus.FAILED
            return result
        except Exception:
            self.status = AgentStatus.FAILED
            raise


# ============================================================
# AgentPool
# ============================================================

class AgentPool:
    """Pool managing agent lifecycle."""

    def __init__(self, max_concurrent: int = 50, idle_timeout: float = 300.0):
        self.agents: dict[str, Agent] = {}
        self.max_concurrent = max_concurrent
        self.idle_timeout = idle_timeout
        self._lock = asyncio.Lock()

    async def acquire(self) -> Agent | None:
        """Acquire an agent from the pool or create one."""
        async with self._lock:
            # Reuse idle agent
            for agent in self.agents.values():
                if agent.status == AgentStatus.IDLE:
                    return agent

            # Create new agent
            if len(self.agents) < self.max_concurrent:
                agent = Agent(agent_id=f"agent:{uuid.uuid4().hex[:8]}")
                self.agents[agent.agent_id] = agent
                return agent

            # Pool full - wait for an agent to become idle
            return None

    async def release(self, agent_id: str) -> None:
        """Release agent back to pool."""
        async with self._lock:
            agent = self.agents.get(agent_id)
            if agent:
                agent.status = AgentStatus.IDLE
                agent.branch_space_id = None

    async def destroy(self, agent_id: str) -> None:
        """Destroy agent and remove from pool."""
        async with self._lock:
            agent = self.agents.pop(agent_id, None)
            if agent:
                agent.status = AgentStatus.DESTROYED
                logger.info("Agent %s destroyed", agent_id)

    async def cleanup_idle(self) -> int:
        """Remove agents idle longer than idle_timeout."""
        now = datetime.now()
        removed = 0
        async with self._lock:
            to_remove = []
            for agent_id, agent in self.agents.items():
                if agent.status == AgentStatus.IDLE:
                    idle_seconds = (now - agent.created_at).total_seconds()
                    if idle_seconds > self.idle_timeout:
                        to_remove.append(agent_id)

            for agent_id in to_remove:
                self.agents.pop(agent_id)
                removed += 1
        return removed


# ============================================================
# AgentRouter (MoE-style)
# ============================================================

class AgentRouter:
    """MoE-style router: match tasks to best-fit agents."""

    def __init__(self, memory: Any, pool: AgentPool):
        self.memory = memory
        self.pool = pool

    async def route(self, task_scope: str, priority: int = 3,
                    top_k: int = 3) -> list[Agent]:
        """Route a task to the best matching agent(s).

        Scoring = similarity * 0.5 + success_rate * 0.3 + load_factor * 0.2
        """
        # Get task embedding for similarity matching
        try:
            task_embedding = await self.memory.embed(task_scope)
        except Exception:
            task_embedding = None

        scores: dict[str, float] = {}
        for agent_id, agent in self.pool.agents.items():
            if agent.status not in (AgentStatus.IDLE, AgentStatus.DONE):
                continue

            # Similarity score
            sim = 0.0
            if task_embedding and agent.expert_profile:
                sim = self._cosine_similarity(task_embedding, agent.expert_profile.embedding)

            # Success rate
            success_rate = (
                agent.success_count / max(agent.task_count, 1)
                if agent.task_count > 0 else 0.5  # Default for new agents
            )

            # Load factor
            load = sum(1 for a in self.pool.agents.values() if a.status == AgentStatus.RUNNING)
            load_factor = 1.0 / (load + 1)

            scores[agent_id] = sim * 0.5 + success_rate * 0.3 + load_factor * 0.2

        # Sort and return top-k
        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [self.pool.agents[aid] for aid, _ in sorted_agents[:top_k] if aid in self.pool.agents]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        na = (sum(x * x for x in a)) ** 0.5
        nb = (sum(y * y for y in b)) ** 0.5
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)


# ============================================================
# AgentEvaluator
# ============================================================

class AgentEvaluator:
    """Quality evaluation of agent outputs."""

    def __init__(self, accept_threshold: float = 0.7,
                 weights: dict[str, float] | None = None):
        self.accept_threshold = accept_threshold
        self.weights = weights or {
            "completeness": 0.30,
            "correctness": 0.30,
            "relevance": 0.25,
            "efficiency": 0.15,
        }

    def evaluate(self, result: TaskResult, task_scope: str) -> EvaluationResult:
        """Evaluate agent output quality."""
        steps = result.steps
        if not steps:
            return EvaluationResult(
                scores={},
                overall=0.0,
                action="discard",
                reason="No execution steps produced"
            )

        # Completeness: did agent attempt to address the task?
        completeness = 1.0 if any(s.tool_name for s in steps) else 0.3

        # Correctness: any errors?
        error_count = sum(1 for s in steps if s.error)
        correctness = max(0.0, 1.0 - error_count / max(len(steps), 1) * 0.5)

        # Relevance: how many steps match the task scope?
        relevance = 1.0 if len(steps) >= 2 else 0.5

        # Efficiency: steps within reasonable bounds
        efficiency = 1.0 if len(steps) <= 10 else max(0.3, 1.0 - (len(steps) - 10) * 0.1)

        overall = (
            completeness * self.weights["completeness"] +
            correctness * self.weights["correctness"] +
            relevance * self.weights["relevance"] +
            efficiency * self.weights["efficiency"]
        )

        action = "accept" if overall >= self.accept_threshold else "discard"
        return EvaluationResult(
            scores={
                "completeness": round(completeness, 2),
                "correctness": round(correctness, 2),
                "relevance": round(relevance, 2),
                "efficiency": round(efficiency, 2),
            },
            overall=round(overall, 2),
            action=action,
            reason="Accepted" if action == "accept" else f"Score {overall:.2f} below threshold {self.accept_threshold}"
        )


# ============================================================
# AgentOrchestrator (Facade)
# ============================================================

class AgentOrchestrator:
    """Top-level orchestrator: routes tasks → agents → evaluates → collects."""

    def __init__(self, memory: Any, agent_loop: Any, config: Any):
        self.memory = memory
        self.agent_loop = agent_loop
        self.config = config

        self.pool = AgentPool(max_concurrent=config.max_agent_concurrent)
        self.router = AgentRouter(memory, self.pool)
        self.evaluator = AgentEvaluator(
            accept_threshold=config.accept_threshold,
            weights=config.evaluation_weights,
        )

        # In-flight tasks
        self._task_agents: dict[str, str] = {}  # task_id → agent_id

    async def dispatch(self, task_id: str, scope: str, priority: int = 3) -> str | None:
        """Dispatch a task to the best available agent.

        Returns agent_id if dispatched, None if waiting.
        """
        # Find best agent
        candidates = await self.router.route(scope, priority)
        if not candidates:
            # Create a new agent
            agent = await self.pool.acquire()
            if agent:
                candidates = [agent]
            else:
                logger.warning("Agent pool full, task %s waiting", task_id)
                return None

        # Assign task to agent
        agent = candidates[0]
        self._task_agents[task_id] = agent.agent_id

        # Build context from memory
        context = {
            "task_id": task_id,
            "scope": scope,
            "priority": priority,
        }

        # Launch agent asynchronously
        asyncio.create_task(self._execute_agent(agent, task_id, scope, context))
        logger.info("Dispatched task %s to agent %s", task_id, agent.agent_id)
        return agent.agent_id

    async def _execute_agent(self, agent: Agent, task_id: str, scope: str,
                             context: dict) -> None:
        """Execute agent and handle results."""
        try:
            result = await agent.run(
                self.agent_loop, scope, context, allowed_tools=[]
            )

            # Evaluate
            eval_result = self.evaluator.evaluate(result, scope)

            if eval_result.action == "accept":
                await self.memory.update_task_status(task_id, "done", {
                    "summary": result.summary,
                    "artifacts": result.artifacts,
                    "score": eval_result.overall,
                })
            else:
                # Discard
                await self.memory.discard_result(
                    agent_id=agent.agent_id,
                    task_id=task_id,
                    reason=eval_result.reason,
                    result={"summary": result.summary},
                    agent_log=[s.__dict__ for s in result.steps],
                )
                await self.memory.update_task_status(task_id, "discarded")
                await self.pool.destroy(agent.agent_id)

        except Exception as e:
            logger.error("Agent %s failed for task %s: %s", agent.agent_id, task_id, e)
            await self.memory.discard_result(
                agent_id=agent.agent_id,
                task_id=task_id,
                reason=f"Agent exception: {e}",
            )
            await self.pool.destroy(agent.agent_id)
        else:
            # Release agent back to pool
            await self.pool.release(agent.agent_id)

        # Clean up
        self._task_agents.pop(task_id, None)

    async def collect(self, task_id: str) -> TaskResult | None:
        """Wait for and collect task result."""
        # Check if task is done
        task_data = await self.memory._db.select(task_id)
        if not task_data:
            return None

        if task_data.get("status") == "done":
            return TaskResult(
                task_id=task_id,
                agent_id=self._task_agents.get(task_id, ""),
                status=TaskStatus.DONE,
                summary=task_data.get("result", {}).get("summary", ""),
                artifacts=task_data.get("result", {}).get("artifacts", {}),
            )

        # Task still running or failed
        return None

    async def cancel(self, task_id: str) -> None:
        """Cancel a task and destroy its agent."""
        agent_id = self._task_agents.pop(task_id, None)
        if agent_id:
            await self.pool.destroy(agent_id)
        await self.memory.update_task_status(task_id, "cancelled")
        logger.info("Cancelled task %s", task_id)
