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
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pathlib import Path

import yaml

from .core import (
    TaskStatus, AgentStatus, TaskResult, EvaluationResult,
    StepLog, AgentRole, LoopPhase,
)
from .agent import Agent, AgentPool, AgentEvaluator
from .agent_soul import AgentSoul, SoulBuilder
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


# ============================================================
# AgentTemplate — per-agent LLM + config spec
# ============================================================

from dataclasses import dataclass as _dc

@_dc
class AgentTemplate:
    """Structured spec for agent creation (from config/agent_templates.yaml)."""
    name: str
    llm_strategy: str = "balanced"
    llm_capabilities_required: list = None  # type: ignore
    max_steps: int = 10
    ttl: int = 300
    tools_allowed: list = None  # type: ignore

    def __post_init__(self):
        if self.llm_capabilities_required is None:
            self.llm_capabilities_required = ["general"]
        if self.tools_allowed is None:
            self.tools_allowed = []


class AgentTemplateRegistry:
    """Loads agent templates from config/agent_templates.yaml."""

    DEFAULT_TEMPLATE = AgentTemplate(name="default_worker")

    def __init__(self, config_path: str | Path = "config/agent_templates.yaml"):
        self._path = Path(config_path)
        self._templates: dict[str, AgentTemplate] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if not self._path.exists():
            logger.warning("AgentTemplateRegistry: %s not found, using defaults", self._path)
            self._loaded = True
            return
        with open(self._path) as f:
            data = yaml.safe_load(f)
        for name, d in data.get("templates", {}).items():
            self._templates[name] = AgentTemplate(
                name=name,
                llm_strategy=d.get("llm_strategy", "balanced"),
                llm_capabilities_required=d.get("llm_capabilities_required", ["general"]),
                max_steps=d.get("max_steps", 10),
                ttl=d.get("ttl", 300),
                tools_allowed=d.get("tools_allowed", []),
            )
        self._loaded = True
        logger.info("AgentTemplateRegistry: loaded %d templates from %s",
                    len(self._templates), self._path)

    def get(self, name: str) -> AgentTemplate:
        """Get template by name, fallback to default_worker."""
        if not self._loaded:
            self.load()
        return self._templates.get(name, self._templates.get("default_worker", self.DEFAULT_TEMPLATE))

    def infer_template(self, task: "ManagedTask") -> AgentTemplate:
        """Infer the best template from task scope and required_tools."""
        if not self._loaded:
            self.load()
        scope_lower = task.scope.lower()
        tools = set(task.required_tools or [])

        if any(kw in scope_lower for kw in ["code", "debug", "implement", "refactor", "fix"]):
            return self.get("coding_expert")
        if any(kw in scope_lower for kw in ["reason", "analyze", "plan", "math", "calculate"]):
            return self.get("reasoning_expert")
        if any(kw in scope_lower for kw in ["quick", "fast", "brief", "summary"]):
            return self.get("fast_responder")
        return self.get("default_worker")


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
                 state_store: Any | None = None,
                 interaction_hub: Any | None = None,
                 mail_router: Any | None = None,
                 persistence: Any | None = None,
                 permission_checker: Any | None = None,
                 sandbox: Any | None = None):
        self.memory = memory
        self.agent_loop = agent_loop
        self.config = config
        self.registry = registry

        # Optional LLM Pool for provider selection
        self.llm_pool = llm_pool

        # Optional StateStore for structured persistence
        self.state_store = state_store

        # Human-in-the-Loop hub — single instance for all agents
        self.interaction_hub = interaction_hub

        # Agent-to-Agent MailRouter — single instance for all agents
        self.mail_router = mail_router

        # PersistenceManager — cross-session state persistence
        self.persistence = persistence

        # PermissionChecker — agent capability-based access control
        self.permission_checker = permission_checker

        # SandboxManager — tiered code execution isolation
        self.sandbox = sandbox

        # Central event bus for progress events across all agents
        self._event_bus: asyncio.Queue = asyncio.Queue()

        # Internal agent pool
        self.pool = AgentPool(max_concurrent=config.max_agent_concurrent)
        self.evaluator = AgentEvaluator(
            accept_threshold=config.accept_threshold,
            weights=config.evaluation_weights,
        )

        # External agent bridge (Codex, Claude Code, Gemini, etc.)
        self.external_bridge = external_bridge or ExternalAgentBridge()

        # Agent Forker — clone agents for parallel tasks
        self.forker = None  # type: ignore  # Set via enable_forker()

        # In-flight tracking: task_id → (agent, asyncio.Task)
        self._inflight: dict[str, tuple[Agent, asyncio.Task]] = {}

        # Lock for thread-safe mutation of pool and in-flight state
        self._lock = asyncio.Lock()

    # ── Persistence: startup restore ───────────────────────────────────

    async def restore_agents_on_startup(self) -> int:
        """Restore agent states from snapshots on system startup.

        Scans all snapshot directories and restores each agent's state
        from the latest snapshot. Returns the number of agents restored.
        """
        if not self.persistence:
            logger.debug("AgentManager: no PersistenceManager, skipping restore")
            return 0

        restored = 0
        snap_base = self.persistence.snapshot_dir
        if not snap_base.exists():
            return 0

        for agent_dir in snap_base.iterdir():
            if not agent_dir.is_dir():
                continue
            agent_id = agent_dir.name
            try:
                success = await self.persistence.restore(agent_id)
                if success:
                    logger.info("AgentManager: restored agent '%s' from snapshot", agent_id)
                    restored += 1
            except Exception as e:
                logger.warning("AgentManager: failed to restore '%s': %s", agent_id, e)

        logger.info("AgentManager: restored %d agents on startup", restored)
        return restored

    # ── Assignment ──────────────────────────────────────────────────────

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
        async with self._lock:
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
        """Find idle agent or create new one with autonomous decision-making.

        The AgentManagerAgent autonomously decides:
          1. Task type classification (coding/reasoning/ops/general)
          2. Personality + role from cards/
          3. Build AgentSoul via SoulBuilder
          4. Select LLM provider via LLMPool
          5. Create ProgressEmitter for streaming

        All decisions are rule-driven (no LLM inference), ensuring fast and
        deterministic agent creation.
        """
        task_type = self._classify_task(task)
        personality = self._choose_personality(task_type)
        role = self._choose_role(task_type)
        llm_strategy = self._choose_llm_strategy(task_type)

        # Try to reuse idle agent with matching expertise
        for agent in self.pool.agents.values():
            if agent.status == AgentStatus.IDLE:
                return agent

        # Create new worker
        agent = await self.pool.acquire()
        if not agent:
            return None

        agent.role = AgentRole.WORKER
        agent.expertise = task.required_tools or []

        # ── Build AgentSoul ──────────────────────────────────────
        from .agent_soul import SoulBuilder
        try:
            soul = SoulBuilder(agent_id=agent.agent_id)\
                .with_personality(personality)\
                .with_role(role)\
                .build()
            agent._agent_soul = soul
            logger.info("AgentManager: agent %s bound to soul (%s/%s)",
                       agent.agent_id, personality, role)
        except Exception as e:
            logger.warning("AgentManager: soul build failed for %s: %s", agent.agent_id, e)

        # ── Create ProgressEmitter ───────────────────────────────
        from .streaming import ProgressEmitter
        emitter = ProgressEmitter(agent_id=agent.agent_id)
        # Subscribe to forward events to the central event bus
        emitter.on_event(lambda ev: self._event_bus.put_nowait(ev))
        agent._progress_emitter = emitter

        # ── Bind EvolutionEngine ─────────────────────────────────
        from .evolution import EvolutionEngine
        agent._evolution_engine = EvolutionEngine(agent_id=agent.agent_id)

        # ── Bind permissions (PermissionChecker → AgentPermissions) ─
        if self.permission_checker:
            try:
                perms = self.permission_checker.get_permissions(
                    agent.agent_id, role
                )
                agent._permissions = perms
                logger.info(
                    "AgentManager: agent %s bound to permissions (template=%s, trust_level=%s)",
                    agent.agent_id, role, perms.trust_level
                )
            except Exception as e:
                logger.warning(
                    "AgentManager: permission bind failed for %s: %s", agent.agent_id, e
                )

        # ── Structured LLM selection via LLMPool ─────────────────
        if self.llm_pool:
            try:
                provider = await self.llm_pool.acquire(
                    capabilities=self._infer_capabilities(task),
                    strategy=llm_strategy,
                )
                # Bind LLM provider to agent for use in AgentLoop
                agent._llm_provider = provider
                agent._llm_provider_id = provider.provider_id
                agent._task_type = task_type
                logger.info(
                    "AgentManager: agent %s bound to provider '%s' (task_type=%s, strategy=%s)",
                    agent.agent_id, provider.provider_id, task_type, llm_strategy
                )
            except Exception as e:
                logger.warning(
                    "AgentManager: llm_pool acquire failed for task_type='%s': %s — using default LLM",
                    task_type, e
                )

        return agent

    # ── Autonomous decision helpers (rule-driven, no LLM) ────────

    @staticmethod
    def _classify_task(task: ManagedTask) -> str:
        """Infer task type from task.scope and required_tools.

        Returns one of: 'coding', 'reasoning', 'ops', 'general'.
        """
        scope = (task.scope or "").lower()
        tools = [t.lower() for t in (task.required_tools or [])]

        if any(k in scope for k in ["code", "implement", "debug", "refactor", "build"]):
            return "coding"
        if any(k in scope for k in ["analyze", "reason", "compare", "evaluate"]):
            return "reasoning"
        if any(k in scope for k in ["deploy", "restart", "check", "monitor"]):
            return "ops"
        return "general"

    @staticmethod
    def _choose_personality(task_type: str) -> str:
        """Select personality card based on task type."""
        return {
            "coding": "executor",
            "reasoning": "analyst",
            "ops": "guardian",
            "general": "executor",
        }.get(task_type, "executor")

    @staticmethod
    def _choose_role(task_type: str) -> str:
        """Select role card based on task type."""
        return {
            "coding": "coder",
            "reasoning": "researcher",
            "ops": "ops",
            "general": "coder",
        }.get(task_type, "coder")

    @staticmethod
    def _choose_llm_strategy(task_type: str) -> str:
        """Select LLM strategy based on task type."""
        return {
            "reasoning": "most_capable",
            "coding": "cheapest",
            "ops": "balanced",
            "general": "balanced",
        }.get(task_type, "balanced")

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
                "template": getattr(agent, "_template_name", "default_worker"),
                "llm_provider_id": getattr(agent, "_llm_provider_id", None),
            }

            # Register mailbox for this agent if router is available
            agent_mailbox = None
            if self.mail_router is not None:
                agent_mailbox = self.mail_router.register(agent.agent_id)

            # Use agent-bound LLM if available (from llm_pool + autonomous selection)
            # Otherwise fall back to the shared agent_loop with its default LLM
            bound_llm = getattr(agent, "_llm_provider", None)
            agent_emitter = getattr(agent, "_progress_emitter", None)
            if bound_llm is not None:
                from .loop_engine import AgentLoop as _AgentLoop
                dedicated_loop = _AgentLoop(
                    tool_loop=self.agent_loop.tool_loop,
                    llm=bound_llm,
                    config=self.config,
                    interaction_hub=self.interaction_hub,
                    mailbox=agent_mailbox,
                    emitter=agent_emitter,
                    sandbox=self.sandbox,
                    agent_permissions=getattr(agent, '_permissions', None),
                )
                result = await agent.run(dedicated_loop, task.scope, context,
                                        allowed_tools=task.required_tools)
            else:
                # Still pass the emitter to the shared loop
                loop_with_emitter = type(self.agent_loop)(
                    tool_loop=self.agent_loop.tool_loop,
                    llm=self.agent_loop.llm,
                    config=self.config,
                    interaction_hub=self.interaction_hub,
                    mailbox=agent_mailbox,
                    emitter=agent_emitter,
                    sandbox=self.sandbox,
                    agent_permissions=getattr(agent, '_permissions', None),
                )
                result = await agent.run(loop_with_emitter, task.scope, context,
                                        allowed_tools=task.required_tools)

            # Unregister mailbox after agent completes
            if self.mail_router is not None:
                self.mail_router.unregister(agent.agent_id)

            branch.log("task_completed", {"summary": result.summary})

            # Evaluate quality
            eval_result = self.evaluator.evaluate(result, task.scope)
            task.evaluation = eval_result

            # ── Evolution: record structured journal entry ──────
            evolution_engine = getattr(agent, '_evolution_engine', None)
            if evolution_engine:
                from src.evolution import JournalEntry
                import uuid as _uuid
                elapsed = (datetime.now(timezone.utc) - task.started_at).total_seconds() if task.started_at else 0.0
                task_type = getattr(agent, '_task_type', self._classify_task(task))
                entry = JournalEntry(
                    id=str(_uuid.uuid4()),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    task_scope=task.scope,
                    task_type=task_type,
                    outcome="success" if eval_result.action == "accept" else "failure",
                    score=eval_result.overall if hasattr(eval_result, 'overall') else 0.8,
                    duration_seconds=elapsed,
                    tools_used=task.required_tools or [],
                    llm_provider=getattr(agent, '_llm_provider_id', 'unknown'),
                    cost_estimate=getattr(result, 'cost_estimate', 0.0) if hasattr(result, 'cost_estimate') else 0.0,
                    lessons=[],
                    tags=[],
                )
                try:
                    await evolution_engine.record_entry(entry)
                    await evolution_engine.adjust_traits(entry)
                    stats = evolution_engine.get_stats()
                    if stats.get('total_tasks', 0) % 5 == 0:
                        await evolution_engine.extract_knowledge()
                    logger.debug("AgentManager: evolution recorded for %s (%s, score=%.2f)",
                               task.task_id, entry.outcome, entry.score)
                except Exception as e:
                    logger.warning("AgentManager: evolution recording failed: %s", e)

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
            # Snapshot agent state for cross-session persistence
            if self.persistence:
                try:
                    await self.persistence.snapshot(agent.agent_id)
                except Exception as e:
                    logger.warning("AgentManager: persistence.snapshot failed: %s", e)
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

    # ── Persistent Agent lifecycle ────────────────────────────────────

    async def create_persistent_agent(self, name: str, personality: str = "executor",
                                      role: str = "coder") -> tuple[Agent, AgentSoul]:
        """Create a persistent Agent — not destroyed, continuously evolves.

        Args:
            name: Human-readable agent name
            personality: Personality card (executor/analyst/creative/guardian)
            role: Role card (coder/researcher/ops)

        Returns (agent, soul) tuple.
        """
        agent = Agent(agent_id=f"agent:persistent:{name}")
        agent.role = AgentRole.WORKER
        agent.status = AgentStatus.IDLE

        # Build AgentSoul with personality + role cards
        soul = SoulBuilder(agent_id=agent.agent_id)\
            .with_personality(personality)\
            .with_role(role)\
            .build()

        # Store the soul reference on the agent
        agent._soul = soul

        # Add to pool
        self.pool.agents[agent.agent_id] = agent

        logger.info("AgentManager: created persistent agent '%s' (%s/%s)",
                    name, personality, role)
        return agent, soul

    async def destroy_agent(self, agent: Agent, reason: str = "manual") -> None:
        """Destroy an agent → move private files to trash/ → record reason.

        Steps:
          1. Remove from pool
          2. Move state/agents/{id}/ → trash/agents/{id}/
          3. Write destroyed_at + destroy_reason to meta.json
        """
        async with self._lock:
            agent_id = agent.agent_id

            # Remove from pool
            self.pool.agents.pop(agent_id, None)
            agent.status = AgentStatus.DESTROYED

        # Move files to trash (I/O outside lock)
        state_dir = Path("state/agents") / agent_id
        trash_dir = Path("trash/agents") / agent_id

        if state_dir.exists():
            try:
                trash_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(state_dir), str(trash_dir))

                # Update meta.json in trash
                meta_path = trash_dir / "meta.json"
                if meta_path.exists():
                    import json
                    meta = json.loads(meta_path.read_text())
                    meta["destroyed_at"] = datetime.now(timezone.utc).isoformat()
                    meta["destroy_reason"] = reason
                    meta_path.write_text(json.dumps(meta, indent=2))

                logger.info("AgentManager: destroyed '%s' → trash (reason: %s)",
                           agent_id, reason)
            except Exception as e:
                logger.warning("AgentManager: destroy_agent failed to move: %s", e)
        else:
            logger.info("AgentManager: destroyed '%s' (no state files) (reason: %s)",
                       agent_id, reason)

    # ── Collection ────────────────────────────────────────────────────

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
        async with self._lock:
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

    # ── Agent switching (user-facing) ─────────────────────────────

    def list_agents(self) -> list[dict]:
        """List all persistent agents in the pool.

        Returns a list of agent summary dicts:
          [{agent_id, status, role, expertise, task_count, created_at}, ...]
        """
        agents = []
        for agent_id, agent in self.pool.agents.items():
            agents.append({
                "agent_id": agent.agent_id,
                "status": agent.status.value if hasattr(agent.status, 'value') else str(agent.status),
                "role": agent.role.value if hasattr(agent.role, 'value') else str(agent.role),
                "expertise": agent.expertise,
                "task_count": agent.task_count,
                "success_count": agent.success_count,
                "created_at": getattr(agent, 'created_at', None),
            })
        return agents

    def get_active_agent(self) -> Agent | None:
        """Get the currently active agent (first RUNNING agent, or first IDLE).

        Returns an Agent instance, or None if pool is empty.
        """
        # Prefer RUNNING agent
        for agent in self.pool.agents.values():
            if agent.status == AgentStatus.RUNNING:
                return agent
        # Fallback to IDLE
        for agent in self.pool.agents.values():
            if agent.status == AgentStatus.IDLE:
                return agent
        # Any agent
        if self.pool.agents:
            return next(iter(self.pool.agents.values()))
        return None

    async def switch_agent(self, agent_id: str) -> bool:
        """Switch the active agent context.

        Steps:
          1. Save current agent's conversation context (if any)
          2. Load target agent's context from state/
          3. Update state/session.json to reflect new active agent

        Returns True on success, False if target agent not found.
        """
        async with self._lock:
            agent = self.pool.agents.get(agent_id)
            if not agent:
                logger.warning("AgentManager: switch_agent failed — '%s' not found", agent_id)
                return False

            # Save current active agent context
            current = self.get_active_agent()
        session_path = Path("state/session.json")
        session_data = {}
        if session_path.exists():
            try:
                session_data = json.loads(session_path.read_text())
            except (json.JSONDecodeError, Exception):
                pass

        if current:
            session_data["previous_agent_id"] = current.agent_id

        session_data["active_agent_id"] = agent_id
        session_data["switched_at"] = datetime.now(timezone.utc).isoformat()

        import json as _json  # already imported at top
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.write_text(_json.dumps(session_data, indent=2))

        logger.info(
            "AgentManager: switched active agent '%s' → '%s'",
            current.agent_id if current else "none", agent_id,
        )
        return True

    # ── Agent Forking ────────────────────────────────────────────────

    def enable_forker(self, state_dir: str = "state/agents") -> None:
        """Enable agent forking capability.

        Creates an AgentForker instance that can clone agents for
        parallel tasks, experiments, or specialization.
        """
        from .agent_fork import AgentForker
        self.forker = AgentForker(state_dir=state_dir)

    async def fork_and_dispatch(self, parent_id: str, task: ManagedTask,
                                fork_reason: str = "parallel_task",
                                inherit_soul: bool = True,
                                inherit_knowledge: bool = True,
                                role_override: str | None = None) -> str | None:
        """Fork a child agent from a parent and dispatch a task to it.

        Steps:
          1. Create ForkConfig from parent agent
          2. Fork child agent via AgentForker
          3. Create a ManagedTask for the child
          4. Dispatch task to child agent
          5. Return child agent_id

        Returns child agent_id on success, None if forker not enabled.
        """
        if not self.forker:
            logger.warning("AgentManagerAgent: forker not enabled, call enable_forker() first")
            return None

        from .agent_fork import ForkConfig

        config = ForkConfig(
            parent_id=parent_id,
            fork_reason=fork_reason,
            inherit_soul=inherit_soul,
            inherit_knowledge=inherit_knowledge,
            role_override=role_override,
        )

        child_id = await self.forker.fork(config)

        async with self._lock:
            task.assigned_agent_id = child_id
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc)

        logger.info("AgentManagerAgent: forked %s → %s for task %s (reason: %s)",
                    parent_id, child_id, task.task_id, fork_reason)
        return child_id

    async def merge_child(self, child_id: str, parent_id: str) -> None:
        """Merge a child agent's knowledge back into the parent.

        Called after a forked child completes its task.
        """
        if not self.forker:
            logger.warning("AgentManagerAgent: forker not enabled")
            return
        await self.forker.merge_back(child_id, parent_id)

    def get_family_tree(self, agent_id: str) -> dict:
        """Get the family tree for an agent."""
        if not self.forker:
            return {"agent_id": agent_id, "error": "forker not enabled"}
        return self.forker.get_family_tree(agent_id)
