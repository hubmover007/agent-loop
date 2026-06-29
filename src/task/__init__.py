"""Task management system - TaskTree, decomposition, scheduling, branch spaces."""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..core import TaskStatus

logger = logging.getLogger(__name__)


# ============================================================
# Task Node
# ============================================================

@dataclass
class TaskNode:
    """A node in the TaskTree."""
    task_id: str
    scope: str
    parent_id: str | None = None
    children: list[TaskNode] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 3
    dependencies: list[str] = field(default_factory=list)
    required_tools: list[str] = field(default_factory=list)

    # Result tracking
    assigned_agent: str | None = None
    result: dict | None = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: datetime | None = None

    @property
    def is_ready(self) -> bool:
        """Check if task is ready to execute (all dependencies satisfied)."""
        return self.status == TaskStatus.PENDING and not self.dependencies

    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf task (no children)."""
        return len(self.children) == 0


# ============================================================
# TaskTree
# ============================================================

class TaskTree:
    """Hierarchical tree of tasks with dependency tracking."""

    def __init__(self, root_scope: str):
        self.root = TaskNode(
            task_id=f"task:{uuid.uuid4().hex[:12]}",
            scope=root_scope,
        )
        self._node_index: dict[str, TaskNode] = {self.root.task_id: self.root}

    def add_child(self, parent_id: str, scope: str, priority: int = 3,
                  dependencies: list[str] | None = None,
                  required_tools: list[str] | None = None) -> str:
        """Add a child task to a parent."""
        parent = self._node_index.get(parent_id)
        if not parent:
            raise ValueError(f"Parent task {parent_id} not found")

        task_id = f"task:{uuid.uuid4().hex[:12]}"
        node = TaskNode(
            task_id=task_id,
            scope=scope,
            parent_id=parent_id,
            priority=priority,
            dependencies=dependencies or [],
            required_tools=required_tools or [],
        )
        parent.children.append(node)
        self._node_index[task_id] = node
        return task_id

    def get(self, task_id: str) -> TaskNode | None:
        """Get a task node by ID."""
        return self._node_index.get(task_id)

    def get_ready_tasks(self) -> list[TaskNode]:
        """Get all tasks that are ready to execute (dependencies satisfied)."""
        ready = []
        for node in self._node_index.values():
            if node.is_ready:
                # Verify dependencies are all done
                deps_done = all(
                    self._node_index.get(d) and self._node_index[d].status == TaskStatus.DONE
                    for d in node.dependencies
                )
                if deps_done:
                    ready.append(node)
        return ready

    def mark_done(self, task_id: str, result: dict | None = None) -> None:
        """Mark a task as done with its result."""
        node = self._node_index.get(task_id)
        if node:
            node.status = TaskStatus.DONE
            node.result = result
            node.completed_at = datetime.now()

            # Check if all children done → mark parent
            if node.parent_id:
                parent = self._node_index.get(node.parent_id)
                if parent and all(c.status == TaskStatus.DONE for c in parent.children):
                    parent.status = TaskStatus.DONE
                    parent.completed_at = datetime.now()

    def mark_failed(self, task_id: str) -> None:
        """Mark a task as failed."""
        node = self._node_index.get(task_id)
        if node:
            node.status = TaskStatus.FAILED

            # Propagate failure upward
            current = node
            while current.parent_id:
                parent = self._node_index.get(current.parent_id)
                if parent:
                    # If parent has no more pending children, mark failed
                    pending = [c for c in parent.children if c.status == TaskStatus.PENDING]
                    if not pending and any(c.status == TaskStatus.FAILED for c in parent.children):
                        parent.status = TaskStatus.FAILED
                    current = parent
                else:
                    break

    def all_tasks(self) -> list[TaskNode]:
        """Return all task nodes."""
        return list(self._node_index.values())

    def to_dict(self) -> dict:
        """Serialize to dict for storage."""
        def _serialize(node: TaskNode) -> dict:
            return {
                "task_id": node.task_id,
                "scope": node.scope,
                "parent_id": node.parent_id,
                "status": node.status.value,
                "priority": node.priority,
                "dependencies": node.dependencies,
                "children": [_serialize(c) for c in node.children],
            }
        return _serialize(self.root)

    @classmethod
    def from_dict(cls, data: dict) -> TaskTree:
        """Deserialize from dict."""
        tree = cls(data["scope"])
        tree.root.task_id = data["task_id"]

        def _deserialize(parent_id: str, children: list[dict]):
            for child in children:
                child_id = tree.add_child(
                    parent_id=parent_id,
                    scope=child["scope"],
                    priority=child.get("priority", 3),
                    dependencies=child.get("dependencies", []),
                )
                # Override the generated ID with stored one
                tree._node_index[child_id].task_id = child["task_id"]
                del tree._node_index[child_id]
                tree._node_index[child["task_id"]] = tree._node_index.pop(child_id)
                tree._node_index[child["task_id"]].task_id = child["task_id"]

        _deserialize(tree.root.task_id, data.get("children", []))
        return tree


# ============================================================
# TaskScheduler
# ============================================================

class TaskScheduler:
    """Schedules tasks based on dependencies and priority."""

    def __init__(self, tree: TaskTree):
        self.tree = tree
        self._dispatch_queue: deque[TaskNode] = deque()

    async def decompose(self, task_scope: str, llm: Any) -> TaskTree:
        """Use LLM to decompose a complex task into subtasks.

        Args:
            task_scope: The overall task description
            llm: LLM provider for decomposition

        Returns:
            TaskTree with decomposed subtasks
        """
        tree = TaskTree(task_scope)

        try:
            prompt = f"""Analyze this task and decompose it into subtasks if it's complex.

Task: {task_scope}

Rules:
- If the task is SIMPLE (single action), return an empty array
- If complex, break down into 2-5 subtasks
- Each subtask must be a concrete, independently executable unit
- Mark dependencies: if subtask B needs subtask A's result, list A as dependency

Respond as JSON:
{{
  "is_complex": true/false,
  "subtasks": [
    {{
      "scope": "concrete subtask description",
      "priority": 1-5,
      "dependencies": [],
      "required_tools": []
    }}
  ]
}}"""

            response = await llm.chat([{"role": "user", "content": prompt}])
            import json

            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            plan = json.loads(content.strip())

            if plan.get("is_complex") and plan.get("subtasks"):
                for st in plan["subtasks"]:
                    tree.add_child(
                        parent_id=tree.root.task_id,
                        scope=st["scope"],
                        priority=st.get("priority", 3),
                        dependencies=st.get("dependencies", []),
                        required_tools=st.get("required_tools", []),
                    )

        except Exception as e:
            logger.warning("Task decomposition failed: %s, using single task", e)

        return tree

    def schedule(self) -> list[TaskNode]:
        """Get the next batch of ready-to-execute tasks."""
        ready = self.tree.get_ready_tasks()
        # Sort by priority (higher first)
        ready.sort(key=lambda t: t.priority, reverse=True)
        return ready

    def get_next(self) -> TaskNode | None:
        """Get the next task to execute (highest priority, ready)."""
        ready = self.schedule()
        return ready[0] if ready else None

    @property
    def is_complete(self) -> bool:
        """Check if all tasks are done."""
        return self.tree.root.status in (TaskStatus.DONE, TaskStatus.FAILED)


# ============================================================
# BranchSpace
# ============================================================

class BranchSpace:
    """Isolated workspace for an agent's task execution."""

    def __init__(self, task_id: str, agent_id: str, base_dir: Path | None = None):
        self.task_id = task_id
        self.agent_id = agent_id
        self.base_dir = base_dir or Path(f"/tmp/agent_loop/branches/{task_id}")

        # Workspace state
        self.memory_snapshot: dict = {}
        self.execution_log: list = []
        self.artifacts: dict[str, Any] = {}
        self.created_at = datetime.now()

    async def init(self, memory: Any) -> None:
        """Initialize branch space: create directory + capture memory snapshot."""
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # Capture relevant memory snapshot (read-only for agent)
        try:
            # Get recent episodes related to this task
            result = await memory._db.query("""
                SELECT * FROM episode
                ORDER BY created_at DESC
                LIMIT 10
            """)
            episodes = result if isinstance(result, list) else result.get("result", [])
            self.memory_snapshot = {
                "episodes": [
                    {"title": e.get("title"), "summary": e.get("summary")}
                    for e in episodes
                ],
                "captured_at": self.created_at.isoformat(),
            }
        except Exception as e:
            logger.warning("Memory snapshot failed: %s", e)
            self.memory_snapshot = {}

    async def commit(self, memory: Any) -> None:
        """Commit branch changes back to mainline memory."""
        # Write execution log as an episode
        if self.execution_log:
            try:
                await memory.write_episode(
                    title=f"Branch: {self.agent_id} - {self.task_id}",
                    summary=f"Agent {self.agent_id} executed task {self.task_id}",
                    content=str(self.execution_log),
                )
            except Exception as e:
                logger.warning("Branch commit failed: %s", e)

    async def cleanup(self) -> None:
        """Clean up temporary files."""
        import shutil
        try:
            if self.base_dir.exists():
                shutil.rmtree(self.base_dir)
        except Exception as e:
            logger.warning("Branch cleanup failed: %s", e)

    def log(self, action: str, data: dict | None = None) -> None:
        """Log an action in this branch space."""
        self.execution_log.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "data": data,
        })
