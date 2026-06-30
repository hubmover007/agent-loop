"""StateStore — structured JSON state persistence for tasks and agents.

Directory layout:
  state/
    tasks/{task_id}.json
    agents/{agent_id}.json
    sessions/{session_id}.json
    llm_pool/usage.jsonl
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class StateStore:
    """Structured JSON state persistence for tasks, agents, and sessions."""

    def __init__(self, base_dir: str = "state"):
        self.base_dir = Path(base_dir)
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Create state directory structure."""
        for sub in ("tasks", "agents", "sessions", "llm_pool"):
            (self.base_dir / sub).mkdir(parents=True, exist_ok=True)

    # ----- Helpers -----

    def _serialize_datetime(self, dt: datetime | None) -> str | None:
        """Serialize datetime to ISO 8601 string."""
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.isoformat()

    def _serialize_enum(self, value: Any) -> Any:
        """Serialize enum to its .value."""
        if hasattr(value, "value"):
            return value.value
        return value

    def _atom_write(self, path: Path, data: dict) -> None:
        """Atomic write via temp file + rename."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    # ----- Task Persistence -----

    async def save_task(self, task: Any) -> None:
        """Serialize a ManagedTask to state/tasks/{task_id}.json.

        Accepts any object with task_id and to_dict() method.
        """
        path = self.base_dir / "tasks" / f"{task.task_id}.json"

        # Attempt full serialization from the ManagedTask object
        try:
            data = {
                "task_id": task.task_id,
                "scope": getattr(task, "scope", ""),
                "priority": getattr(task, "priority", 3),
                "dependencies": getattr(task, "dependencies", []),
                "required_tools": getattr(task, "required_tools", []),
                "parent_id": getattr(task, "parent_id", None),
                "status": self._serialize_enum(getattr(task, "status", None)),
                "assigned_agent_id": getattr(task, "assigned_agent_id", None),
                "created_at": self._serialize_datetime(getattr(task, "created_at", None)),
                "started_at": self._serialize_datetime(getattr(task, "started_at", None)),
                "completed_at": self._serialize_datetime(getattr(task, "completed_at", None)),
                "retry_count": getattr(task, "retry_count", 0),
                "error": getattr(task, "error", None),
                "branch_id": getattr(task, "branch_id", None),
            }

            # Include result summary if available
            result = getattr(task, "result", None)
            if result is not None:
                data["result_summary"] = getattr(result, "summary", str(result))

            # Include evaluation if available
            evaluation = getattr(task, "evaluation", None)
            if evaluation is not None:
                data["evaluation"] = {
                    "overall": getattr(evaluation, "overall", None),
                    "action": getattr(evaluation, "action", None),
                    "reason": getattr(evaluation, "reason", None),
                }
        except Exception as e:
            logger.error("StateStore: failed to serialize task %s: %s", getattr(task, "task_id", "?"), e)
            return

        self._atom_write(path, data)
        logger.debug("StateStore: saved task %s", task.task_id)

    async def load_task(self, task_id: str) -> dict | None:
        """Load a task from state/tasks/{task_id}.json."""
        path = self.base_dir / "tasks" / f"{task_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_tasks(self, status: str | None = None) -> list[dict]:
        """List all tasks, optionally filtered by status."""
        tasks_dir = self.base_dir / "tasks"
        if not tasks_dir.exists():
            return []

        results = []
        for f in sorted(tasks_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if status is None or data.get("status") == status:
                    results.append(data)
            except Exception as e:
                logger.warning("StateStore: failed to read %s: %s", f, e)

        return results

    # ----- Agent Persistence -----

    async def save_agent(self, agent_id: str, info: dict) -> None:
        """Serialize agent info to state/agents/{agent_id}.json.

        Expected info keys:
          agent_id, status, role, expertise, task_count, success_count,
          created_at, llm_provider_id
        """
        path = self.base_dir / "agents" / f"{agent_id}.json"

        data = {
            "agent_id": agent_id,
            "status": self._serialize_enum(info.get("status")),
            "role": self._serialize_enum(info.get("role")),
            "expertise": info.get("expertise", []),
            "task_count": info.get("task_count", 0),
            "success_count": info.get("success_count", 0),
            "created_at": self._serialize_datetime(info.get("created_at")),
            "llm_provider_id": info.get("llm_provider_id"),
        }
        # Merge any extra fields
        for k, v in info.items():
            if k not in data:
                data[k] = v

        self._atom_write(path, data)
        logger.debug("StateStore: saved agent %s", agent_id)

    async def load_agent(self, agent_id: str) -> dict | None:
        """Load an agent from state/agents/{agent_id}.json."""
        path = self.base_dir / "agents" / f"{agent_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_agents(self, status: str | None = None) -> list[dict]:
        """List all agents, optionally filtered by status."""
        agents_dir = self.base_dir / "agents"
        if not agents_dir.exists():
            return []

        results = []
        for f in sorted(agents_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if status is None or data.get("status") == status:
                    results.append(data)
            except Exception as e:
                logger.warning("StateStore: failed to read %s: %s", f, e)

        return results

    # ----- Session Persistence -----

    async def save_session(self, session_id: str, info: dict) -> None:
        """Serialize session info to state/sessions/{session_id}.json."""
        path = self.base_dir / "sessions" / f"{session_id}.json"

        data = {
            "session_id": session_id,
            **info,
        }

        self._atom_write(path, data)
        logger.debug("StateStore: saved session %s", session_id)

    def session_summary(self, session_id: str) -> dict | None:
        """Load session summary."""
        path = self.base_dir / "sessions" / f"{session_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ----- LLM Usage Logging -----

    async def log_llm_usage(self, record: dict) -> None:
        """Append an LLM usage record to state/llm_pool/usage.jsonl.

        Expected record keys:
          timestamp, provider_id, model, input_tokens, output_tokens,
          latency_ms, success, task_id, agent_id
        """
        path = self.base_dir / "llm_pool" / "usage.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.debug("StateStore: logged LLM usage for %s", record.get("provider_id", "?"))
