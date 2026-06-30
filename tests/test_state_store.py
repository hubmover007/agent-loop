"""Tests for StateStore — structured JSON state persistence."""

import json
import tempfile
from pathlib import Path

import pytest

from src.state_store import StateStore


# ============================================================
# Test helpers
# ============================================================

class FakeTask:
    """Minimal ManagedTask-like object for testing StateStore.save_task."""

    def __init__(self, task_id, scope, status, priority=3,
                 dependencies=None, required_tools=None,
                 assigned_agent_id=None, retry_count=0, error=None):
        self.task_id = task_id
        self.scope = scope
        self.priority = priority
        self.dependencies = dependencies or []
        self.required_tools = required_tools or []
        self.parent_id = None
        self.status = status
        self.assigned_agent_id = assigned_agent_id
        self.retry_count = retry_count
        self.error = error
        self.branch_id = None
        self.created_at = None
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.evaluation = None


# ============================================================
# Test: save and load task
# ============================================================

@pytest.mark.asyncio
async def test_save_and_load_task():
    """Saving a task and loading it back preserves all fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = StateStore(base_dir=tmpdir)

        task = FakeTask(
            task_id="task:abc123",
            scope="Test task scope",
            status="done",
            priority=5,
            dependencies=["task:dep1"],
            assigned_agent_id="agent:xyz",
            retry_count=1,
            error=None,
        )

        await store.save_task(task)

        loaded = await store.load_task("task:abc123")
        assert loaded is not None
        assert loaded["task_id"] == "task:abc123"
        assert loaded["scope"] == "Test task scope"
        assert loaded["status"] == "done"
        assert loaded["priority"] == 5
        assert loaded["dependencies"] == ["task:dep1"]
        assert loaded["assigned_agent_id"] == "agent:xyz"
        assert loaded["retry_count"] == 1
        assert loaded["error"] is None

        # Non-existent task returns None
        missing = await store.load_task("task:nonexistent")
        assert missing is None


# ============================================================
# Test: save agent
# ============================================================

@pytest.mark.asyncio
async def test_save_agent():
    """Agent JSON file is written with correct format and fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = StateStore(base_dir=tmpdir)

        agent_info = {
            "agent_id": "agent:test1",
            "status": "idle",
            "role": "worker",
            "expertise": ["ssh", "web"],
            "task_count": 5,
            "success_count": 4,
            "created_at": None,
            "llm_provider_id": "deepseek-chat",
        }

        await store.save_agent("agent:test1", agent_info)

        # Check file exists and has correct content
        path = Path(tmpdir) / "agents" / "agent:test1.json"
        assert path.exists()

        with open(path) as f:
            data = json.load(f)

        assert data["agent_id"] == "agent:test1"
        assert data["status"] == "idle"
        assert data["role"] == "worker"
        assert data["expertise"] == ["ssh", "web"]
        assert data["task_count"] == 5
        assert data["success_count"] == 4
        assert data["llm_provider_id"] == "deepseek-chat"

        # Load back
        loaded = await store.load_agent("agent:test1")
        assert loaded == data


# ============================================================
# Test: log LLM usage
# ============================================================

@pytest.mark.asyncio
async def test_log_llm_usage():
    """LLM usage records are correctly appended to usage.jsonl."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = StateStore(base_dir=tmpdir)

        record1 = {
            "timestamp": "2026-06-30T08:00:00Z",
            "provider_id": "deepseek-chat",
            "model": "deepseek-chat",
            "input_tokens": 100,
            "output_tokens": 200,
            "latency_ms": 800.5,
            "success": True,
            "task_id": "task:abc",
            "agent_id": "agent:xyz",
        }

        record2 = {
            "timestamp": "2026-06-30T08:01:00Z",
            "provider_id": "deepseek-reasoner",
            "model": "deepseek-reasoner",
            "input_tokens": 500,
            "output_tokens": 1000,
            "latency_ms": 2500.0,
            "success": False,
            "task_id": "task:def",
            "agent_id": None,
        }

        await store.log_llm_usage(record1)
        await store.log_llm_usage(record2)

        # Read back usage.jsonl
        path = Path(tmpdir) / "llm_pool" / "usage.jsonl"
        assert path.exists()

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        parsed1 = json.loads(lines[0])
        assert parsed1["provider_id"] == "deepseek-chat"
        assert parsed1["input_tokens"] == 100
        assert parsed1["success"] is True

        parsed2 = json.loads(lines[1])
        assert parsed2["provider_id"] == "deepseek-reasoner"
        assert parsed2["success"] is False


# ============================================================
# Test: list tasks by status
# ============================================================

@pytest.mark.asyncio
async def test_list_tasks_by_status():
    """list_tasks filters by status correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = StateStore(base_dir=tmpdir)

        task_done = FakeTask(task_id="task:1", scope="s1", status="done")
        task_pending = FakeTask(task_id="task:2", scope="s2", status="pending")
        task_failed = FakeTask(task_id="task:3", scope="s3", status="failed")
        task_done2 = FakeTask(task_id="task:4", scope="s4", status="done")

        await store.save_task(task_done)
        await store.save_task(task_pending)
        await store.save_task(task_failed)
        await store.save_task(task_done2)

        # All tasks
        all_tasks = store.list_tasks()
        assert len(all_tasks) == 4

        # Filter by status: done
        done_tasks = store.list_tasks(status="done")
        assert len(done_tasks) == 2
        assert {t["task_id"] for t in done_tasks} == {"task:1", "task:4"}

        # Filter by status: pending
        pending_tasks = store.list_tasks(status="pending")
        assert len(pending_tasks) == 1
        assert pending_tasks[0]["task_id"] == "task:2"

        # Filter by status: failed
        failed_tasks = store.list_tasks(status="failed")
        assert len(failed_tasks) == 1
        assert failed_tasks[0]["task_id"] == "task:3"


# ============================================================
# Test: session save and summary
# ============================================================

@pytest.mark.asyncio
async def test_save_and_load_session():
    """Session is saved and can be loaded back."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = StateStore(base_dir=tmpdir)

        session_info = {
            "session_id": "sess-001",
            "user_input": "What is the weather?",
            "final_output": "Sunny, 25°C",
            "task_count": 3,
            "task_ids": ["task:1", "task:2", "task:3"],
            "error_count": 0,
            "errors": [],
            "reason_confidence": 0.92,
            "phase": "output",
        }

        await store.save_session("sess-001", session_info)

        # Load back
        summary = store.session_summary("sess-001")
        assert summary is not None
        assert summary["session_id"] == "sess-001"
        assert summary["user_input"] == "What is the weather?"
        assert summary["task_count"] == 3
        assert summary["reason_confidence"] == 0.92


# ============================================================
# Test: list agents
# ============================================================

@pytest.mark.asyncio
async def test_list_agents_by_status():
    """list_agents filters by status correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = StateStore(base_dir=tmpdir)

        await store.save_agent("agent:1", {"status": "idle", "role": "worker", "expertise": []})
        await store.save_agent("agent:2", {"status": "running", "role": "worker", "expertise": []})
        await store.save_agent("agent:3", {"status": "idle", "role": "expert", "expertise": ["ssh"]})

        all_agents = store.list_agents()
        assert len(all_agents) == 3

        idle_agents = store.list_agents(status="idle")
        assert len(idle_agents) == 2
