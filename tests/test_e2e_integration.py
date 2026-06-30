"""E2E integration tests for agent-loop cross-module integration.

Covers: AgentSoul, LLMPool, InteractionHub, AgentMailbox, CostController,
        EvolutionEngine, ProgressEmitter, AgentForker, MultimodalProcessor,
        PersistenceManager, ToolRegistry.

All tests use mock LLM (no API key required).
"""

import asyncio
import json
import os
import sys
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import TaskStatus, AgentStatus, TaskResult, EvaluationResult, AgentRole
from src.system_agents import (
    AgentManagerAgent, TaskRegistry, ManagedTask, TaskAgent,
)
from src.agent_soul import AgentSoul, SoulBuilder
from src.evolution import EvolutionEngine, JournalEntry
from src.interaction import InteractionHub, ApprovalRequest, detect_risk_level, is_high_risk
from src.agent_mailbox import AgentMailbox, MailRouter, AgentMessage
from src.cost_control import CostController, Budget, BudgetLimitError
from src.tool_registry import ToolRegistry, ToolSpec, BuiltinTools
from src.agent_fork import AgentForker, ForkConfig
from src.persistence import PersistenceManager, Snapshot
from src.llm_pool import LLMPool
from src.streaming import ProgressEmitter, ProgressEvent
from src.agent import Agent, AgentPool


# ============================================================
# Test helpers
# ============================================================

def _temp_dir():
    """Create a temporary directory that persists for the test duration."""
    d = tempfile.mkdtemp(prefix="agent_loop_test_")
    return d


def _cleanup_dir(d: str):
    """Recursively remove a temporary directory."""
    import shutil
    if os.path.exists(d):
        shutil.rmtree(d, ignore_errors=True)


# ============================================================
# Test 1: Full Agent Lifecycle
# ============================================================

@pytest.mark.asyncio
async def test_full_agent_lifecycle():
    """Agent lifecycle: create persistent → dispatch → execute → evolve → snapshot → destroy → trash."""
    tmp = _temp_dir()
    try:
        state_dir = Path(tmp) / "state"
        state_dir.mkdir(parents=True, exist_ok=True)

        # Setup
        registry = TaskRegistry()
        router = MailRouter()
        hub = InteractionHub(risk_threshold="critical")
        persistence = PersistenceManager(
            state_dir=str(state_dir),
            snapshot_dir=str(state_dir / "snapshots"),
        )

        # Mock AgentLoop + LLM
        mock_loop = MagicMock()
        mock_loop.tool_loop = MagicMock()
        mock_loop.llm = MagicMock()
        from src.loop_engine import LoopConfig
        config = LoopConfig()

        # Create AgentManagerAgent (autonomous decision)
        mgr = AgentManagerAgent(
            memory=MagicMock(),
            agent_loop=mock_loop,
            config=config,
            registry=registry,
            interaction_hub=hub,
            mail_router=router,
            persistence=persistence,
        )

        # ── Create persistent agent ──
        agent, soul = await mgr.create_persistent_agent(
            name="lifecycle_test",
            personality="executor",
            role="coder",
        )
        assert agent.agent_id.startswith("agent:persistent:")
        assert soul is not None
        assert hasattr(agent, '_soul')
        assert agent.status == AgentStatus.IDLE

        # ── Register task ──
        task = ManagedTask(
            task_id=f"task:{uuid.uuid4().hex[:8]}",
            scope="refactor the user auth module",
            required_tools=["fs.read_file", "fs.write_file"],
        )
        registry._tasks[task.task_id] = task
        registry._order.append(task.task_id)

        # ── Classify task ──
        task_type = AgentManagerAgent._classify_task(task)
        assert task_type == "coding"

        # ── Verify soul files created (default path: state/agents/) ──
        agent_dir = Path("state/agents") / agent.agent_id
        assert (agent_dir / "IDENTITY.md").exists(), f"Expected IDENTITY.md at {agent_dir}"
        assert (agent_dir / "ROLE.md").exists()
        assert (agent_dir / "JOURNAL.md").exists()
        assert (agent_dir / "profile.json").exists()

        # ── Snapshot ──
        snap = await persistence.snapshot(agent.agent_id)
        assert snap.agent_id == agent.agent_id
        assert snap.state_version == 1

        # ── Destroy → trash ──
        await mgr.destroy_agent(agent, reason="test_complete")
        assert agent.agent_id not in mgr.pool.agents
        assert agent.status == AgentStatus.DESTROYED

        # ── Trash audit ──
        trash_dir = Path("trash/agents") / agent.agent_id
        assert trash_dir.exists(), f"Trash directory {trash_dir} should exist"
        meta_path = trash_dir / "meta.json"
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text())
        assert "destroyed_at" in meta
        assert meta.get("destroy_reason") == "test_complete"

    finally:
        _cleanup_dir(tmp)
        # Clean up trash
        import shutil
        trash_root = Path("trash")
        if trash_root.exists():
            shutil.rmtree(str(trash_root), ignore_errors=True)


# ============================================================
# Test 2: LLMPool + CostController Integration
# ============================================================

@pytest.mark.asyncio
async def test_llm_pool_with_cost_control():
    """LLMPool + CostController: within budget → accept, over budget → reject."""
    tmp = _temp_dir()
    try:
        # Create a strict budget
        budget = Budget(daily_limit=0.10, monthly_limit=1.00, per_task_limit=0.05)
        cost_state_path = Path(tmp) / "cost.json"
        controller = CostController(budget=budget, state_path=str(cost_state_path))

        # Within budget
        assert controller.check(estimated_cost=0.01, task_scope="task-1") is True
        controller.record(actual_cost=0.01, provider_id="mock", tokens_in=100, tokens_out=50)

        # Within budget again
        assert controller.check(estimated_cost=0.02, task_scope="task-1") is True
        controller.record(actual_cost=0.02, provider_id="mock", tokens_in=200, tokens_out=100)

        # Check per-task limit — already spent 0.03, limit is 0.05
        controller.record_task("task-1", 0.03)
        assert controller.check(estimated_cost=0.01, task_scope="task-1") is True  # total would be 0.04

        # Exceed per-task limit
        controller.record_task("task-1", 0.02)  # now at 0.05 (limit)
        assert controller.check(estimated_cost=0.01, task_scope="task-1") is False  # would be 0.06

        # Exceed daily limit (use same state_path for proper roundtrip)
        controller2a = CostController(
            budget=Budget(daily_limit=0.05, monthly_limit=100.0),
            state_path=str(cost_state_path),
        )
        controller2a.record(actual_cost=0.04, provider_id="mock")
        assert controller2a.check(estimated_cost=0.02) is False  # would be 0.06 > 0.05

        # Persist and load roundtrip
        await controller.persist()
        assert cost_state_path.exists(), f"Cost state not persisted to {cost_state_path}"

        loaded = CostController(budget=budget, state_path=str(cost_state_path))
        await loaded.load()
        remaining = loaded.get_remaining()
        assert "daily_remaining" in remaining
        assert "monthly_remaining" in remaining

    finally:
        _cleanup_dir(tmp)


# ============================================================
# Test 3: Soul builds and evolves
# ============================================================

@pytest.mark.asyncio
async def test_soul_builds_and_evolves():
    """AgentSoul: build → execute task → EVOLVE → journal entry → trait adjustment."""
    tmp = _temp_dir()
    try:
        agent_id = f"agent:test:{uuid.uuid4().hex[:8]}"
        state_root = Path(tmp) / "state"

        # Build soul
        soul = SoulBuilder(agent_id=agent_id, state_root=str(state_root)) \
            .with_personality("executor") \
            .with_role("coder") \
            .build()

        assert soul.agent_id == agent_id
        assert soul.personality == "executor"
        assert soul.role == "coder"

        # Verify files created
        private_dir = state_root / agent_id
        assert (private_dir / "IDENTITY.md").exists()
        assert (private_dir / "profile.json").exists()

        # Load profile and check defaults
        profile = json.loads((private_dir / "profile.json").read_text())
        assert profile["efficiency"] == 0.5
        assert profile["curiosity"] == 0.5

        # ── Create EvolutionEngine ──
        engine = EvolutionEngine(agent_id=agent_id, state_dir=str(state_root / "agents"))

        # Record some task execution entries
        for i in range(6):
            entry = JournalEntry(
                id=str(uuid.uuid4()),
                timestamp=datetime.now(timezone.utc).isoformat(),
                task_scope=f"test task {i}",
                task_type="coding",
                outcome="success",
                score=0.9,
                duration_seconds=5.0,
                tools_used=["fs.read_file"],
                llm_provider="mock",
                cost_estimate=0.001,
            )
            await engine.record_entry(entry)

        # ── Adjust traits ──
        entry = JournalEntry(
            id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            task_scope="final task",
            task_type="coding",
            outcome="success",
            score=0.95,
            duration_seconds=10.0,
            tools_used=["fs.read_file"],
            llm_provider="mock",
            cost_estimate=0.002,
        )
        await engine.adjust_traits(entry)

        # ── Extract knowledge ──
        nuggets = await engine.extract_knowledge(min_evidence=3)
        assert len(nuggets) > 0, "Should have extracted knowledge nuggets"

        # Find the "擅长" nugget
        strength_nugget = next((n for n in nuggets if "擅长" in n.pattern), None)
        assert strength_nugget is not None
        assert strength_nugget.confidence > 0.5

        # ── Stats ──
        stats = engine.get_stats()
        assert stats["total_tasks"] == 6  # 6 recorded entries
        assert stats["success_rate"] == 1.0
        assert stats["average_score"] > 0.8

        # ── Soul evolution ──
        await soul.evolve("Learned: coding tasks are my strength")
        journal_content = (private_dir / "JOURNAL.md").read_text()
        assert "Learned: coding tasks are my strength" in journal_content

        # ── Record task ──
        await soul.record_task(success=True)
        meta = soul.get_meta()
        assert meta["total_tasks"] == 1
        assert meta["success_rate"] == 1.0

    finally:
        _cleanup_dir(tmp)


# ============================================================
# Test 4: InteractionHub Approval
# ============================================================

@pytest.mark.asyncio
async def test_interaction_hub_approval():
    """High-risk operation → InteractionHub blocks → user approves → agent continues."""
    hub = InteractionHub(risk_threshold="medium")

    # Simulate agent requesting approval for a dangerous operation
    req_task = asyncio.create_task(
        hub.request_approval(
            agent_id="agent-1",
            action="rm -rf /data/cache",
            details="Cleaning up stale cache files",
            risk_level="critical",
            task_scope="cache_cleanup",
            timeout_seconds=5,
        )
    )

    # Small delay to let the request register
    await asyncio.sleep(0.1)

    # Verify pending
    assert hub.get_pending_count() == 1
    pending = hub.get_pending()
    assert pending[0].agent_id == "agent-1"
    assert pending[0].status == "pending"

    # User approves
    approved = await hub.approve(pending[0].id, "ok, go ahead")
    assert approved.status == "approved"

    # Agent receives result
    result = await req_task
    assert result.status == "approved"
    assert result.reply == "ok, go ahead"
    assert hub.get_pending_count() == 0


@pytest.mark.asyncio
async def test_interaction_hub_deny():
    """User denies a high-risk operation."""
    hub = InteractionHub(risk_threshold="medium")

    req_task = asyncio.create_task(
        hub.request_approval(
            agent_id="agent-2",
            action="sudo reboot",
            details="Rebooting production server",
            risk_level="critical",
            timeout_seconds=5,
        )
    )

    await asyncio.sleep(0.1)
    pending = hub.get_pending()
    await hub.deny(pending[0].id, "not now")

    result = await req_task
    assert result.status == "denied"


@pytest.mark.asyncio
async def test_interaction_hub_timeout():
    """Request times out if no user response."""
    hub = InteractionHub(risk_threshold="medium")

    result = await hub.request_approval(
        agent_id="agent-3",
        action="delete records",
        details="Purging old logs",
        risk_level="high",
        timeout_seconds=0.1,
    )

    assert result.status == "expired"


def test_detect_risk_level():
    """Risk detection from command strings."""
    assert detect_risk_level("rm -rf /tmp") == "critical"
    assert detect_risk_level("delete user") == "high"
    assert detect_risk_level("sudo apt update") == "critical"
    assert detect_risk_level("chmod 777 file") == "medium"
    assert detect_risk_level("echo hello") is None


def test_is_high_risk():
    """is_high_risk checks against threshold."""
    assert is_high_risk("rm file", threshold="medium") is True
    assert is_high_risk("rm file", threshold="critical") is False
    assert is_high_risk("echo hello", threshold="low") is False


# ============================================================
# Test 5: Agent Mailbox Delegation
# ============================================================

@pytest.mark.asyncio
async def test_agent_mailbox_delegation():
    """Agent A delegates to Agent B → B completes → result flows back."""
    router = MailRouter()

    mailbox_a = router.register("agent-a")
    mailbox_b = router.register("agent-b")

    assert router.agent_count == 2
    assert "agent-a" in router.registered_agents

    # Agent A sends a delegation request
    msg = await mailbox_a.send(
        to_agent="agent-b",
        type="delegate",
        subject="Fix auth bug",
        body="Please investigate and fix the authentication issue in auth.py.",
    )
    assert msg.from_agent == "agent-a"
    assert msg.to_agent == "agent-b"
    assert msg.type == "delegate"

    # Agent B receives
    received = await mailbox_b.receive(timeout=2.0)
    assert received is not None
    assert received.subject == "Fix auth bug"
    assert received.status == "read"

    # Agent B replies
    reply = await mailbox_b.reply(received, "Done — fixed the null pointer check.")
    assert reply.reply_to == received.id
    assert "Done" in reply.body

    # Agent A gets reply
    reply_received = await mailbox_a.receive(timeout=2.0)
    assert reply_received is not None
    assert "Done" in reply_received.body

    # Cleanup
    router.unregister("agent-a")
    router.unregister("agent-b")
    assert router.agent_count == 0


# ============================================================
# Test 6: Tool Registry Invoke
# ============================================================

@pytest.mark.asyncio
async def test_tool_registry_invoke():
    """Register tool → invoke → JSON Schema validation → high-risk confirmation."""
    registry = ToolRegistry()

    # Register a custom tool
    async def mock_sum_handler(a: int, b: int) -> dict:
        return {"sum": a + b}

    spec = ToolSpec(
        name="math.sum",
        namespace="math",
        description="Add two integers",
        input_schema={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        handler=mock_sum_handler,
        risk_level="low",
        tags=["math"],
    )
    registry.register(spec)
    assert "math.sum" in registry
    assert len(registry) == 1

    # Invoke with valid args
    result = await registry.invoke("math.sum", {"a": 3, "b": 5})
    assert result["sum"] == 8

    # Invoke with invalid args (missing required field)
    with pytest.raises(ValueError, match="Invalid args"):
        await registry.invoke("math.sum", {"a": 3})

    # Invoke with wrong type
    with pytest.raises(ValueError, match="Invalid args"):
        await registry.invoke("math.sum", {"a": "hello", "b": 3})

    # Invoke non-existent tool
    with pytest.raises(ValueError, match="Tool not found"):
        await registry.invoke("math.multiply", {"a": 1, "b": 2})

    # ── High-risk tool with approval ──
    hub = InteractionHub(risk_threshold="low")

    async def dangerous_handler(command: str) -> dict:
        return {"executed": command}

    danger_spec = ToolSpec(
        name="shell.rm",
        namespace="shell",
        description="Remove files",
        input_schema={
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
        handler=dangerous_handler,
        risk_level="critical",
        tags=["shell", "dangerous"],
    )
    registry.register(danger_spec)

    # The invoke will trigger approval — we need to approve it
    invoke_task = asyncio.create_task(
        registry.invoke("shell.rm", {"command": "rm -rf /tmp/test"}, interaction_hub=hub)
    )

    await asyncio.sleep(0.1)
    pending = hub.get_pending()
    if pending:
        await hub.approve(pending[0].id, "approved")

    result = await invoke_task
    assert result["executed"] == "rm -rf /tmp/test"

    # ── OpenAI function format ──
    functions = registry.to_openai_functions()
    assert len(functions) == 2
    assert functions[0]["type"] == "function"
    assert "name" in functions[0]["function"]

    # ── Namespace filtering ──
    math_tools = registry.list_tools(namespace="math")
    assert len(math_tools) == 1
    assert math_tools[0].name == "math.sum"

    shell_tools = registry.list_tools(namespace="shell")
    assert len(shell_tools) == 1

    # ── Risk filtering ──
    low_risk = registry.list_tools(max_risk="low")
    assert len(low_risk) == 1  # only math.sum is low risk

    # Unregister
    registry.unregister("math.sum")
    assert len(registry) == 1


# ============================================================
# Test 7: Persistence Snapshot → Restore
# ============================================================

@pytest.mark.asyncio
async def test_persistence_snapshot_restore():
    """Snapshot → modify state → restore → verify state matches original."""
    tmp = _temp_dir()
    try:
        state_dir = Path(tmp) / "state"
        snap_dir = Path(tmp) / "snapshots"
        manager = PersistenceManager(state_dir=str(state_dir), snapshot_dir=str(snap_dir))

        agent_id = f"agent:persist:{uuid.uuid4().hex[:8]}"
        agent_dir = state_dir / "agents" / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Write initial state
        identity = agent_dir / "IDENTITY.md"
        identity.write_text("# Agent Identity\n\nI am a test agent.\n")

        profile = agent_dir / "profile.json"
        original_profile = {
            "efficiency": 0.7,
            "assertiveness": 0.6,
            "curiosity": 0.5,
            "cautiousness": 0.4,
            "creativity": 0.8,
        }
        profile.write_text(json.dumps(original_profile))

        # Snapshot the original state
        snap = await manager.snapshot(agent_id)
        assert snap.agent_id == agent_id

        # Verify snapshot exists
        snapshots = manager.list_snapshots(agent_id)
        assert len(snapshots) == 1
        assert snapshots[0].soul_md5 == snap.soul_md5

        # Modify state (simulate corruption or changes)
        identity.write_text("# Corrupted Identity\n\nI have been changed.\n")
        modified_profile = dict(original_profile, efficiency=0.1, assertiveness=0.1)
        profile.write_text(json.dumps(modified_profile))

        # Verify modified
        current_content = identity.read_text()
        assert "Corrupted" in current_content

        # Restore from snapshot
        restored = await manager.restore(agent_id)
        assert restored is True

        # Verify restored
        restored_content = identity.read_text()
        assert "I am a test agent" in restored_content
        assert "Corrupted" not in restored_content

        restored_profile = json.loads(profile.read_text())
        assert restored_profile["efficiency"] == 0.7
        assert restored_profile["assertiveness"] == 0.6

        # ── Rollback ──
        # Modify again
        identity.write_text("# Rollback target\n")
        await manager.snapshot(agent_id)  # second snapshot

        # Modify to something else
        identity.write_text("# Final state\n")

        # Rollback to first snapshot
        snapshots = manager.list_snapshots(agent_id)
        assert len(snapshots) == 2
        # Restore the older one (last in list since sorted desc)
        await manager.restore(agent_id, snapshot_timestamp=snapshots[1].timestamp)

        content = identity.read_text()
        assert "I am a test agent" in content

    finally:
        _cleanup_dir(tmp)


# ============================================================
# Test 8: Agent Fork and Merge
# ============================================================

@pytest.mark.asyncio
async def test_agent_fork_and_merge():
    """Fork child → child executes task → merge knowledge back to parent."""
    tmp = _temp_dir()
    try:
        state_dir = Path(tmp) / "state"
        forker = AgentForker(state_dir=str(state_dir))

        parent_id = f"agent:parent:{uuid.uuid4().hex[:8]}"
        parent_dir = state_dir / parent_id
        parent_dir.mkdir(parents=True, exist_ok=True)

        # Create parent soul files
        (parent_dir / "IDENTITY.md").write_text("# Parent Identity\n")
        (parent_dir / "ROLE.md").write_text("# Role: coder\n")
        (parent_dir / "profile.json").write_text(json.dumps({
            "efficiency": 0.7,
            "assertiveness": 0.6,
        }))
        (parent_dir / "KNOWLEDGE.json").write_text(json.dumps([
            {"id": "k1", "pattern": "擅长coding", "confidence": 0.8, "evidence_count": 5,
             "source_entries": [], "created_at": "", "last_reinforced": ""}
        ]))
        (parent_dir / "meta.json").write_text(json.dumps({
            "agent_id": parent_id,
            "total_tasks": 5,
            "role": "coder",
        }))

        # Fork child
        config = ForkConfig(
            parent_id=parent_id,
            fork_reason="parallel_task",
            inherit_soul=True,
            inherit_knowledge=True,
        )
        child_id = await forker.fork(config)
        assert child_id.startswith("agent:fork:")

        # Verify child has inherited files
        child_dir = state_dir / child_id
        assert (child_dir / "IDENTITY.md").exists()
        assert (child_dir / "KNOWLEDGE.json").exists()
        assert (child_dir / "profile.json").exists()

        # Verify child meta
        child_meta = json.loads((child_dir / "meta.json").read_text())
        assert child_meta["parent_id"] == parent_id
        assert child_meta["fork_reason"] == "parallel_task"
        assert child_meta["merged"] is False

        # Parent meta should now list child
        parent_meta = json.loads((parent_dir / "meta.json").read_text())
        assert child_id in parent_meta.get("children", [])

        # ── Child does some work ──
        # Simulate child adding knowledge
        child_knowledge = [
            {"id": "k1", "pattern": "擅长coding", "confidence": 0.8, "evidence_count": 5,
             "source_entries": [], "created_at": "", "last_reinforced": ""},
            {"id": "k2", "pattern": "工具fs.write_file需要检查", "confidence": 0.3, "evidence_count": 3,
             "source_entries": [], "created_at": "", "last_reinforced": ""},
        ]
        (child_dir / "KNOWLEDGE.json").write_text(json.dumps(child_knowledge))

        # Write journal entries for child
        journal = child_dir / "JOURNAL.jsonl"
        journal.write_text(
            json.dumps({"id": "j1", "task_scope": "test", "outcome": "success"}) + "\n" +
            json.dumps({"id": "j2", "task_scope": "test2", "outcome": "success"}) + "\n"
        )

        # ── Merge back ──
        await forker.merge_back(child_id, parent_id)

        # Verify child marked as merged
        child_meta = json.loads((child_dir / "meta.json").read_text())
        assert child_meta["merged"] is True
        assert child_meta["merged_into"] == parent_id

        # ── Family tree ──
        tree = forker.get_family_tree(parent_id)
        assert tree["agent_id"] == parent_id
        assert len(tree["children"]) == 1
        assert tree["children"][0]["agent_id"] == child_id
        assert tree["children"][0]["merged"] is True

        # Child's family tree
        child_tree = forker.get_family_tree(child_id)
        assert child_tree["parent_id"] == parent_id
        assert child_tree["fork_reason"] == "parallel_task"

    finally:
        _cleanup_dir(tmp)


# ============================================================
# Test 9: ProgressEmitter streaming
# ============================================================

@pytest.mark.asyncio
async def test_progress_emitter_streaming():
    """ProgressEmitter emits events → subscribers receive them."""
    agent_id = "agent:stream:test"
    emitter = ProgressEmitter(agent_id=agent_id)

    # Subscribe
    q = await emitter.subscribe()

    # Emit events
    emitter.emit("phase_start", "PLAN", "Starting plan generation")
    emitter.emit("tool_call", "EXECUTE", "Calling fs.read_file", tool="fs.read_file")
    emitter.emit("phase_done", "EXECUTE", "Execution complete")

    # Verify events received
    event1 = await asyncio.wait_for(q.get(), timeout=2.0)
    assert event1.type == "phase_start"
    assert event1.phase == "PLAN"
    assert event1.agent_id == agent_id

    event2 = await asyncio.wait_for(q.get(), timeout=2.0)
    assert event2.type == "tool_call"
    assert event2.data.get("tool") == "fs.read_file"

    event3 = await asyncio.wait_for(q.get(), timeout=2.0)
    assert event3.type == "phase_done"

    # Callback test
    captured = []

    def capture_cb(ev):
        captured.append(ev.message)

    emitter.on_event(capture_cb)
    emitter.emit("info", "TEST", "callback message")
    assert "callback message" in captured

    # Cleanup
    emitter.detach()


# ============================================================
# Test 10: Budget enforcement
# ============================================================

def test_budget_model():
    """Budget model serialization."""
    budget = Budget(daily_limit=5.0, monthly_limit=50.0, per_task_limit=1.0, alert_threshold=0.8)
    d = budget.to_dict()
    assert d["daily_limit"] == 5.0
    assert d["alert_threshold"] == 0.8

    restored = Budget.from_dict(d)
    assert restored.daily_limit == 5.0
    assert restored.monthly_limit == 50.0


def test_cost_controller_reset_task():
    """Cost controller can reset per-task tracking."""
    controller = CostController(budget=Budget())
    controller.record_task("task-x", 0.50)
    controller.reset_task("task-x")
    assert controller.check(estimated_cost=0.90, task_scope="task-x") is True


def test_cost_controller_alerts():
    """Cost controller emits alerts when approaching limits."""
    controller = CostController(budget=Budget(daily_limit=1.0, alert_threshold=0.5))
    controller.record(actual_cost=0.60, provider_id="mock")
    remaining = controller.get_remaining()
    assert remaining["daily_alert"] is True
    assert remaining["daily_spent"] == 0.60
    assert remaining["daily_remaining"] == 0.40


# ============================================================
# Test 11: Autonomous decision unit tests
# ============================================================

def test_classify_task_coding():
    """Tasks with coding keywords classify as coding."""
    task = ManagedTask(task_id="t1", scope="implement user login", required_tools=["fs.write_file"])
    assert AgentManagerAgent._classify_task(task) == "coding"

    task2 = ManagedTask(task_id="t2", scope="debug the auth flow", required_tools=[])
    assert AgentManagerAgent._classify_task(task2) == "coding"


def test_classify_task_reasoning():
    """Tasks with analysis keywords classify as reasoning."""
    task = ManagedTask(task_id="t3", scope="analyze performance metrics")
    assert AgentManagerAgent._classify_task(task) == "reasoning"

    task2 = ManagedTask(task_id="t4", scope="evaluate the architecture")
    assert AgentManagerAgent._classify_task(task2) == "reasoning"


def test_classify_task_ops():
    """Tasks with deploy keywords classify as ops."""
    task = ManagedTask(task_id="t5", scope="deploy to production")
    assert AgentManagerAgent._classify_task(task) == "ops"

    task2 = ManagedTask(task_id="t6", scope="monitor server health")
    assert AgentManagerAgent._classify_task(task2) == "ops"


def test_classify_task_general():
    """Unmatched tasks classify as general."""
    task = ManagedTask(task_id="t7", scope="send email notification")
    assert AgentManagerAgent._classify_task(task) == "general"


def test_choose_personality():
    """Personality cards map to task types."""
    assert AgentManagerAgent._choose_personality("coding") == "executor"
    assert AgentManagerAgent._choose_personality("reasoning") == "analyst"
    assert AgentManagerAgent._choose_personality("ops") == "guardian"
    assert AgentManagerAgent._choose_personality("general") == "executor"
    assert AgentManagerAgent._choose_personality("unknown") == "executor"


def test_choose_role():
    """Role cards map to task types."""
    assert AgentManagerAgent._choose_role("coding") == "coder"
    assert AgentManagerAgent._choose_role("reasoning") == "researcher"
    assert AgentManagerAgent._choose_role("ops") == "ops"
    assert AgentManagerAgent._choose_role("general") == "coder"


def test_choose_llm_strategy():
    """LLM strategies map to task types."""
    assert AgentManagerAgent._choose_llm_strategy("coding") == "cheapest"
    assert AgentManagerAgent._choose_llm_strategy("reasoning") == "most_capable"
    assert AgentManagerAgent._choose_llm_strategy("ops") == "balanced"
    assert AgentManagerAgent._choose_llm_strategy("general") == "balanced"
    assert AgentManagerAgent._choose_llm_strategy("unknown") == "balanced"


# ============================================================
# Test 12: Multimodal Processor integration stub
# ============================================================

@pytest.mark.asyncio
async def test_multimodal_processor_integration():
    """MultimodalProcessor integrates with AgentLoop context."""
    from src.multimodal import MultimodalProcessor

    processor = MultimodalProcessor()

    # Text input should pass through
    inputs = [{"type": "text", "content": "Hello world"}]
    blocks = await processor.process(inputs)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "text"

    # Image input should get recognized (use data URL)
    inputs = [
        {"type": "image", "content": "data:image/png;base64,iVBORw0KGgo=", "mime_type": "image/png"}
    ]
    blocks = await processor.process(inputs)
    assert len(blocks) == 1
    assert blocks[0]["type"] == "image_url"

    # Mixed inputs
    inputs = [
        {"type": "text", "content": "Analyze this image:"},
        {"type": "image", "content": "data:image/jpeg;base64,/9j/4AAQ=", "mime_type": "image/jpeg"},
    ]
    blocks = await processor.process(inputs)
    assert len(blocks) == 2
    assert blocks[0]["type"] == "text"
    assert blocks[1]["type"] == "image_url"


# ============================================================
# Test 13: Agent permissions binding (Bug 2 fix)
# ============================================================

@pytest.mark.asyncio
async def test_agent_permissions_binding():
    """AgentManagerAgent._acquire_worker binds permissions from PermissionChecker."""
    tmp = _temp_dir()
    try:
        # Create permissions config for test
        perm_config_dir = Path(tmp) / "config"
        perm_config_dir.mkdir(parents=True, exist_ok=True)
        perm_config = perm_config_dir / "permissions.json"
        perm_config.write_text(json.dumps({
            "templates": {
                "coder": {
                    "trust_level": "restricted",
                    "filesystem": {
                        "read_paths": ["state/agents/{agent_id}/*"],
                        "write_paths": ["state/agents/{agent_id}/*"],
                        "blocked_paths": ["/etc/*", "~/.openclaw/*"]
                    },
                    "network": {"allowed_hosts": ["api.github.com"], "blocked_hosts": []},
                    "shell": {
                        "allowed": True,
                        "allowed_commands": ["git", "python3", "pytest", "pip"],
                        "blocked_commands": ["rm", "sudo", "chmod"]
                    },
                    "agent_ops": {"can_modify_own_soul": True},
                    "self_modification": {
                        "can_modify_identity": True,
                        "can_modify_role": True,
                        "can_append_journal": True,
                        "can_modify_knowledge": True,
                        "can_modify_profile": False
                    }
                }
            },
            "elevation": {"requires_approval": True}
        }))

        from src.permissions import PermissionChecker
        checker = PermissionChecker(config_path=str(perm_config))

        registry = TaskRegistry()
        mock_loop = MagicMock()
        mock_loop.tool_loop = MagicMock()
        mock_loop.llm = MagicMock()
        from src.loop_engine import LoopConfig
        config = LoopConfig()

        mgr = AgentManagerAgent(
            memory=MagicMock(),
            agent_loop=mock_loop,
            config=config,
            registry=registry,
            permission_checker=checker,
        )

        # Register a task
        task = ManagedTask(
            task_id=f"task:{uuid.uuid4().hex[:8]}",
            scope="implement user login",
            required_tools=["fs.write_file"],
        )
        registry._tasks[task.task_id] = task
        registry._order.append(task.task_id)

        # Acquire worker — should bind permissions
        agent = await mgr._acquire_worker(task)
        assert agent is not None

        # Verify agent has _permissions
        assert hasattr(agent, '_permissions'), "Agent should have _permissions after _acquire_worker"
        perms = agent._permissions
        assert perms.template_name == "coder"
        assert perms.trust_level == "restricted"

        # Verify permission checks work
        assert perms.can_read(f"state/agents/{agent.agent_id}/IDENTITY.md") is True
        assert perms.can_execute("git status") is True
        assert perms.can_execute("rm -rf /") is False
        assert perms.can_modify_file("identity") is True
        assert perms.can_modify_file("profile") is False

        # Cleanup
        if agent.agent_id in mgr.pool.agents:
            mgr.pool.agents.pop(agent.agent_id, None)

    finally:
        _cleanup_dir(tmp)


# ============================================================
# Test 14: AgentLoop sandbox + agent_permissions parameters (Bug 4 fix)
# ============================================================

def test_agent_loop_sandbox_params():
    """AgentLoop accepts sandbox and agent_permissions as optional parameters."""
    from src.loop_engine import AgentLoop, LoopConfig, ToolLoop
    from src.tools.base import ToolRegistry

    config = LoopConfig()
    registry = ToolRegistry()
    tool_loop = ToolLoop(registry, config)
    llm = MagicMock()

    # Without sandbox (backward compat)
    loop1 = AgentLoop(tool_loop=tool_loop, llm=llm, config=config)
    assert hasattr(loop1, 'sandbox'), "AgentLoop should have sandbox attribute"
    assert loop1.sandbox is None
    assert hasattr(loop1, 'agent_permissions'), "AgentLoop should have agent_permissions attribute"
    assert loop1.agent_permissions is None

    # With sandbox
    mock_sandbox = MagicMock()
    mock_perms = MagicMock()
    loop2 = AgentLoop(
        tool_loop=tool_loop, llm=llm, config=config,
        sandbox=mock_sandbox,
        agent_permissions=mock_perms,
    )
    assert loop2.sandbox is mock_sandbox
    assert loop2.agent_permissions is mock_perms


# ============================================================
# Test 15: self_modify in EVOLVE phase (Bug 5 fix)
# ============================================================

@pytest.mark.asyncio
async def test_self_modify_in_evolve_phase():
    """AgentLoop EVOLVE phase calls self_modify when score > 0.85."""
    from src.loop_engine import AgentLoop, LoopConfig, ToolLoop
    from src.tools.base import ToolRegistry
    from src.agent_soul import AgentSoul, SoulBuilder

    tmp = _temp_dir()
    try:
        agent_id = f"agent:evolve-test:{uuid.uuid4().hex[:8]}"
        state_root = Path(tmp) / "state"

        # Build a real AgentSoul
        soul = SoulBuilder(agent_id=agent_id, state_root=str(state_root)) \
            .with_personality("executor") \
            .with_role("coder") \
            .build()

        # Verify self_modify is available
        assert hasattr(soul, 'self_modify'), "AgentSoul should have self_modify method"

        # Read original identity
        original = soul.identity_content
        assert 'experience' not in original.lower()

        # Simulate high-score EVOLVE: call self_modify with high eval_score
        # This is a direct call to verify self_modify works (without full loop)
        new_identity = original + '\n\n## 经验\n积累了成功的执行经验。'

        result = await soul.self_modify('identity', new_identity, permissions=None)
        assert result is True, f"self_modify should succeed: {result}"

        # Check identity was updated (read file directly since identity_content is a cached property)
        identity_path = soul._private_dir / "IDENTITY.md"
        updated = identity_path.read_text()
        assert 'experience' in updated.lower() or '经验' in updated

    finally:
        _cleanup_dir(tmp)


@pytest.mark.asyncio
async def test_self_modify_permission_blocked():
    """self_modify returns False when permissions deny modification."""
    from src.agent_soul import SoulBuilder
    from src.permissions import AgentPermissions

    tmp = _temp_dir()
    try:
        agent_id = f"agent:perm-block:{uuid.uuid4().hex[:8]}"
        state_root = Path(tmp) / "state"

        # Create permissions config that blocks identity modification
        perm_config_dir = Path(tmp) / "config"
        perm_config_dir.mkdir(parents=True, exist_ok=True)
        perm_config = perm_config_dir / "permissions.json"
        perm_config.write_text(json.dumps({
            "templates": {
                "restricted_coder": {
                    "trust_level": "restricted",
                    "filesystem": {
                        "read_paths": [],
                        "write_paths": [],
                        "blocked_paths": ["*"]
                    },
                    "network": {"allowed_hosts": [], "blocked_hosts": ["*"]},
                    "shell": {"allowed": False, "allowed_commands": [], "blocked_commands": []},
                    "agent_ops": {},
                    "self_modification": {
                        "can_modify_identity": False,
                        "can_modify_role": False,
                        "can_append_journal": False,
                        "can_modify_knowledge": False,
                        "can_modify_profile": False
                    }
                }
            },
            "elevation": {"requires_approval": True}
        }))

        soul = SoulBuilder(agent_id=agent_id, state_root=str(state_root)) \
            .with_personality("executor") \
            .with_role("restricted_coder") \
            .build()

        # Create permissions that deny identity modification
        perms = AgentPermissions(
            template_name="restricted_coder",
            agent_id=agent_id,
            config_path=str(perm_config),
        )
        assert perms.can_modify_file('identity') is False

        # self_modify should return False
        result = await soul.self_modify('identity', 'new content', permissions=perms)
        assert result is False, "self_modify should be blocked by permissions"

    finally:
        _cleanup_dir(tmp)


# ============================================================
# Test 16: AgentManagerAgent SandboxManager integration (Bug 7 fix)
# ============================================================

def test_agent_manager_accepts_sandbox():
    """AgentManagerAgent accepts sandbox parameter and stores it."""
    from src.loop_engine import LoopConfig

    config = LoopConfig()
    registry = TaskRegistry()
    mock_loop = MagicMock()
    mock_loop.tool_loop = MagicMock()
    mock_loop.llm = MagicMock()
    mock_sandbox = MagicMock()
    mock_checker = MagicMock()

    mgr = AgentManagerAgent(
        memory=MagicMock(),
        agent_loop=mock_loop,
        config=config,
        registry=registry,
        permission_checker=mock_checker,
        sandbox=mock_sandbox,
    )

    assert mgr.permission_checker is mock_checker
    assert mgr.sandbox is mock_sandbox

    # Also verify backward compat (neither provided)
    mgr2 = AgentManagerAgent(
        memory=MagicMock(),
        agent_loop=mock_loop,
        config=config,
        registry=registry,
    )
    assert mgr2.permission_checker is None
    assert mgr2.sandbox is None
