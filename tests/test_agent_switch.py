"""Tests for AgentManagerAgent — switch_agent, list_agents, get_active_agent."""

from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path

from src.core import AgentStatus, AgentRole
from src.agent import Agent, AgentPool


# ── helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def simple_pool():
    """Create an AgentPool with a few agents."""
    pool = AgentPool(max_concurrent=10)

    agent_a = Agent(agent_id="agent:aaaa1111")
    agent_a.status = AgentStatus.IDLE
    agent_a.role = AgentRole.WORKER
    pool.agents[agent_a.agent_id] = agent_a

    agent_b = Agent(agent_id="agent:bbbb2222")
    agent_b.status = AgentStatus.RUNNING
    agent_b.role = AgentRole.EXPERT
    agent_b.expertise = ["code"]
    pool.agents[agent_b.agent_id] = agent_b

    agent_c = Agent(agent_id="agent:cccc3333")
    agent_c.status = AgentStatus.IDLE
    agent_c.role = AgentRole.WORKER
    pool.agents[agent_c.agent_id] = agent_c

    return pool


class FakeLoopConfig:
    max_agent_concurrent = 10
    accept_threshold = 0.7
    evaluation_weights = {"completeness": 0.3, "correctness": 0.3, "relevance": 0.25, "efficiency": 0.15}


class FakeLoop:
    def __init__(self):
        self.tool_loop = None
        self.llm = None
        self.config = FakeLoopConfig()


def make_manager(pool):
    """Create an AgentManagerAgent with a pre-populated pool."""
    import asyncio
    from src.system_agents import AgentManagerAgent, TaskRegistry

    registry = TaskRegistry()
    manager = AgentManagerAgent.__new__(AgentManagerAgent)
    # Manually set required attributes (bypassing __init__ which expects real objects)
    manager.pool = pool
    manager.registry = registry
    manager.memory = None
    manager.agent_loop = FakeLoop()
    manager.config = FakeLoopConfig()
    manager.llm_pool = None
    manager.state_store = None
    manager.interaction_hub = None
    manager.mail_router = None
    manager.persistence = None
    manager._inflight = {}
    manager._event_bus = None
    manager.external_bridge = None
    manager.forker = None
    manager._lock = asyncio.Lock()
    return manager


# ── Tests ────────────────────────────────────────────────────────────────────


def test_list_agents(simple_pool):
    """list_agents returns all agents in the pool."""
    manager = make_manager(simple_pool)
    agents = manager.list_agents()
    assert len(agents) == 3
    ids = {a["agent_id"] for a in agents}
    assert "agent:aaaa1111" in ids
    assert "agent:bbbb2222" in ids
    assert "agent:cccc3333" in ids


def test_list_agents_includes_status(simple_pool):
    """list_agents includes status and role for each agent."""
    manager = make_manager(simple_pool)
    agents = manager.list_agents()

    for a in agents:
        assert "agent_id" in a
        assert "status" in a
        assert "role" in a
        assert "expertise" in a
        assert "task_count" in a

    # Find running agent
    running = [a for a in agents if a["agent_id"] == "agent:bbbb2222"]
    assert len(running) == 1
    # Status can be enum value or string
    assert running[0]["status"] in ("running", AgentStatus.RUNNING)


def test_get_active_agent_prefers_running(simple_pool):
    """get_active_agent returns the RUNNING agent first."""
    manager = make_manager(simple_pool)
    active = manager.get_active_agent()
    assert active is not None
    assert active.agent_id == "agent:bbbb2222"  # RUNNING agent


def test_get_active_agent_fallback_to_idle():
    """get_active_agent returns first IDLE agent if none running."""
    pool = AgentPool(max_concurrent=10)
    agent = Agent(agent_id="agent:idle_only")
    agent.status = AgentStatus.IDLE
    pool.agents[agent.agent_id] = agent

    manager = make_manager(pool)
    active = manager.get_active_agent()
    assert active is not None
    assert active.agent_id == "agent:idle_only"


def test_get_active_agent_empty_pool():
    """get_active_agent returns None for empty pool."""
    pool = AgentPool(max_concurrent=10)
    manager = make_manager(pool)
    active = manager.get_active_agent()
    assert active is None


@pytest.mark.asyncio
async def test_switch_agent(tmp_path, simple_pool):
    """switch_agent changes active agent and writes session.json."""
    import os
    from src.system_agents import AgentManagerAgent

    # Run in tmp_path so session.json is created there
    orig_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        manager = make_manager(simple_pool)
        result = await manager.switch_agent("agent:cccc3333")
        assert result is True

        # Check session.json was created
        session_path = Path("state/session.json")
        assert session_path.exists()

        session_data = json.loads(session_path.read_text())
        assert session_data["active_agent_id"] == "agent:cccc3333"
        assert "previous_agent_id" in session_data
        assert "switched_at" in session_data
    finally:
        os.chdir(orig_cwd)


@pytest.mark.asyncio
async def test_switch_agent_not_found(simple_pool):
    """switch_agent returns False for unknown agent."""
    manager = make_manager(simple_pool)
    result = await manager.switch_agent("nonexistent")
    assert result is False


def test_list_agents_empty_pool():
    """list_agents returns empty list for empty pool."""
    pool = AgentPool(max_concurrent=10)
    manager = make_manager(pool)
    agents = manager.list_agents()
    assert agents == []


def test_get_active_agent_with_running_first(simple_pool):
    """A running agent takes priority over idle agents."""
    manager = make_manager(simple_pool)
    active = manager.get_active_agent()
    assert active is not None
    assert active.agent_id == "agent:bbbb2222"
    assert active.status == AgentStatus.RUNNING
