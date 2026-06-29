"""Tests for split system agents: TaskAgent + AgentManagerAgent."""

import asyncio
import pytest
from src.system_agents import TaskAgent, AgentManagerAgent, TaskRegistry, ManagedTask
from src.core import TaskStatus, AgentRole


@pytest.mark.asyncio
async def test_task_agent_role():
    """TaskAgent has MANAGER role."""
    assert TaskAgent.role == AgentRole.MANAGER


def test_agent_manager_role():
    """AgentManagerAgent has MANAGER role."""
    assert AgentManagerAgent.role == AgentRole.MANAGER


@pytest.mark.asyncio
async def test_task_registry_register_and_get():
    """TaskRegistry can register and retrieve tasks."""
    registry = TaskRegistry()
    task = await registry.register(scope="Test", priority=3)
    assert registry.get(task.task_id) is not None
    assert registry.get("nonexistent") is None


@pytest.mark.asyncio
async def test_task_registry_dependency_scheduling():
    """Registry correctly schedules by dependencies."""
    registry = TaskRegistry()
    t1 = await registry.register(scope="T1", priority=5)
    t2 = await registry.register(scope="T2", priority=3, dependencies=[t1.task_id])

    # Only t1 ready initially
    ready = registry.get_ready()
    assert len(ready) == 1
    assert ready[0].task_id == t1.task_id

    # After t1 done, t2 ready
    t1.status = TaskStatus.DONE
    ready = registry.get_ready()
    assert len(ready) == 1
    assert ready[0].task_id == t2.task_id


@pytest.mark.asyncio
async def test_managed_task_lifecycle():
    """ManagedTask transitions through lifecycle states."""
    task = ManagedTask(task_id="task:test", scope="Test")
    assert task.is_ready
    assert not task.is_terminal

    task.status = TaskStatus.RUNNING
    assert not task.is_ready

    task.status = TaskStatus.DONE
    assert task.is_terminal


@pytest.mark.asyncio
async def test_registry_stats():
    """Registry stats are correct."""
    registry = TaskRegistry()
    await registry.register(scope="T1")
    await registry.register(scope="T2")

    stats = registry.stats()
    assert stats["total"] == 2
    assert stats["pending"] == 2


def test_separation_of_concerns():
    """TaskAgent and AgentManagerAgent have non-overlapping methods."""
    task_agent_methods = {m for m in dir(TaskAgent) if not m.startswith('_')}
    agent_mgr_methods = {m for m in dir(AgentManagerAgent) if not m.startswith('_')}

    # TaskAgent should have task-related methods
    assert 'decompose' in task_agent_methods
    assert 'get_ready_tasks' in task_agent_methods
    assert 'replan' in task_agent_methods
    assert 'update_status' in task_agent_methods

    # AgentManagerAgent should have agent-related methods
    assert 'assign' in agent_mgr_methods
    assert 'collect_all' in agent_mgr_methods
    assert 'cancel' in agent_mgr_methods
    assert 'stats' in agent_mgr_methods

    # TaskAgent should NOT have agent management methods
    assert 'assign' not in task_agent_methods
    assert 'collect_all' not in task_agent_methods

    # AgentManagerAgent should NOT have task decomposition methods
    assert 'decompose' not in agent_mgr_methods
    assert 'get_ready_tasks' not in agent_mgr_methods
    assert 'replan' not in agent_mgr_methods
