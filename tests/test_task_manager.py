"""Tests for TaskRegistry and TaskManagerAgent."""

import asyncio
import pytest
from src.task_manager import TaskRegistry, ManagedTask
from src.core import TaskStatus


@pytest.mark.asyncio
async def test_task_registry_register():
    """Test task registration."""
    registry = TaskRegistry()
    task = await registry.register(
        scope="Test task",
        priority=3,
    )
    assert task.task_id.startswith("task:")
    assert task.scope == "Test task"
    assert task.priority == 3
    assert task.status == TaskStatus.PENDING


@pytest.mark.asyncio
async def test_task_registry_get_ready():
    """Test dependency-aware scheduling."""
    registry = TaskRegistry()

    # Create tasks with dependencies
    t1 = await registry.register(scope="Task 1", priority=5)
    t2 = await registry.register(scope="Task 2", priority=3, dependencies=[t1.task_id])
    t3 = await registry.register(scope="Task 3", priority=4, dependencies=[t1.task_id])

    # Initially only t1 is ready (no deps)
    ready = registry.get_ready()
    assert len(ready) == 1
    assert ready[0].task_id == t1.task_id

    # Mark t1 as done
    t1.status = TaskStatus.DONE

    # Now t2 and t3 are ready, sorted by priority (t3 > t2)
    ready = registry.get_ready()
    assert len(ready) == 2
    assert ready[0].task_id == t3.task_id  # priority 4 > 3


@pytest.mark.asyncio
async def test_task_registry_stats():
    """Test registry statistics."""
    registry = TaskRegistry()
    await registry.register(scope="T1")
    await registry.register(scope="T2")

    stats = registry.stats()
    assert stats["total"] == 2
    assert stats["pending"] == 2
    assert stats["done"] == 0


@pytest.mark.asyncio
async def test_managed_task_lifecycle():
    """Test task lifecycle transitions."""
    task = ManagedTask(
        task_id="task:test",
        scope="Test",
    )

    assert task.is_ready
    assert not task.is_terminal

    task.status = TaskStatus.RUNNING
    assert not task.is_ready

    task.status = TaskStatus.DONE
    assert task.is_terminal
