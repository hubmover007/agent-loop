"""Tests for core type definitions."""

import pytest
from src.core import (
    TaskStatus, AgentStatus, LoopPhase, MemoryLayer,
    ToolResult, ToolResultStatus, TaskResult, EvaluationResult,
    StepLog, DiscardRecord, ExpertProfile, AgentRole,
)


def test_task_status_values():
    assert TaskStatus.PENDING.value == "pending"
    assert TaskStatus.RUNNING.value == "running"
    assert TaskStatus.DONE.value == "done"
    assert TaskStatus.FAILED.value == "failed"


def test_agent_status_values():
    assert AgentStatus.IDLE.value == "idle"
    assert AgentStatus.RUNNING.value == "running"
    assert AgentStatus.DESTROYED.value == "destroyed"


def test_loop_phase_values():
    assert LoopPhase.INPUT.value == "input"
    assert LoopPhase.REASON.value == "reason"
    assert LoopPhase.DISPATCH.value == "dispatch"
    assert LoopPhase.OUTPUT.value == "output"


def test_memory_layer_values():
    assert MemoryLayer.FACT.value == "fact"
    assert MemoryLayer.FACET.value == "facet"
    assert MemoryLayer.EPISODE.value == "episode"
    assert MemoryLayer.PROJECT.value == "project"


def test_agent_role_values():
    assert AgentRole.MANAGER.value == "manager"
    assert AgentRole.WORKER.value == "worker"
    assert AgentRole.EXPERT.value == "expert"
    assert AgentRole.OBSERVER.value == "observer"


def test_tool_result_creation():
    result = ToolResult(
        status=ToolResultStatus.SUCCESS,
        data={"output": "test"},
    )
    assert result.status == ToolResultStatus.SUCCESS
    assert result.data["output"] == "test"


def test_task_result_creation():
    result = TaskResult(
        task_id="task:test",
        agent_id="agent:test",
        status=TaskStatus.DONE,
        summary="Test completed",
        artifacts={},
        steps=[],
    )
    assert result.task_id == "task:test"
    assert result.status == TaskStatus.DONE
