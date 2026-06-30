"""Tests for P2.5 plan mode (AgentLoop plan_mode)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.loop_engine import (
    AgentLoop, LLMProvider, LLMResponse, LoopConfig, ToolLoop,
)
from src.core import TaskStatus
from src.tool_registry import ToolRegistry


class PlanModeMockLLM(LLMProvider):
    """Mock LLM for plan mode tests."""

    def __init__(self):
        self.chat_calls = []

    async def chat(self, messages, **kwargs):
        self.chat_calls.append(messages)
        content = messages[0].get("content", "") if messages else ""
        if "execution planner" in content:
            return LLMResponse(
                content='[{"description": "do X", "tool": null}]',
                model="mock",
            )
        if "quality evaluator" in content:
            return LLMResponse(content='{"score": 0.8, "reason": "ok"}', model="mock")
        return LLMResponse(content="Done", model="mock")

    async def embed(self, text):
        return [[0.0] * 128]


class MockApprovalResult:
    def __init__(self, status="approved"):
        self.status = status
        self.reply = "ok"


class MockInteractionHub:
    """Mock interaction hub for plan approval."""

    def __init__(self, approval_status="approved"):
        self.approval_status = approval_status
        self.requests = []

    async def request_approval(self, agent_id, action, details="",
                               risk_level="medium", task_scope=""):
        self.requests.append({
            "agent_id": agent_id,
            "action": action,
            "details": details,
            "risk_level": risk_level,
        })
        return MockApprovalResult(self.approval_status)


@pytest.mark.asyncio
async def test_plan_approved():
    """When user approves, execution proceeds."""
    llm = PlanModeMockLLM()
    config = LoopConfig(accept_threshold=0.6)
    hub = MockInteractionHub(approval_status="approved")

    loop = AgentLoop(
        tool_loop=ToolLoop(ToolRegistry(), config),
        llm=llm,
        config=config,
        interaction_hub=hub,
        plan_mode=True,
    )

    result = await loop.run(
        agent_id="test-1",
        task_scope="Do something",
        context={"task_id": "t-1"},
        allowed_tools=[],
    )

    assert result.status == TaskStatus.DONE
    assert len(hub.requests) >= 1  # Plan approval was requested
    assert hub.requests[0]["action"] == "plan_approval"


@pytest.mark.asyncio
async def test_plan_rejected():
    """When user rejects, execution stops."""
    llm = PlanModeMockLLM()
    config = LoopConfig(accept_threshold=0.6)
    hub = MockInteractionHub(approval_status="denied")

    loop = AgentLoop(
        tool_loop=ToolLoop(ToolRegistry(), config),
        llm=llm,
        config=config,
        interaction_hub=hub,
        plan_mode=True,
    )

    result = await loop.run(
        agent_id="test-2",
        task_scope="Do something dangerous",
        context={"task_id": "t-2"},
        allowed_tools=[],
    )

    assert result.status == TaskStatus.FAILED
    assert "rejected" in result.summary.lower() or "denied" in result.summary.lower()
    assert len(hub.requests) >= 1


@pytest.mark.asyncio
async def test_plan_mode_disabled():
    """Default: plan_mode=False, no approval requested."""
    llm = PlanModeMockLLM()
    config = LoopConfig(accept_threshold=0.6)
    hub = MockInteractionHub(approval_status="denied")  # Would deny if asked

    loop = AgentLoop(
        tool_loop=ToolLoop(ToolRegistry(), config),
        llm=llm,
        config=config,
        interaction_hub=hub,
        plan_mode=False,  # Default
    )

    result = await loop.run(
        agent_id="test-3",
        task_scope="Do stuff",
        context={"task_id": "t-3"},
        allowed_tools=[],
    )

    # Should succeed despite hub being set to deny
    assert result.status == TaskStatus.DONE
    # No plan approval was requested
    plan_approval_requests = [r for r in hub.requests if r["action"] == "plan_approval"]
    assert len(plan_approval_requests) == 0


@pytest.mark.asyncio
async def test_plan_mode_no_hub():
    """plan_mode=True with no interaction_hub executes normally (graceful degradation)."""
    llm = PlanModeMockLLM()
    config = LoopConfig(accept_threshold=0.6)

    loop = AgentLoop(
        tool_loop=ToolLoop(ToolRegistry(), config),
        llm=llm,
        config=config,
        plan_mode=True,
        # No interaction_hub
    )

    result = await loop.run(
        agent_id="test-4",
        task_scope="Do stuff",
        context={"task_id": "t-4"},
        allowed_tools=[],
    )

    assert result.status == TaskStatus.DONE
