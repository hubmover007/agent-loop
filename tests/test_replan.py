"""Tests for P2 replan (AgentLoop self-eval replanning)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.loop_engine import (
    AgentLoop, LLMProvider, LLMResponse, LoopConfig, ToolLoop, DEFAULT_MAX_ITERATIONS,
)
from src.core import TaskStatus, StepLog, ToolResult, ToolResultStatus
from src.tool_registry import ToolRegistry


class ReplanMockLLM(LLMProvider):
    """Mock LLM that returns controllable eval scores."""

    def __init__(self, scores=None, plan_content=None, summary="Done"):
        self.scores = scores or [0.5]  # Default: pass
        self.score_idx = 0
        self.plan_content = plan_content
        self.summary = summary
        self.chat_calls = []

    async def chat(self, messages, **kwargs):
        self.chat_calls.append(messages)
        content = messages[0].get("content", "") if messages else ""

        # Self-eval request?
        if "quality evaluator" in content:
            if self.score_idx < len(self.scores):
                score = self.scores[self.score_idx]
            else:
                score = self.scores[-1]
            return LLMResponse(
                content=f'{{"score": {score}, "reason": "eval {self.score_idx}"}}',
                model="mock",
            )

        # Plan request?
        if "execution planner" in content:
            plan = self.plan_content or [{"description": "do thing", "tool": None}]
            return LLMResponse(content=str(plan), model="mock")

        # Summary request?
        return LLMResponse(content=self.summary, model="mock")

    async def embed(self, text):
        return [[0.0] * 128]


@pytest.fixture
def basic_loop():
    registry = ToolRegistry()
    config = LoopConfig(accept_threshold=0.6)
    return AgentLoop(
        tool_loop=ToolLoop(registry, config),
        llm=ReplanMockLLM(),
        config=config,
    )


@pytest.mark.asyncio
async def test_no_replan_on_success():
    """High score does not trigger replan."""
    llm = ReplanMockLLM(scores=[0.9])
    config = LoopConfig(accept_threshold=0.6)
    loop = AgentLoop(
        tool_loop=ToolLoop(ToolRegistry(), config),
        llm=llm,
        config=config,
    )

    result = await loop.run(
        agent_id="test-1",
        task_scope="Simple task",
        context={"task_id": "t-1"},
        allowed_tools=[],
    )

    # High score → no replan, should succeed
    assert result.status == TaskStatus.DONE
    # Only one plan call, one eval call, one summary call
    plan_calls = [m for m in llm.chat_calls if "execution planner" in m[0].get("content", "")]
    assert len(plan_calls) == 1


@pytest.mark.asyncio
async def test_replan_on_low_score():
    """Low SELF_EVAL score triggers replan."""
    llm = ReplanMockLLM(scores=[0.2, 0.2, 0.2, 0.8])
    config = LoopConfig(accept_threshold=0.6)
    loop = AgentLoop(
        tool_loop=ToolLoop(ToolRegistry(), config),
        llm=llm,
        config=config,
        max_replans=3,
    )

    result = await loop.run(
        agent_id="test-2",
        task_scope="Hard task",
        context={"task_id": "t-2"},
        allowed_tools=[],
    )

    # First 3 attempts scored 0.2 → replan, 4th scored 0.8 → accept
    # Plan is called multiple times
    plan_calls = [m for m in llm.chat_calls if "execution planner" in m[0].get("content", "")]
    assert len(plan_calls) >= 2  # At least initial + 1 replan


@pytest.mark.asyncio
async def test_max_replans_exceeded():
    """When max replans are exceeded, the agent fails."""
    llm = ReplanMockLLM(scores=[0.1, 0.1, 0.1, 0.1])  # All low
    config = LoopConfig(accept_threshold=0.6)
    loop = AgentLoop(
        tool_loop=ToolLoop(ToolRegistry(), config),
        llm=llm,
        config=config,
        max_replans=2,  # Only 2 replans allowed
    )

    result = await loop.run(
        agent_id="test-3",
        task_scope="Impossible task",
        context={"task_id": "t-3"},
        allowed_tools=[],
    )

    # Should fail after exhausting replans
    assert result.status == TaskStatus.FAILED
    # Should have called plan at most 3 times (initial + 2 replans)
    plan_calls = [m for m in llm.chat_calls if "execution planner" in m[0].get("content", "")]
    assert len(plan_calls) <= 3


@pytest.mark.asyncio
async def test_replan_context_in_prompt():
    """Replan context is included in the plan prompt."""
    llm = ReplanMockLLM(scores=[0.2, 0.8])
    config = LoopConfig(accept_threshold=0.6)
    loop = AgentLoop(
        tool_loop=ToolLoop(ToolRegistry(), config),
        llm=llm,
        config=config,
        max_replans=1,
    )
    loop._replan_context = {
        "previous_plan": [{"description": "do A"}],
        "failure_reason": "plan was wrong",
        "attempt": 1,
    }

    # Call _plan directly with replan context set
    plan = await loop._plan("Test task", {})
    assert isinstance(plan, list)

    # The prompt should include the replan feedback
    plan_prompt = [m for m in llm.chat_calls if "execution planner" in m[0].get("content", "")]
    assert len(plan_prompt) > 0
    prompt_text = plan_prompt[-1][0]["content"]
    assert "Previous attempt" in prompt_text
    assert "failure_reason" not in prompt_text  # Value was substituted
    assert "plan was wrong" in prompt_text
