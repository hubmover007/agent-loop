"""Tests for Max Iterations & Timeout (P0 Feature)."""

import asyncio
import json

import pytest

from src.loop_engine import (
    AgentLoop, LoopConfig, ToolLoop, LLMResponse, LLMProvider,
    DEFAULT_MAX_ITERATIONS, DEFAULT_TIMEOUT_SECONDS,
)
from src.core import TaskStatus
from src.tool_registry import ToolRegistry


class SlowMockLLM(LLMProvider):
    """Mock LLM that can be configured for latency testing."""

    def __init__(self, response_content="ok", delay=0.0):
        self.response_content = response_content
        self.delay = delay
        self.call_count = 0

    async def chat(self, messages, **kwargs):
        self.call_count += 1
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        return LLMResponse(
            content=self.response_content,
            model="mock-slow",
        )

    async def embed(self, text):
        return [[0.0]]


class TestMaxIterations:
    """Test max_iterations limit."""

    @pytest.mark.asyncio
    async def test_completes_within_limit(self):
        """Normal completion within iteration limit should succeed."""
        llm = SlowMockLLM(response_content="done")
        config = LoopConfig()
        tool_loop = ToolLoop(ToolRegistry(), config)

        agent = AgentLoop(
            tool_loop=tool_loop,
            llm=llm,
            config=config,
            max_iterations=5,
        )

        result = await agent.run(
            agent_id="test-1",
            task_scope="Simple task",
            context={"task_id": "t-1"},
            allowed_tools=[],
        )

        # Should either complete or self-eval fail, but NOT hit max_iterations
        assert "Max iterations" not in result.summary

    @pytest.mark.asyncio
    async def test_max_iterations_reached(self):
        """When max_iterations is set to 0, run() returns immediately with error."""
        llm = SlowMockLLM(response_content="done")
        config = LoopConfig()
        tool_loop = ToolLoop(ToolRegistry(), config)

        agent = AgentLoop(
            tool_loop=tool_loop,
            llm=llm,
            config=config,
            max_iterations=0,  # Zero means first iteration exceeds
        )

        result = await agent.run(
            agent_id="test-2",
            task_scope="Should fail",
            context={"task_id": "t-2"},
            allowed_tools=[],
        )

        assert result.status == TaskStatus.FAILED
        assert "Max iterations" in result.summary

    @pytest.mark.asyncio
    async def test_custom_iterations(self):
        """Custom iteration count is respected."""
        llm = SlowMockLLM(response_content="done")
        config = LoopConfig()
        tool_loop = ToolLoop(ToolRegistry(), config)

        agent = AgentLoop(
            tool_loop=tool_loop,
            llm=llm,
            config=config,
            max_iterations=100,
        )

        assert agent.max_iterations == 100

        result = await agent.run(
            agent_id="test-3",
            task_scope="Custom limit",
            context={"task_id": "t-3"},
            allowed_tools=[],
        )

        # 100 iterations is plenty, should complete normally
        assert "Max iterations" not in result.summary

    def test_default_iterations(self):
        """Default max_iterations should be DEFAULT_MAX_ITERATIONS."""
        llm = SlowMockLLM()
        config = LoopConfig()
        tool_loop = ToolLoop(ToolRegistry(), config)

        agent = AgentLoop(
            tool_loop=tool_loop,
            llm=llm,
            config=config,
        )

        assert agent.max_iterations == DEFAULT_MAX_ITERATIONS


class TestTimeout:
    """Test timeout_seconds limit."""

    @pytest.mark.asyncio
    async def test_timeout_terminates(self):
        """When timeout is very small, run() returns timeout error."""
        # LLM with significant delay to trigger timeout
        llm = SlowMockLLM(response_content="slow", delay=0.5)
        config = LoopConfig()
        tool_loop = ToolLoop(ToolRegistry(), config)

        agent = AgentLoop(
            tool_loop=tool_loop,
            llm=llm,
            config=config,
            timeout_seconds=0.001,  # Tiny timeout
        )

        result = await agent.run(
            agent_id="test-t1",
            task_scope="Should timeout",
            context={"task_id": "t-t1"},
            allowed_tools=[],
        )

        assert result.status == TaskStatus.FAILED
        assert "Timeout" in result.summary

    def test_default_timeout(self):
        """Default timeout should be DEFAULT_TIMEOUT_SECONDS."""
        llm = SlowMockLLM()
        config = LoopConfig()
        tool_loop = ToolLoop(ToolRegistry(), config)

        agent = AgentLoop(
            tool_loop=tool_loop,
            llm=llm,
            config=config,
        )

        assert agent.timeout_seconds == DEFAULT_TIMEOUT_SECONDS

    @pytest.mark.asyncio
    async def test_completes_within_timeout(self):
        """Task that completes quickly should not trigger timeout."""
        llm = SlowMockLLM(response_content="fast", delay=0.0)
        config = LoopConfig()
        tool_loop = ToolLoop(ToolRegistry(), config)

        agent = AgentLoop(
            tool_loop=tool_loop,
            llm=llm,
            config=config,
            timeout_seconds=60,  # Plenty of time
        )

        result = await agent.run(
            agent_id="test-t2",
            task_scope="Fast task",
            context={"task_id": "t-t2"},
            allowed_tools=[],
        )

        assert "Timeout" not in result.summary

    def test_defaults_are_positive(self):
        """Both defaults must be positive integers."""
        assert DEFAULT_MAX_ITERATIONS > 0
        assert DEFAULT_TIMEOUT_SECONDS > 0
