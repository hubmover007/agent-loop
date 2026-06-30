"""Tests for P2.5 Stop-and-Fix (AgentLoop._verify_step / _fix_step)."""

import asyncio
import pytest

from src.loop_engine import AgentLoop, LLMProvider, LLMResponse, LoopConfig, ToolLoop
from src.tool_registry import ToolRegistry


class StopFixMockLLM(LLMProvider):
    """Mock LLM for stop-and-fix tests."""

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
        # Fix step (from _fix_step) - return success
        return LLMResponse(content="Fixed the issue", model="mock")

    async def embed(self, text):
        return [[0.0] * 128]


@pytest.fixture
def basic_loop():
    llm = StopFixMockLLM()
    config = LoopConfig(accept_threshold=0.6)
    return AgentLoop(
        tool_loop=ToolLoop(ToolRegistry(), config),
        llm=llm,
        config=config,
    )


class TestVerifyStep:
    """Tests for AgentLoop._verify_step."""

    @pytest.mark.asyncio
    async def test_verify_pass(self, basic_loop):
        """Verification passes when command succeeds."""
        result = await basic_loop._verify_step("test", "true")
        assert result.passed is True
        assert result.error is None

    @pytest.mark.asyncio
    async def test_verify_fail(self, basic_loop):
        """Verification fails when command exits non-zero."""
        result = await basic_loop._verify_step("test", "exit 1")
        assert result.passed is False
        assert result.error is not None  # stderr captured

    @pytest.mark.asyncio
    async def test_verify_invalid_command(self, basic_loop):
        """Invalid/nonexistent command returns failure."""
        result = await basic_loop._verify_step("test", "nonexistent_command_xyz_123")
        assert result.passed is False
        assert result.error is not None


class TestFixStep:
    """Tests for AgentLoop._fix_step."""

    @pytest.mark.asyncio
    async def test_fix_step_success(self, basic_loop):
        """Fix step succeeds when LLM responds."""
        result = await basic_loop._fix_step("deploy", "Connection refused")
        assert result.success is True
        assert result.response == "Fixed the issue"

    @pytest.mark.asyncio
    async def test_fix_step_failure(self, basic_loop):
        """Fix step fails when LLM raises."""
        # Make LLM fail on fix call
        basic_loop.llm.chat = lambda *args, **kwargs: asyncio.sleep(0, result=None) or None
        result = await basic_loop._fix_step("deploy", "Fatal error")
        assert result.success is False
        assert result.error is not None


class TestStopAndFixIntegration:
    """Integration tests for stop-and-fix in run()."""

    class IntegrationMockLLM(LLMProvider):
        """Mock LLM with controllable responses."""

        def __init__(self):
            self.chat_calls = []

        async def chat(self, messages, **kwargs):
            self.chat_calls.append(messages)
            content = messages[0].get("content", "") if messages else ""
            if "execution planner" in content:
                return LLMResponse(
                    content='[{"description": "run tests", "tool": "shell.test"}, {"description": "lint", "tool": "shell.lint"}]',
                    model="mock",
                )
            if "quality evaluator" in content:
                return LLMResponse(content='{"score": 0.8, "reason": "ok"}', model="mock")
            return LLMResponse(content="done", model="mock")

        async def embed(self, text):
            return [[0.0] * 128]

    @pytest.mark.asyncio
    async def test_stop_and_fix_with_verify_config(self):
        """When a step has verification config and the step matches, verify is called."""
        llm = self.IntegrationMockLLM()

        from src.tools.base import ToolRegistry as TR
        registry = TR()

        # Register test tools
        call_log = []

        async def test_handler(**kwargs):
            call_log.append(("test", kwargs))
            return {"status": "ok"}

        async def lint_handler(**kwargs):
            call_log.append(("lint", kwargs))
            return {"status": "ok"}

        registry.register(type('Spec', (), {
            'name': 'shell.test',
            'namespace': 'shell',
            'description': 'Run tests',
            'enabled': True,
            'risk_level': 'low',
            'handler': test_handler,
            'input_schema': {},
        })())
        registry.register(type('Spec', (), {
            'name': 'shell.lint',
            'namespace': 'shell',
            'description': 'Run linter',
            'enabled': True,
            'risk_level': 'low',
            'handler': lint_handler,
            'input_schema': {},
        })())

        config = LoopConfig(accept_threshold=0.6)
        loop = AgentLoop(
            tool_loop=ToolLoop(registry, config),
            llm=llm,
            config=config,
        )

        result = await loop.run(
            agent_id="test-sf-1",
            task_scope="Run tests and lint",
            context={"task_id": "t-sf-1", "_verify_config": {"run tests": "true", "lint": "true"}},
            allowed_tools=["shell.test", "shell.lint"],
        )

        assert result.status.name in ("DONE",)
