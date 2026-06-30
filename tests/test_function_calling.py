"""Tests for Function Calling (P0 Feature)."""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.loop_engine import (
    ChatResponse, ChatStreamChunk, ToolCall, ToolFunction,
    LLMResponse, LLMProvider, LoopConfig, ToolLoop, AgentLoop,
)
from src.tool_registry import ToolSpec, ToolRegistry


# ============================================================
# Helper: Mock LLM for testing
# ============================================================

class MockLLM(LLMProvider):
    """Mock LLM provider for testing tool calling."""

    def __init__(self, response_content="", tool_calls=None,
                 finish_reason="stop"):
        self.response_content = response_content
        self.response_tool_calls = tool_calls
        self.response_finish_reason = finish_reason
        self.chat_calls = []
        self.stream_chunks = []

    async def chat(self, messages, **kwargs):
        self.chat_calls.append({"messages": messages, "kwargs": kwargs})
        return LLMResponse(
            content=self.response_content,
            model="mock",
            finish_reason=self.response_finish_reason,
            tool_calls=self.response_tool_calls,
        )

    async def embed(self, text):
        return [[0.0]]

    async def chat_stream(self, messages, tools=None):
        for chunk in self.stream_chunks:
            yield chunk


# ============================================================
# Test 1: chat with tools
# ============================================================

class TestChatWithTools:
    """Test LLM chat requests with tools parameter."""

    def test_chat_with_tools(self):
        """chat() should pass tools to the API and parse tool_calls."""
        tool_def = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "Search the web",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ]
        tool_call = [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": '{"query": "hello"}',
                },
            }
        ]
        llm = MockLLM(
            response_content="",
            tool_calls=tool_call,
            finish_reason="tool_calls",
        )

        import asyncio
        resp = asyncio.run(llm.chat(
            [{"role": "user", "content": "Search for hello"}],
            tools=tool_def,
        ))

        assert resp.finish_reason == "tool_calls"
        assert resp.tool_calls == tool_call
        assert resp.content == ""

    def test_tool_call_parsed(self):
        """Tool calls are correctly parsed into structured format."""
        raw_tool_calls = [
            {
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city": "Beijing"}',
                },
            }
        ]
        llm = MockLLM(tool_calls=raw_tool_calls)

        import asyncio
        resp = asyncio.run(llm.chat([{"role": "user", "content": "Weather?"}]))

        assert resp.tool_calls is not None
        assert len(resp.tool_calls) == 1
        tc = resp.tool_calls[0]
        assert tc["id"] == "call_abc"
        assert tc["function"]["name"] == "get_weather"
        args = json.loads(tc["function"]["arguments"])
        assert args["city"] == "Beijing"

    def test_no_tools_graceful(self):
        """Not passing tools should work normally (graceful)."""
        llm = MockLLM(response_content="Hello, how can I help?")

        import asyncio
        resp = asyncio.run(llm.chat(
            [{"role": "user", "content": "Hi"}],
        ))

        assert resp.content == "Hello, how can I help?"
        assert resp.tool_calls is None
        assert resp.finish_reason == "stop"


# ============================================================
# Test 2: ToolRegistry OpenAI format
# ============================================================

class TestToolRegistryOpenAITools:
    """Test ToolRegistry to_openai_tools output."""

    def test_tool_registry_to_openai_tools(self):
        """to_openai_tools() should return OpenAI-compatible format."""
        reg = ToolRegistry()
        spec = ToolSpec(
            name="test.echo",
            namespace="test",
            description="Echo back input",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            handler=lambda message: message,
        )
        reg.register(spec)

        tools = reg.to_openai_tools()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "test.echo"
        assert "parameters" in tools[0]["function"]

    def test_to_openai_tools_empty(self):
        """Empty registry returns empty list."""
        reg = ToolRegistry()
        tools = reg.to_openai_tools()
        assert tools == []

    def test_to_openai_tools_namespace_filter(self):
        """Namespace filter works on to_openai_tools."""
        reg = ToolRegistry()
        reg.register(ToolSpec(
            "fs.read", "fs", "Read",
            {"type": "object", "properties": {"p": {"type": "string"}}, "required": ["p"]},
            lambda p: p,
        ))
        reg.register(ToolSpec(
            "web.fetch", "web", "Fetch",
            {"type": "object", "properties": {"u": {"type": "string"}}, "required": ["u"]},
            lambda u: u,
        ))

        tools = reg.to_openai_tools(namespace="web")
        assert len(tools) == 1
        assert tools[0]["function"]["name"] == "web.fetch"


# ============================================================
# Test 3: Tool execution in AgentLoop
# ============================================================

class TestToolExecutionInLoop:
    """Test tool calling integration in AgentLoop."""

    @pytest.mark.asyncio
    async def test_tool_execution_in_loop(self):
        """AgentLoop should invoke tools via tool_registry."""
        from src.tools.base import ToolInterface, ToolRegistry as BaseToolRegistry
        from src.core import ToolResult as TR, ToolResultStatus

        reg = ToolRegistry()  # MCP registry for to_openai_tools

        # Register in MCP registry
        called = []
        def handler(message: str) -> str:
            called.append(message)
            return f"Echo: {message}"

        spec = ToolSpec(
            name="test.echo",
            namespace="test",
            description="Echo",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            handler=handler,
            risk_level="low",
        )
        reg.register(spec)

        # ToolLoop uses tools/base.py ToolRegistry (ToolInterface)
        class EchoTool(ToolInterface):
            name = "test.echo"
            description = "Echo"

            async def execute(self, **kwargs):
                called.append(kwargs.get("message", ""))
                return TR(
                    status=ToolResultStatus.SUCCESS,
                    data=f"Echo: {kwargs.get('message', '')}",
                )

        tl_reg = BaseToolRegistry()
        tl_reg.register(EchoTool())

        # LLM that returns a tool-call plan
        plan_json = json.dumps([
            {"description": "Echo message", "tool": "test.echo",
             "params": {"message": "hello"}, "output_key": "echo_result"},
        ])
        llm = MockLLM(response_content=plan_json)

        config = LoopConfig(max_agent_steps=5)
        tool_loop = ToolLoop(tl_reg, config)

        agent = AgentLoop(
            tool_loop=tool_loop,
            llm=llm,
            config=config,
            tool_registry=reg,
        )

        result = await agent.run(
            agent_id="test-agent",
            task_scope="Echo a message",
            context={"task_id": "task-1"},
            allowed_tools=["test.echo"],
        )

        # Verify tool was called (regardless of final status)
        assert any(s.tool_name == "test.echo" and s.error is None
                   for s in result.steps), (
            f"Expected test.echo to be called successfully, got steps: "
            f"{[s.tool_name for s in result.steps]}"
        )
        assert "hello" in called

    @pytest.mark.asyncio
    async def test_tool_call_error_handling(self):
        """Tool execution errors should be captured in step logs."""
        from src.tools.base import ToolInterface, ToolRegistry as BaseToolRegistry
        from src.core import ToolResult as TR, ToolResultStatus

        reg = ToolRegistry()  # MCP registry

        def failing_handler(x: int) -> int:
            raise RuntimeError("Tool broken!")

        spec = ToolSpec(
            name="test.fail",
            namespace="test",
            description="Always fails",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            handler=failing_handler,
            risk_level="low",
        )
        reg.register(spec)

        # ToolLoop uses tools/base.py ToolRegistry (ToolInterface)
        class FailTool(ToolInterface):
            name = "test.fail"
            description = "Always fails"

            async def execute(self, **kwargs):
                raise RuntimeError("Tool broken!")

        tl_reg = BaseToolRegistry()
        tl_reg.register(FailTool())

        plan_json = json.dumps([
            {"description": "Will fail", "tool": "test.fail",
             "params": {"x": 1}, "output_key": None},
        ])
        llm = MockLLM(response_content=plan_json)

        config = LoopConfig(max_agent_steps=5)
        tool_loop = ToolLoop(tl_reg, config)

        agent = AgentLoop(
            tool_loop=tool_loop,
            llm=llm,
            config=config,
            tool_registry=reg,
        )

        result = await agent.run(
            agent_id="test-agent",
            task_scope="Fail task",
            context={"task_id": "task-2"},
            allowed_tools=["test.fail"],
        )

        # Should have error in some step
        errors = [s.error for s in result.steps if s.error]
        assert len(errors) > 0


# ============================================================
# Test 4: ChatStreamChunk
# ============================================================

class TestChatStreamChunk:
    """Test ChatStreamChunk dataclass."""

    def test_create_chunk(self):
        chunk = ChatStreamChunk(
            delta_content="Hello",
            finish_reason=None,
        )
        assert chunk.delta_content == "Hello"
        assert chunk.delta_tool_calls is None
        assert chunk.finish_reason is None

    def test_chunk_with_tool_call(self):
        chunk = ChatStreamChunk(
            delta_content="",
            delta_tool_calls=[{"index": 0, "function": {"name": "search", "arguments": "qu"}}],
            finish_reason=None,
        )
        assert chunk.delta_content == ""
        assert chunk.delta_tool_calls is not None
        assert chunk.delta_tool_calls[0]["function"]["name"] == "search"

    def test_chunk_finish(self):
        chunk = ChatStreamChunk(
            delta_content="",
            finish_reason="stop",
        )
        assert chunk.finish_reason == "stop"


# ============================================================
# ChatResponse parsing
# ============================================================

class TestChatResponseParsing:
    """Test ChatResponse and ToolCall dataclasses."""

    def test_chat_response_with_tool_calls(self):
        tc = ToolCall(
            id="call_1",
            function=ToolFunction(name="search", arguments='{"query":"x"}'),
        )
        resp = ChatResponse(
            content="",
            tool_calls=[tc],
            finish_reason="tool_calls",
        )
        assert resp.finish_reason == "tool_calls"
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].function.name == "search"
        args = json.loads(resp.tool_calls[0].function.arguments)
        assert args["query"] == "x"

    def test_chat_response_no_tools(self):
        resp = ChatResponse(content="Hello world")
        assert resp.content == "Hello world"
        assert resp.tool_calls is None
        assert resp.finish_reason == "stop"
