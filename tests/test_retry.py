"""Tests for ToolRegistry retry mechanism with exponential backoff."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.tool_registry import ToolSpec, ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def async_tool_spec():
    """An async tool that can be configured to fail/succeed."""
    mock_handler = AsyncMock()
    return ToolSpec(
        name="test.async_echo",
        namespace="test",
        description="Echo back",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        handler=mock_handler,
    )


@pytest.fixture
def sync_tool_spec():
    """A sync tool that can be configured to fail/succeed."""
    mock_handler = MagicMock()
    return ToolSpec(
        name="test.sync_echo",
        namespace="test",
        description="Echo back (sync)",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        handler=mock_handler,
    )


class TestRetryMechanism:
    """Tests for ToolRegistry invoke retry with exponential backoff."""

    @pytest.mark.asyncio
    async def test_invoke_success_first_attempt(self, registry, async_tool_spec):
        """Normal invocation should succeed on first try."""
        async_tool_spec.handler.return_value = "echo: hello"
        registry.register(async_tool_spec)

        result = await registry.invoke("test.async_echo", {"message": "hello"})

        assert result == "echo: hello"
        assert async_tool_spec.handler.call_count == 1

    @pytest.mark.asyncio
    async def test_invoke_retry_on_failure(self, registry, async_tool_spec):
        """Should retry on failure and succeed on subsequent attempt."""
        async_tool_spec.handler.side_effect = [
            RuntimeError("temporary failure"),
            "echo: recovered",
        ]
        registry.register(async_tool_spec)

        result = await registry.invoke(
            "test.async_echo", {"message": "hello"},
            max_retries=3, retry_delay=0.01,
        )

        assert result == "echo: recovered"
        assert async_tool_spec.handler.call_count == 2

    @pytest.mark.asyncio
    async def test_invoke_max_retries_exceeded(self, registry, async_tool_spec):
        """Should raise last error after exhausting max_retries."""
        async_tool_spec.handler.side_effect = RuntimeError("persistent failure")
        registry.register(async_tool_spec)

        with pytest.raises(RuntimeError, match="persistent failure"):
            await registry.invoke(
                "test.async_echo", {"message": "hello"},
                max_retries=2, retry_delay=0.01,
            )

        # 1 initial + 2 retries = 3 attempts
        assert async_tool_spec.handler.call_count == 3

    @pytest.mark.asyncio
    async def test_invoke_retry_delay_exponential(self, registry, async_tool_spec):
        """Verify retry delays follow exponential backoff pattern."""
        call_count = [0]
        async def fail_always(**kwargs):
            call_count[0] += 1
            raise RuntimeError(f"fail {call_count[0]}")
        
        async_tool_spec.handler.side_effect = fail_always
        registry.register(async_tool_spec)

        with patch("asyncio.sleep") as mock_sleep:
            with pytest.raises(RuntimeError):
                await registry.invoke(
                    "test.async_echo", {"message": "hello"},
                    max_retries=3, retry_delay=1.0,
                )

        # Delays: 1.0 * 2^0 = 1.0, 1.0 * 2^1 = 2.0, 1.0 * 2^2 = 4.0
        assert mock_sleep.call_count == 3
        delay_call_args = [call.args[0] for call in mock_sleep.call_args_list]
        assert delay_call_args == [1.0, 2.0, 4.0]

    @pytest.mark.asyncio
    async def test_invoke_sync_handler_retry(self, registry, sync_tool_spec):
        """Sync handler should also be retried on failure."""
        sync_tool_spec.handler.side_effect = [
            ValueError("sync fail"),
            "sync: recovered",
        ]
        registry.register(sync_tool_spec)

        result = await registry.invoke(
            "test.sync_echo", {"message": "hello"},
            max_retries=3, retry_delay=0.01,
        )

        assert result == "sync: recovered"
        assert sync_tool_spec.handler.call_count == 2

    @pytest.mark.asyncio
    async def test_invoke_no_retry_when_disabled(self, registry, async_tool_spec):
        """max_retries=0 should mean no retry at all."""
        async_tool_spec.handler.side_effect = RuntimeError("fail")
        registry.register(async_tool_spec)

        with pytest.raises(RuntimeError, match="fail"):
            await registry.invoke(
                "test.async_echo", {"message": "hello"},
                max_retries=0, retry_delay=0.01,
            )

        assert async_tool_spec.handler.call_count == 1
