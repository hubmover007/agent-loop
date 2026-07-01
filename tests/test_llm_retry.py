"""Tests for LLMProvider.chat_with_retry()."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.loop_engine import LLMProvider, LLMResponse


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing retry logic."""

    def __init__(self, fail_count=0, delay=0.0, error_type="timeout"):
        self._fail_count = fail_count
        self._delay = delay
        self._error_type = error_type
        self._call_count = 0

    async def chat(self, messages, **kwargs):
        self._call_count += 1
        if self._delay > 0:
            await asyncio.sleep(self._delay)

        if self._call_count <= self._fail_count:
            if self._error_type == "timeout":
                raise asyncio.TimeoutError()
            else:
                raise RuntimeError(f"Mock error on call {self._call_count}")

        return LLMResponse(content="OK", model="mock", usage={"input_tokens": 10, "output_tokens": 5})

    async def chat_stream(self, messages, tools=None):
        yield {"type": "content", "content": "OK"}

    async def embed(self, texts):
        return [[0.0] * 768 for _ in texts]

    @property
    def call_count(self):
        return self._call_count


class TestLLMRetry:
    """Test LLM retry logic."""

    @pytest.mark.asyncio
    async def test_retry_success_on_third_attempt(self):
        """LLM fails twice, succeeds on third attempt."""
        llm = MockLLMProvider(fail_count=2)
        resp = await llm.chat_with_retry(
            [{"role": "user", "content": "hi"}],
            max_retries=3,
            timeout=5.0,
        )
        assert resp.content == "OK"
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_retry_exhausted(self):
        """LLM always fails, retry exhausted."""
        llm = MockLLMProvider(fail_count=99)
        with pytest.raises(asyncio.TimeoutError):
            await llm.chat_with_retry(
                [{"role": "user", "content": "hi"}],
                max_retries=3,
                timeout=1.0,
            )
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_no_retry_needed(self):
        """LLM succeeds on first attempt."""
        llm = MockLLMProvider(fail_count=0)
        resp = await llm.chat_with_retry(
            [{"role": "user", "content": "hi"}],
            max_retries=3,
            timeout=5.0,
        )
        assert resp.content == "OK"
        assert llm.call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_generic_error(self):
        """LLM fails with generic error, then succeeds."""
        llm = MockLLMProvider(fail_count=1, error_type="runtime")
        resp = await llm.chat_with_retry(
            [{"role": "user", "content": "hi"}],
            max_retries=3,
            timeout=5.0,
        )
        assert resp.content == "OK"
        assert llm.call_count == 2

    @pytest.mark.asyncio
    async def test_retry_with_backoff(self):
        """Test that retry uses exponential backoff (fast test)."""
        llm = MockLLMProvider(fail_count=1)
        import time
        start = time.time()
        resp = await llm.chat_with_retry(
            [{"role": "user", "content": "hi"}],
            max_retries=2,
            timeout=5.0,
        )
        elapsed = time.time() - start
        # Should have waited ~1s (2^0 = 1s backoff)
        assert elapsed >= 0.8  # Allow some tolerance
        assert resp.content == "OK"

    @pytest.mark.asyncio
    async def test_timeout_actually_works(self):
        """Test that timeout parameter actually enforces timeout."""
        llm = MockLLMProvider(delay=5.0)  # 5 second delay
        with pytest.raises(asyncio.TimeoutError):
            await llm.chat_with_retry(
                [{"role": "user", "content": "hi"}],
                max_retries=1,
                timeout=0.5,  # 0.5s timeout
            )

    @pytest.mark.asyncio
    async def test_default_retry_count(self):
        """Test default retry count is 3."""
        llm = MockLLMProvider(fail_count=99)
        with pytest.raises(asyncio.TimeoutError):
            await llm.chat_with_retry(
                [{"role": "user", "content": "hi"}],
                timeout=1.0,
            )
        # Default max_retries=3
        assert llm.call_count == 3

    @pytest.mark.asyncio
    async def test_usage_preserved_on_success(self):
        """Test that usage info is preserved on successful retry."""
        llm = MockLLMProvider(fail_count=1)
        resp = await llm.chat_with_retry(
            [{"role": "user", "content": "hi"}],
            max_retries=3,
            timeout=5.0,
        )
        assert resp.usage is not None
        assert resp.usage["input_tokens"] == 10
        assert resp.usage["output_tokens"] == 5
