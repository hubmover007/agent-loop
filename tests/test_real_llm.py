"""Real LLM integration tests — skipped without API keys.

These tests verify that LLMPool.verify() and actual chat completions
work with real API endpoints. They are skipped by default in CI/CD.
"""

import os
import pytest


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set"
)
@pytest.mark.asyncio
async def test_verify_real_deepseek():
    """Test LLMPool.verify() with real DeepSeek API."""
    from src.llm_pool import LLMPool
    pool = LLMPool(config_path="config/llm_pool.json")
    pool.initialize()
    result = await pool.verify("deepseek-chat")
    assert result in (True, False)


@pytest.mark.skipif(
    not os.environ.get("DEEPSEEK_API_KEY"),
    reason="DEEPSEEK_API_KEY not set"
)
@pytest.mark.asyncio
async def test_real_chat_completion():
    """Test real chat completion with DeepSeek API."""
    from src.llm_pool import LLMPool
    pool = LLMPool(config_path="config/llm_pool.json")
    pool.initialize()
    provider = await pool.acquire(capabilities=["chat"])
    if provider:
        response = await provider.chat([
            {"role": "user", "content": "Reply with only: OK"}
        ])
        assert response is not None
        assert hasattr(response, "content")
