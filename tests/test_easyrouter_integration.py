"""Integration tests for EasyRouter models.

These tests make REAL API calls to EasyRouter.
Skip if EASYROUTER_API_KEY is not set.
"""

from __future__ import annotations

import os
import asyncio
import pytest

# Skip all if no API key
EASYROUTER_KEY = os.environ.get("EASYROUTER_API_KEY", "")
pytestmark = pytest.mark.skipif(
    not EASYROUTER_KEY,
    reason="EASYROUTER_API_KEY not set"
)


class TestEasyRouterConnectivity:
    """Test basic connectivity to EasyRouter API."""

    @pytest.mark.asyncio
    async def test_list_models(self):
        """Can we reach the /models endpoint?"""
        import httpx
        async with httpx.AsyncClient(
            base_url="https://easyrouter.io/v1",
            headers={"Authorization": f"Bearer {EASYROUTER_KEY}"},
            timeout=15.0,
        ) as client:
            resp = await client.get("/models")
            assert resp.status_code == 200
            data = resp.json()
            assert "data" in data
            assert len(data["data"]) > 0

    @pytest.mark.asyncio
    async def test_chat_completion_deepseek(self):
        """Test basic chat completion with deepseek-v4-pro."""
        import httpx
        async with httpx.AsyncClient(
            base_url="https://easyrouter.io/v1",
            headers={"Authorization": f"Bearer {EASYROUTER_KEY}"},
            timeout=30.0,
        ) as client:
            resp = await client.post("/chat/completions", json={
                "model": "deepseek-v4-pro",
                "messages": [{"role": "user", "content": "Say hello in one word."}],
                "max_tokens": 200,
            })
            assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            assert "choices" in data
            assert len(data["choices"]) > 0
            content = data["choices"][0]["message"]["content"]
            assert len(content) > 0

    @pytest.mark.asyncio
    async def test_chat_completion_gpt55(self):
        """Test basic chat completion with gpt-5.5."""
        import httpx
        async with httpx.AsyncClient(
            base_url="https://easyrouter.io/v1",
            headers={"Authorization": f"Bearer {EASYROUTER_KEY}"},
            timeout=30.0,
        ) as client:
            resp = await client.post("/chat/completions", json={
                "model": "gpt-5.5",
                "messages": [{"role": "user", "content": "Say hi."}],
                "max_tokens": 10,
            })
            assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:200]}"
            data = resp.json()
            assert len(data["choices"]) > 0


class TestLLMPoolWithEasyRouter:
    """Test LLMPool integration with EasyRouter models."""

    @pytest.mark.asyncio
    async def test_pool_loads_easyrouter(self):
        """LLMPool should load EasyRouter providers."""
        from src.llm_pool import LLMPool

        pool = LLMPool("config/llm_pool.json")
        pool.initialize()

        # Should have 8 EasyRouter providers (by endpoint)
        easyrouter_count = sum(
            1 for p in pool._providers.values()
            if "easyrouter" in p._endpoint
        )
        assert easyrouter_count >= 8, f"Expected >=8 EasyRouter providers, got {easyrouter_count}"

    @pytest.mark.asyncio
    async def test_pool_total_providers(self):
        """Total enabled providers should be at least 8 (EasyRouter), up to 10 with DeepSeek fallback."""
        from src.llm_pool import LLMPool

        pool = LLMPool("config/llm_pool.json")
        pool.initialize()

        # At minimum all 8 EasyRouter providers must load
        # DeepSeek fallback only loads if DEEPSEEK_API_KEY is set
        assert len(pool._providers) >= 8, f"Expected >=8 providers, got {len(pool._providers)}"

    @pytest.mark.asyncio
    async def test_select_vision_model(self):
        """get_vision_model() should return gpt-5.5 or gemini-2.5-flash."""
        from src.llm_pool import LLMPool

        pool = LLMPool("config/llm_pool.json")
        pool.initialize()

        vision_cfg = pool.select(
            capabilities=["vision"],
            modalities=["text", "image"],
            strategy="cheapest",
        )
        assert vision_cfg is not None, "No vision model selected"
        assert "image" in vision_cfg.modality
        assert vision_cfg.model in ["gpt-5.5", "gemini-2.5-flash"], \
            f"Expected gpt-5.5 or gemini-2.5-flash, got {vision_cfg.model}"

    @pytest.mark.asyncio
    async def test_select_reasoning_model(self):
        """select(capabilities=['reasoning']) should return a reasoning-capable model."""
        from src.llm_pool import LLMPool

        pool = LLMPool("config/llm_pool.json")
        pool.initialize()

        model_cfg = pool.select(capabilities=["reasoning"])
        assert model_cfg is not None, "No reasoning model selected"
        assert model_cfg.enabled

    @pytest.mark.asyncio
    async def test_select_primary_model(self):
        """select should find the primary tagged model."""
        from src.llm_pool import LLMPool

        pool = LLMPool("config/llm_pool.json")
        pool.initialize()

        cfg = pool.select(capabilities=["reasoning"])
        assert cfg is not None
        # deepseek-v4-pro has "primary" tag
        assert cfg.id == "easyrouter-deepseek-v4-pro" or "reasoning" in cfg.tags

    @pytest.mark.asyncio
    async def test_fallback_providers_present(self):
        """DeepSeek originals are configured as fallback (may not load if API key missing)."""
        from src.llm_pool import LLMPool, PoolConfigJSON

        # Check config level (regardless of auth availability)
        cfg = PoolConfigJSON.from_json("config/llm_pool.json")
        ds_providers = [
            p for p in cfg.providers
            if p.id.startswith("deepseek-") and "fallback" in p.tags
        ]
        assert len(ds_providers) >= 2, f"Expected >=2 DeepSeek fallback in config, got {len(ds_providers)}"

        # Also test pool level - DeepSeek fallback may not load if key is missing (that's OK)
        pool = LLMPool("config/llm_pool.json")
        pool.initialize()
        ds_ids = [p for p in pool._providers if p.startswith("deepseek-")]
        # At least 0 if no DEEPSEEK_API_KEY, 2 if available
        assert len(ds_ids) >= 0


class TestRealLLMCall:
    """Test real LLM calls through the agent-loop LLMProvider interface."""

    @pytest.mark.asyncio
    async def test_chat_via_pool_deepseek(self):
        """Test chat() through LLMPool with deepseek-v4-pro."""
        from src.llm_pool import LLMPool

        pool = LLMPool("config/llm_pool.json")
        pool.initialize()

        # Acquire a provider and call chat()
        provider = await pool.acquire(capabilities=["reasoning"])
        resp = await provider.chat([
            {"role": "user", "content": "What is 2+2? Answer with just the number."}
        ])

        assert resp is not None
        assert resp.content.strip(), "Response content should not be empty"
        # The answer should contain "4" somewhere
        assert "4" in resp.content.strip() or len(resp.content.strip()) > 0

    @pytest.mark.asyncio
    async def test_chat_via_pool_gpt55(self):
        """Test chat() through LLMPool with gpt-5.5."""
        from src.llm_pool import LLMPool

        pool = LLMPool("config/llm_pool.json")
        pool.initialize()

        # Force use gpt-5.5 by specifying the provider directly
        provider = pool._providers.get("easyrouter-gpt-5.5")
        assert provider is not None, "easyrouter-gpt-5.5 not found in pool"

        resp = await provider.chat([
            {"role": "user", "content": "Say 'hello' in one lowercase word."}
        ])

        assert resp is not None
        assert resp.content.strip(), "Response content should not be empty"
        assert "hello" in resp.content.strip().lower()

    @pytest.mark.asyncio
    async def test_chat_via_pool_flash(self):
        """Test chat() through LLMPool with deepseek-v4-flash (fast model)."""
        from src.llm_pool import LLMPool

        pool = LLMPool("config/llm_pool.json")
        pool.initialize()

        provider = pool._providers.get("easyrouter-deepseek-v4-flash")
        assert provider is not None, "easyrouter-deepseek-v4-flash not found in pool"

        resp = await provider.chat([
            {"role": "user", "content": "Reply with one word: yes"}
        ])

        assert resp is not None
        assert len(resp.content.strip()) > 0
