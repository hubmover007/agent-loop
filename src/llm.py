"""Concrete LLM Provider implementations.

Supported providers:
  - DeepSeek (OpenAI-compatible API)
  - Anthropic (Bedrock + direct API)
  - OpenAIClient (any OpenAI-compatible endpoint, e.g. local llama.cpp, vLLM)

Usage:
    from agent_loop.llm import DeepSeekProvider, AnthropicBedrockProvider

    llm = DeepSeekProvider(api_key="sk-...")
    response = await llm.chat([{"role": "user", "content": "Hello"}])
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from .loop_engine import LLMProvider, LLMResponse

logger = logging.getLogger(__name__)


# ============================================================
# DeepSeek Provider
# ============================================================

class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider (OpenAI-compatible endpoint)."""

    provider_name = "deepseek"

    def __init__(self, api_key: str, base_url: str = "https://api.deepseek.com",
                 default_model: str = "deepseek-chat", embedding_model: str = "deepseek-chat"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.embedding_model = embedding_model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )

    async def chat(self, messages: list[dict], thinking: bool = False,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   model: str | None = None, tools: list[dict] | None = None,
                   tool_choice: str = "auto", **kwargs) -> LLMResponse:
        self._ensure_client()

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        if thinking:
            # DeepSeek-R1 style: force thinking via system message
            payload["messages"] = [
                {"role": "system", "content": "Think step by step before answering. Show your reasoning."}
            ] + messages

        try:
            resp = await self._client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            content = choice["message"].get("content", "") or ""
            tool_calls = choice["message"].get("tool_calls")

            # Extract thinking if present (DeepSeek-R1)
            thinking_text = None
            if "reasoning_content" in choice["message"]:
                thinking_text = choice["message"]["reasoning_content"]

            return LLMResponse(
                content=content,
                model=data["model"],
                usage=data.get("usage", {}),
                thinking=thinking_text,
                finish_reason=choice.get("finish_reason", "stop"),
                tool_calls=tool_calls,
            )
        except Exception as e:
            logger.error("DeepSeek chat failed: %s", e)
            raise

    async def embed(self, text: str | list[str], model: str | None = None) -> list[list[float]]:
        """DeepSeek doesn't have a dedicated embedding model yet.
        Falls back to using chat model for simple embedding estimation.
        For production, configure a separate embedding provider."""
        logger.warning("DeepSeek embedding not natively supported, using placeholder")
        # Return zero vectors as placeholder - configure real embedder in production
        texts = [text] if isinstance(text, str) else text
        return [[0.0] * 1536 for _ in texts]


# ============================================================
# Anthropic Bedrock Provider
# ============================================================

class AnthropicBedrockProvider(LLMProvider):
    """Anthropic Claude via AWS Bedrock."""

    provider_name = "anthropic-bedrock"

    def __init__(self, aws_access_key: str, aws_secret_key: str,
                 region: str = "us-west-2",
                 default_model: str = "us.anthropic.claude-sonnet-4-6-v1"):
        self.aws_access_key = aws_access_key
        self.aws_secret_key = aws_secret_key
        self.region = region
        self.default_model = default_model
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client(
                "bedrock-runtime",
                region_name=self.region,
                aws_access_key_id=self.aws_access_key,
                aws_secret_access_key=self.aws_secret_key,
            )

    async def chat(self, messages: list[dict], thinking: bool = False,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   model: str | None = None, tools: list[dict] | None = None,
                   tool_choice: str = "auto", **kwargs) -> LLMResponse:
        self._ensure_client()

        # Anthropic uses a different tool format; gracefully fall back if tools requested
        if tools:
            logger.warning(
                "AnthropicBedrockProvider: tool calling requested but Anthropic format differs. "
                "Tools will be ignored for this provider. Use an OpenAI-compatible provider for "
                "full tool calling support."
            )
        self._ensure_client()

        model_id = model or self.default_model

        # Convert messages to Anthropic format
        system = ""
        anthropic_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system += msg["content"] + "\n"
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": [{"type": "text", "text": msg["content"]}],
                })

        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": anthropic_messages,
        }
        if system.strip():
            request_body["system"] = system.strip()

        if thinking:
            # Enable extended thinking
            request_body["thinking"] = {
                "type": "enabled",
                "budget_tokens": min(max_tokens // 2, 4096),
            }

        try:
            import asyncio
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: self._client.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body),
                )
            )
            body = json.loads(resp["body"].read())

            # Extract content and thinking
            content = ""
            thinking_text = None
            for block in body.get("content", []):
                if block["type"] == "text":
                    content += block["text"]
                elif block["type"] == "thinking":
                    thinking_text = block.get("thinking", "")

            return LLMResponse(
                content=content,
                model=model_id,
                usage=body.get("usage", {}),
                thinking=thinking_text,
                finish_reason=body.get("stop_reason", "stop"),
            )
        except Exception as e:
            logger.error("Bedrock chat failed: %s", e)
            raise

    async def embed(self, text: str | list[str], model: str | None = None) -> list[list[float]]:
        """Use Amazon Titan Embeddings via Bedrock."""
        self._ensure_client()

        texts = [text] if isinstance(text, str) else text
        embeddings = []

        for t in texts:
            request_body = json.dumps({"inputText": t})
            import asyncio
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: self._client.invoke_model(
                    modelId="amazon.titan-embed-text-v2:0",
                    body=request_body,
                )
            )
            body = json.loads(resp["body"].read())
            embeddings.append(body["embedding"])

        return embeddings


# ============================================================
# OpenAI-Compatible Provider (通用)
# ============================================================

class OpenAICompatibleProvider(LLMProvider):
    """Any OpenAI-compatible API endpoint.

    Works with: vLLM, llama.cpp server, Ollama, GLM, local models, etc.
    """

    provider_name = "openai-compatible"

    def __init__(self, base_url: str, api_key: str = "not-needed",
                 default_model: str = "default",
                 embedding_model: str | None = None,
                 extra_headers: dict | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.embedding_model = embedding_model
        self.extra_headers = extra_headers or {}
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import httpx
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                **self.extra_headers,
            }
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=300.0,  # Long timeout for local models
            )

    async def chat(self, messages: list[dict], thinking: bool = False,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   model: str | None = None, tools: list[dict] | None = None,
                   tool_choice: str = "auto", **kwargs) -> LLMResponse:
        self._ensure_client()

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            **kwargs,
        }

        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        try:
            resp = await self._client.post("/v1/chat/completions", json=payload)
            resp.raise_for_status()
            data = resp.json()

            choice = data["choices"][0]
            content = choice["message"].get("content", "") or ""
            tool_calls = choice["message"].get("tool_calls")

            return LLMResponse(
                content=content,
                model=data.get("model", payload["model"]),
                usage=data.get("usage", {}),
                finish_reason=choice.get("finish_reason", "stop"),
                tool_calls=tool_calls,
            )
        except Exception as e:
            logger.error("OpenAI-compatible chat failed: %s", e)
            raise

    async def chat_stream(self, messages: list[dict],
                          tools: list[dict] | None = None,
                          tool_choice: str = "auto",
                          model: str | None = None,
                          **kwargs):
        """Stream chat completion via SSE.

        Yields:
            ChatStreamChunk
        """
        from .loop_engine import ChatStreamChunk

        self._ensure_client()

        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "stream": True,
            **kwargs,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        try:
            async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                resp.raise_for_status()
                buffer = b""
                async for chunk in resp.aiter_bytes():
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith(b"data: "):
                            data_str = line[6:].decode("utf-8", errors="replace")
                            if data_str == "[DONE]":
                                return
                            try:
                                data = json.loads(data_str)
                            except json.JSONDecodeError:
                                continue
                            choice = data["choices"][0]
                            delta = choice.get("delta", {})
                            yield ChatStreamChunk(
                                delta_content=delta.get("content", "") or "",
                                delta_tool_calls=delta.get("tool_calls"),
                                finish_reason=choice.get("finish_reason"),
                            )
        except Exception as e:
            logger.error("OpenAI-compatible chat_stream failed: %s", e)
            raise

    async def embed(self, text: str | list[str], model: str | None = None) -> list[list[float]]:
        """Generate embeddings via OpenAI-compatible API."""
        if not self.embedding_model and self.base_url:
            # Try the same endpoint for embeddings
            pass

        self._ensure_client()
        texts = [text] if isinstance(text, str) else text

        try:
            # Build the embedding URL explicitly to avoid httpx base_url
            # path resolution issues (double /v1 prefix).
            embed_url = f"{self.base_url}/embeddings"
            resp = await self._client.post(embed_url, json={
                "model": model or self.embedding_model or self.default_model,
                "input": texts,
            })
            resp.raise_for_status()
            data = resp.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            logger.warning("Embedding failed: %s, using zero vectors", e)
            return [[0.0] * 1536 for _ in texts]


# ============================================================
# Provider Factory
# ============================================================

def create_provider(provider_type: str, **kwargs) -> LLMProvider:
    """Factory function to create LLM providers from configuration."""
    providers = {
        "deepseek": DeepSeekProvider,
        "anthropic-bedrock": AnthropicBedrockProvider,
        "openai-compatible": OpenAICompatibleProvider,
        "easyrouter": OpenAICompatibleProvider,  # EasyRouter via OpenAI-compatible interface
    }

    cls = providers.get(provider_type)
    if cls is None:
        raise ValueError(f"Unknown provider type: {provider_type}. Available: {list(providers.keys())}")

    # EasyRouter: set default endpoint and API key
    if provider_type == "easyrouter":
        kwargs.setdefault("base_url", "https://easyrouter.io/v1")
        kwargs.setdefault("api_key", os.environ.get("EASYROUTER_API_KEY", ""))
        kwargs.setdefault("default_model", "deepseek-v4-pro")

    return cls(**kwargs)
