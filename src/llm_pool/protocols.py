"""Protocol Adapters — one adapter per wire protocol.

Protocols:
  openai-completions      → OpenAI-compatible REST API
  bedrock-converse-stream → AWS Bedrock Converse API
  google-gemini           → Google Generative Language API

Each adapter implements the LLMProvider interface from loop_engine.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import abstractmethod
from typing import Any

logger = logging.getLogger(__name__)

# Import base types from loop_engine
from ..loop_engine import LLMProvider, LLMResponse


class ProtocolAdapter(LLMProvider):
    """Base class for all protocol adapters."""

    def __init__(self, config, auth_kwargs: dict):
        self._config = config
        self._auth_kwargs = auth_kwargs

    @property
    def provider_id(self) -> str:
        return self._config.id

    @property
    def model_id(self) -> str:
        return self._config.endpoint.model

    @property
    def supports_vision(self) -> bool:
        return "vision" in self._config.capabilities

    @property
    def supports_streaming(self) -> bool:
        return True  # override in subclasses if not supported

    @abstractmethod
    async def chat(self, messages: list[dict], **kwargs) -> LLMResponse: ...

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        raise NotImplementedError(f"{self.__class__.__name__} does not support embeddings")


# ============================================================
# OpenAI-compatible adapter (openai-completions)
# ============================================================

class OpenAICompatibleAdapter(ProtocolAdapter):
    """Handles any OpenAI-compatible REST API (OpenAI, DeepSeek, EasyRouter, local, etc.)"""

    def __init__(self, config, auth_kwargs: dict):
        super().__init__(config, auth_kwargs)
        self._client: Any = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError:
            raise RuntimeError("openai package required: pip install openai")

        api_key = self._auth_kwargs.get("api_key") or "not-needed"
        base_url = self._config.endpoint.base_url

        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self._config.endpoint.timeout_s,
            max_retries=0,  # we handle retries via ToolLoop
        )

    async def chat(self, messages: list[dict], thinking: bool = False,
                   max_tokens: int | None = None, temperature: float = 0.7,
                   model: str | None = None, **kwargs) -> LLMResponse:
        self._ensure_client()

        _model = model or self._config.endpoint.model
        _max_tokens = max_tokens or self._config.max_output_tokens

        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=_model,
                    messages=messages,
                    max_tokens=_max_tokens,
                    temperature=temperature,
                    **{k: v for k, v in kwargs.items() if k not in ("thinking",)},
                ),
                timeout=self._config.endpoint.timeout_s,
            )
            content = response.choices[0].message.content or ""
            usage = response.usage
            return LLMResponse(
                content=content,
                model=_model,
                usage={
                    "input_tokens": usage.prompt_tokens if usage else 0,
                    "output_tokens": usage.completion_tokens if usage else 0,
                },
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"OpenAI-compatible chat timed out ({self._config.endpoint.timeout_s}s)")

    async def embed(self, text: str | list[str]) -> list[list[float]]:
        self._ensure_client()
        texts = [text] if isinstance(text, str) else text
        response = await self._client.embeddings.create(
            model=self._config.endpoint.model,
            input=texts,
        )
        return [item.embedding for item in response.data]


# ============================================================
# AWS Bedrock Converse adapter (bedrock-converse-stream)
# ============================================================

class BedrockConverseAdapter(ProtocolAdapter):
    """AWS Bedrock Converse API adapter."""

    def __init__(self, config, auth_kwargs: dict):
        super().__init__(config, auth_kwargs)
        self._client: Any = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import boto3  # type: ignore
        except ImportError:
            raise RuntimeError("boto3 required: pip install boto3")

        profile = self._auth_kwargs.get("aws_profile")
        region = self._auth_kwargs.get("aws_region", "us-east-1")

        session_kwargs: dict = {}
        if profile:
            session_kwargs["profile_name"] = profile

        import os
        # Respect AGENTS.md: unset env AK before using profile
        env_ak = os.environ.get("AWS_ACCESS_KEY_ID")
        env_sk = os.environ.get("AWS_SECRET_ACCESS_KEY")

        if profile and env_ak:
            logger.debug("BedrockAdapter: profile=%s specified, env AK will be overridden by profile", profile)

        session = boto3.Session(**session_kwargs)
        self._client = session.client("bedrock-runtime", region_name=region)

    async def chat(self, messages: list[dict], thinking: bool = False,
                   max_tokens: int | None = None, temperature: float = 0.7,
                   model: str | None = None, **kwargs) -> LLMResponse:
        self._ensure_client()
        model_id = model or self._config.endpoint.model
        _max_tokens = max_tokens or self._config.max_output_tokens

        # Convert OpenAI-style messages to Bedrock Converse format
        system_parts = []
        converse_messages = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_parts.append({"text": content})
            elif role in ("user", "assistant"):
                converse_messages.append({
                    "role": role,
                    "content": [{"text": content}],
                })

        request: dict = {
            "modelId": model_id,
            "messages": converse_messages,
            "inferenceConfig": {"maxTokens": _max_tokens, "temperature": temperature},
        }
        if system_parts:
            request["system"] = system_parts

        try:
            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: self._client.converse(**request)),
                timeout=self._config.endpoint.timeout_s,
            )
            content = response["output"]["message"]["content"][0]["text"]
            usage_raw = response.get("usage", {})
            return LLMResponse(
                content=content,
                model=_model,
                usage={
                    "input_tokens": usage_raw.get("inputTokens", 0),
                    "output_tokens": usage_raw.get("outputTokens", 0),
                    "cache_read_tokens": usage_raw.get("cacheReadInputTokens", 0),
                    "cache_write_tokens": usage_raw.get("cacheWriteInputTokens", 0),
                },
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Bedrock chat timed out ({self._config.endpoint.timeout_s}s)")


# ============================================================
# Google Gemini adapter (google-gemini)
# ============================================================

class GoogleGeminiAdapter(ProtocolAdapter):
    """Google Generative Language API adapter."""

    def __init__(self, config, auth_kwargs: dict):
        super().__init__(config, auth_kwargs)
        self._client: Any = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError:
            raise RuntimeError("google-generativeai required: pip install google-generativeai")

        if self._auth_kwargs.get("use_adc"):
            pass  # ADC handles auth automatically
        else:
            api_key = self._auth_kwargs.get("api_key")
            if api_key:
                genai.configure(api_key=api_key)

        self._client = genai

    async def chat(self, messages: list[dict], thinking: bool = False,
                   max_tokens: int | None = None, temperature: float = 0.7,
                   model: str | None = None, **kwargs) -> LLMResponse:
        self._ensure_client()
        model_name = model or self._config.endpoint.model

        # Convert to Gemini format
        gemini_messages = []
        system_instruction = None
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = content
            elif role == "user":
                gemini_messages.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                gemini_messages.append({"role": "model", "parts": [content]})

        try:
            genai_model = self._client.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction,
            )
            gen_config = self._client.types.GenerationConfig(
                max_output_tokens=max_tokens or self._config.max_output_tokens,
                temperature=temperature,
            )

            loop = asyncio.get_event_loop()
            response = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: genai_model.generate_content(
                    gemini_messages, generation_config=gen_config
                )),
                timeout=self._config.endpoint.timeout_s,
            )
            content = response.text or ""
            usage = response.usage_metadata
            return LLMResponse(
                content=content,
                model=_model,
                usage={
                    "input_tokens": getattr(usage, "prompt_token_count", 0),
                    "output_tokens": getattr(usage, "candidates_token_count", 0),
                },
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Gemini chat timed out ({self._config.endpoint.timeout_s}s)")


# ============================================================
# Adapter factory
# ============================================================

_PROTOCOL_MAP: dict[str, type[ProtocolAdapter]] = {
    "openai-completions": OpenAICompatibleAdapter,
    "bedrock-converse-stream": BedrockConverseAdapter,
    "google-gemini": GoogleGeminiAdapter,
}


def create_adapter(config, auth_kwargs: dict) -> ProtocolAdapter:
    """Create the appropriate protocol adapter for a provider config."""
    protocol = config.protocol
    adapter_cls = _PROTOCOL_MAP.get(protocol)
    if not adapter_cls:
        raise ValueError(
            f"Unknown protocol '{protocol}' for provider '{config.id}'. "
            f"Supported: {list(_PROTOCOL_MAP.keys())}"
        )
    return adapter_cls(config, auth_kwargs)
