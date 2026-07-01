"""Embedding service for memory vectorization.

Supports multiple providers:
  1. OpenAI text-embedding-004 (default, 1536 dims)
  2. OpenAI text-embedding-3-large (3072 dims)
  3. Local sentence-transformers (bge-large-zh, 1024 dims)
  4. Mock embedding (for testing, deterministic random vectors)

Configured via agent-loop.yaml:
  memory:
    embedding:
      provider: "openai"  # or "local" or "mock"
      model: "text-embedding-004"
      api_key: "${OPENAI_API_KEY}"
      dimensions: 1536
      batch_size: 100
"""

import asyncio
import hashlib
import logging
import os
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Embedding service configuration."""
    provider: str = "mock"  # mock | openai | local
    model: str = "text-embedding-004"
    api_key: str = ""
    dimensions: int = 768
    batch_size: int = 100
    cache_enabled: bool = True


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensions."""
        ...

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        ...


class MockEmbeddingProvider(EmbeddingProvider):
    """Deterministic mock embedding for testing.

    Uses hash-based pseudo-random vectors. Same text → same vector.
    """

    def __init__(self, dimensions: int = 768):
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, text: str) -> list[float]:
        # Deterministic: hash text → seed → random vector
        seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        vec = [rng.gauss(0, 1) for _ in range(self._dimensions)]
        # Normalize to unit length
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [await self.embed(t) for t in texts]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """OpenAI text-embedding API provider."""

    def __init__(self, api_key: str, model: str = "text-embedding-004",
                 dimensions: int = 768, batch_size: int = 100):
        self._api_key = api_key
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size
        self._client = None  # lazy init

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(
                base_url="https://api.openai.com/v1",
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=30.0,
            )
        return self._client

    async def embed(self, text: str) -> list[float]:
        results = await self.embed_batch([text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        client = await self._get_client()
        all_embeddings = []

        for i in range(0, len(texts), self._batch_size):
            batch = texts[i:i + self._batch_size]
            resp = await client.post(
                "/embeddings",
                json={
                    "model": self._model,
                    "input": batch,
                    "dimensions": self._dimensions,
                }
            )
            resp.raise_for_status()
            data = resp.json()
            all_embeddings.extend([d["embedding"] for d in data["data"]])

        return all_embeddings


class LocalEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers embedding (bge-large-zh etc).

    Requires: pip install sentence-transformers
    """

    def __init__(self, model_name: str = "BAAI/bge-large-zh-v1.5"):
        self._model_name = model_name
        self._model = None  # lazy init
        self._dimensions = 1024  # bge-large default

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._dimensions = self._model.get_sentence_embedding_dimension()
        return self._model

    async def embed(self, text: str) -> list[float]:
        model = await self._get_model()
        # Run in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(None, model.encode, text)
        return embedding.tolist()

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        model = await self._get_model()
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, model.encode, texts)
        return [e.tolist() for e in embeddings]


class EmbeddingService:
    """High-level embedding service with caching.

    Features:
    - Provider abstraction (mock/openai/local)
    - LRU cache for repeated embeddings
    - Batch processing
    - Auto-fallback: openai → mock (if no key)
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self._provider: EmbeddingProvider | None = None
        self._cache: dict[str, list[float]] = {}  # text → embedding
        self._cache_max = 10000  # max cached embeddings

    def _create_provider(self) -> EmbeddingProvider:
        """Create provider based on config."""
        provider = self.config.provider.lower()

        if provider == "mock":
            return MockEmbeddingProvider(self.config.dimensions)

        elif provider == "openai":
            api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning("OpenAI embedding: no API key, falling back to mock")
                return MockEmbeddingProvider(self.config.dimensions)
            return OpenAIEmbeddingProvider(
                api_key=api_key,
                model=self.config.model,
                dimensions=self.config.dimensions,
                batch_size=self.config.batch_size,
            )

        elif provider == "local":
            try:
                return LocalEmbeddingProvider(self.config.model)
            except ImportError:
                logger.warning(
                    "Local embedding: sentence-transformers not installed, "
                    "falling back to mock. Install with: pip install sentence-transformers"
                )
                return MockEmbeddingProvider(self.config.dimensions)

        else:
            logger.warning("Unknown embedding provider: %s, using mock", provider)
            return MockEmbeddingProvider(self.config.dimensions)

    @property
    def provider(self) -> EmbeddingProvider:
        if self._provider is None:
            self._provider = self._create_provider()
        return self._provider

    @property
    def dimensions(self) -> int:
        return self.provider.dimensions

    async def embed(self, text: str) -> list[float]:
        """Embed text with caching."""
        if self.config.cache_enabled and text in self._cache:
            return self._cache[text]

        embedding = await self.provider.embed(text)

        if self.config.cache_enabled:
            if len(self._cache) >= self._cache_max:
                # Simple eviction: clear half
                keys = list(self._cache.keys())
                for k in keys[:len(keys) // 2]:
                    del self._cache[k]
            self._cache[text] = embedding

        return embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts, using cache where possible."""
        results = []
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        for i, text in enumerate(texts):
            if self.config.cache_enabled and text in self._cache:
                results.append(self._cache[text])
            else:
                results.append([])  # placeholder
                uncached_indices.append(i)
                uncached_texts.append(text)

        if uncached_texts:
            new_embeddings = await self.provider.embed_batch(uncached_texts)
            for pos, idx in enumerate(uncached_indices):
                embedding = new_embeddings[pos]
                results[idx] = embedding
                if self.config.cache_enabled:
                    self._cache[uncached_texts[pos]] = embedding

        return results
