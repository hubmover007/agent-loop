"""Tests for P6-E: Embedding service (mock/openai/local providers with caching)."""

import math
import pytest

from src.memory.embedding import (
    EmbeddingConfig,
    EmbeddingService,
    MockEmbeddingProvider,
    OpenAIEmbeddingProvider,
    LocalEmbeddingProvider,
)
from src.memory import MemoryPool


# ────────────────────────────────────────────────────────────
# MockEmbeddingProvider
# ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_mock_deterministic():
    """Same text → same vector (deterministic)."""
    provider = MockEmbeddingProvider(dimensions=128)
    v1 = await provider.embed("hello world")
    v2 = await provider.embed("hello world")
    assert v1 == v2
    assert len(v1) == 128


@pytest.mark.asyncio
async def test_mock_different_texts_different():
    """Different texts → different vectors."""
    provider = MockEmbeddingProvider(dimensions=64)
    v1 = await provider.embed("hello")
    v2 = await provider.embed("world")
    assert v1 != v2


@pytest.mark.asyncio
async def test_mock_normalized():
    """Vectors should be unit length."""
    provider = MockEmbeddingProvider(dimensions=256)
    for text in ["a", "hello world", "testing 123", "longer sentence here"]:
        v = await provider.embed(text)
        norm = math.sqrt(sum(x * x for x in v))
        assert math.isclose(norm, 1.0, rel_tol=1e-6)


@pytest.mark.asyncio
async def test_mock_batch():
    """Batch embedding returns same results as individual."""
    provider = MockEmbeddingProvider(dimensions=64)
    texts = ["a", "b", "c", "d", "e"]
    batch = await provider.embed_batch(texts)
    individual = [await provider.embed(t) for t in texts]
    assert batch == individual


@pytest.mark.asyncio
async def test_mock_dimensions():
    """Provider respects configured dimensions."""
    for dim in [64, 128, 256, 1536]:
        provider = MockEmbeddingProvider(dimensions=dim)
        assert provider.dimensions == dim
        v = await provider.embed("test")
        assert len(v) == dim


# ────────────────────────────────────────────────────────────
# EmbeddingConfig
# ────────────────────────────────────────────────────────────

def test_config_defaults():
    """Default config uses mock provider."""
    cfg = EmbeddingConfig()
    assert cfg.provider == "mock"
    assert cfg.dimensions == 1536
    assert cfg.batch_size == 100
    assert cfg.cache_enabled is True


def test_config_custom():
    """Custom config settings."""
    cfg = EmbeddingConfig(
        provider="openai",
        model="text-embedding-3-large",
        dimensions=3072,
        batch_size=50,
        cache_enabled=False,
    )
    assert cfg.provider == "openai"
    assert cfg.model == "text-embedding-3-large"
    assert cfg.dimensions == 3072
    assert cfg.batch_size == 50
    assert cfg.cache_enabled is False


# ────────────────────────────────────────────────────────────
# EmbeddingService
# ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_service_mock_embed():
    """EmbeddingService with mock provider embeds text."""
    service = EmbeddingService(EmbeddingConfig(provider="mock", dimensions=128))
    v = await service.embed("hello")
    assert len(v) == 128
    assert service.dimensions == 128


@pytest.mark.asyncio
async def test_service_caching():
    """Repeated calls for same text use cache."""
    service = EmbeddingService(EmbeddingConfig(provider="mock", dimensions=64))
    v1 = await service.embed("test")
    v2 = await service.embed("test")
    assert v1 == v2
    # Should be cached now
    assert "test" in service._cache


@pytest.mark.asyncio
async def test_service_caching_disabled():
    """When cache is disabled, no entries stored."""
    service = EmbeddingService(
        EmbeddingConfig(provider="mock", dimensions=64, cache_enabled=False)
    )
    await service.embed("test")
    assert not service._cache


@pytest.mark.asyncio
async def test_service_batch():
    """Batch embedding returns correct number of vectors."""
    service = EmbeddingService(EmbeddingConfig(provider="mock", dimensions=64))
    texts = ["a", "b", "c", "d", "e"]
    results = await service.embed_batch(texts)
    assert len(results) == 5
    for r in results:
        assert len(r) == 64


@pytest.mark.asyncio
async def test_service_batch_with_cache():
    """Batch embedding leverages cache for repeated texts."""
    service = EmbeddingService(EmbeddingConfig(provider="mock", dimensions=64))
    # Pre-cache some
    await service.embed("a")
    await service.embed("b")
    # Batch includes cached + uncached
    results = await service.embed_batch(["a", "b", "c"])
    assert len(results) == 3
    assert "a" in service._cache
    assert "b" in service._cache
    assert "c" in service._cache


@pytest.mark.asyncio
async def test_service_cache_eviction():
    """Cache evicts old entries when exceeding max."""
    service = EmbeddingService(
        EmbeddingConfig(provider="mock", dimensions=16, cache_enabled=True)
    )
    service._cache_max = 10  # Small cache for testing
    # Fill cache beyond max
    for i in range(15):
        await service.embed(f"text_{i}")
    # Should have evicted some entries
    assert len(service._cache) <= 10


@pytest.mark.asyncio
async def test_service_unknown_provider_fallback():
    """Unknown provider falls back to mock."""
    service = EmbeddingService(EmbeddingConfig(provider="nonexistent", dimensions=64))
    v = await service.embed("test")
    assert len(v) == 64
    assert isinstance(service.provider, MockEmbeddingProvider)


@pytest.mark.asyncio
async def test_service_openai_fallback_no_key():
    """OpenAI provider with no key falls back to mock."""
    cfg = EmbeddingConfig(provider="openai", dimensions=128, api_key="")
    service = EmbeddingService(cfg)
    v = await service.embed("test")
    assert len(v) == 128
    assert isinstance(service.provider, MockEmbeddingProvider)


@pytest.mark.asyncio
async def test_service_default_config():
    """EmbeddingService with default config works."""
    service = EmbeddingService()
    v = await service.embed("any text")
    assert len(v) == 1536  # default dimension


# ────────────────────────────────────────────────────────────
# OpenAIEmbeddingProvider (mocked via unittest.mock)
# ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_openai_provider_embed():
    """OpenAI provider calls correct API endpoint."""
    import httpx
    from unittest.mock import AsyncMock, patch, MagicMock

    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [{"embedding": [0.1] * 128, "index": 0}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 2, "total_tokens": 2},
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-small", dimensions=128
        )
        # Reset _client so it creates a new one via the patched constructor
        provider._client = None
        v = await provider.embed("test")
        assert len(v) == 128
        assert v[0] == 0.1


@pytest.mark.asyncio
async def test_openai_provider_batch():
    """OpenAI provider batches correctly."""
    import httpx
    from unittest.mock import AsyncMock, patch, MagicMock

    expected = [[float(i) / 100] * 64 for i in range(3)]
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {"embedding": expected[0], "index": 0},
            {"embedding": expected[1], "index": 1},
            {"embedding": expected[2], "index": 2},
        ],
        "model": "text-embedding-3-small",
    }

    mock_client = AsyncMock()
    mock_client.post.return_value = mock_response

    with patch.object(httpx, "AsyncClient", return_value=mock_client):
        provider = OpenAIEmbeddingProvider(api_key="sk-test", dimensions=64)
        provider._client = None
        results = await provider.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert results[0] == expected[0]


# ────────────────────────────────────────────────────────────
# LocalEmbeddingProvider (skip if not installed)
# ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_local_provider_not_installed():
    """Local provider raises clear error when sentence-transformers missing."""
    try:
        import sentence_transformers  # noqa: F401
        pytest.skip("sentence-transformers is installed")
    except ImportError:
        pass

    # Even without the library, constructing the config works
    # (lazy init — only fails when embed() is called)
    provider = LocalEmbeddingProvider(model_name="BAAI/bge-large-zh-v1.5")
    assert provider.dimensions == 1024  # default before model load


# ────────────────────────────────────────────────────────────
# MemoryPool integration
# ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_memorypool_embed_with_service():
    """MemoryPool.embed() uses EmbeddingService when configured."""
    cfg = EmbeddingConfig(provider="mock", dimensions=64)
    pool = MemoryPool(db_path=":memory:", embedding_config=cfg)
    v = await pool.embed("hello")
    assert len(v) == 64
    assert pool.embedding_service is not None


@pytest.mark.asyncio
async def test_memorypool_embed_no_config():
    """MemoryPool without embedding config returns zero vector."""
    pool = MemoryPool(db_path=":memory:")
    v = await pool.embed("hello")
    assert v == [0.0] * 1536


@pytest.mark.asyncio
async def test_memorypool_embed_batch():
    """MemoryPool.embed_batch() works with service."""
    cfg = EmbeddingConfig(provider="mock", dimensions=64)
    pool = MemoryPool(db_path=":memory:", embedding_config=cfg)
    results = await pool.embed_batch(["a", "b", "c"])
    assert len(results) == 3
    for r in results:
        assert len(r) == 64


@pytest.mark.asyncio
async def test_memorypool_embed_batch_no_config():
    """MemoryPool.embed_batch() without config returns zero vectors."""
    pool = MemoryPool(db_path=":memory:")
    results = await pool.embed_batch(["a", "b"])
    assert results == [[0.0] * 1536, [0.0] * 1536]


@pytest.mark.asyncio
async def test_memorypool_configure_embedding_legacy():
    """Legacy configure_embedding() API still works."""
    pool = MemoryPool(db_path=":memory:")
    pool.configure_embedding(lambda text: [float(ord(c)) for c in text.ljust(10)[:10]])
    v = await pool.embed("abc")
    assert len(v) == 10
    assert v[0] == float(ord("a"))


@pytest.mark.asyncio
async def test_memorypool_service_overrides_legacy():
    """EmbeddingService takes priority over legacy configure_embedding."""
    cfg = EmbeddingConfig(provider="mock", dimensions=32)
    pool = MemoryPool(db_path=":memory:", embedding_config=cfg)
    pool.configure_embedding(lambda text: [1.0] * 32)
    # Service should be used, not legacy fn
    v = await pool.embed("hello")
    assert pool.embedding_service is not None
    # The mock provider gives deterministic vectors, not all 1.0
    assert v != [1.0] * 32


@pytest.mark.asyncio
async def test_memorypool_write_fact_with_service():
    """write_fact uses EmbeddingService when configured."""
    cfg = EmbeddingConfig(provider="mock", dimensions=64)
    pool = MemoryPool(db_path=":memory:", embedding_config=cfg)
    fact_id = await pool.write_fact("entity", "test_fact", value="hello")
    assert fact_id is not None
    # Fact should have a proper embedding (not zero vector)
    fact = await pool.get_fact("test_fact")
    assert fact is not None
    emb = fact.get("embedding", [])
    assert len(emb) == 64
    assert any(v != 0 for v in emb)  # Not all zeros


# ────────────────────────────────────────────────────────────
# Vector search with mock embedding
# ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_and_search_by_embedding():
    """Write facts with embeddings, then verify they're stored."""
    cfg = EmbeddingConfig(provider="mock", dimensions=64)
    pool = MemoryPool(db_path=":memory:", embedding_config=cfg)

    # Write several facts
    await pool.write_fact("entity", "apple", value="A fruit")
    await pool.write_fact("entity", "banana", value="Another fruit")
    await pool.write_fact("entity", "car", value="A vehicle")

    # All should have embeddings stored
    for name in ["apple", "banana", "car"]:
        fact = await pool.get_fact(name)
        assert fact is not None, f"Fact {name} not found"
        emb = fact.get("embedding", [])
        assert len(emb) == 64


@pytest.mark.asyncio
async def test_embedding_consistency():
    """Same text always produces same embedding via service."""
    cfg = EmbeddingConfig(provider="mock", dimensions=32)
    pool = MemoryPool(db_path=":memory:", embedding_config=cfg)

    v1 = await pool.embed("consistent text")
    v2 = await pool.embed("consistent text")
    v3 = await pool.embed("different text")

    assert v1 == v2
    assert v1 != v3


@pytest.mark.asyncio
async def test_empty_text_embedding():
    """Empty text embedding produces valid vector."""
    cfg = EmbeddingConfig(provider="mock", dimensions=64)
    pool = MemoryPool(db_path=":memory:", embedding_config=cfg)
    v = await pool.embed("")
    assert len(v) == 64
    # Should be a unit vector
    norm = math.sqrt(sum(x * x for x in v))
    assert math.isclose(norm, 1.0, rel_tol=1e-6)
