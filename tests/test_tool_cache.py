"""Tests for P2 tool result caching in ToolRegistry."""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.tool_registry import ToolSpec, ToolRegistry


@pytest.fixture
def registry():
    reg = ToolRegistry()
    # Register a simple tool
    call_count = [0]  # mutable counter

    async def echo_handler(message: str) -> dict:
        call_count[0] += 1
        return {"echo": message, "count": call_count[0]}

    spec = ToolSpec(
        name="test.echo",
        namespace="test",
        description="Echo",
        input_schema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
        },
        handler=echo_handler,
        risk_level="low",
    )
    reg.register(spec)
    return reg


@pytest.mark.asyncio
async def test_cache_hit(registry):
    """Same arguments should hit the cache on second call."""
    args = {"message": "hello"}

    result1 = await registry.invoke("test.echo", args, use_cache=True)
    result2 = await registry.invoke("test.echo", args, use_cache=True)

    # Both should return the same echo (cached result)
    assert result1["echo"] == "hello"
    assert result2["echo"] == "hello"
    # The handler should only have been called once
    assert result1["count"] == result2["count"]


@pytest.mark.asyncio
async def test_cache_miss(registry):
    """Different arguments should NOT hit the cache."""
    args1 = {"message": "hello"}
    args2 = {"message": "world"}

    result1 = await registry.invoke("test.echo", args1, use_cache=True)
    result2 = await registry.invoke("test.echo", args2, use_cache=True)

    assert result1["echo"] == "hello"
    assert result2["echo"] == "world"
    # Different args → different cache keys → handler called twice
    assert result2["count"] > result1["count"]


@pytest.mark.asyncio
async def test_cache_disabled(registry):
    """When use_cache=False, cache should be bypassed."""
    args = {"message": "hello"}

    result1 = await registry.invoke("test.echo", args, use_cache=False)
    result2 = await registry.invoke("test.echo", args, use_cache=False)

    # Each call should increment the count
    assert result2["count"] > result1["count"]


@pytest.mark.asyncio
async def test_cache_ttl(registry):
    """Expired cache entries should be re-executed."""
    # Set TTL very low
    registry._cache_ttl = -1  # Immediately expired

    args = {"message": "hello"}
    result1 = await registry.invoke("test.echo", args, use_cache=True)
    result2 = await registry.invoke("test.echo", args, use_cache=True)

    # TTL expired → re-executed
    assert result2["count"] > result1["count"]


@pytest.mark.asyncio
async def test_clear_cache(registry):
    """clear_cache() should remove all cached entries."""
    args = {"message": "hello"}

    result1 = await registry.invoke("test.echo", args, use_cache=True)
    assert result1["count"] == 1

    registry.clear_cache()
    stats = registry.cache_stats()
    assert stats["size"] == 0

    # After clearing, next call should execute fresh
    result2 = await registry.invoke("test.echo", args, use_cache=True)
    assert result2["count"] == 2


@pytest.mark.asyncio
async def test_cache_stats(registry):
    """cache_stats() returns correct metadata."""
    await registry.invoke("test.echo", {"message": "a"}, use_cache=True)
    await registry.invoke("test.echo", {"message": "b"}, use_cache=True)

    stats = registry.cache_stats()
    assert stats["size"] == 2
    assert stats["enabled"] is True
    assert isinstance(stats["ttl"], int)


@pytest.mark.asyncio
async def test_cache_key_consistency():
    """Cache keys should be deterministic regardless of dict key order."""
    reg = ToolRegistry()
    key1 = reg._cache_key("test.echo", {"b": 1, "a": 2})
    key2 = reg._cache_key("test.echo", {"a": 2, "b": 1})
    assert key1 == key2
