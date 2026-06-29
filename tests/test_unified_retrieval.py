"""Tests for UnifiedRetriever (M-FLOW + Mythos integration)."""

import asyncio
import pytest
from src.memory.unified_retrieval import UnifiedRetriever, MemoryContext


def test_memory_context_creation():
    """MemoryContext can be created with defaults."""
    ctx = MemoryContext()
    assert ctx.explicit == []
    assert ctx.implicit == ""
    assert ctx.confidence == 0.0


def test_memory_context_to_prompt_empty():
    """Empty MemoryContext produces empty prompt."""
    ctx = MemoryContext()
    assert ctx.to_prompt() == ""


def test_memory_context_to_prompt_with_data():
    """MemoryContext with data produces formatted prompt."""
    ctx = MemoryContext(
        explicit=[
            {"layer": "fact", "title": "Server A", "summary": "Running"},
            {"layer": "episode", "title": "Last incident", "summary": "Fixed"},
        ],
        implicit="The server is stable based on recent patterns.",
    )
    prompt = ctx.to_prompt()
    assert "Retrieved from Memory Graph" in prompt
    assert "Server A" in prompt
    assert "Deep Recall" in prompt
    assert "stable" in prompt


def test_unified_retriever_class_exists():
    """UnifiedRetriever class is importable."""
    assert UnifiedRetriever is not None

    # Verify it has the expected interface
    methods = [m for m in dir(UnifiedRetriever) if not m.startswith('_')]
    assert 'retrieve' in methods
    assert 'store_episode' in methods
    assert 'consolidate' in methods
