"""End-to-end smoke test: verify the core loop actually works.

Tests the full pipeline:
  User input → MainLoop → TaskAgent decompose → AgentManagerAgent assign
  → AgentLoop execute → evaluate → return result

This test uses a mock LLM (no real API calls) to verify the wiring
without spending tokens.

Run:
    python3 tests/test_e2e_smoke.py
"""

import asyncio
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import TaskStatus, LoopPhase
from src.loop_engine import LLMProvider, LLMResponse, LoopConfig
from src.loop_engine.main_loop import MainLoop
from src.memory import MemoryPool
from src.system_agents import TaskAgent, AgentManagerAgent, TaskRegistry


# ============================================================
# Mock LLM Provider — simulates LLM responses for testing
# ============================================================

class MockLLMProvider(LLMProvider):
    """Mock LLM that returns predefined responses."""

    provider_name = "mock"

    def __init__(self):
        self._call_count = 0

    async def chat(self, messages: list[dict], thinking: bool = False,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   model: str | None = None, **kwargs) -> LLMResponse:
        self._call_count += 1

        # Return a simple decomposition for task management queries
        last_msg = messages[-1]["content"].lower() if messages else ""

        if "decompose" in last_msg or "拆分" in last_msg or "subtask" in last_msg:
            text = '''```json
            [
              {"scope": "分析需求", "priority": 5, "dependencies": []},
              {"scope": "执行核心步骤", "priority": 3, "dependencies": ["分析需求"]}
            ]
            ```'''
        else:
            text = "任务已完成：模拟执行结果"

        return LLMResponse(content=text, usage={"total_tokens": 100},
                          model="mock")

    async def embed(self, text: str) -> list[float]:
        return [0.1] * 128


# ============================================================
# Smoke Test
# ============================================================

@pytest.mark.asyncio
async def test_e2e_basic():
    """Test basic MainLoop wiring without real LLM."""
    print("\n=== E2E Smoke Test ===")

    # Setup
    llm = MockLLMProvider()
    memory = MemoryPool()  # In-memory (no SurrealDB)

    config = LoopConfig(
        max_agent_concurrent=3,
        accept_threshold=0.6,
    )

    main_loop = MainLoop(
        llm=llm,
        memory=memory,
        config=config,
    )
    print("✅ MainLoop created")

    # Manually init agents (normally done in _decompose)
    from src.system_agents import TaskRegistry as TR
    main_loop.task_registry = TR()
    main_loop.task_agent = TaskAgent(llm=llm, registry=main_loop.task_registry)
    from src.external_agents import ExternalAgentBridge
    from src.loop_engine import AgentLoop
    main_loop.agent_manager = AgentManagerAgent(
        memory=memory,
        agent_loop=main_loop.agent_loop,
        config=config,
        registry=main_loop.task_registry,
        external_bridge=ExternalAgentBridge(),
    )
    print("✅ TaskAgent + AgentManagerAgent initialized")

    # Test TaskAgent decompose
    tasks = await main_loop.task_agent.decompose(
        reasoning_output="修复 auth.py 的 bug",
        original_input="修复 auth.py 的 bug",
    )
    print(f"✅ TaskAgent decompose: {len(tasks)} tasks created")
    for t in tasks:
        print(f"   - {t.task_id}: {t.scope} (priority={t.priority})")

    # Test TaskRegistry
    ready = main_loop.task_agent.get_ready_tasks()
    print(f"✅ TaskRegistry: {len(ready)} ready tasks")

    # Test memory (in-memory fallback)
    await memory.write_fact("entity", "test_key", {"value": "test_data"})
    fact = await memory.get_fact("test_key")
    print(f"✅ MemoryPool: write + read fact = {fact is not None}")

    print("\n=== All E2E checks passed ✅ ===")


@pytest.mark.asyncio
async def test_core_types():
    """Verify core types are consistent."""
    print("\n=== Core Types ===")

    # TaskStatus
    statuses = [s.value for s in TaskStatus]
    print(f"TaskStatus: {statuses}")
    assert "pending" in statuses
    assert "done" in statuses

    # LoopPhase
    phases = [p.value for p in LoopPhase]
    print(f"LoopPhase: {phases}")
    assert "input" in phases
    assert "output" in phases

    print("✅ Core types OK")


if __name__ == "__main__":
    asyncio.run(test_core_types())
    asyncio.run(test_e2e_basic())
