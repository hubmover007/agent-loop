"""End-to-end full integration test: complete 7-phase MainLoop with MockLLM.

Validates:
  1. MainLoop runs through all 7 phases without errors
  2. ctx.final_output is non-empty
  3. MemoryPool contains a new episode record
  4. DECOMPOSE produced >= 1 task
  5. LLM-driven self-evaluation returns a valid score
  6. LLM-driven summary generation works
  7. Memory consolidation extracts facts from episodes
"""

import asyncio
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import TaskStatus, LoopPhase
from src.loop_engine import LLMProvider, LLMResponse, LoopConfig, LoopContext, AgentLoop, ToolLoop
from src.loop_engine.main_loop import MainLoop
from src.memory import MemoryPool
from src.system_agents import TaskAgent, AgentManagerAgent, TaskRegistry


# ============================================================
# Mock LLM Provider — content-aware responses for full E2E
# ============================================================

class MockLLMProvider(LLMProvider):
    """Mock LLM that returns different JSON responses based on prompt keywords.

    Supports all the prompt types used across the full main loop:
      - Execution planner → returns plan JSON
      - Task decomposition → returns subtask JSON
      - Quality evaluation → returns {"score": 0.9, "reason": "good"}
      - Summarization → returns natural language summary
      - Triple extraction → returns entity-relation triples
      - Generic/other → returns a generic response
    """

    provider_name = "mock"

    def __init__(self):
        self._call_count = 0
        self._call_log: list[str] = []

    async def chat(self, messages: list[dict], thinking: bool = False,
                   max_tokens: int = 4096, temperature: float = 0.7,
                   model: str | None = None, **kwargs) -> LLMResponse:
        self._call_count += 1

        last_msg = messages[-1]["content"].lower() if messages else ""
        self._call_log.append(f"call_{self._call_count}: {last_msg[:80]}")

        # Quality evaluator prompt
        if "quality evaluator" in last_msg or "evaluate" in last_msg:
            text = '{"score": 0.9, "reason": "All steps completed successfully without errors"}'
            return LLMResponse(content=text, usage={"total_tokens": 50}, model="mock")

        # Summarization prompt
        if "summarize" in last_msg or "summary" in last_msg:
            text = "Successfully executed the plan: analyzed the problem, applied fix, and verified results."
            return LLMResponse(content=text, usage={"total_tokens": 50}, model="mock")

        # Execution planner (AgentLoop._plan)
        if "execution planner" in last_msg or "step-by-step plan" in last_msg:
            text = '''```json
[
  {"description": "Analyze task requirements", "tool": null, "params": {}, "output_key": null},
  {"description": "Execute core logic", "tool": null, "params": {}, "output_key": "result"},
  {"description": "Verify output", "tool": null, "params": {}, "output_key": null}
]
```'''
            return LLMResponse(content=text, usage={"total_tokens": 80}, model="mock")

        # Task decomposition (TaskAgent._llm_decompose)
        if "decompose" in last_msg or "拆分" in last_msg or "subtask" in last_msg or "task agent" in last_msg:
            text = '''```json
[
  {"scope": "Analyze and understand the problem", "priority": 5, "required_tools": [], "dependencies": []},
  {"scope": "Implement the solution", "priority": 3, "required_tools": [], "dependencies": ["Analyze and understand the problem"]},
  {"scope": "Verify and validate results", "priority": 2, "required_tools": [], "dependencies": ["Implement the solution"]}
]
```'''
            return LLMResponse(content=text, usage={"total_tokens": 100}, model="mock")

        # Triple extraction (consolidate)
        if "entity-relation triple" in last_msg or "extract triples" in last_msg.lower() or "extract triples" in last_msg.lower():
            text = '''[
  {"entity": "server", "relation": "has_status", "target": "online"},
  {"entity": "auth_bug", "relation": "fixed_in", "target": "auth.py"}
]'''
            return LLMResponse(content=text, usage={"total_tokens": 60}, model="mock")

        # Re-plan prompt
        if "re-plan" in last_msg or "failed" in last_msg:
            text = '''```json
[
  {"scope": "Retry with alternative approach", "priority": 3, "required_tools": [], "dependencies": []}
]
```'''
            return LLMResponse(content=text, usage={"total_tokens": 60}, model="mock")

        # DeepReason prompt (generic analysis)
        if "deep reasoning" in last_msg or "thoroughly" in last_msg or "re-examine" in last_msg:
            text = "The problem requires analysis of code, followed by implementation and verification. Confidence high."
            return LLMResponse(content=text, usage={"total_tokens": 40}, model="mock")

        # Generic fallback
        text = "Task completed: simulated execution result."
        return LLMResponse(content=text, usage={"total_tokens": 30}, model="mock")

    async def embed(self, text: str | list[str], model: str | None = None) -> list[list[float]]:
        texts = [text] if isinstance(text, str) else text
        return [[0.1] * 128 for _ in texts]


# ============================================================
# E2E Full MainLoop Test
# ============================================================

@pytest.mark.asyncio
async def test_e2e_full_mainloop():
    """Test the complete MainLoop cycle: 7 phases with MockLLM."""
    print("\n=== E2E Full MainLoop Test ===")

    llm = MockLLMProvider()
    memory = MemoryPool()

    config = LoopConfig(
        max_agent_concurrent=3,
        accept_threshold=0.6,
        max_reason_loops=2,
        reason_confidence_threshold=0.7,
    )

    main_loop = MainLoop(
        llm=llm,
        memory=memory,
        config=config,
    )
    print("✅ MainLoop created")

    # Manually init agents (normally done in _decompose)
    main_loop.task_registry = TaskRegistry()
    main_loop.task_agent = TaskAgent(llm=llm, registry=main_loop.task_registry)

    from src.external_agents import ExternalAgentBridge
    main_loop.agent_manager = AgentManagerAgent(
        memory=memory,
        agent_loop=main_loop.agent_loop,
        config=config,
        registry=main_loop.task_registry,
        external_bridge=ExternalAgentBridge(),
    )
    print("✅ TaskAgent + AgentManagerAgent initialized")

    # ---- ACT: Run the full main loop ----
    result = await main_loop.run("Fix the auth.py bug in production")
    print(f"✅ MainLoop.run() completed")

    # ---- ASSERTIONS ----

    # 1. result is a LoopContext with final_output
    assert result is not None, "result should not be None"
    from src.loop_engine import LoopContext
    assert isinstance(result, LoopContext), f"result should be LoopContext, got {type(result)}"
    assert result.final_output, "final_output should not be empty"
    print(f"✅ final_output: {result.final_output[:100]}...")

    # 2. LLM was called multiple times (at least for decompose + plan + self-eval + summary)
    assert llm._call_count > 0, "LLM should have been called at least once"
    print(f"✅ LLM call count: {llm._call_count}")

    # 3. DECOMPOSE produced >= 1 task
    task_count = len(result.task_ids)
    assert task_count >= 1, f"DECOMPOSE should produce >= 1 task, got {task_count}"
    print(f"✅ Tasks created: {task_count}")

    # 4. MemoryPool has a new episode
    episodes = memory._mem.get("episode", [])
    assert len(episodes) >= 1, f"Expected >= 1 episode in memory, got {len(episodes)}"
    ep = episodes[-1]
    assert ep.get("type") == "episode"
    assert ep.get("user_input") == "Fix the auth.py bug in production"
    assert ep.get("output") is not None
    assert ep.get("session_id") is not None
    print(f"✅ Memory has {len(episodes)} episode(s)")


@pytest.mark.asyncio
async def test_e2e_memory_consolidation():
    """Test the memory consolidation pipeline: episodes → facts."""
    print("\n=== E2E Memory Consolidation Test ===")

    llm = MockLLMProvider()
    memory = MemoryPool()

    from src.memory.unified_retrieval import UnifiedRetriever
    from src.memory.graph_route import GraphRouter
    from src.loop_engine.deep_reason import DeepReasonLoop, DeepReasonConfig

    config = LoopConfig()
    graph_router = GraphRouter(memory)
    deep_reason = DeepReasonLoop(llm, DeepReasonConfig(
        max_iterations=1,
        confidence_threshold=0.7,
    ))

    retriever = UnifiedRetriever(
        memory_pool=memory,
        graph_router=graph_router,
        deep_reason=deep_reason,
        llm=llm,
    )

    # Store some episodes
    await memory.store({
        "type": "episode",
        "title": "Session: Fix auth.py bug",
        "user_input": "Fix the auth.py bug in production",
        "output": "Bug fixed: null check added to auth.py:42",
        "task_count": 3,
        "session_id": "test-session-001",
        "tags": ["session", "2026-06-30"],
    })

    await memory.store({
        "type": "episode",
        "title": "Session: Server health check",
        "user_input": "Check server health status",
        "output": "Server is running, all services online",
        "task_count": 1,
        "session_id": "test-session-002",
        "tags": ["session", "2026-06-30"],
    })
    print(f"✅ 2 episodes stored in memory")

    # Run consolidation
    stats = await retriever.consolidate()
    print(f"✅ Consolidation stats: {stats}")

    # Assertions
    assert stats["episodes_processed"] >= 1, "Should process at least 1 episode"
    assert stats["facts_created"] >= 1, "Should create at least 1 fact"

    # Check that episodes are now marked consolidated
    for ep in memory._mem.get("episode", []):
        assert ep.get("consolidated") == True, f"Episode {ep.get('id')} should be consolidated"

    # Check that facts were written
    facts = memory._mem.get("fact", [])
    assert len(facts) >= 1, f"Should have at least 1 fact, got {len(facts)}"
    print(f"✅ {len(facts)} facts in memory after consolidation")


@pytest.mark.asyncio
async def test_e2e_agent_loop_eval_summary():
    """Test that AgentLoop._self_evaluate and _generate_summary use LLM."""
    print("\n=== E2E AgentLoop Self-Eval & Summary Test ===")

    llm = MockLLMProvider()
    config = LoopConfig(accept_threshold=0.6)

    from src.tools.base import ToolRegistry
    tool_registry = ToolRegistry()
    tool_registry.register_defaults()
    tool_loop = ToolLoop(tool_registry, config)

    agent_loop = AgentLoop(tool_loop, llm, config)

    from src.core import StepLog
    steps = [
        StepLog(step=1, action="analyze", tool_name=None),
        StepLog(step=2, action="execute core logic", tool_name=None),
        StepLog(step=3, action="verify output", tool_name=None),
    ]
    artifacts = {"result": "test_output"}

    # Test _self_evaluate (LLM path)
    score = await agent_loop._self_evaluate("Test task", steps, artifacts)
    print(f"✅ LLM self-eval score: {score}")
    assert 0.0 <= score <= 1.0, f"Score should be between 0 and 1, got {score}"
    assert score >= 0.5, f"Score should be reasonable (>= 0.5), got {score}"

    # Test _generate_summary (LLM path)
    summary = await agent_loop._generate_summary("Test task", steps, artifacts)
    print(f"✅ LLM summary: {summary}")
    assert len(summary) > 0, "Summary should not be empty"
    assert "Test task" in summary or "Successfully" in summary or "analyzed" in summary.lower()


@pytest.mark.asyncio
async def test_e2e_eval_fallback():
    """Test that self-evaluation falls back to heuristics on LLM failure."""
    print("\n=== E2E Eval Fallback Test ===")

    from src.loop_engine import LLMResponse

    # Mock LLM that always fails
    class FailingMockLLM(LLMProvider):
        provider_name = "failing-mock"
        async def chat(self, messages, **kwargs):
            raise RuntimeError("LLM unavailable")
        async def embed(self, text):
            return [[0.0] * 128]

    llm = FailingMockLLM()
    config = LoopConfig()

    from src.tools.base import ToolRegistry
    tool_registry = ToolRegistry()
    tool_loop = ToolLoop(tool_registry, config)

    agent_loop = AgentLoop(tool_loop, llm, config)

    from src.core import StepLog
    steps = [
        StepLog(step=1, action="analyze", tool_name=None),
        StepLog(step=2, action="execute", tool_name=None),
    ]
    artifacts = {}

    # Should fall back to heuristics without raising
    score = await agent_loop._self_evaluate("Test task", steps, artifacts)
    print(f"✅ Fallback heuristic score: {score}")
    assert 0.0 <= score <= 1.0, f"Fallback score should be between 0 and 1, got {score}"

    # Summary should also fallback gracefully
    summary = await agent_loop._generate_summary("Test task", steps, artifacts)
    print(f"✅ Fallback summary: {summary}")
    assert len(summary) > 0, "Fallback summary should not be empty"


if __name__ == "__main__":
    asyncio.run(test_e2e_full_mainloop())
    asyncio.run(test_e2e_memory_consolidation())
    asyncio.run(test_e2e_agent_loop_eval_summary())
    asyncio.run(test_e2e_eval_fallback())
    print("\n=== ALL E2E TESTS PASSED ===")
