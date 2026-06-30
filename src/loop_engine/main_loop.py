"""MainLoop - the outermost Loop Engine cycle.

INPUT → RETRIEVE → REASON → DECOMPOSE → DISPATCH → COLLECT → OUTPUT
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone

from . import LoopConfig, LoopContext, AgentLoop, ToolLoop, LLMProvider
from .deep_reason import DeepReasonLoop, DeepReasonConfig
from ..core import LoopPhase, TaskStatus, ToolResult, DiscardRecord
from ..memory import MemoryPool
from ..memory.graph_route import GraphRouter
from ..system_agents import TaskAgent, AgentManagerAgent, TaskRegistry
from ..memory.unified_retrieval import UnifiedRetriever, MemoryContext

logger = logging.getLogger(__name__)


class MainLoop:
    """The outermost Loop Engine that orchestrates the full cycle."""

    def __init__(self, memory: MemoryPool, llm: LLMProvider, config: LoopConfig | None = None):
        self.memory = memory
        self.llm = llm
        self.config = config or LoopConfig()

        # Sub-systems
        from ..tools.base import ToolRegistry
        self.tool_registry = ToolRegistry()
        self.tool_registry.register_defaults()
        self.tool_loop = ToolLoop(self.tool_registry, self.config)
        self.graph_router = GraphRouter(memory)  # M-FLOW graph routing
        self.agent_loop = AgentLoop(self.tool_loop, llm, self.config)
        self.deep_reason = DeepReasonLoop(llm, DeepReasonConfig(
            max_iterations=self.config.max_reason_loops,
            confidence_threshold=self.config.reason_confidence_threshold,
        ))

        # Unified Memory Retriever: M-FLOW graph + Mythos deep reasoning
        self.retriever = UnifiedRetriever(
            memory_pool=memory,
            graph_router=self.graph_router,
            deep_reason=self.deep_reason,
            llm=llm,
        )

        # System Agents: TaskAgent (manages tasks) + AgentManagerAgent (manages agents)
        self.task_agent: TaskAgent | None = None
        self.agent_manager: AgentManagerAgent | None = None
        self.task_registry: TaskRegistry | None = None

    # ============================================================
    # 1. INPUT
    # ============================================================

    async def _input(self, ctx: LoopContext) -> None:
        """Parse and validate user input."""
        ctx.current_phase = LoopPhase.INPUT
        logger.info("MainLoop[%s]: INPUT '%s'", ctx.session_id, ctx.user_input[:50])

        if not ctx.user_input.strip():
            ctx.errors.append("Empty input")
            ctx.final_output = "I didn't receive any input. Can you try again?"

    # ============================================================
    # 2. RETRIEVE
    # ============================================================

    async def _retrieve(self, ctx: LoopContext) -> None:
        """Retrieve unified memory context: M-FLOW graph + Mythos deep recall.

        This is the systemic integration point:
          1. GraphRouter queries explicit knowledge graph (M-FLOW)
          2. Results feed into DeepReason for implicit recall (Mythos)
          3. Unified MemoryContext returned with both layers
        """
        ctx.current_phase = LoopPhase.RETRIEVE

        try:
            # Unified retrieval: explicit graph + implicit deep reasoning
            mem_ctx = await self.retriever.retrieve(
                query=ctx.user_input,
                max_hops=3,
                deep_reason_iterations=3,
            )

            # Store unified context for REASON phase
            ctx.memory_context = mem_ctx
            ctx.retrieved_context = [
                {
                    "title": item.get("title", ""),
                    "summary": item.get("summary", ""),
                    "layer": item.get("layer", ""),
                }
                for item in mem_ctx.explicit
            ]

            logger.info(
                "MainLoop[%s]: RETRIVE graph=%d items, reason_iter=%d conf=%.2f",
                ctx.session_id, len(mem_ctx.explicit),
                mem_ctx.reason_iterations, mem_ctx.confidence
            )

        except Exception as e:
            logger.warning("Unified retrieval failed: %s", e)
            ctx.retrieved_context = []
            ctx.errors.append(f"Retrieval degraded: {e}")

    # ============================================================
    # 3. REASON
    # ============================================================

    async def _reason(self, ctx: LoopContext) -> None:
        """Deep reasoning: RDT-style loop via DeepReasonLoop.

        Hybrid mode (C: explicit loop + implicit latent):
          - Internal: model's native latent reasoning (thinking=true)
          - External: iterative refinement with ACT adaptive depth

        Memory integration:
          - Uses unified MemoryContext from _retrieve() (M-FLOW + Mythos)
          - Graph facts provide anchors for reasoning
          - DeepReason activates parametric knowledge
        """
        ctx.current_phase = LoopPhase.REASON

        # Use unified memory context (explicit graph + implicit recall)
        mem_ctx = getattr(ctx, 'memory_context', None)
        if mem_ctx and mem_ctx.explicit:
            context_text = mem_ctx.to_prompt()
        else:
            # Fallback: build from retrieved_context
            context_text = "\n".join(
                f"[{rc.get('layer', '?')}] {rc.get('title', '')}: {rc.get('summary', '')}"
                for rc in ctx.retrieved_context[:3]
            )

        try:
            state = await self.deep_reason.reason(
                query=ctx.user_input,
                context=context_text,
            )

            ctx.reason_iterations = state.iteration
            ctx.reason_confidence = state.confidence
            ctx.reason_output = state.current_thought
            ctx.thought_chain = [state.current_thought]

            logger.info(
                "MainLoop[%s]: REASON completed (iter=%d, conf=%.2f, insights=%d)",
                ctx.session_id, state.iteration, state.confidence, len(state.insights)
            )

        except Exception as e:
            logger.error("DeepReason failed: %s, falling back to simple reasoning", e)
            ctx.reason_output = ctx.user_input
            ctx.errors.append(f"Reasoning degraded: {e}")

    # ============================================================
    # 4. DECOMPOSE
    # ============================================================

    async def _decompose(self, ctx: LoopContext) -> None:
        """TaskAgent decomposes reasoning into subtasks."""
        ctx.current_phase = LoopPhase.DECOMPOSE

        if not self.task_registry:
            self.task_registry = TaskRegistry()
        if not self.task_agent:
            self.task_agent = TaskAgent(
                llm=self.llm,
                registry=self.task_registry,
            )
        if not self.agent_manager:
            from ..external_agents import ExternalAgentBridge
            self.agent_manager = AgentManagerAgent(
                memory=self.memory,
                agent_loop=self.agent_loop,
                config=self.config,
                registry=self.task_registry,
                external_bridge=ExternalAgentBridge(),
            )

        # TaskAgent uses LLM to decompose + register tasks
        tasks = await self.task_agent.decompose(
            reasoning_output=ctx.reason_output,
            original_input=ctx.user_input,
        )

        ctx.task_ids = [t.task_id for t in tasks]
        logger.info("MainLoop[%s]: DECOMPOSE → %d tasks", ctx.session_id, len(tasks))

    # ============================================================
    # 5. DISPATCH
    # ============================================================

    async def _dispatch(self, ctx: LoopContext) -> None:
        """AgentManagerAgent assigns ready tasks to worker agents.

        TaskAgent provides ready tasks (deps satisfied).
        AgentManagerAgent creates/assigns workers and launches execution.
        """
        ctx.current_phase = LoopPhase.DISPATCH

        if not self.task_agent or not self.agent_manager:
            ctx.errors.append("System agents not initialized")
            return

        # TaskAgent provides ready tasks
        ready = self.task_agent.get_ready_tasks()

        # AgentManagerAgent assigns each to a worker
        dispatched = 0
        for task in ready:
            if await self.agent_manager.assign(task):
                dispatched += 1

        logger.info("MainLoop[%s]: DISPATCH → %d workers launched",
                    ctx.session_id, dispatched)

    # ============================================================
    # 6. COLLECT
    # ============================================================

    async def _collect(self, ctx: LoopContext) -> None:
        """AgentManagerAgent collects results from all worker agents.

        Waits for all in-flight tasks, gathers results.
        Failed tasks may trigger re-planning by TaskAgent.
        """
        ctx.current_phase = LoopPhase.COLLECT

        if not self.agent_manager or not self.task_agent:
            ctx.errors.append("System agents not initialized")
            return

        # AgentManagerAgent waits for all workers to finish
        results = await self.agent_manager.collect_all(timeout=300.0)

        for task_id, result in results.items():
            ctx.agent_results.append(result)

        # Check for failed tasks — TaskAgent can re-plan
        for task in self.task_registry.all_tasks():
            if task.status == TaskStatus.FAILED:
                ctx.discarded_results.append(task.task_id)
                ctx.errors.append(f"Task {task.task_id} failed: {task.error}")

                # TaskAgent re-plans failed tasks
                # new_tasks = await self.task_agent.replan(task)
                # if new_tasks:
                #     for nt in new_tasks:
                #         if await self.agent_manager.assign(nt):
                #             pass

        logger.info(
            "MainLoop[%s]: COLLECT → %d results, %d failed",
            ctx.session_id, len(ctx.agent_results), len(ctx.discarded_results)
        )

    # ============================================================
    # 7. OUTPUT
    # ============================================================

    async def _output(self, ctx: LoopContext) -> None:
        """Synthesize final output from all results."""
        ctx.current_phase = LoopPhase.OUTPUT

        if not ctx.agent_results:
            ctx.final_output = "\n".join(ctx.errors) if ctx.errors else "No results produced."
            # Still write an episode even with no results
            try:
                await self.memory.store({
                    "type": "episode",
                    "title": f"Session: {ctx.user_input[:80]}",
                    "user_input": ctx.user_input,
                    "output": ctx.final_output,
                    "task_count": len(ctx.task_ids),
                    "session_id": ctx.session_id,
                    "tags": ["session", "empty"],
                })
            except Exception as e:
                logger.warning("Failed to write episode: %s", e)
            return

        # Synthesize results
        parts = []
        for result in ctx.agent_results:
            parts.append(f"### {result.summary}")

        ctx.final_output = "\n\n".join(parts)
        if ctx.errors:
            ctx.final_output += "\n\n⚠️ Issues encountered:\n" + "\n".join(f"- {e}" for e in ctx.errors)

        # Write a new Episode to memory for future retrieval
        try:
            await self.memory.store({
                "type": "episode",
                "title": f"Session: {ctx.user_input[:80]}",
                "user_input": ctx.user_input,
                "output": ctx.final_output,
                "summary": ctx.final_output[:200],
                "content": ctx.reason_output,
                "task_count": len(ctx.task_ids),
                "session_id": ctx.session_id,
                "tags": ["session", datetime.now(timezone.utc).strftime("%Y-%m-%d")],
            })
        except Exception as e:
            logger.warning("Failed to write episode: %s", e)

        logger.info("MainLoop[%s]: OUTPUT completed", ctx.session_id)

    # ============================================================
    # Main Loop
    # ============================================================

    async def run(self, user_input: str) -> str:
        """Execute the complete MainLoop cycle.

        Args:
            user_input: The user's message

        Returns:
            The final response text
        """
        ctx = LoopContext(user_input=user_input)
        logger.info("=" * 60)
        logger.info("MainLoop[%s]: START", ctx.session_id)

        phases = [
            self._input,
            self._retrieve,
            self._reason,
            self._decompose,
            self._dispatch,
            self._collect,
            self._output,
        ]

        for phase in phases:
            try:
                await phase(ctx)
            except Exception as e:
                logger.error("Phase %s failed: %s", phase.__name__, e)
                ctx.errors.append(f"Error in {phase.__name__}: {e}")

        logger.info("MainLoop[%s]: END", ctx.session_id)
        return ctx.final_output
