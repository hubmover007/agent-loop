"""MainLoop - the outermost Loop Engine cycle.

INPUT → RETRIEVE → REASON → DECOMPOSE → DISPATCH → COLLECT → OUTPUT
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime

from . import LoopConfig, LoopContext, AgentLoop, ToolLoop, LLMProvider
from .deep_reason import DeepReasonLoop, DeepReasonConfig
from ..core import LoopPhase, TaskStatus, ToolResult, DiscardRecord
from ..memory import MemoryPool
from ..memory.graph_route import GraphRouter
from ..task_manager import TaskManagerAgent

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
        self.tool_loop = ToolLoop(self.tool_registry, self.config)
        self.graph_router = GraphRouter(memory)  # M-FLOW graph routing
        self.agent_loop = AgentLoop(self.tool_loop, llm, self.config)
        self.deep_reason = DeepReasonLoop(llm, DeepReasonConfig(
            max_iterations=self.config.max_reason_loops,
            confidence_threshold=self.config.reason_confidence_threshold,
        ))

        # TaskManager Agent (central orchestrator for task lifecycle)
        self.task_manager: TaskManagerAgent | None = None

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
        """Retrieve relevant context from Memory Pool using graph-routed search."""
        ctx.current_phase = LoopPhase.RETRIEVE

        try:
            # Use M-FLOW style graph routing
            results = await self.graph_router.retrieve(ctx.user_input, top_k=5)

            ctx.retrieved_context = [
                {
                    "episode_id": r.episode_id,
                    "title": r.episode_data.get("title", ""),
                    "summary": r.episode_data.get("summary", ""),
                    "score": r.score,
                    "hops": r.path.hops,
                }
                for r in results
            ]
            logger.info("MainLoop[%s]: RETRIEVE found %d episodes", ctx.session_id, len(results))

        except Exception as e:
            logger.warning("Graph route retrieval failed: %s, falling back to keyword search", e)
            ctx.retrieved_context = []
            ctx.errors.append(f"Retrieval degraded: {e}")

    # ============================================================
    # 3. REASON
    # ============================================================

    async def _reason(self, ctx: LoopContext) -> None:
        """Deep reasoning: RDT-style loop via DeepReasonLoop.

        Hybrid mode:
          - Internal: model's native latent reasoning (thinking=true)
          - External: iterative refinement with ACT adaptive depth
        """
        ctx.current_phase = LoopPhase.REASON

        # Build context from retrieved memories
        context_text = ""
        for rc in ctx.retrieved_context[:3]:
            context_text += f"\n[Relevant: {rc['title']}]\n{rc['summary']}\n"

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
        """TaskManager Agent decomposes reasoning into subtasks."""
        ctx.current_phase = LoopPhase.DECOMPOSE

        if not self.task_manager:
            self.task_manager = TaskManagerAgent(
                memory=self.memory,
                llm=self.llm,
                agent_loop=self.agent_loop,
                config=self.config,
            )

        # TaskManager uses LLM to decompose + register tasks
        tasks = await self.task_manager.decompose(
            reasoning_output=ctx.reason_output,
            original_input=ctx.user_input,
        )

        ctx.task_ids = [t.task_id for t in tasks]
        logger.info("MainLoop[%s]: DECOMPOSE → %d tasks", ctx.session_id, len(tasks))

    # ============================================================
    # 5. DISPATCH
    # ============================================================

    async def _dispatch(self, ctx: LoopContext) -> None:
        """TaskManager Agent dispatches ready tasks to worker agents.

        The TaskManager handles:
          - Dependency-aware scheduling
          - Worker agent routing (MoE)
          - Async execution launch
        """
        ctx.current_phase = LoopPhase.DISPATCH

        if not self.task_manager:
            ctx.errors.append("TaskManager not initialized")
            return

        dispatched = await self.task_manager.dispatch_ready()
        logger.info("MainLoop[%s]: DISPATCH → %d workers launched",
                    ctx.session_id, len(dispatched))

    # ============================================================
    # 6. COLLECT
    # ============================================================

    async def _collect(self, ctx: LoopContext) -> None:
        """TaskManager Agent collects results from all worker agents.

        Waits for all in-flight tasks, gathers results.
        Failed tasks may trigger re-planning by the TaskManager.
        """
        ctx.current_phase = LoopPhase.COLLECT

        if not self.task_manager:
            ctx.errors.append("TaskManager not initialized")
            return

        # Wait for all tasks to complete
        results = await self.task_manager.collect_all(timeout=300.0)

        for task_id, result in results.items():
            ctx.agent_results.append(result)

        # Check for failed tasks — attempt re-planning
        for task in self.task_manager.registry.all_tasks():
            if task.status == TaskStatus.FAILED:
                ctx.discarded_results.append(task.task_id)
                ctx.errors.append(f"Task {task.task_id} failed: {task.error}")

                # TaskManager can re-plan failed tasks
                # new_tasks = await self.task_manager.replan(task)
                # if new_tasks:
                #     await self.task_manager.dispatch_ready()

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
            return

        # Synthesize results
        parts = []
        for result in ctx.agent_results:
            parts.append(f"### {result.summary}")

        # Write a new Episode to memory for future retrieval
        try:
            await self.memory.write_episode(
                title=f"Session: {ctx.user_input[:80]}",
                summary="\n\n".join(parts),
                content=ctx.reason_output,
                tags=["session", datetime.now().strftime("%Y-%m-%d")],
            )
        except Exception as e:
            logger.warning("Failed to write episode: %s", e)

        ctx.final_output = "\n\n".join(parts)
        if ctx.errors:
            ctx.final_output += "\n\n⚠️ Issues encountered:\n" + "\n".join(f"- {e}" for e in ctx.errors)

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
