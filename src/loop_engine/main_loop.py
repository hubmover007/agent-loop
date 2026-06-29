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
from ..agent import AgentOrchestrator
from ..task import TaskTree, TaskScheduler

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

        # Will be set up lazily
        self.task_scheduler: TaskScheduler | None = None
        self.agent_orchestrator: AgentOrchestrator | None = None

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
        """Decompose reasoning output into TaskTree."""
        ctx.current_phase = LoopPhase.DECOMPOSE

        try:
            decompose_prompt = f"""Based on the analysis below, decompose the task into subtasks.

Analysis: {ctx.reason_output}

Output structured subtasks. Each subtask should be a concrete, actionable unit.
Respond as JSON array:
[
  {{
    "scope": "specific subtask description",
    "priority": 1-5 (5=highest),
    "required_tools": ["tool1", "tool2"],
    "dependencies": [] or ["other_subtask_id"]
  }}
]

If the task is already simple, return a single subtask array."""

            response = await self.llm.chat([{"role": "user", "content": decompose_prompt}])
            import json

            # Extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]

            subtasks = json.loads(content.strip())
            if not isinstance(subtasks, list):
                subtasks = [subtasks]

            # Register tasks in memory
            for i, st in enumerate(subtasks):
                task_id = f"task:{uuid.uuid4().hex[:12]}"
                await self.memory.register_task(
                    task_id=task_id,
                    parent_id=None,
                    scope=st["scope"],
                    priority=st.get("priority", 3),
                )
                ctx.task_ids.append(task_id)

            logger.info("MainLoop[%s]: DECOMPOSE → %d subtasks", ctx.session_id, len(subtasks))

        except Exception as e:
            logger.warning("Task decomposition failed: %s, using single task", e)
            task_id = f"task:{uuid.uuid4().hex[:12]}"
            await self.memory.register_task(
                task_id=task_id, parent_id=None, scope=ctx.user_input
            )
            ctx.task_ids.append(task_id)

    # ============================================================
    # 5. DISPATCH
    # ============================================================

    async def _dispatch(self, ctx: LoopContext) -> None:
        """Dispatch tasks to agents via the orchestrator."""
        ctx.current_phase = LoopPhase.DISPATCH

        if not self.agent_orchestrator:
            # Lazy init
            from ..agent import AgentOrchestrator
            self.agent_orchestrator = AgentOrchestrator(
                memory=self.memory,
                agent_loop=self.agent_loop,
                config=self.config,
            )

        for task_id in ctx.task_ids:
            task_data = await self.memory._db.select(task_id)
            if task_data:
                await self.agent_orchestrator.dispatch(
                    task_id=task_id,
                    scope=task_data.get("scope", ""),
                    priority=task_data.get("priority", 3),
                )

        logger.info("MainLoop[%s]: DISPATCH → %d agents", ctx.session_id, len(ctx.task_ids))

    # ============================================================
    # 6. COLLECT
    # ============================================================

    async def _collect(self, ctx: LoopContext) -> None:
        """Collect and evaluate agent results."""
        ctx.current_phase = LoopPhase.COLLECT

        if not self.agent_orchestrator:
            ctx.errors.append("No orchestrator available")
            return

        for task_id in ctx.task_ids:
            result = await self.agent_orchestrator.collect(task_id)
            if result:
                ctx.agent_results.append(result)
            else:
                ctx.discarded_results.append(task_id)
                ctx.errors.append(f"Task {task_id} had no valid result")

        logger.info(
            "MainLoop[%s]: COLLECT → %d results, %d discarded",
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
