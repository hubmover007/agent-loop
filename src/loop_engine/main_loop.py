"""MainLoop - the outermost Loop Engine cycle.

INPUT → RETRIEVE → REASON → DECOMPOSE → DISPATCH → COLLECT → OUTPUT
"""

from __future__ import annotations

import asyncio
import logging
import time
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

# Phase timeout defaults (seconds)
PHASE_TIMEOUTS: dict[str, float] = {
    "_input": 5.0,
    "_retrieve": 15.0,
    "_reason": 20.0,
    "_decompose": 15.0,
    "_dispatch": 10.0,
    "_collect": 60.0,
    "_output": 10.0,
}

# Keywords that indicate a complex query requiring full pipeline
COMPLEX_KEYWORDS: list[str] = [
    "搜索", "分析", "写代码", "部署", "创建", "修复", "生成",
    "search", "analyze", "code", "deploy", "create", "fix", "generate",
    "build", "run", "execute", "compile", "debug", "refactor",
]


class MainLoop:
    """The outermost Loop Engine that orchestrates the full cycle."""

    def __init__(self, memory: MemoryPool, llm: LLMProvider,
                 config: LoopConfig | None = None,
                 state_store: Any | None = None,
                 tracer: Any | None = None,
                 anchor_manager: Any | None = None):
        self.memory = memory
        self.llm = llm
        self.config = config or LoopConfig()
        self.state_store = state_store
        self.tracer = tracer  # Optional Tracer for distributed tracing
        self.anchor_manager = anchor_manager  # Optional AnchorManager for O(1) key-fact lookup

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

    def _check_anchors(self, query: str) -> dict[str, str] | None:
        """Check if query matches any anchor key.

        Simple heuristic: if query contains an anchor name or any entry key,
        return the matching anchor file contents keyed by anchor name.
        """
        if not self.anchor_manager:
            return None

        query_lower = query.lower()
        hits: dict[str, str] = {}

        for name in self.anchor_manager.list_anchors():
            anchor = self.anchor_manager.read_anchor(name)
            if not anchor:
                continue

            # Check if query mentions the anchor name
            name_match = (
                name.replace("-", "_") in query_lower
                or name.replace("_", "-") in query_lower
                or name.replace("-", " ") in query_lower
            )

            # Check if query mentions any entry key
            key_match = any(
                e.key.lower() in query_lower or e.value.lower() in query_lower
                for e in anchor.entries
            )

            if name_match or key_match:
                hits[name] = anchor.to_markdown()

        return hits if hits else None

    async def _retrieve(self, ctx: LoopContext) -> None:
        """Retrieve unified memory context: M-FLOW graph + Mythos deep recall.

        This is the systemic integration point:
          1. Anchor layer: O(1) precise lookup for stable key facts
          2. GraphRouter queries explicit knowledge graph (M-FLOW)
          3. Results feed into DeepReason for implicit recall (Mythos)
          4. Unified MemoryContext returned with both layers
        """
        ctx.current_phase = LoopPhase.RETRIEVE

        # Step 0: Check anchor layer (precise O(1) lookup for stable key facts)
        if self.anchor_manager:
            anchor_hits = self._check_anchors(ctx.user_input)
            if anchor_hits:
                ctx.anchor_context = anchor_hits
                logger.info(
                    "MainLoop[%s]: RETRIEVE anchor hit → %s",
                    ctx.session_id, list(anchor_hits.keys()),
                )

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

        # AgentManagerAgent waits for all workers to finish (limited to 30s per phase + extra asyncio.wait_for)
        results = await self.agent_manager.collect_all(timeout=30.0)

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
            # Save session state
            await self._save_session(ctx)
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

        # Save session state
        await self._save_session(ctx)

        logger.info("MainLoop[%s]: OUTPUT completed", ctx.session_id)

    # ============================================================
    # Simple Run (fast path for trivial questions)
    # ============================================================

    async def _simple_run(self, ctx: LoopContext,
                          task_handle: Any = None) -> LoopContext:
        """Fast path for simple queries: direct LLM response.

        Skips: REASON, DECOMPOSE, DISPATCH, COLLECT phases.
        Only runs: INPUT (no-op) + minimal RETRIEVE + direct LLM call.

        Returns in <5 seconds for trivial questions like "1+1=?".
        """
        t0 = time.time()
        logger.info("MainLoop[%s]: SIMPLE_RUN for '%s'", ctx.session_id, ctx.user_input[:50])

        async def _check():
            if task_handle:
                await task_handle.wait_if_paused()
                return await task_handle.check_cancelled()
            return False

        # Phase: INPUT
        if await _check():
            ctx.final_output = "任务已取消"
            return ctx

        # Phase: RETRIEVE (lightweight — skip DeepReason in retriever)
        try:
            ctx.current_phase = LoopPhase.RETRIEVE
            mem_ctx = await asyncio.wait_for(
                self.retriever.retrieve(
                    query=ctx.user_input,
                    max_hops=2,
                    deep_reason_iterations=0,  # No deep reasoning for simple queries
                ),
                timeout=10.0,
            )
            ctx.memory_context = mem_ctx
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("Simple retrieve skipped: %s", e)

        if await _check():
            ctx.final_output = "任务已取消"
            return ctx

        # Build concise context
        context_text = ""
        if ctx.memory_context and ctx.memory_context.explicit:
            context_text = "\n".join(
                f"[{rc.get('layer', '?')}] {rc.get('title', '')}: {rc.get('summary', '')}"
                for rc in ctx.memory_context.explicit[:3]
            )

        # Single LLM call — no iterative reasoning
        try:
            system_prompt = "You are a helpful assistant. Answer concisely and directly."
            if context_text:
                system_prompt += f"\n\nRelevant context:\n{context_text}"

            resp = await asyncio.wait_for(
                self.llm.chat([
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ctx.user_input},
                ]),
                timeout=25.0,
            )
            ctx.final_output = resp.content
            ctx.reason_output = resp.content
            logger.info(
                "MainLoop[%s]: SIMPLE_RUN done (%.1fs)",
                ctx.session_id, time.time() - t0,
            )
        except asyncio.TimeoutError:
            ctx.final_output = "抱歉，回复超时了，请再试一次。"
            ctx.errors.append("Simple query LLM timeout")
        except Exception as e:
            ctx.final_output = f"处理出错: {e}"
            ctx.errors.append(str(e))

        # Save episode
        try:
            await asyncio.wait_for(
                self.memory.store({
                    "type": "episode",
                    "title": f"Session: {ctx.user_input[:80]}",
                    "user_input": ctx.user_input,
                    "output": ctx.final_output,
                    "session_id": ctx.session_id,
                    "tags": ["session", "simple", datetime.now(timezone.utc).strftime("%Y-%m-%d")],
                }),
                timeout=3.0,
            )
        except Exception:
            pass

        await self._save_session(ctx)
        return ctx

    async def _save_session(self, ctx: LoopContext) -> None:
        """Save session state to StateStore if available."""
        if not self.state_store:
            return
        try:
            await self.state_store.save_session(ctx.session_id, {
                "session_id": ctx.session_id,
                "user_input": ctx.user_input,
                "final_output": ctx.final_output,
                "task_count": len(ctx.task_ids),
                "task_ids": ctx.task_ids,
                "error_count": len(ctx.errors),
                "errors": ctx.errors,
                "reason_confidence": ctx.reason_confidence,
                "reason_iterations": ctx.reason_iterations,
                "phase": ctx.current_phase.value,
            })
        except Exception as e:
            logger.warning("MainLoop: state_store.save_session failed: %s", e)

    # ============================================================
    # Main Loop
    # ============================================================

    def _is_simple_query(self, text: str) -> bool:
        """Check if a query should skip the complex pipeline.

        Simple queries: short, no action keywords → direct LLM response.
        Complex queries: long, or contains action keywords → full pipeline.
        """
        if len(text) > 50:
            return False
        lower = text.lower()
        return not any(k in lower for k in COMPLEX_KEYWORDS)

    async def run(self, user_input: str | list[MediaInput], task_handle: Any = None) -> LoopContext:
        """Execute the complete MainLoop cycle.

        Args:
            user_input: The user's message (str) or a list of MediaInput objects
            task_handle: Optional TaskHandle for cancellation/pause/resume support

        Returns:
            LoopContext with final output and metadata
        """
        # ── Multimodal routing ─────────────────────────────────
        if isinstance(user_input, list):
            # Multimodal input: route via MultimodalRouter
            media_inputs: list[MediaInput] = user_input
            text_parts = [m.source for m in media_inputs if m.type == "text"]
            text_input = " ".join(text_parts) or "[multimodal input]"

            ctx = LoopContext(user_input=text_input)
            ctx.media_blocks = await self.multimodal_router.process(media_inputs)
            logger.info(
                "MainLoop[%s]: multimodal routing → %d blocks",
                ctx.session_id, len(ctx.media_blocks),
            )
        else:
            ctx = LoopContext(user_input=user_input)
            ctx.media_blocks = []

        async def _check_handle():
            """Check task_handle at safe points during execution."""
            if task_handle is not None:
                await task_handle.wait_if_paused()
                if await task_handle.check_cancelled():
                    return True
            return False
        run_start = time.time()
        logger.info("=" * 60)
        logger.info("MainLoop[%s]: START", ctx.session_id)

        # ── Simple query fast path ────────────────────────────
        text_input = ctx.user_input if not isinstance(user_input, list) else (
            " ".join(m.source for m in user_input if m.type == "text")
        )
        if (not isinstance(user_input, list)
                and not ctx.media_blocks
                and self._is_simple_query(text_input)):
            return await self._simple_run(ctx, task_handle)

        # Tracing: start root span
        loop_span = None
        if self.tracer:
            loop_span = self.tracer.start_span("main_loop.run", input=user_input[:100])

        # Metrics: task started
        try:
            from ..metrics import get_collector
            get_collector().inc("agent_loop_tasks_total", {"status": "started"})
            get_collector().inc("agent_loop_iterations_total")
        except ImportError:
            pass

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
            phase_name = phase.__name__
            phase_timeout = PHASE_TIMEOUTS.get(phase_name, 30.0)

            # Tracing: phase span
            phase_span = None
            if self.tracer:
                phase_span = self.tracer.start_span(f"phase.{phase_name}")

            # Check for cancellation/pause before each phase
            if await _check_handle():
                logger.info("MainLoop[%s]: cancelled during %s", ctx.session_id, phase_name)
                ctx.errors.append("Task cancelled by user")
                ctx.final_output = "任务已取消"
                break

            phase_start = time.time()
            logger.info("MainLoop[%s]: phase=%s START", ctx.session_id, phase_name)

            try:
                # Run phase with timeout
                await asyncio.wait_for(phase(ctx), timeout=phase_timeout)
                phase_elapsed = time.time() - phase_start

                if self.tracer and phase_span:
                    self.tracer.end_span(phase_span, "ok")

                # Log per-phase wall-clock time
                logger.info(
                    "MainLoop[%s]: phase=%s DONE (%.1fs)",
                    ctx.session_id, phase_name, phase_elapsed,
                )

            except asyncio.TimeoutError:
                logger.error(
                    "MainLoop[%s]: phase=%s TIMEOUT after %.1fs",
                    ctx.session_id, phase_name, phase_timeout,
                )
                ctx.errors.append(f"Phase {phase_name} timed out ({phase_timeout}s)")
                if self.tracer and phase_span:
                    self.tracer.end_span(phase_span, "timeout")

            except Exception as e:
                logger.error("Phase %s failed: %s", phase_name, e)
                ctx.errors.append(f"Error in {phase_name}: {e}")
                if self.tracer and phase_span:
                    self.tracer.end_span(phase_span, "error")

        # Metrics: task completed
        try:
            from ..metrics import get_collector
            status = "success" if not ctx.errors else "failed"
            get_collector().inc("agent_loop_tasks_total", {"status": status})
        except ImportError:
            pass

        # Tracing: end root span
        if self.tracer and loop_span:
            self.tracer.end_span(loop_span, "error" if ctx.errors else "ok")

        elapsed = time.time() - run_start
        logger.info("MainLoop[%s]: END (%.1fs)", ctx.session_id, elapsed)
        return ctx
