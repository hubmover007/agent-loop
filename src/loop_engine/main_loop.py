"""MainLoop - the outermost Loop Engine cycle.

INPUT → RETRIEVE → REASON → DECOMPOSE → DISPATCH → COLLECT → OUTPUT

Intent classification:
  - Level 1 (chat):    simple Q&A → _simple_run (<5s)
  - Level 2 (analysis): reasoning/comparison → _analysis_run (<15s)
  - Level 3 (complex): multi-step tasks → full pipeline with decompose
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import LoopConfig, LoopContext, AgentLoop, ToolLoop, LLMProvider
from .deep_reason import DeepReasonLoop, DeepReasonConfig
from .metrics import LoopMetrics
from ..core import LoopPhase, TaskStatus, ToolResult, DiscardRecord
from ..memory import MemoryPool
from ..memory.graph_route import GraphRouter
from ..system_agents import TaskAgent, AgentManagerAgent, TaskRegistry
from ..memory.unified_retrieval import UnifiedRetriever, MemoryContext
from ..project import Project, ProjectCard
from ..ammo import AmmoBox, AmmoRefiller

logger = logging.getLogger(__name__)

# Phase timeout defaults (seconds)
PHASE_TIMEOUTS: dict[str, float] = {
    "_input": 5.0,
    "_retrieve": 10.0,
    "_reason": 25.0,
    "_decompose": 20.0,
    "_dispatch": 5.0,
    "_collect": 60.0,   # Was 30s, now configurable via LoopConfig.collect_timeout
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
                 anchor_manager: Any | None = None,
                 cost_controller: Any | None = None,
                 llm_pool: Any | None = None,
                 project: Project | None = None,
                 project_root: str | Path | None = None):
        self.memory = memory
        self.llm = llm
        self.config = config or LoopConfig()
        self.state_store = state_store
        self.tracer = tracer  # Optional Tracer for distributed tracing
        self.anchor_manager = anchor_manager  # Optional AnchorManager for O(1) key-fact lookup
        self.cost_controller = cost_controller  # Optional CostController for budget management
        self.llm_pool = llm_pool  # Optional LLMPool for capability-based model selection

        # Project workspace (shared space for all agents)
        if project:
            self.project = project
        elif project_root:
            self.project = Project(project_root)
        else:
            self.project = None  # No project context (backward compat)

        # System Agents: TaskAgent (manages tasks) + AgentManagerAgent (manages agents)
        self.task_agent: TaskAgent | None = None
        self.agent_manager: AgentManagerAgent | None = None
        self.task_registry: TaskRegistry | None = None

        # Sub-systems
        from ..tools.base import ToolRegistry
        self.tool_registry = ToolRegistry()
        if self.project:
            self.tool_registry.register_defaults(code_workspace=str(self.project.workspace))
        else:
            self.tool_registry.register_defaults()
        self.tool_loop = ToolLoop(self.tool_registry, self.config)
        self.graph_router = GraphRouter(memory)  # M-FLOW graph routing
        self.agent_loop = AgentLoop(self.tool_loop, llm, self.config, llm_pool=llm_pool)
        self.deep_reason = DeepReasonLoop(llm, DeepReasonConfig(
            max_iterations=self.config.max_reason_loops,
            confidence_threshold=self.config.reason_confidence_threshold,
            enable_latent_thinking=False,  # Disable for speed; use multi-iteration instead
        ), llm_pool=llm_pool)

        # Unified Memory Retriever: M-FLOW graph + Mythos deep reasoning
        self.retriever = UnifiedRetriever(
            memory_pool=memory,
            graph_router=self.graph_router,
            deep_reason=self.deep_reason,
            llm=llm,
            llm_pool=llm_pool,
        )

    # ============================================================
    # LLM selection helper
    # ============================================================

    def _get_llm(self, capabilities: list[str], strategy: str) -> LLMProvider:
        """Get the best LLM for a given capability+strategy, falling back to default."""
        if self.llm_pool:
            provider = self.llm_pool.get_provider(
                capabilities=capabilities, strategy=strategy
            )
            if provider:
                return provider
        return self.llm

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
            # Unified retrieval: explicit graph only (no deep reasoning during retrieve)
            # Deep reasoning happens in _reason phase, not retrieve
            mem_ctx = await self.retriever.retrieve(
                query=ctx.user_input,
                max_hops=3,
                deep_reason_iterations=0,
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
                llm_pool=self.llm_pool,
            )
        if not self.agent_manager:
            from ..external_agents import ExternalAgentBridge
            self.agent_manager = AgentManagerAgent(
                memory=self.memory,
                agent_loop=self.agent_loop,
                config=self.config,
                registry=self.task_registry,
                external_bridge=ExternalAgentBridge(),
                project=self.project,
                tool_registry=self.tool_registry,
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
        results = await self.agent_manager.collect_all(timeout=self.config.collect_timeout)

        for task_id, result in results.items():
            ctx.agent_results.append(result)

        # Check for failed tasks — TaskAgent can re-plan
        for task in self.task_registry.all_tasks():
            if task.status == TaskStatus.FAILED:
                ctx.discarded_results.append(task.task_id)
                ctx.errors.append(f"Task {task.task_id} failed: {task.error}")

                # TaskAgent re-plans failed tasks (with retry limit)
                if task.retry_count < self.config.max_task_retries:
                    task.retry_count += 1
                    try:
                        new_tasks = await self.task_agent.replan(task)
                        if new_tasks:
                            for nt in new_tasks:
                                await self.agent_manager.assign(nt)
                            logger.info("MainLoop[%s]: replan task %s → %d new tasks",
                                       ctx.session_id, task.task_id, len(new_tasks))
                    except Exception as e:
                        logger.warning("MainLoop[%s]: replan failed for %s: %s",
                                      ctx.session_id, task.task_id, e)
                else:
                    logger.warning("MainLoop[%s]: task %s exceeded max retries (%d)",
                                 ctx.session_id, task.task_id, self.config.max_task_retries)

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

        # Write a new Episode to memory with project_id
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
                "project": self.project.project_id if self.project else "",
                "tags": ["session", datetime.now(timezone.utc).strftime("%Y-%m-%d")],
            })
        except Exception as e:
            logger.warning("Failed to write episode: %s", e)

        # Update project card
        if self.project:
            self.project.save_session_summary(
                ctx.session_id,
                ctx.final_output[:200],
                ctx.user_input[:200],
                ctx.final_output[:200],
            )
            card = self.project.load_card()
            card.update_session(
                ctx.session_id,
                ctx.final_output[:200],
                ctx.user_input[:200],
                ctx.final_output[:200],
            )
            # Cleanup worker findings
            self.project.cleanup_findings()

        # Save session state
        await self._save_session(ctx)

        logger.info("MainLoop[%s]: OUTPUT completed", ctx.session_id)

        # ── Auto-consolidation (async, non-blocking) ─────────────
        if self.config.auto_consolidate:
            asyncio.create_task(self._auto_consolidate())

    async def _auto_consolidate(self) -> None:
        """Run memory consolidation in background after session.

        Only triggers if enough unconsolidated episodes have accumulated.
        """
        try:
            # Check if enough episodes to justify consolidation
            if hasattr(self.memory, 'get_unconsolidated_episodes'):
                episodes = await asyncio.wait_for(
                    self.memory.get_unconsolidated_episodes(limit=50),
                    timeout=5.0,
                )
                pending = [e for e in episodes if not e.get("consolidated", False)]
                if len(pending) < self.config.consolidate_min_episodes:
                    return  # Not enough episodes yet

            logger.info("MainLoop: auto-consolidation starting (%d+ episodes)",
                       self.config.consolidate_min_episodes)
            result = await asyncio.wait_for(
                self.memory.consolidate(
                    llm_provider=self.llm,
                    min_episodes=self.config.consolidate_min_episodes,
                ),
                timeout=120.0,
            )
            logger.info("MainLoop: auto-consolidation done — %s",
                       result.to_summary() if hasattr(result, 'to_summary') else str(result)[:200])
        except asyncio.TimeoutError:
            logger.warning("MainLoop: auto-consolidation timed out")
        except Exception as e:
            logger.warning("MainLoop: auto-consolidation failed: %s", e)

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

        # Metrics for simple run
        ctx.metrics.start_phase("input")
        ctx.metrics.end_phase("input")

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
        ctx.metrics.start_phase("retrieve")
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
        ctx.metrics.end_phase("retrieve")

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
        ctx.metrics.start_phase("reason")
        try:
            system_prompt = "You are a helpful assistant. Answer concisely and directly."
            if context_text:
                system_prompt += f"\n\nRelevant context:\n{context_text}"

            llm = self._get_llm(["quick"], "cheapest")
            resp = await llm.chat_with_retry(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": ctx.user_input},
                ],
                max_retries=self.config.llm_max_retries,
                timeout=self.config.llm_timeout,
            )
            ctx.final_output = resp.content
            ctx.reason_output = resp.content

            # Record token usage
            if hasattr(resp, 'usage') and resp.usage:
                pt = resp.usage.get('input_tokens', resp.usage.get('prompt_tokens', 0))
                ct = resp.usage.get('output_tokens', resp.usage.get('completion_tokens', 0))
                ctx.metrics.record_llm_call(prompt_tokens=pt, completion_tokens=ct)
            else:
                ctx.metrics.record_llm_call()

            ctx.metrics.end_phase("reason")
            logger.info(
                "MainLoop[%s]: SIMPLE_RUN done (%.1fs)",
                ctx.session_id, time.time() - t0,
            )
        except asyncio.TimeoutError:
            ctx.metrics.end_phase("reason")
            ctx.metrics.record_error()
            ctx.final_output = "抱歉，回复超时了，请再试一次。"
            ctx.errors.append("Simple query LLM timeout")
        except Exception as e:
            ctx.metrics.end_phase("reason")
            ctx.metrics.record_error()
            ctx.final_output = f"处理出错: {e}"
            ctx.errors.append(str(e))

        ctx.metrics.start_phase("output")
        ctx.metrics.end_phase("output")
        ctx.metrics.finish()
        logger.info("MainLoop[%s]: SIMPLE_RUN metrics\n%s", ctx.session_id, ctx.metrics.summary())

        # Save episode with project_id
        await self._save_episode(ctx, tags=["session", "simple"])

        # Update project card
        if self.project:
            self.project.save_session_summary(
                ctx.session_id,
                ctx.final_output[:200],
                ctx.user_input[:200],
                ctx.final_output[:200],
            )
            card = self.project.load_card()
            card.update_session(
                ctx.session_id,
                ctx.final_output[:200],
                ctx.user_input[:200],
                ctx.final_output[:200],
            )

        await self._save_session(ctx)
        return ctx

    async def _analysis_run(self, ctx: LoopContext,
                           task_handle: Any = None) -> LoopContext:
        """Level 2: Analysis/comparison queries — single agent + ammo box.

        Skips: DECOMPOSE, DISPATCH, COLLECT.
        Runs: INPUT + RETRIEVE + REASON (with ammo) + OUTPUT.
        Returns in <15 seconds.
        """
        t0 = time.time()
        logger.info("MainLoop[%s]: ANALYSIS_RUN for '%s'", ctx.session_id, ctx.user_input[:50])

        # Init ammo box
        ammo = self._init_ammo_box()

        async def _check():
            if task_handle:
                await task_handle.wait_if_paused()
                return await task_handle.check_cancelled()
            return False

        # Phase: INPUT
        ctx.metrics.start_phase("input")
        ctx.metrics.end_phase("input")

        if await _check():
            ctx.final_output = "任务已取消"
            return ctx

        # Phase: RETRIEVE (with project filter)
        ctx.metrics.start_phase("retrieve")
        try:
            ctx.current_phase = LoopPhase.RETRIEVE
            mem_ctx = await asyncio.wait_for(
                self.retriever.retrieve(
                    query=ctx.user_input,
                    max_hops=3,
                    deep_reason_iterations=0,
                ),
                timeout=10.0,
            )
            ctx.memory_context = mem_ctx
            # Add retrieved context to ammo
            if mem_ctx and mem_ctx.explicit:
                for item in mem_ctx.explicit[:3]:
                    summary = item.get("summary", "")[:200]
                    if summary:
                        if ammo:
                            ammo.add_fact(summary, source="retrieve")
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("Analysis retrieve skipped: %s", e)
        ctx.metrics.end_phase("retrieve")

        if await _check():
            ctx.final_output = "任务已取消"
            return ctx

        # Phase: REASON (with ammo context)
        ctx.metrics.start_phase("reason")
        try:
            # Build context from ammo box
            ammo_context = ammo.to_context() if ammo else ""

            system_prompt = "You are a helpful assistant. Answer thoroughly and clearly."
            if ammo_context:
                system_prompt += f"\n\n--- Project Context ---\n{ammo_context}"

            # Check for @project mention → deeper retrieval
            if self._detect_project_mention(ctx.user_input) and self.project:
                deeper = await self._search_project_memory(ctx.user_input)
                if deeper:
                    system_prompt += f"\n\n--- Deep Memory ---\n{deeper}"

            llm = self._get_llm(["reasoning"], "balanced")
            resp = await asyncio.wait_for(
                llm.chat_with_retry(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": ctx.user_input},
                    ],
                    max_retries=2,
                    timeout=20.0,
                ),
                timeout=25.0,
            )
            ctx.final_output = resp.content
            ctx.reason_output = resp.content

            # Record token usage
            if hasattr(resp, 'usage') and resp.usage:
                pt = resp.usage.get('input_tokens', resp.usage.get('prompt_tokens', 0))
                ct = resp.usage.get('output_tokens', resp.usage.get('completion_tokens', 0))
                ctx.metrics.record_llm_call(prompt_tokens=pt, completion_tokens=ct)
            else:
                ctx.metrics.record_llm_call()

            ctx.metrics.end_phase("reason")
            logger.info(
                "MainLoop[%s]: ANALYSIS_RUN done (%.1fs)",
                ctx.session_id, time.time() - t0,
            )
        except asyncio.TimeoutError:
            ctx.metrics.end_phase("reason")
            ctx.metrics.record_error()
            ctx.final_output = "抱歉，分析超时了，请简化问题再试。"
            ctx.errors.append("Analysis LLM timeout")
        except Exception as e:
            ctx.metrics.end_phase("reason")
            ctx.metrics.record_error()
            ctx.final_output = f"处理出错: {e}"
            ctx.errors.append(str(e))

        ctx.metrics.start_phase("output")
        ctx.metrics.end_phase("output")
        ctx.metrics.finish()

        # Save episode with project_id
        await self._save_episode(ctx, tags=["session", "analysis"])

        # Update project card
        if self.project:
            self.project.save_session_summary(
                ctx.session_id,
                ctx.final_output[:200],
                ctx.user_input[:200],
                ctx.final_output[:200],
            )
            card = self.project.load_card()
            card.update_session(
                ctx.session_id,
                ctx.final_output[:200],
                ctx.user_input[:200],
                ctx.final_output[:200],
            )

        await self._save_session(ctx)
        return ctx

    async def _search_project_memory(self, query: str) -> str:
        """Deep search project memory for @project queries."""
        if not self.memory:
            return ""

        project_filter = ""
        if self.project:
            project_filter = f"AND project = '{self.project.project_id}'"

        try:
            if hasattr(self.memory, '_db') and self.memory._db:
                result = await self.memory._db.query(f"""
                    SELECT * FROM episode
                    WHERE summary != ''
                    {project_filter}
                    ORDER BY created_at DESC
                    LIMIT 5
                """)
                if isinstance(result, list):
                    items = result
                elif isinstance(result, dict) and "result" in result:
                    items = result["result"]
                else:
                    items = []

                parts = []
                for item in items:
                    summary = item.get("summary", "")[:200]
                    if summary:
                        parts.append(f"- {summary}")
                return "\n".join(parts)
        except Exception as e:
            logger.warning("Project memory search failed: %s", e)

        return ""

    async def _save_episode(self, ctx: LoopContext, tags: list[str] | None = None) -> None:
        """Save session as episode to memory."""
        try:
            await asyncio.wait_for(
                self.memory.store({
                    "type": "episode",
                    "title": f"Session: {ctx.user_input[:80]}",
                    "user_input": ctx.user_input,
                    "output": ctx.final_output,
                    "session_id": ctx.session_id,
                    "project": self.project.project_id if self.project else "",
                    "tags": tags or ["session", datetime.now(timezone.utc).strftime("%Y-%m-%d")],
                }),
                timeout=3.0,
            )
        except Exception:
            pass

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

    def _classify_intent(self, text: str) -> str:
        """Classify user intent into 3 levels.

        Returns:
            "chat"     — Level 1: simple Q&A, direct LLM response
            "analysis"  — Level 2: reasoning/analysis, single agent + ammo
            "complex"   — Level 3: multi-step tasks, full pipeline
        """
        lower = text.lower()

        # Multi-step keywords: deploy, create project, fix bug, build, migrate
        multi_step_keywords = [
            "部署", "创建项目", "修复", "修bug", "构建", "迁移",
            "deploy", "create project", "build", "migrate", "install",
            "write code", "写代码", "实现", "implement", "fix", "debug",
        ]
        if any(kw in lower for kw in multi_step_keywords):
            return "complex"

        # Analysis keywords: comparison, analysis, summary, explain
        analysis_keywords = [
            "分析", "对比", "比较", "总结", "解释", "区别",
            "analyze", "compare", "summary", "explain", "difference",
            "vs", "versus", "pros", "cons",
        ]
        if any(kw in lower for kw in analysis_keywords):
            return "analysis"

        # Level 1: short + no complex keywords → chat
        if self._is_simple_query(text):
            return "chat"

        # Medium length without keywords → analysis (not complex enough for full pipeline)
        if len(text) < 200:
            return "analysis"

        return "complex"

    def _detect_project_mention(self, text: str) -> bool:
        """Detect if user explicitly mentions @project."""
        return bool(re.search(r'@project', text, re.IGNORECASE))

    def _init_ammo_box(self) -> AmmoBox | None:
        """Initialize AmmoBox with project card and recent context."""
        if not self.project:
            return None

        ammo = AmmoBox(max_tokens=2000)

        # Layer 1: Project Card (auto-injected)
        card = self.project.load_card()
        ammo.add_pinned(card.to_context(), source="project_card")

        # Layer 2: Recent context (auto-injected)
        recent = self.project.load_recent_sessions(limit=3)
        if recent:
            recent_text = "## 最近会话\n"
            for s in recent:
                recent_text += f"- {s.get('summary', '')[:100]}\n"
            ammo.add_pinned(recent_text, source="recent")

        return ammo

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

        # ── Loop metrics ────────────────────────────────────────
        ctx.metrics = LoopMetrics()
        ctx.metrics.start()

        # ── Budget check (optional) ──────────────────────────────
        if self.cost_controller:
            if not self.cost_controller.check(estimated_cost=0.01, task_scope="main"):
                ctx.final_output = "预算超限，停止执行"
                ctx.metrics.finish()
                return ctx

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

        # ── Intent classification: 3-level routing ─────────
        text_input = ctx.user_input if not isinstance(user_input, list) else (
            " ".join(m.source for m in user_input if m.type == "text")
        )
        if (not isinstance(user_input, list)
                and not ctx.media_blocks):
            intent = self._classify_intent(text_input)
            logger.info("MainLoop[%s]: intent='%s'", ctx.session_id, intent)
            if intent == "chat":
                return await self._simple_run(ctx, task_handle)
            elif intent == "analysis":
                return await self._analysis_run(ctx, task_handle)
            # else: complex → full pipeline below

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
            ctx.metrics.start_phase(phase_name)
            logger.info("MainLoop[%s]: phase=%s START", ctx.session_id, phase_name)

            try:
                # Run phase with timeout
                await asyncio.wait_for(phase(ctx), timeout=phase_timeout)
                phase_elapsed = time.time() - phase_start
                ctx.metrics.end_phase(phase_name)

                if self.tracer and phase_span:
                    self.tracer.end_span(phase_span, "ok")

                # Log per-phase wall-clock time
                logger.info(
                    "MainLoop[%s]: phase=%s DONE (%.1fs)",
                    ctx.session_id, phase_name, phase_elapsed,
                )

            except asyncio.TimeoutError:
                ctx.metrics.end_phase(phase_name)
                ctx.metrics.record_error()
                logger.error(
                    "MainLoop[%s]: phase=%s TIMEOUT after %.1fs",
                    ctx.session_id, phase_name, phase_timeout,
                )
                ctx.errors.append(f"Phase {phase_name} timed out ({phase_timeout}s)")
                if self.tracer and phase_span:
                    self.tracer.end_span(phase_span, "timeout")

            except Exception as e:
                ctx.metrics.end_phase(phase_name)
                ctx.metrics.record_error()
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
        ctx.metrics.finish()
        logger.info("MainLoop[%s]: END (%.1fs)\n%s", ctx.session_id, elapsed, ctx.metrics.summary())
        return ctx
