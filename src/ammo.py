"""AmmoBox & AmmoRefiller — context management for worker agents.

Complete ammo lifecycle:
  1. Init:     Project Card + Recent Sessions (auto, pinned)
  2. Retrieve: RAG from MemoryPool (auto, on-demand)
  3. Fill:     Web search when knowledge gap detected (on-demand)
  4. Share:    Worker findings via project temp dir (auto)
  5. Compress: Context > 70% → summarize old steps (auto)
  6. Clean:    Remove failed/duplicate data (per-step)
  7. Learn:    Consolidation extracts facts → better ammo next time

AmmoBox layers (by priority):
  pinned:     Project Card + Recent Context (never compressed)
  decisions:  Verified conclusions (high priority)
  facts:      Key facts from RAG/search (medium, evictable)
  findings:   Other workers' findings (medium-low)
  workspace:  Current execution steps (low, compressible)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AmmoItem:
    """A single ammo item."""
    content: str
    source: str  # "project_card" | "recent" | "rag" | "web" | "finding" | "decision"
    priority: int = 5  # 1=highest (pinned), 10=lowest (workspace)
    timestamp: float = field(default_factory=time.time)
    token_est: int = 0
    relevance: float = 1.0  # 0-1, used for eviction
    tags: list[str] = field(default_factory=list)


class AmmoBox:
    """Pinned context that survives compaction.

    Token budget: ~2000 tokens total.
    Layers (eviction order: workspace → findings → facts → decisions → pinned):
      pinned:     ~400 tokens (Project Card + Recent)
      decisions:  ~300 tokens (verified conclusions)
      facts:      ~600 tokens (RAG/search results)
      findings:   ~400 tokens (other workers)
      workspace:  ~300 tokens (current execution, compressible)
    """

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.pinned: list[AmmoItem] = []
        self.decisions: list[AmmoItem] = []
        self.facts: list[AmmoItem] = []
        self.findings: list[AmmoItem] = []
        self.workspace: list[AmmoItem] = []  # current execution steps
        self._total_tokens: int = 0
        self._compaction_count: int = 0
        self._refill_count: int = 0

    # ── Add items ────────────────────────────────────────────────

    def add_pinned(self, content: str, source: str = "project_card") -> None:
        item = AmmoItem(content=content, source=source, priority=1,
                        token_est=self._est_tokens(content))
        self.pinned.append(item)
        self._total_tokens += item.token_est

    def add_decision(self, content: str) -> None:
        item = AmmoItem(content=content, source="decision", priority=2,
                        token_est=self._est_tokens(content))
        self.decisions.append(item)
        self._total_tokens += item.token_est
        self._evict_if_needed()

    def add_fact(self, content: str, source: str = "rag",
                 relevance: float = 1.0) -> None:
        item = AmmoItem(content=content, source=source, priority=3,
                        token_est=self._est_tokens(content),
                        relevance=relevance)
        self.facts.append(item)
        self._total_tokens += item.token_est
        self._evict_if_needed()

    def add_finding(self, content: str, task_id: str = "") -> None:
        item = AmmoItem(content=content, source=f"finding:{task_id}",
                        priority=4, token_est=self._est_tokens(content))
        self.findings.append(item)
        self._total_tokens += item.token_est
        self._evict_if_needed()

    def add_workspace(self, content: str, step_num: int = 0) -> None:
        item = AmmoItem(content=content, source=f"step:{step_num}",
                        priority=5, token_est=self._est_tokens(content))
        self.workspace.append(item)
        self._total_tokens += item.token_est
        self._evict_if_needed()

    # ── Render to context ────────────────────────────────────────

    def to_context(self, include_workspace: bool = True) -> str:
        """Render ammo box to compact string for LLM context."""
        parts = []

        # Pinned (always)
        for item in self.pinned:
            parts.append(item.content)

        # Decisions (high priority)
        if self.decisions:
            parts.append("## 已验证结论")
            for d in self.decisions[-5:]:
                parts.append(f"- {d.content[:200]}")

        # Key facts (medium)
        if self.facts:
            parts.append("## 关键事实")
            for f in self.facts[-10:]:
                parts.append(f"- {f.content[:200]}")

        # Findings from others (medium-low)
        if self.findings:
            parts.append("## 其他 worker 发现")
            for f in self.findings[-5:]:
                parts.append(f"- {f.content[:200]}")

        # Workspace (current execution, compressible)
        if include_workspace and self.workspace:
            parts.append("## 执行记录")
            for w in self.workspace[-5:]:  # last 5 steps
                parts.append(f"- {w.content[:150]}")

        return "\n\n".join(parts)

    def to_pinned_only(self) -> str:
        """Render only pinned + decisions (for context injection after compaction)."""
        parts = []
        for item in self.pinned:
            parts.append(item.content)
        if self.decisions:
            parts.append("## 已验证结论")
            for d in self.decisions[-3:]:
                parts.append(f"- {d.content[:200]}")
        return "\n\n".join(parts)

    # ── Compaction ───────────────────────────────────────────────

    def compact_workspace(self) -> str:
        """Compress workspace items into a summary (rule-based, fast).

        Only called when context > 70%. For smarter compaction,
        use smart_compact() with LLM.
        """
        if not self.workspace:
            return ""

        # Build summary from workspace items
        summary_parts = []
        for w in self.workspace:
            summary_parts.append(w.content[:100])

        # Remove old workspace tokens
        for w in self.workspace:
            self._total_tokens -= w.token_est

        # Create compressed summary
        summary = " | ".join(summary_parts)
        summary_item = AmmoItem(
            content=f"[前{len(self.workspace)}步摘要] {summary[:500]}",
            source="compacted",
            priority=5,
            token_est=self._est_tokens(summary),
        )

        self.workspace = [summary_item]
        self._total_tokens += summary_item.token_est
        self._compaction_count += 1

        logger.debug("AmmoBox: compacted %d workspace items → 1 summary (%d→%d tokens)",
                     len(summary_parts), self._compaction_count, summary_item.token_est)
        return summary_item.content

    async def smart_compact(self, llm: Any, task_scope: str) -> str:
        """LLM-driven intelligent compaction.

        Asks LLM to classify workspace items into:
          - KEEP: valuable for task completion
          - COMPRESS: can be summarized
          - DISCARD: no value

        Called when context > 70% and LLM is available.
        """
        if not self.workspace or not llm:
            return self.compact_workspace()

        items_text = "\n".join(
            f"[{i}] {w.content[:200]}"
            for i, w in enumerate(self.workspace)
        )

        prompt = f"""任务: {task_scope}

以下是执行过程中的 workspace 记录:

{items_text}

请分类:
1. KEEP: 对完成任务有价值的记录（关键发现、成功验证、重要中间结果、错误中的教训）
2. COMPRESS: 可压缩成摘要的记录（冗长输出、重复信息）
3. DISCARD: 可丢弃的记录（完全无关、空内容）

只返回 JSON，格式: {{"keep": [序号], "compress": [序号], "discard": [序号]}}"""

        try:
            resp = await llm.chat([
                {"role": "system", "content": "You are a context management assistant."},
                {"role": "user", "content": prompt},
            ])

            from .utils import extract_json_from_llm_response
            classification = extract_json_from_llm_response(resp.content, default={})

            keep_ids = set(classification.get("keep", []))
            compress_ids = set(classification.get("compress", []))
            discard_ids = set(classification.get("discard", []))

            # Execute classification
            new_workspace = []
            compressed_parts = []

            for i, w in enumerate(self.workspace):
                if i in keep_ids:
                    new_workspace.append(w)
                elif i in compress_ids:
                    compressed_parts.append(w.content[:100])
                # discard: skip

            # Add compressed summary if any items were compressed
            if compressed_parts:
                summary = " | ".join(compressed_parts)
                new_workspace.append(AmmoItem(
                    content=f"[压缩摘要] {summary[:400]}",
                    source="smart_compacted",
                    priority=5,
                    token_est=self._est_tokens(summary),
                ))

            # Recalculate tokens
            for w in self.workspace:
                self._total_tokens -= w.token_est
            for w in new_workspace:
                self._total_tokens += w.token_est

            self.workspace = new_workspace
            self._compaction_count += 1

            logger.info("AmmoBox: smart_compact — keep=%d, compress=%d, discard=%d",
                       len(keep_ids), len(compress_ids), len(discard_ids))
            return f"[智能压缩] 保留{len(keep_ids)}条, 压缩{len(compress_ids)}条, 丢弃{len(discard_ids)}条"

        except Exception as e:
            logger.warning("AmmoBox: smart_compact failed, falling back to rule-based: %s", e)
            return self.compact_workspace()

    def clean_workspace_rules(self) -> int:
        """Rule-based cleaning — only remove clearly useless items.

        Safe rules (no LLM needed):
          1. Empty content
          2. Exact duplicates
          3. Tool output > 500 tokens (keep first 100 chars as summary)

        Does NOT remove error/failed items (they may contain valuable info).
        """
        before = len(self.workspace)
        seen_content: set[str] = set()
        cleaned: list[AmmoItem] = []

        for w in self.workspace:
            content = w.content.strip()
            # Rule 1: skip empty
            if not content:
                self._total_tokens -= w.token_est
                continue
            # Rule 2: skip exact duplicates
            content_key = content[:200]  # first 200 chars as dedup key
            if content_key in seen_content:
                self._total_tokens -= w.token_est
                continue
            seen_content.add(content_key)
            # Rule 3: truncate overly long tool outputs
            if w.token_est > 500:
                old_est = w.token_est
                w.content = w.content[:200] + "...[truncated]"
                w.token_est = self._est_tokens(w.content)
                self._total_tokens -= (old_est - w.token_est)

            cleaned.append(w)

        self.workspace = cleaned
        removed = before - len(self.workspace)
        if removed > 0:
            logger.debug("AmmoBox: rule-cleaned %d workspace items", removed)
        return removed

    def clean_failed_steps(self) -> int:
        """Deprecated: use clean_workspace_rules() instead.

        Kept for backward compat. Now just calls clean_workspace_rules().
        """
        return self.clean_workspace_rules()

    # ── Stats ────────────────────────────────────────────────────

    def token_usage(self) -> int:
        return self._total_tokens

    def usage_ratio(self) -> float:
        return self._total_tokens / self.max_tokens if self.max_tokens > 0 else 0.0

    def stats(self) -> dict:
        return {
            "total_tokens": self._total_tokens,
            "usage_ratio": round(self.usage_ratio(), 2),
            "pinned": len(self.pinned),
            "decisions": len(self.decisions),
            "facts": len(self.facts),
            "findings": len(self.findings),
            "workspace": len(self.workspace),
            "compactions": self._compaction_count,
            "refills": self._refill_count,
        }

    # ── Internal ─────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        """Evict low-priority items if over budget. Priority: workspace > findings > facts."""
        while self._total_tokens > self.max_tokens:
            # Try workspace first (lowest priority)
            if self.workspace:
                old = self.workspace.pop(0)
                self._total_tokens -= old.token_est
                continue
            # Then findings
            if self.findings:
                old = self.findings.pop(0)
                self._total_tokens -= old.token_est
                continue
            # Then low-relevance facts
            if self.facts:
                # Remove lowest relevance fact
                self.facts.sort(key=lambda f: f.relevance)
                old = self.facts.pop(0)
                self._total_tokens -= old.token_est
                continue
            break  # Can't evict pinned/decisions

    @staticmethod
    def _est_tokens(text: str) -> int:
        return max(1, len(text) // 4)


class AmmoRefiller:
    """Refills ammo during agent execution.

    6 trigger conditions:
      1. context > 70% → compact workspace
      2. PLAN finds knowledge gap → RAG search
      3. EXECUTE hits error → web search for solution
      4. Every N steps → review findings from other workers
      5. @project mention → deep project memory search
      6. Step success → write finding to shared temp dir

    Ammo sources:
      - RAG: MemoryPool (episode/fact/graph) — fast, local
      - Web: WebTool (Brave Search) — slower, external
      - Findings: project .agent_loop/temp/ — from other workers
    """

    def __init__(self, ammo_box: AmmoBox,
                 project: Any = None,
                 memory: Any = None,
                 llm: Any = None,
                 web_tool: Any = None,
                 task_id: str = "",
                 max_refills: int = 10):
        self.ammo = ammo_box
        self.project = project
        self.memory = memory
        self.llm = llm
        self.web_tool = web_tool  # WebTool for web search
        self.task_id = task_id
        self.max_refills = max_refills
        self._step_count = 0
        self._review_interval = 3  # review every 3 steps
        self._refill_count = 0

    async def check_and_refill(self, ctx: dict) -> dict:
        """Check all triggers and refill ammo if needed.

        Args:
            ctx: execution context:
              - phase: "PLAN" | "EXECUTE" | "SELF_EVAL"
              - plan: list of steps (optional)
              - last_error: str (optional)
              - last_step_success: bool (optional)
              - step_num: int (optional)
              - workspace_tokens: int (optional)
              - user_input: str (optional, for @project check)

        Returns:
            dict with actions taken
        """
        actions = {
            "compacted": False,
            "rag_searched": False,
            "web_searched": False,
            "reviewed": False,
            "finding_written": False,
            "project_searched": False,
        }

        if self._refill_count >= self.max_refills:
            return actions

        phase = ctx.get("phase", "")
        step_num = ctx.get("step_num", 0)

        # Trigger 1: context too full → smart compact (LLM-driven)
        ws_tokens = ctx.get("workspace_tokens", 0)
        total = self.ammo.token_usage() + ws_tokens
        if total > 0 and total / (self.ammo.max_tokens + 4000) > 0.7:
            task_scope = ctx.get("user_input", "")
            if self.llm and task_scope:
                # Use LLM to intelligently classify what to keep/compress/discard
                await self.ammo.smart_compact(self.llm, task_scope)
            else:
                self.ammo.compact_workspace()
            actions["compacted"] = True
            self._refill_count += 1

        # Trigger 2: PLAN finds knowledge gap
        if phase == "PLAN" and ctx.get("plan"):
            gaps = self._detect_gaps(ctx["plan"])
            if gaps:
                await self._rag_search(gaps)
                actions["rag_searched"] = True
                self._refill_count += 1

        # Trigger 3: EXECUTE hits error → web search
        if ctx.get("last_error"):
            await self._web_search_solution(ctx["last_error"])
            actions["web_searched"] = True
            self._refill_count += 1

        # Trigger 4: periodic review
        if phase == "EXECUTE":
            self._step_count += 1
            if self._step_count % self._review_interval == 0:
                await self._review_findings()
                actions["reviewed"] = True

        # Trigger 5: @project mention
        user_input = ctx.get("user_input", "")
        if user_input and "@project" in user_input.lower():
            await self._deep_project_search(user_input)
            actions["project_searched"] = True
            self._refill_count += 1

        # Trigger 6: step success → write finding to shared
        if ctx.get("last_step_success") and self.project:
            step_desc = ctx.get("step_desc", "")
            if step_desc:
                self._write_finding(step_desc)
                actions["finding_written"] = True

        # Rule-based cleaning (safe, no LLM) — only removes clearly useless items
        removed = self.ammo.clean_workspace_rules()
        if removed > 0:
            actions["cleaned"] = removed

        return actions

    # ── Gap detection ────────────────────────────────────────────

    def _detect_gaps(self, plan: list[dict]) -> list[str]:
        """Detect knowledge gaps in the plan.

        Looks for steps that need external info (search, docs, reference).
        """
        gaps = []
        gap_keywords = [
            "搜索", "查找", "调研", "文档", "参考", "了解",
            "search", "find", "lookup", "document", "reference",
            "research", "investigate",
        ]

        for step in plan:
            desc = ""
            if isinstance(step, dict):
                desc = step.get("description", "") or step.get("tool", "") or str(step)
            else:
                desc = str(step)

            desc_lower = desc.lower()
            if any(kw in desc_lower for kw in gap_keywords):
                # Extract the core intent
                gaps.append(desc[:200])

        return gaps[:3]  # cap at 3 gaps

    # ── RAG search (MemoryPool) ──────────────────────────────────

    async def _rag_search(self, gaps: list[str]) -> None:
        """Search MemoryPool for gap-filling info."""
        if not self.memory:
            return

        for gap in gaps:
            try:
                results = await self._search_memory(gap)
                for r in results[:2]:
                    content = r.get("summary", r.get("content", ""))[:300]
                    if content:
                        self.ammo.add_fact(content, source=f"rag:{gap[:50]}",
                                           relevance=0.8)
                        logger.debug("AmmoRefiller: RAG filled gap '%s'", gap[:50])
            except Exception as e:
                logger.warning("AmmoRefiller: RAG search failed for '%s': %s", gap[:50], e)

    async def _search_memory(self, query: str) -> list[dict]:
        """Search MemoryPool for relevant memories."""
        if not self.memory:
            return []

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
                    return result
                if isinstance(result, dict) and "result" in result:
                    return result["result"]
                return []
        except Exception:
            pass

        # Fallback: in-memory
        if hasattr(self.memory, '_mem'):
            episodes = self.memory._mem.get("episode", [])
            query_lower = query.lower()
            matched = [
                e for e in episodes
                if query_lower in e.get("summary", "").lower()
                or query_lower in e.get("title", "").lower()
            ]
            return matched[:5]

        return []

    # ── Web search ───────────────────────────────────────────────

    async def _web_search_solution(self, error: str) -> None:
        """Search the web for error solutions."""
        # First try RAG (maybe we've seen this error before)
        if self.memory:
            try:
                results = await self._search_memory(f"error: {error[:200]}")
                for r in results[:1]:
                    fix = r.get("fix", r.get("summary", ""))[:300]
                    if fix:
                        self.ammo.add_fact(fix, source=f"rag_error:{error[:50]}",
                                           relevance=0.9)
                        return  # RAG hit, no need for web search
            except Exception:
                pass

        # Web search if no RAG hit
        if not self.web_tool:
            return

        try:
            from ..core import ToolResultStatus
            result = await self.web_tool.execute(
                action="search",
                query=f"fix: {error[:200]}",
                count=3,
            )
            if result.status == ToolResultStatus.SUCCESS and result.data:
                for item in result.data.get("results", [])[:2]:
                    title = item.get("title", "")[:100]
                    desc = item.get("description", "")[:200]
                    if desc:
                        self.ammo.add_fact(
                            f"{title}: {desc}",
                            source=f"web:{error[:50]}",
                            relevance=0.7,
                        )
                logger.debug("AmmoRefiller: web search filled error solution")
        except Exception as e:
            logger.warning("AmmoRefiller: web search failed: %s", e)

    # ── Findings review ──────────────────────────────────────────

    async def _review_findings(self) -> None:
        """Review findings from other workers."""
        if not self.project:
            return

        try:
            findings = self.project.read_findings(exclude_task_id=self.task_id)
            if findings:
                # Clear old findings, add new ones
                self.ammo.findings.clear()
                for f in findings[:3]:
                    self.ammo.add_finding(f["content"], task_id=f.get("task_id", ""))
                logger.debug("AmmoRefiller: reviewed %d findings from other workers", len(findings))
        except Exception as e:
            logger.warning("AmmoRefiller: review failed: %s", e)

    # ── Deep project search ──────────────────────────────────────

    async def _deep_project_search(self, query: str) -> None:
        """Deep search project memory for @project queries."""
        if not self.memory or not self.project:
            return

        try:
            results = await self._search_memory(query)
            for r in results[:3]:
                content = r.get("summary", r.get("content", ""))[:300]
                if content:
                    self.ammo.add_fact(content, source=f"project_deep:{query[:50]}",
                                       relevance=0.9)
        except Exception as e:
            logger.warning("AmmoRefiller: deep project search failed: %s", e)

    # ── Write finding ────────────────────────────────────────────

    def _write_finding(self, content: str) -> None:
        """Write a finding to shared temp dir for other workers."""
        if not self.project or not self.task_id:
            return

        try:
            self.project.write_finding(self.task_id, content[:500])
        except Exception as e:
            logger.warning("AmmoRefiller: write finding failed: %s", e)


class AmmoLearner:
    """Learns what ammo fills are effective.

    Tracks:
      - Which RAG results were actually used in the final output
      - Which web searches led to successful step completion
      - Which findings from other workers were relevant

    Used by Consolidation to improve future ammo selection.
    """

    def __init__(self):
        self.refill_history: list[dict] = []
        self.effectiveness_scores: dict[str, float] = {}  # source → avg effectiveness

    def record_refill(self, source: str, content_preview: str,
                      phase: str, trigger: str) -> None:
        """Record a refill event."""
        self.refill_history.append({
            "timestamp": time.time(),
            "source": source,
            "content_preview": content_preview[:100],
            "phase": phase,
            "trigger": trigger,
        })

    def record_effectiveness(self, source: str, was_used: bool) -> None:
        """Record whether ammo from a source was used in the final output."""
        current = self.effectiveness_scores.get(source, 0.5)
        # Exponential moving average
        alpha = 0.1
        self.effectiveness_scores[source] = (
            alpha * (1.0 if was_used else 0.0) + (1 - alpha) * current
        )

    def get_effective_sources(self, top_n: int = 3) -> list[str]:
        """Get the most effective ammo sources."""
        sorted_sources = sorted(
            self.effectiveness_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        return [s[0] for s in sorted_sources[:top_n]]

    def to_summary(self) -> dict:
        return {
            "total_refills": len(self.refill_history),
            "effectiveness": self.effectiveness_scores,
            "top_sources": self.get_effective_sources(3),
        }
