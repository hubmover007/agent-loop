"""AmmoBox & AmmoRefiller — context management for worker agents.

Ammo lifecycle:
  1. Init:     Project Card + Recent Sessions (auto, pinned)
  2. Retrieve: RAG from MemoryPool (auto, on-demand)
  3. Fill:     Web search when knowledge gap detected (on-demand)
  4. Share:    Worker findings via project temp dir (auto)
  5. Learn:    Consolidation extracts facts → better ammo next time

No active cleaning or compaction. Data flows in naturally,
old items are evicted by FIFO/relevance when capacity is exceeded.
Importance judgment is left to the agent's own reasoning (constrained
via AGENTS.md / system prompt, not code-level rules).

AmmoBox layers (by eviction order: workspace → findings → facts):
  pinned:     Project Card + Recent Context (never evicted)
  decisions:  Verified conclusions (never evicted)
  facts:      Key facts from RAG/search (evicted by lowest relevance)
  findings:   Other workers' findings (FIFO eviction)
  workspace:  Current execution steps (FIFO eviction)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AmmoItem:
    """A single ammo item."""
    content: str
    source: str  # "project_card" | "recent" | "rag" | "web" | "finding" | "decision" | "step"
    priority: int = 5  # 1=highest (pinned), 10=lowest (workspace)
    timestamp: float = field(default_factory=time.time)
    token_est: int = 0
    relevance: float = 1.0  # 0-1, used for eviction
    step_num: int = 0  # which step produced this item


class AmmoBox:
    """Pinned context for a worker agent.

    No active cleaning or compaction. Eviction is automatic:
      - workspace: FIFO (oldest first)
      - findings: FIFO (oldest first)
      - facts: lowest relevance first

    Token budget: ~2000 tokens. Pinned + decisions are never evicted.
    """

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.pinned: list[AmmoItem] = []
        self.decisions: list[AmmoItem] = []
        self.facts: list[AmmoItem] = []
        self.findings: list[AmmoItem] = []
        self.workspace: list[AmmoItem] = []
        self._total_tokens: int = 0

    # ── Add items ────────────────────────────────────────────────

    def add_pinned(self, content: str, source: str = "project_card") -> None:
        """Add pinned content (never evicted)."""
        item = AmmoItem(content=content, source=source, priority=1,
                        token_est=self._est_tokens(content))
        self.pinned.append(item)
        self._total_tokens += item.token_est

    def add_decision(self, content: str) -> None:
        """Add a verified decision (never evicted)."""
        item = AmmoItem(content=content, source="decision", priority=2,
                        token_est=self._est_tokens(content))
        self.decisions.append(item)
        self._total_tokens += item.token_est
        self._evict_if_needed()

    def add_fact(self, content: str, source: str = "rag",
                 relevance: float = 1.0) -> None:
        """Add a key fact (evicted by lowest relevance when over budget)."""
        item = AmmoItem(content=content, source=source, priority=3,
                        token_est=self._est_tokens(content),
                        relevance=relevance)
        self.facts.append(item)
        self._total_tokens += item.token_est
        self._evict_if_needed()

    def add_finding(self, content: str, task_id: str = "") -> None:
        """Add a finding from another worker (FIFO eviction)."""
        item = AmmoItem(content=content, source=f"finding:{task_id}",
                        priority=4, token_est=self._est_tokens(content))
        self.findings.append(item)
        self._total_tokens += item.token_est
        self._evict_if_needed()

    def add_workspace(self, content: str, step_num: int = 0) -> None:
        """Add a workspace item (FIFO eviction)."""
        item = AmmoItem(content=content, source=f"step:{step_num}",
                        priority=5, token_est=self._est_tokens(content),
                        step_num=step_num)
        self.workspace.append(item)
        self._total_tokens += item.token_est
        self._evict_if_needed()

    # ── Render to context ────────────────────────────────────────

    def to_context(self, include_workspace: bool = True) -> str:
        """Render ammo box to compact string for LLM context."""
        parts = []

        for item in self.pinned:
            parts.append(item.content)

        if self.decisions:
            parts.append("## 已验证结论")
            for d in self.decisions[-5:]:
                parts.append(f"- {d.content[:200]}")

        if self.facts:
            parts.append("## 关键事实")
            for f in self.facts[-10:]:
                parts.append(f"- {f.content[:200]}")

        if self.findings:
            parts.append("## 其他 worker 发现")
            for f in self.findings[-5:]:
                parts.append(f"- {f.content[:200]}")

        if include_workspace and self.workspace:
            parts.append("## 执行记录")
            for w in self.workspace[-5:]:
                parts.append(f"- {w.content[:150]}")

        return "\n\n".join(parts)

    def to_pinned_only(self) -> str:
        """Render only pinned + decisions."""
        parts = []
        for item in self.pinned:
            parts.append(item.content)
        if self.decisions:
            parts.append("## 已验证结论")
            for d in self.decisions[-3:]:
                parts.append(f"- {d.content[:200]}")
        return "\n\n".join(parts)

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
        }

    # ── Internal: natural eviction ───────────────────────────────

    def _evict_if_needed(self) -> None:
        """Evict low-priority items when over budget.

        Order: workspace (oldest first) → findings (oldest first) → facts (lowest relevance).
        Pinned and decisions are never evicted.
        """
        while self._total_tokens > self.max_tokens:
            # Workspace: FIFO
            if self.workspace:
                old = self.workspace.pop(0)
                self._total_tokens -= old.token_est
                continue
            # Findings: FIFO
            if self.findings:
                old = self.findings.pop(0)
                self._total_tokens -= old.token_est
                continue
            # Facts: lowest relevance first
            if self.facts:
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

    Triggers:
      1. PLAN finds knowledge gap → RAG search (MemoryPool)
      2. EXECUTE hits error → web search for solution
      3. Every N steps → review findings from other workers
      4. @project mention → deep project memory search
      5. Step success → write finding to shared temp dir

    No active compaction or cleaning. AmmoBox handles eviction naturally.
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
        self.web_tool = web_tool
        self.task_id = task_id
        self.max_refills = max_refills
        self._step_count = 0
        self._review_interval = 3
        self._refill_count = 0

    async def check_and_refill(self, ctx: dict) -> dict:
        """Check triggers and refill ammo if needed.

        Args:
            ctx: execution context:
              - phase: "PLAN" | "EXECUTE" | "SELF_EVAL"
              - plan: list of steps (optional)
              - last_error: str (optional)
              - last_step_success: bool (optional)
              - step_num: int (optional)
              - step_desc: str (optional)
              - user_input: str (optional, for @project check)

        Returns:
            dict with actions taken
        """
        actions = {
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

        # Trigger 1: PLAN finds knowledge gap
        if phase == "PLAN" and ctx.get("plan"):
            gaps = self._detect_gaps(ctx["plan"])
            if gaps:
                await self._rag_search(gaps)
                actions["rag_searched"] = True
                self._refill_count += 1

        # Trigger 2: EXECUTE hits error → web search
        if ctx.get("last_error"):
            await self._web_search_solution(ctx["last_error"])
            actions["web_searched"] = True
            self._refill_count += 1

        # Trigger 3: periodic review
        if phase == "EXECUTE":
            self._step_count += 1
            if self._step_count % self._review_interval == 0:
                await self._review_findings()
                actions["reviewed"] = True

        # Trigger 4: @project mention
        user_input = ctx.get("user_input", "")
        if user_input and "@project" in user_input.lower():
            await self._deep_project_search(user_input)
            actions["project_searched"] = True
            self._refill_count += 1

        # Trigger 5: step success → write finding to shared
        if ctx.get("last_step_success") and self.project:
            step_desc = ctx.get("step_desc", "")
            if step_desc:
                self._write_finding(step_desc)
                actions["finding_written"] = True

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
                gaps.append(desc[:200])

        return gaps[:3]

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
        """Search for error solutions — RAG first, web second."""
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
                for f in self.ammo.findings:
                    self.ammo._total_tokens -= f.token_est
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

    Tracks refill effectiveness for future optimization.
    Used by Consolidation to improve ammo selection over time.
    """

    def __init__(self):
        self.refill_history: list[dict] = []
        self.effectiveness_scores: dict[str, float] = {}

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
