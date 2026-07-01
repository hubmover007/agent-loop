"""AmmoBox & AmmoRefiller — context management for worker agents.

AmmoBox: pinned context that survives compaction. Contains:
  - Project Card (Layer 1, auto)
  - Recent context (Layer 2, auto)
  - Key facts (Layer 3, on-demand)
  - Findings from other workers

AmmoRefiller: triggers during execution to refill ammo:
  - context > 70% → compress workspace + re-inject ammo
  - PLAN finds knowledge gap → search and fill
  - EXECUTE hits error → search solution
  - Every N steps → review ammo
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class AmmoItem:
    """A single ammo item in the ammo box."""
    content: str
    source: str  # "project_card" | "recent" | "search" | "finding" | "user"
    priority: int = 5  # 1=highest, 10=lowest
    timestamp: float = field(default_factory=time.time)
    token_est: int = 0  # estimated token count


class AmmoBox:
    """Pinned context that survives compaction.

    Structure:
      pinned:  Project Card + Recent Context (always present)
      facts:   Key facts from memory search (medium priority)
      findings: Findings from other workers
      decisions: Verified conclusions
      free:    Remaining budget for workspace
    """

    def __init__(self, max_tokens: int = 2000):
        self.max_tokens = max_tokens
        self.pinned: list[AmmoItem] = []
        self.facts: list[AmmoItem] = []
        self.findings: list[AmmoItem] = []
        self.decisions: list[AmmoItem] = []
        self._total_tokens: int = 0

    # ── Add items ────────────────────────────────────────────────

    def add_pinned(self, content: str, source: str = "project_card") -> None:
        """Add pinned content (never compressed)."""
        item = AmmoItem(
            content=content,
            source=source,
            priority=1,
            token_est=self._est_tokens(content),
        )
        self.pinned.append(item)
        self._total_tokens += item.token_est

    def add_fact(self, content: str, source: str = "search") -> None:
        """Add a key fact (can be evicted if low relevance)."""
        item = AmmoItem(
            content=content,
            source=source,
            priority=3,
            token_est=self._est_tokens(content),
        )
        self.facts.append(item)
        self._total_tokens += item.token_est
        self._evict_if_needed()

    def add_finding(self, content: str, task_id: str = "") -> None:
        """Add a finding from another worker."""
        item = AmmoItem(
            content=content,
            source=f"finding:{task_id}",
            priority=4,
            token_est=self._est_tokens(content),
        )
        self.findings.append(item)
        self._total_tokens += item.token_est
        self._evict_if_needed()

    def add_decision(self, content: str) -> None:
        """Add a verified decision (high priority, not evicted)."""
        item = AmmoItem(
            content=content,
            source="decision",
            priority=2,
            token_est=self._est_tokens(content),
        )
        self.decisions.append(item)
        self._total_tokens += item.token_est

    # ── Render to context ────────────────────────────────────────

    def to_context(self) -> str:
        """Render ammo box to a compact string for LLM context."""
        parts = []

        # Pinned (always)
        for item in self.pinned:
            parts.append(item.content)

        # Decisions (high priority)
        if self.decisions:
            parts.append("## 已验证结论")
            for d in self.decisions[-5:]:  # last 5
                parts.append(f"- {d.content[:200]}")

        # Key facts (medium)
        if self.facts:
            parts.append("## 关键事实")
            for f in self.facts[-10:]:  # last 10
                parts.append(f"- {f.content[:200]}")

        # Findings from others (medium-low)
        if self.findings:
            parts.append("## 其他 worker 的发现")
            for f in self.findings[-5:]:  # last 5
                parts.append(f"- {f.content[:200]}")

        return "\n\n".join(parts)

    # ── Management ───────────────────────────────────────────────

    def token_usage(self) -> int:
        return self._total_tokens

    def usage_ratio(self) -> float:
        return self._total_tokens / self.max_tokens if self.max_tokens > 0 else 0.0

    def clear_facts(self) -> None:
        """Clear facts (called after compaction)."""
        for f in self.facts:
            self._total_tokens -= f.token_est
        self.facts.clear()

    def clear_findings(self) -> None:
        """Clear findings (called after absorbing)."""
        for f in self.findings:
            self._total_tokens -= f.token_est
        self.findings.clear()

    # ── Internal ─────────────────────────────────────────────────

    def _evict_if_needed(self) -> None:
        """Evict low-priority items if over budget."""
        while self._total_tokens > self.max_tokens and self.facts:
            # Remove oldest fact
            old = self.facts.pop(0)
            self._total_tokens -= old.token_est
            logger.debug("AmmoBox: evicted fact from %s (%d tokens)",
                        old.source, old.token_est)

    @staticmethod
    def _est_tokens(text: str) -> int:
        """Rough token estimate: 1 token ≈ 3.5 chars for mixed text."""
        return max(1, len(text) // 4)


class AmmoRefiller:
    """Refills ammo during agent execution.

    Trigger conditions:
      1. context > 70% → compress workspace
      2. PLAN finds knowledge gap → search memory
      3. EXECUTE hits error → search solution
      4. Every N steps → review ammo
    """

    def __init__(self, ammo_box: AmmoBox, project: Any = None,
                 memory: Any = None, llm: Any = None):
        self.ammo = ammo_box
        self.project = project
        self.memory = memory
        self.llm = llm
        self._step_count = 0
        self._review_interval = 5  # review every 5 steps

    async def check_and_refill(self, ctx: dict) -> dict:
        """Check triggers and refill ammo if needed.

        Args:
            ctx: execution context with keys:
              - phase: "PLAN" | "EXECUTE" | "SELF_EVAL"
              - plan: list of steps (optional)
              - last_error: str (optional)
              - workspace_tokens: int (optional)

        Returns:
            dict with actions taken
        """
        actions = {"compressed": False, "searched": False, "reviewed": False}
        phase = ctx.get("phase", "")

        # Trigger 1: context too full
        ws_tokens = ctx.get("workspace_tokens", 0)
        total = self.ammo.token_usage() + ws_tokens
        if total > 0 and total / (self.ammo.max_tokens + 4000) > 0.7:
            await self._compress(ctx)
            actions["compressed"] = True

        # Trigger 2: PLAN finds knowledge gap
        if phase == "PLAN" and ctx.get("plan"):
            gaps = self._detect_gaps(ctx["plan"])
            if gaps:
                await self._search_and_fill(gaps)
                actions["searched"] = True

        # Trigger 3: EXECUTE hits error
        if ctx.get("last_error"):
            await self._search_solution(ctx["last_error"])
            actions["searched"] = True

        # Trigger 4: periodic review
        if phase == "EXECUTE":
            self._step_count += 1
            if self._step_count % self._review_interval == 0:
                await self._review()
                actions["reviewed"] = True

        return actions

    def _detect_gaps(self, plan: list[dict]) -> list[str]:
        """Detect knowledge gaps in the plan.

        Simple heuristic: look for keywords that suggest external info needed.
        """
        gaps = []
        gap_keywords = ["搜索", "查找", "调研", "search", "find", "lookup",
                        "文档", "document", "参考", "reference"]

        for step in plan:
            desc = ""
            if isinstance(step, dict):
                desc = step.get("description", "") or step.get("tool", "") or str(step)
            else:
                desc = str(step)

            desc_lower = desc.lower()
            if any(kw in desc_lower for kw in gap_keywords):
                gaps.append(desc[:200])

        return gaps[:3]  # cap at 3 gaps

    async def _search_and_fill(self, gaps: list[str]) -> None:
        """Search memory for gap-filling info."""
        if not self.memory:
            return

        for gap in gaps:
            try:
                # Search MemoryPool
                results = await self._search_memory(gap)
                for r in results[:2]:  # top 2 per gap
                    content = r.get("summary", r.get("content", ""))[:300]
                    if content:
                        self.ammo.add_fact(content, source=f"gap_search:{gap[:50]}")
            except Exception as e:
                logger.warning("AmmoRefiller: search failed for gap '%s': %s", gap[:50], e)

    async def _search_solution(self, error: str) -> None:
        """Search for solution when EXECUTE hits error."""
        if not self.memory:
            return
        try:
            results = await self._search_memory(f"error: {error[:200]}")
            for r in results[:2]:
                content = r.get("fix", r.get("summary", ""))[:300]
                if content:
                    self.ammo.add_fact(content, source=f"error_search:{error[:50]}")
        except Exception as e:
            logger.warning("AmmoRefiller: solution search failed: %s", e)

    async def _compress(self, ctx: dict) -> None:
        """Compress workspace — clear old facts to make room."""
        self.ammo.clear_facts()
        logger.info("AmmoRefiller: compressed workspace (cleared facts)")

    async def _review(self) -> None:
        """Review ammo box — refresh findings from other workers."""
        if not self.project:
            return
        try:
            findings = self.project.read_findings()
            if findings:
                self.ammo.clear_findings()
                for f in findings[:3]:
                    self.ammo.add_finding(f["content"], task_id=f.get("task_id", ""))
        except Exception as e:
            logger.warning("AmmoRefiller: review failed: %s", e)

    async def _search_memory(self, query: str) -> list[dict]:
        """Search MemoryPool for relevant memories."""
        if not self.memory:
            return []

        # Try SurrealDB query
        try:
            if hasattr(self.memory, '_db') and self.memory._db:
                project_filter = ""
                if self.project:
                    project_filter = f"AND project = '{self.project.project_id}'"

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
            # Simple keyword match
            query_lower = query.lower()
            matched = [
                e for e in episodes
                if query_lower in e.get("summary", "").lower()
                or query_lower in e.get("title", "").lower()
            ]
            return matched[:5]

        return []
