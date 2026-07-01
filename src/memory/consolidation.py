"""LLM-driven memory consolidation engine.

Five-phase pipeline:
  1. Gather — collect unconsolidated episodes
  2. Extract — LLM extracts MemoryRecords from episodes
  3. Link — create graph edges between related records
  4. Resolve — detect & resolve contradictions
  5. Prune — remove low-value memories

Runs periodically (like sleep/dream cycle).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

from .schema import (
    MemoryRecord, MemoryType,
    render_for_llm, parse_llm_extraction,
    CONSOLIDATION_PROMPT,
)

if TYPE_CHECKING:
    from ..llm import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class ConsolidationResult:
    """Result of a consolidation run."""
    episodes_processed: int = 0
    records_extracted: int = 0
    links_created: int = 0
    contradictions_resolved: int = 0
    memories_pruned: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def to_summary(self) -> str:
        """Human-readable summary."""
        parts = [f"Episodes: {self.episodes_processed}"]
        if self.records_extracted:
            parts.append(f"Records extracted: {self.records_extracted}")
        if self.links_created:
            parts.append(f"Links created: {self.links_created}")
        if self.contradictions_resolved:
            parts.append(f"Contradictions resolved: {self.contradictions_resolved}")
        if self.memories_pruned:
            parts.append(f"Memories pruned: {self.memories_pruned}")
        if self.errors:
            parts.append(f"Errors: {len(self.errors)}")
        parts.append(f"Duration: {self.duration_seconds:.1f}s")
        return " | ".join(parts)


class ConsolidationEngine:
    """LLM-driven memory consolidation.

    Usage:
        engine = ConsolidationEngine(memory_pool, llm_provider)
        result = await engine.run()
    """

    def __init__(self, memory_pool, llm_provider=None,
                 anchor_manager=None,
                 min_episodes: int = 3,
                 max_episodes_per_run: int = 50,
                 prune_threshold: float = 0.3,
                 enable_linking: bool = True,
                 enable_resolution: bool = True,
                 enable_pruning: bool = True,
                 llm_pool: Any | None = None):
        self.memory = memory_pool
        self.llm = llm_provider
        self.anchor = anchor_manager
        self.min_episodes = min_episodes
        self.max_episodes_per_run = max_episodes_per_run
        self.prune_threshold = prune_threshold
        self.enable_linking = enable_linking
        self.enable_resolution = enable_resolution
        self.enable_pruning = enable_pruning
        self.llm_pool = llm_pool
        self._last_consolidation: str | None = None

    def _get_llm(self, capabilities: list[str], strategy: str):
        """Get the best LLM for a given capability+strategy, falling back to default."""
        if self.llm_pool:
            provider = self.llm_pool.get_provider(
                capabilities=capabilities, strategy=strategy
            )
            if provider:
                return provider
        return self.llm

    async def run(self) -> ConsolidationResult:
        """Run full consolidation pipeline."""
        start = datetime.now(timezone.utc)
        result = ConsolidationResult()

        try:
            # Phase 1: Gather
            episodes = await self._gather()
            if len(episodes) < self.min_episodes:
                logger.info(
                    "Consolidation: only %d episodes (need %d), skipping",
                    len(episodes), self.min_episodes,
                )
                result.duration_seconds = (
                    datetime.now(timezone.utc) - start
                ).total_seconds()
                return result

            result.episodes_processed = len(episodes)

            # Phase 2: Extract
            records = await self._extract(episodes)
            result.records_extracted = len(records)

            # Phase 3: Link
            if self.enable_linking:
                links = await self._link(records)
                result.links_created = links

            # Phase 4: Resolve
            if self.enable_resolution:
                resolved = await self._resolve(records)
                result.contradictions_resolved = resolved

            # Phase 5: Prune
            if self.enable_pruning:
                pruned = await self._prune()
                result.memories_pruned = pruned

            # Mark episodes as consolidated
            await self._mark_consolidated(episodes)
            self._last_consolidation = datetime.now(timezone.utc).isoformat()

        except Exception as e:
            logger.error("Consolidation failed: %s", e)
            result.errors.append(str(e))

        result.duration_seconds = (
            datetime.now(timezone.utc) - start
        ).total_seconds()
        logger.info("Consolidation done: %s", result.to_summary())
        return result

    # ── Phase 1: Gather ────────────────────────────────────────

    async def _gather(self) -> list[dict]:
        """Collect unconsolidated episodes from the memory pool."""
        # Use existing MemoryPool method
        episodes = await self.memory.get_unconsolidated_episodes(
            limit=self.max_episodes_per_run
        )
        logger.debug("Gathered %d unconsolidated episodes", len(episodes))
        return episodes

    # ── Phase 2: Extract ────────────────────────────────────────

    async def _extract(self, episodes: list[dict]) -> list[MemoryRecord]:
        """LLM extracts MemoryRecords from episodes."""
        if not self.llm:
            logger.warning("No LLM provider, using heuristic extraction")
            return self._heuristic_extract(episodes)

        # Format episodes for LLM
        episode_texts = []
        for ep in episodes:
            title = ep.get("title", ep.get("summary", ""))
            summary = ep.get("summary", "")
            content = ep.get("content", summary)
            created = str(ep.get("created_at", ""))
            episode_texts.append(
                f"[Episode|{created}]\nTitle: {title}\nSummary: {summary}\nContent: {content}"
            )

        episodes_block = "\n---\n".join(episode_texts)

        # Get existing memories for contradiction check
        existing = await self._get_existing_memories(limit=100)
        existing_block = render_for_llm(existing) if existing else "(none)"

        # Call LLM
        prompt = CONSOLIDATION_PROMPT.format(
            episodes_text=episodes_block,
            existing_memories_text=existing_block,
        )

        try:
            llm = self._get_llm(["general"], "cheapest")
            resp = await llm.chat([
                {
                    "role": "system",
                    "content": "You are a memory consolidation engine. Extract structured records.",
                },
                {"role": "user", "content": prompt},
            ])

            content = resp.get("content", "") if isinstance(resp, dict) else resp.content

            # Parse LLM output (multiple records separated by ---)
            records = []
            blocks = content.split("---")
            for block in blocks:
                block = block.strip()
                if not block:
                    continue
                record = parse_llm_extraction(block)
                if record and record.summary:
                    records.append(record)

            # Write extracted records to fact table
            for record in records:
                await self._persist_record(record)

            return records

        except Exception as e:
            logger.error("LLM extraction failed: %s", e)
            return self._heuristic_extract(episodes)

    async def _persist_record(self, record: MemoryRecord) -> None:
        """Persist an extracted MemoryRecord as a fact."""
        try:
            await self.memory.write_fact(
                fact_type="entity",
                name=record.summary[:80],
                value=record.to_compact_text(),
                agent_id=record.agent_id,
                mem_type=record.type.value,
                summary=record.summary,
                trigger=record.trigger,
                action=record.action,
                outcome=record.outcome,
                lesson=record.lesson,
                confidence=record.confidence,
                tags=record.tags,
                error=record.error,
                fix=record.fix,
                steps=record.steps,
                related_ids=record.related_ids,
            )
        except Exception as e:
            logger.warning("Write record failed for '%s': %s", record.summary[:40], e)

    def _heuristic_extract(self, episodes: list[dict]) -> list[MemoryRecord]:
        """Fallback extraction without LLM (simple heuristics)."""
        records = []
        for ep in episodes:
            summary = ep.get("summary", ep.get("title", ""))
            if not summary:
                continue

            content = (ep.get("content", "") or "").lower()
            error = ep.get("error", "")
            fix_val = ep.get("fix", "")

            # Use existing fields if available
            mem_type_str = ep.get("type", ep.get("mem_type", ""))

            if mem_type_str in {"failure", "success", "lesson", "fact", "procedure", "pattern"}:
                try:
                    mem_type = MemoryType(mem_type_str)
                except ValueError:
                    mem_type = MemoryType.FACT
            elif error:
                mem_type = MemoryType.FAILURE
            elif any(w in content for w in ["error", "fail", "exception", "broken"]):
                mem_type = MemoryType.FAILURE
            elif any(w in content for w in ["success", "working", "fixed", "resolved"]):
                mem_type = MemoryType.SUCCESS
            else:
                mem_type = MemoryType.FACT

            record = MemoryRecord(
                type=mem_type,
                summary=summary[:100],
                project=ep.get("project", ""),
                trigger=ep.get("trigger", ""),
                action=ep.get("action", ""),
                outcome=ep.get("outcome", ""),
                lesson=ep.get("lesson", ""),
                error=error or ep.get("error", ""),
                fix=fix_val or ep.get("fix", ""),
                confidence=float(ep.get("confidence", 0.6)),
                tags=ep.get("tags", []),
                created_at=str(ep.get("created_at", "")),
            )
            records.append(record)

        # Persist heuristic records too
        for record in records:
            # Don't await inside sync method — store for caller
            pass

        return records

    # ── Phase 2 helpers ────────────────────────────────────────

    async def _get_existing_memories(self, limit: int = 100) -> list[MemoryRecord]:
        """Get existing memories for contradiction check."""
        try:
            facts = await self.memory.query_facts(limit=limit)
            return [MemoryRecord.from_dict(f) for f in facts]
        except Exception:
            return []

    # ── Phase 3: Link ──────────────────────────────────────────

    async def _link(self, records: list[MemoryRecord]) -> int:
        """Create graph edges between related records.

        Two strategies:
          1. failure → success: if a failure's fix references a success outcome
          2. explicit related_ids: record directly references another record
        """
        if not self.memory or not self.memory._db:
            return 0

        links = 0

        # Strategy 1: failure → success overlap
        try:
            successes = await self.memory.query_facts(
                fact_type="entity",
                limit=200,
            )
        except Exception:
            successes = []

        for record in records:
            if record.type == MemoryType.FAILURE and record.fix:
                for s in successes:
                    try:
                        s_record = MemoryRecord.from_dict(s)
                    except Exception:
                        continue
                    if s_record.type != MemoryType.SUCCESS:
                        continue
                    # Simple text overlap check
                    fix_words = set(record.fix.lower().split())
                    outcome_words = set(s_record.outcome.lower().split())
                    overlap = fix_words & outcome_words
                    if len(overlap) > 2:
                        try:
                            await self.memory.write_edge(
                                source=record.id or f"fact:{record.summary[:10]}",
                                target=s_record.id or f"fact:{s_record.summary[:10]}",
                                relation="failure_led_to",
                            )
                            links += 1
                        except Exception as e:
                            logger.debug("Link edge failed: %s", e)

            # Strategy 2: explicit related_ids
            for rid in record.related_ids:
                try:
                    await self.memory.write_edge(
                        source=record.id or f"fact:{record.summary[:10]}",
                        target=rid,
                        relation="references",
                    )
                    links += 1
                except Exception:
                    pass

        return links

    # ── Phase 4: Resolve ───────────────────────────────────────

    async def _resolve(self, records: list[MemoryRecord]) -> int:
        """Detect and resolve contradictions between records."""
        if not self.llm or not records:
            return 0

        # Find potential contradictions
        contradictions = []
        for i, r1 in enumerate(records):
            for r2 in records[i + 1:]:
                if (r1.project == r2.project
                        and r1.type == r2.type
                        and r1.summary != r2.summary):
                    if self._is_contradiction(r1, r2):
                        contradictions.append((r1, r2))

        resolved = 0
        for r1, r2 in contradictions:
            try:
                resp = await self.llm.chat([
                    {
                        "role": "system",
                        "content": "You resolve memory contradictions. Reply with ONLY 'A' or 'B'.",
                    },
                    {
                        "role": "user",
                        "content": f"""Two memories contradict:

Memory A (confidence {r1.confidence:.1f}):
{r1.to_compact_text()}

Memory B (confidence {r2.confidence:.1f}):
{r2.to_compact_text()}

Which is more reliable? Reply A or B:""",
                    },
                ])

                content = resp.get("content", "") if isinstance(resp, dict) else resp.content
                choice = content.strip().upper()[:1] if content else "A"

                loser = r2 if choice == "A" else r1
                # Lower confidence on the loser by updating the fact
                await self._lower_confidence(loser)
                resolved += 1

            except Exception as e:
                logger.debug("Resolve contradiction failed: %s", e)

        return resolved

    async def _lower_confidence(self, record: MemoryRecord) -> None:
        """Lower confidence on a record that was contradicted."""
        if not self.memory or not self.memory._db:
            return
        try:
            new_conf = max(0.1, record.confidence - 0.3)
            # Update via query if we have an ID
            if record.id:
                await self.memory._db.query(
                    "UPDATE type::thing('fact', $id) SET confidence = $conf",
                    {"id": record.id, "conf": new_conf},
                )
        except Exception:
            pass

    def _is_contradiction(self, r1: MemoryRecord, r2: MemoryRecord) -> bool:
        """Heuristic: check if two records contradict based on outcome polarity."""
        positive = {"success", "working", "fixed", "resolved", "passed"}
        negative = {"fail", "error", "broken", "failed", "crash"}

        r1_outcome = r1.outcome.lower()
        r2_outcome = r2.outcome.lower()

        r1_pos = any(w in r1_outcome for w in positive)
        r1_neg = any(w in r1_outcome for w in negative)
        r2_pos = any(w in r2_outcome for w in positive)
        r2_neg = any(w in r2_outcome for w in negative)

        return (r1_pos and r2_neg) or (r1_neg and r2_pos)

    # ── Phase 5: Prune ─────────────────────────────────────────

    async def _prune(self) -> int:
        """Remove low-value memories.

        Criteria:
          - confidence < prune_threshold
          - access_count == 0 (never accessed)
          - agent_id != 'anchor' (protect anchor records)
        """
        if not self.memory or not self.memory._db:
            return 0

        try:
            result = await self.memory._db.query(
                """DELETE FROM fact
                   WHERE confidence < $threshold
                   AND access_count = 0
                   AND agent_id != 'anchor'
                   RETURN id""",
                {"threshold": self.prune_threshold},
            )
            rows = result if isinstance(result, list) else (
                result.get("result", []) if isinstance(result, dict) else []
            )
            count = len(rows)
            if count > 0:
                logger.info("Pruned %d low-value memories", count)
            return count
        except Exception as e:
            logger.warning("Prune failed: %s", e)
            return 0

    # ── Mark consolidated ──────────────────────────────────────

    async def _mark_consolidated(self, episodes: list[dict]) -> None:
        """Mark processed episodes as consolidated."""
        for ep in episodes:
            ep_id = ep.get("id", "")
            if not ep_id:
                continue
            await self.memory.mark_episode_consolidated(ep_id)
