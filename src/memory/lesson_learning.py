"""Success/Failure associative learning system.

Generates StructuredLessons from failures, links them to future successes,
and provides retrieval for "what did we learn from similar failures?"

Key concepts:
  - StructuredLesson: root_cause + fix_approach + applicable_conditions
  - FailureChain: failure → lesson → success (or failure → lesson → another failure)
  - Similarity matching: find similar past failures when new failure occurs

Architecture:
  1. Failure occurs → extract StructuredLesson (root_cause + fix_approach)
  2. Store as fact (type=lesson, related_ids=[failure_id])
  3. Later, similar scenario retrieves lesson → avoid repeating failure
  4. Success → link back to lesson → failure→lesson→success chain

Inspired by OpenViking Structured Lesson pattern.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .schema import MemoryRecord, MemoryType

logger = logging.getLogger(__name__)

# ── Time budget for LLM lesson extraction (seconds) ──────────────
LLM_EXTRACTION_TIMEOUT = 15.0


@dataclass
class StructuredLesson:
    """A lesson extracted from a failure.

    Unlike a simple MemoryRecord, StructuredLesson has:
      - root_cause: why the failure happened (not just "what")
      - fix_approach: how to fix it (not just "try something else")
      - applicable_conditions: when this lesson applies
      - verified: whether the fix was confirmed by a later success
    """

    id: str = ""
    failure_id: str = ""  # reference to the failure record
    root_cause: str = ""
    fix_approach: str = ""
    applicable_conditions: str = ""  # when does this lesson apply?

    # The fix steps (actionable)
    fix_steps: list[str] = field(default_factory=list)

    # Verification
    verified: bool = False
    verified_by_success_id: str = ""
    verified_at: str = ""

    # Metadata
    confidence: float = 0.5
    created_at: str = ""
    tags: list[str] = field(default_factory=list)

    def to_record(self) -> MemoryRecord:
        """Convert to MemoryRecord for storage."""
        summary = f"Lesson: {self.root_cause[:60]}"
        return MemoryRecord(
            type=MemoryType.LESSON,
            summary=summary,
            trigger=self.applicable_conditions,
            action=self.fix_approach,
            outcome="Learned from failure",
            lesson=self.fix_approach,
            confidence=self.confidence,
            related_ids=[self.failure_id] if self.failure_id else [],
            related_type="cause",
            steps=self.fix_steps,
            tags=self.tags + ["lesson", "structured"],
            created_at=self.created_at,
        )

    @classmethod
    def from_record(cls, record: MemoryRecord, raw_value: str = "") -> "StructuredLesson":
        """Create from a MemoryRecord (type=lesson).

        If raw_value is provided (the to_compact_text() output stored in
        the fact's 'value' field), parse it to recover the original fields.
        """
        if raw_value:
            parsed = cls._parse_compact_text(raw_value)
            return cls(
                id=record.id,
                root_cause=parsed.get("root_cause") or cls._extract_root_cause(record),
                fix_approach=parsed.get("fix_approach", record.action or record.lesson),
                applicable_conditions=parsed.get(
                    "applicable_conditions", record.trigger
                ),
                fix_steps=parsed.get("fix_steps", record.steps),
                verified=parsed.get("verified", False),
                verified_by_success_id=parsed.get("verified_by_success_id", ""),
                confidence=parsed.get("confidence", record.confidence),
                created_at=record.created_at,
                tags=record.tags,
            )
        return cls(
            id=record.id,
            root_cause=cls._extract_root_cause(record),
            fix_approach=record.action or record.lesson,
            applicable_conditions=record.trigger,
            fix_steps=record.steps,
            confidence=record.confidence,
            created_at=record.created_at,
            tags=record.tags,
        )

    @staticmethod
    def _extract_root_cause(record: MemoryRecord) -> str:
        """Extract root_cause from a MemoryRecord fallback fields.

        Priority: summary (stripped of "Lesson: " prefix) > error > summary raw.
        """
        summary = record.summary
        if summary.startswith("Lesson: "):
            return summary[len("Lesson: "):]
        if record.error:
            return record.error
        return record.trigger or summary

    @staticmethod
    def _parse_compact_text(text: str) -> dict:
        """Parse to_compact_text() output back into a dict."""
        result: dict = {}
        lines = text.strip().split("\n")
        for line in lines:
            if line.startswith("[LESSON|"):
                # Parse header: [LESSON|verified=True|conf:0.9]
                inner = line.strip("[]")
                parts = inner.split("|")
                for part in parts[1:]:  # skip "LESSON"
                    if "=" in part:
                        k, v = part.split("=", 1)
                        if k == "verified":
                            result["verified"] = v.lower() == "true"
                        elif k == "conf":
                            try:
                                result["confidence"] = float(v)
                            except ValueError:
                                pass
            elif ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if key == "root_cause":
                    result["root_cause"] = value
                elif key == "fix":
                    result["fix_approach"] = value
                elif key == "applies_when":
                    result["applicable_conditions"] = value
                elif key == "steps":
                    result["fix_steps"] = [
                        s.strip() for s in value.split(",") if s.strip()
                    ]
                elif key == "verified_by":
                    result["verified_by_success_id"] = value
        return result

    def to_compact_text(self) -> str:
        """Render for LLM consumption."""
        lines = [
            f"[LESSON|verified={self.verified}|conf:{self.confidence:.1f}]",
            f"root_cause: {self.root_cause}",
            f"fix: {self.fix_approach}",
        ]
        if self.applicable_conditions:
            lines.append(f"applies_when: {self.applicable_conditions}")
        if self.fix_steps:
            lines.append(f"steps: {', '.join(self.fix_steps)}")
        if self.verified:
            lines.append(f"verified_by: {self.verified_by_success_id}")
        return "\n".join(lines)


@dataclass
class FailureChain:
    """A chain of related failure → lesson → success/failure.

    Tracks the full lifecycle of a problem:
      1. Initial failure
      2. Lesson extracted
      3. Retry (success or another failure)
      4. If success: mark lesson as verified
      5. If another failure: refine lesson
    """

    failure_ids: list[str] = field(default_factory=list)
    lesson_ids: list[str] = field(default_factory=list)
    success_ids: list[str] = field(default_factory=list)

    def is_resolved(self) -> bool:
        """Is the chain resolved (has at least one success)?"""
        return len(self.success_ids) > 0

    def retry_count(self) -> int:
        """Number of failure attempts before success."""
        return len(self.failure_ids)

    def to_summary(self) -> str:
        """Human-readable summary."""
        status = "✅ resolved" if self.is_resolved() else "❌ unresolved"
        return (
            f"FailureChain({status}, "
            f"failures={len(self.failure_ids)}, "
            f"lessons={len(self.lesson_ids)}, "
            f"successes={len(self.success_ids)})"
        )


class LessonLearner:
    """Extracts StructuredLessons from failures using LLM.

    Usage:
        learner = LessonLearner(memory_pool, llm_provider)
        lesson = await learner.extract_lesson(failure_record)
        await learner.store_lesson(lesson)

        # Later, when searching for relevant lessons:
        lessons = await learner.find_similar_lessons(context="deploying to ECS")
    """

    def __init__(
        self,
        memory_pool,
        llm_provider=None,
        similarity_threshold: float = 0.5,
        max_lessons_per_query: int = 5,
    ):
        self.memory = memory_pool
        self.llm = llm_provider
        self.similarity_threshold = similarity_threshold
        self.max_lessons = max_lessons_per_query

        self._embedding_service = getattr(memory_pool, "_embedding_service", None)

    async def extract_lesson(self, failure: MemoryRecord) -> StructuredLesson:
        """Extract a StructuredLesson from a failure record.

        Uses LLM if available, falls back to heuristic.
        """
        if self.llm:
            return await self._llm_extract(failure)
        return self._heuristic_extract(failure)

    async def _llm_extract(self, failure: MemoryRecord) -> StructuredLesson:
        """Use LLM to extract structured lesson."""
        prompt = f"""Analyze this failure and extract a structured lesson.

Failure record:
{failure.to_compact_text()}

Extract:
1. root_cause: Why did this fail? (the fundamental reason, not surface symptom)
2. fix_approach: How to fix it? (actionable steps, not vague advice)
3. applicable_conditions: When does this lesson apply? (what scenarios)
4. fix_steps: Step-by-step fix procedure (numbered list)
5. confidence: 0-1, how confident are you in this lesson?

Respond in this exact format:
root_cause: ...
fix_approach: ...
applicable_conditions: ...
fix_steps: step1, step2, step3
confidence: 0.8
"""
        try:
            import asyncio

            resp = await asyncio.wait_for(
                self.llm.chat(
                    [
                        {"role": "system", "content": "You are a failure analysis engine."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                ),
                timeout=LLM_EXTRACTION_TIMEOUT,
            )

            return self._parse_llm_response(resp.content, failure)
        except asyncio.TimeoutError:
            logger.warning("LLM lesson extraction timed out after %.1fs", LLM_EXTRACTION_TIMEOUT)
            return self._heuristic_extract(failure)
        except Exception as e:
            logger.warning("LLM lesson extraction failed: %s", e)
            return self._heuristic_extract(failure)

    def _parse_llm_response(self, text: str, failure: MemoryRecord) -> StructuredLesson:
        """Parse LLM response into StructuredLesson."""
        fields: dict[str, Any] = {}
        for line in text.strip().split("\n"):
            if ":" in line:
                key, _, value = line.partition(":")
                key = key.strip()
                value = value.strip()
                if key == "fix_steps":
                    fields["fix_steps"] = [s.strip() for s in value.split(",") if s.strip()]
                elif key == "confidence":
                    try:
                        fields["confidence"] = float(value)
                    except ValueError:
                        fields["confidence"] = 0.5
                else:
                    fields[key] = value

        return StructuredLesson(
            failure_id=failure.id,
            root_cause=fields.get("root_cause", failure.error or failure.summary),
            fix_approach=fields.get("fix_approach", failure.fix or "Unknown fix"),
            applicable_conditions=fields.get("applicable_conditions", failure.trigger),
            fix_steps=fields.get("fix_steps", []),
            confidence=fields.get("confidence", 0.5),
            created_at=datetime.now(timezone.utc).isoformat(),
            tags=failure.tags,
        )

    def _heuristic_extract(self, failure: MemoryRecord) -> StructuredLesson:
        """Fallback extraction without LLM.

        Uses all available failure fields to build a reasonably complete lesson.
        """
        # Combine summary and error for richer root_cause
        if failure.summary and failure.error:
            root_cause = f"{failure.summary}: {failure.error}"
        else:
            root_cause = failure.error or failure.summary

        # Combine fix and lesson for richer fix_approach
        if failure.fix and failure.lesson:
            fix_approach = f"{failure.fix} (lesson: {failure.lesson})"
        else:
            fix_approach = failure.fix or failure.lesson or "Investigate and retry"

        return StructuredLesson(
            failure_id=failure.id,
            root_cause=root_cause,
            fix_approach=fix_approach,
            applicable_conditions=failure.trigger,
            fix_steps=failure.steps or [],
            confidence=0.4,  # low confidence for heuristic
            created_at=datetime.now(timezone.utc).isoformat(),
            tags=failure.tags,
        )

    async def store_lesson(self, lesson: StructuredLesson) -> str:
        """Store a StructuredLesson in the memory pool.

        Returns the fact ID.
        """
        record = lesson.to_record()

        fact_id = await self.memory.write_fact(
            fact_type="entity",
            name=record.summary[:80],
            value=lesson.to_compact_text(),
            agent_id="shared",
            mem_type="lesson",
            summary=record.summary,
            trigger=record.trigger,
            action=record.action,
            outcome=record.outcome,
            lesson=record.lesson,
            confidence=record.confidence,
            tags=record.tags,
            steps=record.steps,
            related_ids=record.related_ids,
        )

        # Create graph edge: failure → lesson
        if lesson.failure_id and fact_id and hasattr(self.memory, "_db") and self.memory._db:
            try:
                await self.memory._db.query(
                    "RELATE type::thing('fact', $from)->references_episode->type::thing('fact', $to)",
                    {"from": lesson.failure_id, "to": fact_id},
                )
            except Exception as e:
                logger.debug("Edge creation failed: %s", e)

        return fact_id

    async def find_similar_lessons(
        self, context: str, project: str = ""
    ) -> list[StructuredLesson]:
        """Find lessons relevant to a given context.

        Uses embedding similarity if available, falls back to tag/keyword match.
        """
        # Try embedding similarity first
        if self._embedding_service:
            return await self._embedding_search(context, project)

        # Fallback: keyword/tag search
        return await self._keyword_search(context, project)

    async def _embedding_search(self, context: str, project: str) -> list[StructuredLesson]:
        """Search lessons using embedding similarity."""
        try:
            query_embedding = await self.memory.embed(context)
            if not query_embedding:
                return await self._keyword_search(context, project)

            # Search fact table for lessons
            if hasattr(self.memory, "_db") and self.memory._db:
                results = await self.memory._db.query(
                    """SELECT *, vector::distance::cosine(embedding, $query_vector) AS _dist
                       FROM fact
                       WHERE type = 'lesson'
                       AND ($project = '' OR project = $project)
                       ORDER BY _dist ASC
                       LIMIT $limit""",
                    {
                        "query_vector": query_embedding,
                        "project": project,
                        "limit": self.max_lessons,
                    },
                )
            else:
                return await self._keyword_search(context, project)

            lessons = []
            for row in results:
                record = MemoryRecord.from_dict(dict(row))
                raw_value = str(row.get("value", ""))
                lessons.append(StructuredLesson.from_record(record, raw_value=raw_value))

            return lessons
        except Exception as e:
            logger.warning("Embedding search failed: %s", e)
            return await self._keyword_search(context, project)

    async def _keyword_search(self, context: str, project: str) -> list[StructuredLesson]:
        """Fallback: keyword search for lessons."""
        try:
            facts = await self.memory.query_facts(limit=200)
            lessons = []

            context_lower = context.lower()
            for f in facts:
                if f.get("type") != "lesson":
                    continue
                if project and f.get("project") and f.get("project", "") != project:
                    continue

                record = MemoryRecord.from_dict(f)
                raw_value = f.get("value")
                raw_value = str(raw_value) if raw_value else ""
                lesson = StructuredLesson.from_record(record, raw_value=raw_value)

                # Use raw value text for search if available, for better matching
                search_text = raw_value if raw_value else (
                    lesson.root_cause
                    + " "
                    + lesson.fix_approach
                    + " "
                    + lesson.applicable_conditions
                )
                search_text = search_text.lower()
                context_words = set(context_lower.split())
                lesson_words = set(search_text.split())
                overlap = context_words & lesson_words

                if len(overlap) >= 2:  # at least 2 words overlap
                    lessons.append(lesson)

                if len(lessons) >= self.max_lessons:
                    break

            return lessons
        except Exception as e:
            logger.warning("Keyword search failed: %s", e)
            return []

    async def verify_lesson(self, lesson_id: str, success_id: str) -> bool:
        """Mark a lesson as verified by a success.

        Creates graph edge: lesson → success
        Updates lesson record: verified=true, verified_by=success_id
        """
        try:
            db = getattr(self.memory, "_db", None)
            if db:
                # Update lesson record
                await db.query(
                    """UPDATE type::thing('fact', $id)
                       SET verified = true,
                           verified_by_success_id = $success_id,
                           verified_at = time::now(),
                           confidence = MIN(1.0, confidence + 0.2)
                       WHERE type = 'lesson'""",
                    {"id": lesson_id, "success_id": success_id},
                )

                # Create graph edge
                await db.query(
                    "RELATE type::thing('fact', $from)->references_episode->type::thing('fact', $to)",
                    {"from": lesson_id, "to": success_id},
                )
            else:
                # In-memory mode: mark verified in the record
                facts = self.memory._mem.get("fact", [])
                for f in facts:
                    if f.get("id") == lesson_id or str(f.get("id", "")) == lesson_id:
                        f["verified"] = True
                        f["verified_by_success_id"] = success_id
                        break

            logger.info("Lesson %s verified by success %s", lesson_id, success_id)
            return True
        except Exception as e:
            logger.warning("Verify lesson failed: %s", e)
            return False

    async def get_failure_chain(self, failure_id: str) -> FailureChain:
        """Get the full failure chain for a failure.

        Traverses graph edges to find all related lessons and successes.
        """
        chain = FailureChain(failure_ids=[failure_id])

        db = getattr(self.memory, "_db", None)
        if not db:
            return chain

        try:
            # Find lessons linked to this failure
            lessons = await db.query(
                """SELECT out FROM references_episode
                   WHERE in = type::thing('fact', $fid)""",
                {"fid": failure_id},
            )
            for row in lessons:
                chain.lesson_ids.append(str(row.get("out", "")))

            # For each lesson, find successes
            for lesson_id in chain.lesson_ids:
                successes = await db.query(
                    """SELECT out FROM references_episode
                       WHERE in = type::thing('fact', $lid)""",
                    {"lid": lesson_id},
                )
                for row in successes:
                    sid = str(row.get("out", ""))
                    if sid:
                        chain.success_ids.append(sid)

        except Exception as e:
            logger.debug("Get failure chain failed: %s", e)

        return chain

    async def get_unresolved_failures(self, project: str = "") -> list[MemoryRecord]:
        """Find failures that don't have any verified lessons yet.

        These are failures where we haven't learned the fix.
        """
        try:
            # Get all failures
            facts = await self.memory.query_facts(limit=500)
            failures = []

            for f in facts:
                if f.get("type") != "failure":
                    continue
                if project and f.get("project") and f.get("project", "") != project:
                    continue

                record = MemoryRecord.from_dict(f)

                # Check if this failure has any verified lesson
                chain = await self.get_failure_chain(record.id)
                has_verified = False
                for lid in chain.lesson_ids:
                    if await self._is_lesson_verified(lid):
                        has_verified = True
                        break

                if not has_verified:
                    failures.append(record)

            return failures
        except Exception as e:
            logger.warning("Get unresolved failures failed: %s", e)
            return []

    async def _is_lesson_verified(self, lesson_id: str) -> bool:
        """Check if a lesson has been verified."""
        db = getattr(self.memory, "_db", None)
        if db:
            try:
                result = await db.query(
                    "SELECT verified FROM type::thing('fact', $id)",
                    {"id": lesson_id},
                )
                if result:
                    return bool(result[0].get("verified", False))
            except Exception:
                pass
        else:
            facts = self.memory._mem.get("fact", [])
            for f in facts:
                if str(f.get("id", "")) == lesson_id:
                    return bool(f.get("verified", False))
        return False


class AssociativeLearner:
    """High-level associative learning: connect failures to successes.

    When a success occurs, search for prior similar failures and their lessons,
    and verify those lessons. This closes the feedback loop.

    Usage:
        learner = AssociativeLearner(memory_pool, llm_provider)

        # When something fails:
        lesson = await learner.on_failure(failure_record)

        # When something succeeds:
        linked = await learner.on_success(success_record)

        # Before attempting something risky:
        warnings = await learner.get_relevant_lessons("deploy to production")
    """

    def __init__(self, memory_pool, llm_provider=None):
        self.memory = memory_pool
        self.llm = llm_provider
        self.lesson_learner = LessonLearner(memory_pool, llm_provider)

    async def on_failure(self, failure: MemoryRecord) -> StructuredLesson:
        """Handle a failure: extract and store a lesson.

        This is the entry point called when a task/action fails. It:
        1. Extracts a StructuredLesson from the failure
        2. Stores it in the memory pool
        3. Returns the lesson for immediate use (e.g., retry guidance)

        Args:
            failure: The failure MemoryRecord (type=FAILURE).

        Returns:
            The extracted StructuredLesson.
        """
        logger.info("Extracting lesson from failure: %s", failure.summary[:80])
        lesson = await self.lesson_learner.extract_lesson(failure)
        lesson_id = await self.lesson_learner.store_lesson(lesson)
        lesson.id = lesson_id
        logger.info("Lesson stored: %s (id=%s)", lesson.root_cause[:60], lesson_id)
        return lesson

    async def on_success(self, success: MemoryRecord) -> list[tuple[str, bool]]:
        """Handle a success: find and verify related past lessons.

        When a task succeeds, search for past similar failures. If the success
        is related (e.g., same trigger or similar context), verify the lesson
        extracted from that failure. This closes the feedback loop.

        Args:
            success: The success MemoryRecord (type=SUCCESS).

        Returns:
            List of (lesson_id, verified) tuples showing which lessons were linked.
        """
        # Build context from success record for searching
        context = (
            f"{success.summary} {success.trigger} {success.action} {success.outcome}"
        )
        project = success.project

        # Find similar lessons (these are from past failures)
        lessons = await self.lesson_learner.find_similar_lessons(context, project)

        results: list[tuple[str, bool]] = []
        for lesson in lessons:
            # Only verify lessons that haven't been verified yet
            if not lesson.verified and lesson.id:
                verified = await self.lesson_learner.verify_lesson(
                    lesson.id, success.id
                )
                results.append((lesson.id, verified))
                if verified:
                    logger.info("Success %s verified lesson %s", success.id, lesson.id)

        if lessons:
            logger.info(
                "on_success: found %d relevant lessons, verified %d",
                len(lessons),
                sum(1 for _, v in results if v),
            )

        return results

    async def get_relevant_lessons(
        self, context: str, project: str = ""
    ) -> list[StructuredLesson]:
        """Get lessons relevant to a context before attempting an action.

        Call this before risky operations to check "what did we learn from
        similar past failures?"

        Args:
            context: Description of what you're about to do.
            project: Optional project filter.

        Returns:
            Relevant lessons, sorted by relevance (most relevant first).
        """
        all_lessons = await self.lesson_learner.find_similar_lessons(context, project)

        # Filter: prefer verified lessons with higher confidence
        verified = [l for l in all_lessons if l.verified]
        unverified = [l for l in all_lessons if not l.verified]

        # Sort verified by confidence descending, unverified follow
        verified.sort(key=lambda l: l.confidence, reverse=True)
        unverified.sort(key=lambda l: l.confidence, reverse=True)

        combined = verified + unverified
        return combined[: self.lesson_learner.max_lessons]

    async def on_failure_with_retry_guidance(
        self, failure: MemoryRecord
    ) -> tuple[StructuredLesson, list[StructuredLesson]]:
        """Handle a failure and return both the new lesson and past relevant lessons.

        This is useful when you want to immediately retry with guidance from
        both the current failure's lesson and past similar lessons.

        Args:
            failure: The failure MemoryRecord.

        Returns:
            Tuple of (new_lesson, past_relevant_lessons).
        """
        new_lesson = await self.on_failure(failure)

        # Search for past relevant lessons
        context = (
            f"{failure.summary} {failure.trigger} {failure.error}"
        )
        past_lessons = await self.get_relevant_lessons(context, failure.project)

        return new_lesson, past_lessons

    async def get_learning_summary(self, project: str = "") -> str:
        """Get a summary of what the system has learned.

        Returns:
            Human-readable summary of resolved chains and unresolved failures.
        """
        facts = await self.memory.query_facts(limit=500)

        # Lesson facts
        all_lessons = [f for f in facts if f.get("type") == "lesson"]
        if project:
            all_lessons = [f for f in all_lessons if f.get("project", "") == project]

        verified_count = sum(1 for f in all_lessons if f.get("verified", False))
        unverified_count = len(all_lessons) - verified_count

        # Failure facts (unresolved = failures without verified lessons)
        failure_facts = [
            f for f in facts if f.get("type") == "failure"
            and (not project or f.get("project", "") == project)
        ]

        lines = [
            "# Learning Summary",
            "",
            f"Total lessons learned: {len(all_lessons)}",
            f"Verified (proven): {verified_count}",
            f"Unverified (speculative): {unverified_count}",
            f"Unresolved failures: {len(failure_facts)}",
        ]

        if failure_facts:
            lines.append("")
            lines.append("## Unresolved Failures")
            for uf in failure_facts[:5]:
                rec = MemoryRecord.from_dict(uf)
                lines.append(f"- {rec.summary}")

        if all_lessons:
            lines.append("")
            lines.append("## Verified Lessons")
            verified_lessons = [f for f in all_lessons if f.get("verified", False)]
            for f in verified_lessons:
                rec = MemoryRecord.from_dict(f)
                lines.append(f"- [{rec.confidence:.0%}] {rec.lesson or rec.summary}")

        return "\n".join(lines)
