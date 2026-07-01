"""Tests for P6-B: Unified Memory Schema (src/memory/schema.py).

Covers:
  - MemoryType enum values
  - MemoryRecord.to_compact_text() formatting
  - Empty field omission (token efficiency)
  - parse_llm_extraction() round-trip
  - render_for_llm() multi-record rendering + token truncation
  - from_dict() backward compatibility (name/title → summary fallback)
  - Success and failure share same format, only type differs
  - write_fact / write_episode with new schema fields
  - render_records_for_llm via MemoryPool
"""

import pytest
from src.memory.schema import (
    MemoryType,
    MemoryRecord,
    render_for_llm,
    parse_llm_extraction,
    EXTRACTION_PROMPT,
    CONSOLIDATION_PROMPT,
)


class TestMemoryType:
    """Test MemoryType enum values."""

    def test_all_types(self):
        """All six memory types are defined."""
        assert MemoryType.SUCCESS.value == "success"
        assert MemoryType.FAILURE.value == "failure"
        assert MemoryType.LESSON.value == "lesson"
        assert MemoryType.FACT.value == "fact"
        assert MemoryType.PROCEDURE.value == "procedure"
        assert MemoryType.PATTERN.value == "pattern"

    def test_from_string(self):
        """MemoryType can be constructed from string values."""
        assert MemoryType("success") == MemoryType.SUCCESS
        assert MemoryType("failure") == MemoryType.FAILURE
        assert MemoryType("lesson") == MemoryType.LESSON

    def test_unknown_type_falls_back(self):
        """Unknown string values raise ValueError (handled by from_dict)."""
        with pytest.raises(ValueError):
            MemoryType("unknown")


class TestMemoryRecordToCompactText:
    """Test MemoryRecord.to_compact_text() rendering."""

    def test_minimal_record(self):
        """Minimal record renders header + summary."""
        r = MemoryRecord(
            id="1",
            type=MemoryType.FACT,
            summary="nginx listens on 8000",
            created_at="2026-07-01",
        )
        text = r.to_compact_text()
        assert "[FACT" in text
        assert "summary: nginx listens on 8000" in text

    def test_full_success_record(self):
        """Success record renders all non-empty fields."""
        r = MemoryRecord(
            id="2",
            type=MemoryType.SUCCESS,
            summary="Fixed auth.py null pointer",
            project="agent-loop",
            trigger="production crash on auth endpoint",
            action="Added null check at auth.py:42",
            outcome="No more crashes, 200 OK restored",
            lesson="Always validate Session.user before dereference",
            confidence=0.95,
            tags=["bugfix", "auth"],
            steps=["Reproduce crash", "Add null guard", "Deploy", "Verify"],
            created_at="2026-07-01",
        )
        text = r.to_compact_text()

        # Header
        assert text.startswith("[SUCCESS|agent-loop|2026-07-01|conf:0.9]")
        # All fields present
        assert "summary: Fixed auth.py null pointer" in text
        assert "trigger: production crash on auth endpoint" in text
        assert "action: Added null check at auth.py:42" in text
        assert "outcome: No more crashes, 200 OK restored" in text
        assert "lesson: Always validate Session.user before dereference" in text
        assert "steps: Reproduce crash, Add null guard, Deploy, Verify" in text
        assert "tags: [bugfix, auth]" in text

    def test_full_failure_record(self):
        """Failure record includes error + fix fields."""
        r = MemoryRecord(
            id="3",
            type=MemoryType.FAILURE,
            summary="Deploy failed due to missing env var",
            project="agent-loop",
            trigger="git push to main",
            action="Ran deploy script",
            outcome="Service 502 — DATABASE_URL not set",
            lesson="Check env vars before deploy",
            error="Environment variable DATABASE_URL was not configured",
            fix="Add DATABASE_URL to .env.production and re-deploy",
            confidence=0.9,
            tags=["deploy", "env"],
            created_at="2026-07-01",
        )
        text = r.to_compact_text()

        assert "[FAILURE" in text
        assert "error: Environment variable DATABASE_URL was not configured" in text
        assert "fix: Add DATABASE_URL to .env.production and re-deploy" in text

    def test_success_and_failure_format_consistent(self):
        """Success and failure records use the same field layout, only type differs."""
        success = MemoryRecord(
            type=MemoryType.SUCCESS,
            summary="Did X, worked",
            project="p",
            trigger="t",
            action="a",
            outcome="o",
            lesson="l",
            confidence=1.0,
            tags=["tag"],
            created_at="2026-07-01",
        )
        failure = MemoryRecord(
            type=MemoryType.FAILURE,
            summary="Did X, failed",
            project="p",
            trigger="t",
            action="a",
            outcome="o",
            lesson="l",
            error="e",
            fix="f",
            confidence=0.3,
            tags=["tag"],
            created_at="2026-07-01",
        )

        s_text = success.to_compact_text()
        f_text = failure.to_compact_text()

        # Both have header, summary, trigger, action, outcome, lesson, tags
        for field in ["summary:", "trigger:", "action:", "outcome:", "lesson:", "tags:"]:
            assert field in s_text, f"Success missing field: {field}"
            assert field in f_text, f"Failure missing field: {field}"

        # Type differs
        assert "[SUCCESS" in s_text
        assert "[FAILURE" in f_text

        # Failure has error + fix; success doesn't
        assert "error:" in f_text
        assert "fix:" in f_text
        assert "error:" not in s_text
        assert "fix:" not in s_text

    def test_empty_fields_omitted(self):
        """Empty fields are not rendered (token efficiency)."""
        r = MemoryRecord(
            id="4",
            type=MemoryType.FACT,
            summary="Simple fact",
            created_at="2026-07-01",
        )
        text = r.to_compact_text()

        # These should NOT appear since they're empty
        assert "trigger:" not in text
        assert "action:" not in text
        assert "outcome:" not in text
        assert "lesson:" not in text
        assert "error:" not in text
        assert "fix:" not in text
        assert "steps:" not in text
        assert "related:" not in text

    def test_date_truncation(self):
        """created_at is truncated to date only (YYYY-MM-DD)."""
        r = MemoryRecord(
            type=MemoryType.FACT,
            summary="test",
            created_at="2026-07-01T12:34:56+00:00",
        )
        text = r.to_compact_text()
        assert "2026-07-01" in text
        assert "12:34:56" not in text  # Time stripped

    def test_no_date_when_empty(self):
        """Empty created_at renders empty date part."""
        r = MemoryRecord(
            type=MemoryType.FACT,
            summary="test",
            created_at="",
        )
        text = r.to_compact_text()
        assert "[FACT|||" in text  # Empty project and date


class TestMemoryRecordFromDict:
    """Test MemoryRecord.from_dict() compatibility."""

    def test_from_episode_dict_with_title(self):
        """Episode dict uses title as summary fallback."""
        data = {
            "id": "episode:1",
            "type": "episode",
            "title": "Session: Fix auth bug",
            "content": "Fixed the bug.",
            "created_at": "2026-07-01T10:00:00Z",
            "tags": ["session"],
        }
        r = MemoryRecord.from_dict(data)
        assert r.id == "episode:1"
        assert r.type == MemoryType.FACT  # "episode" is not a MemoryType → falls back to FACT
        assert r.summary == "Session: Fix auth bug"
        assert r.tags == ["session"]

    def test_from_fact_dict_with_name(self):
        """Fact dict uses name as summary fallback when summary is missing."""
        data = {
            "id": "fact:1",
            "type": "entity",  # Not a valid MemoryType → falls back to FACT
            "name": "nginx_config",
            "value": {"port": 8000},
            "created_at": "2026-07-01",
        }
        r = MemoryRecord.from_dict(data)
        assert r.summary == "nginx_config"  # name → summary fallback

    def test_from_dict_with_explicit_summary(self):
        """Explicit summary takes priority over title/name."""
        data = {
            "id": "1",
            "type": "success",
            "title": "Original title",
            "name": "original_name",
            "summary": "Explicit summary",
        }
        r = MemoryRecord.from_dict(data)
        assert r.type == MemoryType.SUCCESS
        assert r.summary == "Explicit summary"

    def test_from_dict_defaults(self):
        """Empty dict produces valid record with defaults."""
        r = MemoryRecord.from_dict({})
        assert r.id == ""
        assert r.type == MemoryType.FACT
        assert r.summary == ""
        assert r.confidence == 0.5
        assert r.tags == []
        assert r.steps == []
        assert r.related_ids == []

    def test_from_dict_with_all_unified_fields(self):
        """Full unified schema dict maps correctly."""
        data = {
            "id": "ep:42",
            "type": "failure",
            "summary": "Deploy failed",
            "project": "agent-loop",
            "trigger": "push to main",
            "action": "ran deploy.sh",
            "outcome": "502 error",
            "lesson": "check env",
            "confidence": 0.8,
            "related_ids": ["ep:40", "ep:41"],
            "related_type": "cause",
            "steps": ["push", "deploy", "fail"],
            "error": "missing env",
            "fix": "add env var",
            "agent_id": "test-agent",
            "tags": ["deploy", "urgent"],
        }
        r = MemoryRecord.from_dict(data)
        assert r.type == MemoryType.FAILURE
        assert r.summary == "Deploy failed"
        assert r.project == "agent-loop"
        assert r.trigger == "push to main"
        assert r.action == "ran deploy.sh"
        assert r.outcome == "502 error"
        assert r.lesson == "check env"
        assert r.confidence == 0.8
        assert r.related_ids == ["ep:40", "ep:41"]
        assert r.related_type == "cause"
        assert r.steps == ["push", "deploy", "fail"]
        assert r.error == "missing env"
        assert r.fix == "add env var"
        assert r.agent_id == "test-agent"
        assert r.tags == ["deploy", "urgent"]


class TestParseLLMExtraction:
    """Test parse_llm_extraction() round-trip."""

    def test_parse_success_extraction(self):
        """Parse a success record from LLM output."""
        text = """[SUCCESS|agent-loop|2026-07-01|conf:0.9]
summary: Fixed auth bug
trigger: crash report
action: added null check
outcome: no more crashes
lesson: validate before deref
tags: [bugfix, auth]"""

        r = parse_llm_extraction(text)
        assert r is not None
        assert r.type == MemoryType.SUCCESS
        assert r.summary == "Fixed auth bug"
        assert r.project == "agent-loop"
        assert r.trigger == "crash report"
        assert r.action == "added null check"
        assert r.outcome == "no more crashes"
        assert r.lesson == "validate before deref"
        assert r.confidence == 0.9
        assert r.tags == ["bugfix", "auth"]

    def test_parse_failure_extraction(self):
        """Parse a failure record with error + fix."""
        text = """[FAILURE|ops|2026-07-01|conf:1.0]
summary: Deploy failed
trigger: git push
action: ran deploy
outcome: 502
lesson: check env vars
error: DB_URL missing
fix: add to .env
tags: [deploy, critical]"""

        r = parse_llm_extraction(text)
        assert r is not None
        assert r.type == MemoryType.FAILURE
        assert r.error == "DB_URL missing"
        assert r.fix == "add to .env"
        assert r.tags == ["deploy", "critical"]

    def test_parse_roundtrip(self):
        """Parse output of to_compact_text() produces equivalent record."""
        original = MemoryRecord(
            type=MemoryType.SUCCESS,
            summary="Fixed bug",
            project="test",
            trigger="issue reported",
            action="patched code",
            outcome="resolved",
            lesson="test first",
            confidence=0.9,  # 0.9 avoids floating-point rounding in :.1f formatting
            tags=["bug", "test"],
            created_at="2026-07-01",
        )
        text = original.to_compact_text()
        parsed = parse_llm_extraction(text)
        assert parsed is not None
        assert parsed.type == original.type
        assert parsed.summary == original.summary
        assert parsed.project == original.project
        assert parsed.trigger == original.trigger
        assert parsed.action == original.action
        assert parsed.outcome == original.outcome
        assert parsed.lesson == original.lesson
        assert parsed.confidence == original.confidence
        assert parsed.tags == original.tags

    def test_parse_empty_text(self):
        """Empty text returns None."""
        assert parse_llm_extraction("") is None
        assert parse_llm_extraction("   ") is None

    def test_parse_invalid_header(self):
        """Text without [TYPE|...] header returns None."""
        text = "summary: just a summary"
        assert parse_llm_extraction(text) is None

    def test_parse_unknown_type(self):
        """Unknown type falls back to FACT."""
        text = """[UNKNOWN|proj|2026-07-01|conf:0.5]
summary: something"""
        r = parse_llm_extraction(text)
        assert r is not None
        assert r.type == MemoryType.FACT

    def test_parse_short_header(self):
        """Header with too few parts returns None."""
        text = "[SUCCESS]\nsummary: x"
        assert parse_llm_extraction(text) is None

    def test_parse_confidence_edge_cases(self):
        """Edge cases for confidence parsing."""
        # Empty header parts
        text = "[SUCCESS||2026-07-01|conf:0.5]\nsummary: x"
        r = parse_llm_extraction(text)
        assert r is not None
        assert r.confidence == 0.5  # default

        # Invalid confidence
        text = "[SUCCESS||2026-07-01|conf:abc]\nsummary: x"
        r = parse_llm_extraction(text)
        assert r is not None
        assert r.confidence == 0.5  # default on parse error

    def test_parse_tags_and_related(self):
        """Parse tags and related_ids from list format."""
        text = """[FACT||2026-07-01|conf:0.5]
summary: multi-tag test
tags: [tag1, tag2, tag3]
related: [ep:1, ep:2]
steps: [step1, step2, step3]"""
        r = parse_llm_extraction(text)
        assert r is not None
        assert r.tags == ["tag1", "tag2", "tag3"]
        assert r.related_ids == ["ep:1", "ep:2"]
        assert r.steps == ["step1", "step2", "step3"]


class TestRenderForLLM:
    """Test render_for_llm() function."""

    def test_single_record(self):
        r = MemoryRecord(
            type=MemoryType.FACT,
            summary="Test fact",
            created_at="2026-07-01",
        )
        text = render_for_llm([r])
        assert "[FACT" in text
        assert "summary: Test fact" in text

    def test_multiple_records(self):
        records = [
            MemoryRecord(type=MemoryType.FACT, summary="Fact 1",
                         confidence=1.0, created_at="2026-07-01"),
            MemoryRecord(type=MemoryType.SUCCESS, summary="Success 1",
                         project="proj", confidence=0.9, created_at="2026-07-02"),
        ]
        text = render_for_llm(records)
        assert "---" in text  # Separator between records
        assert "[FACT" in text
        assert "[SUCCESS" in text

    def test_empty_list(self):
        assert render_for_llm([]) == ""

    def test_token_truncation(self):
        """Long records are truncated at max_tokens."""
        # Create 100 records, each ~100 chars → ~2500 tokens
        records = [
            MemoryRecord(
                type=MemoryType.FACT,
                summary=f"Fact number {i} with some extra text to consume tokens",
                created_at="2026-07-01",
            )
            for i in range(100)
        ]
        text = render_for_llm(records, max_tokens=50)  # Very low limit
        # Should have only a few records
        block_count = text.count("---") + 1
        assert block_count < 50  # Should be heavily truncated

    def test_render_preserves_failure_fields(self):
        """Failure error + fix are preserved in rendering."""
        records = [
            MemoryRecord(
                type=MemoryType.FAILURE,
                summary="Deploy failed",
                error="ENV missing",
                fix="Add to config",
                created_at="2026-07-01",
            ),
        ]
        text = render_for_llm(records)
        assert "error: ENV missing" in text
        assert "fix: Add to config" in text


class TestPromptTemplates:
    """Test EXTRACTION_PROMPT and CONSOLIDATION_PROMPT templates."""

    def test_extraction_prompt_contains_placeholders(self):
        assert "{episode_content}" in EXTRACTION_PROMPT
        assert "Extracted records:" in EXTRACTION_PROMPT

    def test_extraction_prompt_includes_all_types(self):
        for t in ["success", "failure", "lesson", "fact", "procedure"]:
            assert t in EXTRACTION_PROMPT.lower()

    def test_consolidation_prompt_contains_placeholders(self):
        assert "{episodes_text}" in CONSOLIDATION_PROMPT
        assert "{existing_memories_text}" in CONSOLIDATION_PROMPT

    def test_extraction_prompt_mentions_same_format(self):
        assert "SAME format" in EXTRACTION_PROMPT


class TestWriteWithSchemaFields:
    """Test write_fact / write_episode with new unified schema fields via MemoryPool."""

    @pytest.mark.asyncio
    async def test_write_fact_with_schema_fields(self):
        """write_fact accepts and stores unified schema fields."""
        from src.memory import MemoryPool
        pool = MemoryPool(db_path=":memory:")
        try:
            fid = await pool.write_fact(
                fact_type="facetpoint",
                name="auth_bug_fix",
                value={"file": "auth.py", "line": 42},
                mem_type="success",
                summary="Fixed auth bug in production",
                trigger="500 errors on /auth endpoint",
                action="Added null check at auth.py:42",
                outcome="No more 500 errors",
                lesson="Validate session before access",
                confidence=0.95,
                tags=["bugfix", "auth"],
                steps=["Reproduce", "Fix", "Deploy", "Verify"],
            )
            assert fid is not None
            assert "fact:" in fid or ":" in str(fid)
        finally:
            pool.clear()

    @pytest.mark.asyncio
    async def test_write_episode_with_schema_fields(self):
        """write_episode accepts and stores unified schema fields."""
        from src.memory import MemoryPool
        pool = MemoryPool(db_path=":memory:")
        try:
            eid = await pool.write_episode(
                title="Session: Fix auth bug",
                summary="Fixed auth.py null pointer exception",
                content="Full session transcript here...",
                mem_type="success",
                project="agent-loop",
                trigger="Production crash on /auth",
                action="Added null guard",
                outcome="Service recovered",
                lesson="Always validate",
                confidence=0.9,
                tags=["session", "bugfix"],
                steps=["diagnose", "fix", "deploy"],
            )
            assert eid is not None
            assert "episode:" in str(eid)
        finally:
            pool.clear()

    @pytest.mark.asyncio
    async def test_write_fact_backward_compatible(self):
        """Old write_fact calls (no new params) still work."""
        from src.memory import MemoryPool
        pool = MemoryPool(db_path=":memory:")
        try:
            fid = await pool.write_fact("entity", "test_key", {"v": 1})
            assert fid is not None
        finally:
            pool.clear()

    @pytest.mark.asyncio
    async def test_write_episode_backward_compatible(self):
        """Old write_episode calls (no new params) still work."""
        from src.memory import MemoryPool
        pool = MemoryPool(db_path=":memory:")
        try:
            eid = await pool.write_episode("Test", "Summary text")
            assert eid is not None
        finally:
            pool.clear()

    @pytest.mark.asyncio
    async def test_store_with_schema_fields(self):
        """store() passes through unified schema fields."""
        from src.memory import MemoryPool
        pool = MemoryPool(db_path=":memory:")
        try:
            eid = await pool.store({
                "type": "episode",
                "mem_type": "failure",
                "title": "Deploy failed",
                "summary": "DB_URL missing",
                "trigger": "git push",
                "action": "ran deploy.sh",
                "outcome": "502 error",
                "lesson": "check env vars",
                "error": "DATABASE_URL not set",
                "fix": "Add to .env.production",
                "confidence": 0.8,
                "tags": ["deploy"],
                "steps": ["push", "deploy", "fail"],
            })
            assert eid is not None
            assert "episode:" in str(eid)
        finally:
            pool.clear()


class TestRenderRecordsForLLM:
    """Test MemoryPool.render_records_for_llm() method."""

    @pytest.mark.asyncio
    async def test_render_episodes(self):
        """Render stored episodes as LLM-friendly text."""
        from src.memory import MemoryPool
        pool = MemoryPool(db_path=":memory:")
        try:
            # Write episodes with unified fields
            await pool.write_episode(
                title="Session 1",
                summary="Fixed auth bug",
                mem_type="success",
                project="agent-loop",
                trigger="crash",
                action="added null check",
                outcome="resolved",
                confidence=0.9,
            )
            await pool.write_episode(
                title="Session 2",
                summary="Deploy failed",
                mem_type="failure",
                project="agent-loop",
                error="env missing",
                fix="add to config",
                confidence=0.8,
            )

            # Get records from in-memory store and render
            episodes = pool._mem.get("episode", [])
            text = pool.render_records_for_llm(episodes)
            assert "summary: Fixed auth bug" in text
            assert "summary: Deploy failed" in text
        finally:
            pool.clear()
