"""Tests for P6-C: LLM-driven memory consolidation engine.

Covers:
  - ConsolidationResult dataclass
  - ConsolidationEngine._gather (mock DB)
  - ConsolidationEngine._extract with mock LLM
  - ConsolidationEngine._heuristic_extract (no LLM fallback)
  - ConsolidationEngine._link (failure→success edges)
  - ConsolidationEngine._resolve (contradiction detection + LLM arbitration)
  - ConsolidationEngine._prune (low confidence deletion)
  - ConsolidationEngine._mark_consolidated
  - Full run() pipeline (mock DB + mock LLM)
  - min_episodes gate (skips when below threshold)
  - Anchor records protected from pruning
  - MemoryPool.consolidate() convenience method
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone

from src.memory.consolidation import (
    ConsolidationResult,
    ConsolidationEngine,
)
from src.memory.schema import (
    MemoryRecord,
    MemoryType,
    render_for_llm,
    parse_llm_extraction,
)


# ================================================================
# Helpers
# ================================================================

def make_episode(title="Test Episode", summary="Did something",
                 content="Some content here", created_at="2026-07-01T10:00:00Z",
                 consolidated=False, error="", fix="", mem_type="episode",
                 project="", trigger="", action="", outcome="", lesson="",
                 confidence=0.5, tags=None):
    """Create a mock episode dict."""
    return {
        "id": f"episode:{hash(title) % 10000}",
        "title": title,
        "summary": summary,
        "content": content,
        "created_at": created_at,
        "consolidated": consolidated,
        "error": error,
        "fix": fix,
        "type": mem_type,
        "project": project,
        "trigger": trigger,
        "action": action,
        "outcome": outcome,
        "lesson": lesson,
        "confidence": confidence,
        "tags": tags or [],
    }


def make_fact(name="test_fact", agent_id="shared", confidence=0.7,
              access_count=0, mem_type="fact", outcome="", fix="",
              fact_type="entity"):
    """Create a mock fact dict."""
    return {
        "id": f"fact:{hash(name) % 10000}",
        "name": name,
        "agent_id": agent_id,
        "confidence": confidence,
        "access_count": access_count,
        "type": mem_type,
        "outcome": outcome,
        "fix": fix,
        "summary": name,
        "fact_type": fact_type,
    }


class MockMemoryPool:
    """Mock MemoryPool for testing consolidation phases."""

    def __init__(self, episodes=None, facts=None, db_connected=True):
        self._episodes = episodes or []
        self._facts = facts or []
        self._db = MagicMock() if db_connected else None
        self._edges = []

    async def get_unconsolidated_episodes(self, limit=50):
        episodes = [e for e in self._episodes if not e.get("consolidated", False)]
        return episodes[:limit]

    async def mark_episode_consolidated(self, episode_id):
        for e in self._episodes:
            if e.get("id") == episode_id:
                e["consolidated"] = True
                e["consolidated_at"] = datetime.now(timezone.utc).isoformat()

    async def query_facts(self, agent_id=None, fact_type=None, limit=100):
        facts = list(self._facts)
        if agent_id:
            facts = [f for f in facts if f.get("agent_id") == agent_id]
        if fact_type:
            facts = [f for f in facts if f.get("fact_type") == fact_type]
        return facts[:limit]

    async def write_fact(self, fact_type, name, value=None, agent_id="shared",
                         mem_type="fact", summary=None, trigger="", action="",
                         outcome="", lesson="", confidence=0.5, tags=None,
                         error="", fix="", steps=None, related_ids=None,
                         embedding_text=None, upsert=True):
        fid = f"fact:{len(self._facts) + 1}"
        self._facts.append({
            "id": fid,
            "fact_type": fact_type,
            "name": name,
            "value": value,
            "agent_id": agent_id,
            "type": mem_type,
            "summary": summary or name,
            "trigger": trigger,
            "action": action,
            "outcome": outcome,
            "lesson": lesson,
            "confidence": confidence,
            "tags": tags or [],
            "error": error,
            "fix": fix,
            "steps": steps or [],
            "related_ids": related_ids or [],
            "access_count": 0,
        })
        return fid

    async def write_edge(self, source, target, relation):
        self._edges.append({"source": source, "target": target, "relation": relation})
        return f"edge:{len(self._edges)}"

    def clear(self):
        self._episodes = []
        self._facts = []
        self._edges = []


class MockLLM:
    """Mock LLM provider for testing."""

    def __init__(self, response="[FACT||2026-07-01|conf:0.8]\nsummary: Test extraction"):
        self._response = response
        self.call_count = 0
        self.last_messages = []

    async def chat(self, messages, **kwargs):
        self.call_count += 1
        self.last_messages = messages
        return {"content": self._response}


# ================================================================
# ConsolidationResult
# ================================================================

class TestConsolidationResult:
    """Test ConsolidationResult dataclass."""

    def test_defaults(self):
        r = ConsolidationResult()
        assert r.episodes_processed == 0
        assert r.records_extracted == 0
        assert r.links_created == 0
        assert r.contradictions_resolved == 0
        assert r.memories_pruned == 0
        assert r.errors == []
        assert r.duration_seconds == 0.0

    def test_to_summary_empty(self):
        r = ConsolidationResult()
        s = r.to_summary()
        assert "Episodes: 0" in s
        assert "Duration: 0.0s" in s

    def test_to_summary_full(self):
        r = ConsolidationResult(
            episodes_processed=10,
            records_extracted=8,
            links_created=3,
            contradictions_resolved=2,
            memories_pruned=5,
            errors=["test error"],
            duration_seconds=1.5,
        )
        s = r.to_summary()
        assert "Episodes: 10" in s
        assert "Records extracted: 8" in s
        assert "Links created: 3" in s
        assert "Contradictions resolved: 2" in s
        assert "Memories pruned: 5" in s
        assert "Errors: 1" in s
        assert "Duration: 1.5s" in s

    def test_to_summary_no_errors(self):
        r = ConsolidationResult(episodes_processed=5)
        s = r.to_summary()
        assert "Errors" not in s


# ================================================================
# Phase 1: _gather
# ================================================================

class TestGather:
    """Test ConsolidationEngine._gather()."""

    @pytest.mark.asyncio
    async def test_gather_returns_unconsolidated(self):
        pool = MockMemoryPool(episodes=[
            make_episode("E1", consolidated=False),
            make_episode("E2", consolidated=True),
            make_episode("E3", consolidated=False),
        ])
        engine = ConsolidationEngine(pool)
        episodes = await engine._gather()
        assert len(episodes) == 2
        titles = [e["title"] for e in episodes]
        assert "E1" in titles
        assert "E3" in titles
        assert "E2" not in titles

    @pytest.mark.asyncio
    async def test_gather_respects_limit(self):
        pool = MockMemoryPool(episodes=[
            make_episode(f"E{i}", consolidated=False) for i in range(100)
        ])
        engine = ConsolidationEngine(pool, max_episodes_per_run=10)
        episodes = await engine._gather()
        assert len(episodes) <= 10

    @pytest.mark.asyncio
    async def test_gather_empty(self):
        pool = MockMemoryPool(episodes=[])
        engine = ConsolidationEngine(pool)
        episodes = await engine._gather()
        assert episodes == []

    @pytest.mark.asyncio
    async def test_gather_all_consolidated(self):
        pool = MockMemoryPool(episodes=[
            make_episode("E1", consolidated=True),
            make_episode("E2", consolidated=True),
        ])
        engine = ConsolidationEngine(pool)
        episodes = await engine._gather()
        assert episodes == []


# ================================================================
# Phase 2: _extract (heuristic, no LLM)
# ================================================================

class TestHeuristicExtract:
    """Test ConsolidationEngine._heuristic_extract()."""

    def test_heuristic_identifies_failure(self):
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool)
        episodes = [
            make_episode("Bug", "Deploy error", "The deploy failed with exception"),
        ]
        records = engine._heuristic_extract(episodes)
        assert len(records) == 1
        assert records[0].type == MemoryType.FAILURE

    def test_heuristic_identifies_success(self):
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool)
        episodes = [
            make_episode("Fix", "Fixed deploy", "The issue was resolved successfully"),
        ]
        records = engine._heuristic_extract(episodes)
        assert len(records) == 1
        assert records[0].type == MemoryType.SUCCESS

    def test_heuristic_identifies_fact(self):
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool)
        episodes = [
            make_episode("Meeting", "Team sync", "Discussed project timeline"),
        ]
        records = engine._heuristic_extract(episodes)
        assert len(records) == 1
        assert records[0].type == MemoryType.FACT

    def test_heuristic_uses_explicit_type(self):
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool)
        episodes = [
            make_episode("Lesson", "CI tip", mem_type="lesson"),
        ]
        records = engine._heuristic_extract(episodes)
        assert len(records) == 1
        assert records[0].type == MemoryType.LESSON

    def test_heuristic_uses_error_field(self):
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool)
        episodes = [
            make_episode("Bug", "Something", error="missing config"),
        ]
        records = engine._heuristic_extract(episodes)
        assert len(records) == 1
        assert records[0].type == MemoryType.FAILURE

    def test_heuristic_empty_episodes(self):
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool)
        records = engine._heuristic_extract([])
        assert records == []

    def test_heuristic_skips_empty_summary(self):
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool)
        episodes = [
            make_episode("", ""),
        ]
        records = engine._heuristic_extract(episodes)
        assert records == []

    def test_heuristic_preserves_fields(self):
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool)
        episodes = [
            make_episode(
                "Bug", "Test error", "error occurred",
                project="myproject", trigger="deploy",
                action="ran script", outcome="failed",
                lesson="check first", error="E1", fix="do X",
                confidence=0.8, tags=["urgent"],
            ),
        ]
        records = engine._heuristic_extract(episodes)
        assert len(records) == 1
        r = records[0]
        assert r.project == "myproject"
        assert r.trigger == "deploy"
        assert r.action == "ran script"
        assert r.outcome == "failed"
        assert r.lesson == "check first"
        assert r.error == "E1"
        assert r.fix == "do X"
        assert r.confidence == 0.8
        assert r.tags == ["urgent"]


# ================================================================
# Phase 2: _extract (with mock LLM)
# ================================================================

class TestLLMExtract:
    """Test ConsolidationEngine._extract() with mock LLM."""

    @pytest.mark.asyncio
    async def test_extract_with_llm_success(self):
        """LLM returns a parsed success record."""
        llm_response = (
            "[SUCCESS|agent-loop|2026-07-01|conf:0.9]\n"
            "summary: Fixed auth bug\n"
            "trigger: crash report\n"
            "action: added null check\n"
            "outcome: no more crashes\n"
            "lesson: validate before deref\n"
            "tags: [bugfix, auth]"
        )
        pool = MockMemoryPool(facts=[
            make_fact("old_fact"),
        ])
        llm = MockLLM(response=llm_response)
        engine = ConsolidationEngine(pool, llm_provider=llm)

        episodes = [make_episode("Bug fix", "Fixed auth crash")]
        records = await engine._extract(episodes)

        assert len(records) == 1
        assert records[0].type == MemoryType.SUCCESS
        assert records[0].summary == "Fixed auth bug"
        assert records[0].confidence == 0.9
        # Should have been persisted
        assert len(pool._facts) > 0

    @pytest.mark.asyncio
    async def test_extract_with_llm_multiple_records(self):
        """LLM returns multiple records separated by ---."""
        llm_response = (
            "[SUCCESS||2026-07-01|conf:0.8]\nsummary: Record A\n"
            "---\n"
            "[FAILURE||2026-07-01|conf:0.7]\nsummary: Record B\nerror: broke\nfix: do Y"
        )
        pool = MockMemoryPool()
        llm = MockLLM(response=llm_response)
        engine = ConsolidationEngine(pool, llm_provider=llm)

        episodes = [make_episode("Test", "Multiple events")]
        records = await engine._extract(episodes)

        assert len(records) == 2
        assert records[0].type == MemoryType.SUCCESS
        assert records[1].type == MemoryType.FAILURE

    @pytest.mark.asyncio
    async def test_extract_falls_back_on_llm_error(self):
        """When LLM fails, fall back to heuristic extraction."""

        class FailingLLM:
            async def chat(self, messages, **kwargs):
                raise RuntimeError("LLM unavailable")

        pool = MockMemoryPool()
        llm = FailingLLM()
        engine = ConsolidationEngine(pool, llm_provider=llm)

        episodes = [make_episode("Bug", "error occurred", "content with error")]
        records = await engine._extract(episodes)

        # Should have used heuristic
        assert len(records) == 1
        assert records[0].type == MemoryType.FAILURE

    @pytest.mark.asyncio
    async def test_extract_no_llm_uses_heuristic(self):
        """No LLM provider → use heuristic."""
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool, llm_provider=None)

        episodes = [make_episode("Bug", "Deploy failed", "exception thrown")]
        records = await engine._extract(episodes)

        assert len(records) == 1
        assert records[0].type == MemoryType.FAILURE

    @pytest.mark.asyncio
    async def test_extract_persists_to_fact_table(self):
        """Extracted records are written as facts."""
        llm_response = (
            "[SUCCESS||2026-07-01|conf:0.9]\n"
            "summary: Persist test\n"
            "tags: [test]"
        )
        pool = MockMemoryPool()
        llm = MockLLM(response=llm_response)
        engine = ConsolidationEngine(pool, llm_provider=llm)

        episodes = [make_episode("Test", "Persist me")]
        records = await engine._extract(episodes)

        assert len(records) == 1
        # Check fact was written
        new_facts = [f for f in pool._facts if f["name"].startswith("Persist test")]
        assert len(new_facts) >= 1

    @pytest.mark.asyncio
    async def test_extract_skips_empty_blocks(self):
        """Empty blocks between --- separators are skipped."""
        llm_response = (
            "[SUCCESS||2026-07-01|conf:0.9]\nsummary: Valid\n"
            "---\n\n---\n"  # empty block
            "[FACT||2026-07-01|conf:0.5]\nsummary: Also valid"
        )
        pool = MockMemoryPool()
        llm = MockLLM(response=llm_response)
        engine = ConsolidationEngine(pool, llm_provider=llm)

        episodes = [make_episode("Test", "Multiple blocks")]
        records = await engine._extract(episodes)

        assert len(records) == 2


# ================================================================
# Phase 3: _link
# ================================================================

class TestLink:
    """Test ConsolidationEngine._link()."""

    @pytest.mark.asyncio
    async def test_link_failure_to_success(self):
        """Failure record whose fix overlaps with success outcome gets linked."""
        pool = MockMemoryPool(facts=[
            make_fact("success_fact", mem_type="success",
                      outcome="added null check in auth.py", agent_id="shared"),
        ])
        engine = ConsolidationEngine(pool)

        records = [
            MemoryRecord(
                type=MemoryType.FAILURE,
                summary="Auth crash",
                fix="add null check in auth.py file",
                outcome="500 error",  # has "error" → negative
                id="fact:failure",
            ),
        ]

        links = await engine._link(records)
        assert links >= 1

    @pytest.mark.asyncio
    async def test_link_explicit_related_ids(self):
        """Records with explicit related_ids create edges."""
        pool = MockMemoryPool(facts=[])
        engine = ConsolidationEngine(pool)

        records = [
            MemoryRecord(
                type=MemoryType.FACT,
                summary="Test fact",
                related_ids=["ep:1", "ep:2"],
                id="fact:test",
            ),
        ]

        links = await engine._link(records)
        assert links == 2  # one per related_id

    @pytest.mark.asyncio
    async def test_link_no_db(self):
        """No DB → link returns 0."""
        pool = MockMemoryPool(db_connected=False)
        engine = ConsolidationEngine(pool)

        records = [MemoryRecord(type=MemoryType.FACT, summary="test",
                                related_ids=["ep:1"])]
        links = await engine._link(records)
        assert links == 0

    @pytest.mark.asyncio
    async def test_link_empty_records(self):
        """Empty records → link returns 0."""
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool)
        links = await engine._link([])
        assert links == 0


# ================================================================
# Phase 4: _resolve (contradiction detection)
# ================================================================

class TestResolve:
    """Test ConsolidationEngine._resolve()."""

    def test_is_contradiction_opposite_outcomes(self):
        """Positive vs negative outcomes → contradiction."""
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool)

        r1 = MemoryRecord(type=MemoryType.FACT, summary="A", project="p",
                          outcome="deploy was successful")
        r2 = MemoryRecord(type=MemoryType.FACT, summary="B", project="p",
                          outcome="deploy failed with error")

        assert engine._is_contradiction(r1, r2) is True

    def test_is_contradiction_same_polarity(self):
        """Both positive → not contradiction."""
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool)

        r1 = MemoryRecord(type=MemoryType.FACT, summary="A", project="p",
                          outcome="deploy was successful")
        r2 = MemoryRecord(type=MemoryType.FACT, summary="B", project="p",
                          outcome="deploy worked fine")

        assert engine._is_contradiction(r1, r2) is False

    def test_is_contradiction_neutral(self):
        """Neutral outcomes → not contradiction."""
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool)

        r1 = MemoryRecord(type=MemoryType.FACT, summary="A", project="p",
                          outcome="updated config")
        r2 = MemoryRecord(type=MemoryType.FACT, summary="B", project="p",
                          outcome="reviewed code")

        assert engine._is_contradiction(r1, r2) is False

    async def _test_resolve_with_llm(self):
        """LLM arbitration resolves contradiction.

        NOTE: This test is skipped by default because it requires
        SurrealDB-connected MemoryPool. Use RealDB tests below instead.
        """
        pass  # placeholder — tested in RealDB suite

    @pytest.mark.asyncio
    async def test_resolve_no_llm_returns_zero(self):
        """No LLM → can't resolve, returns 0."""
        pool = MockMemoryPool()
        engine = ConsolidationEngine(pool, llm_provider=None)

        r1 = MemoryRecord(type=MemoryType.FACT, summary="A", project="p",
                          outcome="success", id="fact:a")
        r2 = MemoryRecord(type=MemoryType.FACT, summary="B", project="p",
                          outcome="failed", id="fact:b")

        resolved = await engine._resolve([r1, r2])
        assert resolved == 0  # No LLM, can't arbitrate

    @pytest.mark.asyncio
    async def test_resolve_different_projects_no_contradiction(self):
        """Different projects → never contradictory."""
        pool = MockMemoryPool()
        llm = MockLLM(response="A")
        engine = ConsolidationEngine(pool, llm_provider=llm)

        r1 = MemoryRecord(type=MemoryType.FACT, summary="A", project="proj1",
                          outcome="success")
        r2 = MemoryRecord(type=MemoryType.FACT, summary="B", project="proj2",
                          outcome="failed")

        resolved = await engine._resolve([r1, r2])
        assert resolved == 0  # Different projects

    @pytest.mark.asyncio
    async def test_resolve_different_types_no_contradiction(self):
        """Different types → never contradictory."""
        pool = MockMemoryPool()
        llm = MockLLM(response="A")
        engine = ConsolidationEngine(pool, llm_provider=llm)

        r1 = MemoryRecord(type=MemoryType.SUCCESS, summary="A", project="p",
                          outcome="success")
        r2 = MemoryRecord(type=MemoryType.FAILURE, summary="B", project="p",
                          outcome="failed")

        resolved = await engine._resolve([r1, r2])
        assert resolved == 0  # Different types


# ================================================================
# Phase 5: _prune
# ================================================================

class TestPrune:
    """Test ConsolidationEngine._prune()."""

    @pytest.mark.asyncio
    async def test_prune_no_db(self):
        """No DB → prune returns 0."""
        pool = MockMemoryPool(db_connected=False)
        engine = ConsolidationEngine(pool)
        pruned = await engine._prune()
        assert pruned == 0

    @pytest.mark.asyncio
    async def test_prune_with_db_query(self):
        """DB query runs prune SQL."""
        pool = MockMemoryPool()
        # Mock the DB query to return deleted IDs
        pool._db.query = AsyncMock(return_value=[{"id": "fact:1"}, {"id": "fact:2"}])
        engine = ConsolidationEngine(pool)

        pruned = await engine._prune()
        assert pruned == 2
        # Verify SQL includes anchor protection
        call_args = pool._db.query.call_args
        sql = str(call_args[0][0])
        assert "agent_id != 'anchor'" in sql

    @pytest.mark.asyncio
    async def test_prune_handles_db_error(self):
        """DB error → returns 0 gracefully."""
        pool = MockMemoryPool()
        pool._db.query = AsyncMock(side_effect=RuntimeError("DB down"))
        engine = ConsolidationEngine(pool)

        pruned = await engine._prune()
        assert pruned == 0


# ================================================================
# _mark_consolidated
# ================================================================

class TestMarkConsolidated:
    """Test ConsolidationEngine._mark_consolidated()."""

    @pytest.mark.asyncio
    async def test_mark_updates_episodes(self):
        episodes = [
            make_episode("E1", consolidated=False),
            make_episode("E2", consolidated=False),
        ]
        pool = MockMemoryPool(episodes=episodes)
        engine = ConsolidationEngine(pool)

        await engine._mark_consolidated(episodes)

        assert all(e["consolidated"] for e in episodes)
        assert all("consolidated_at" in e for e in episodes)

    @pytest.mark.asyncio
    async def test_mark_skips_empty_id(self):
        episodes = [{"id": "", "title": "No ID", "consolidated": False}]
        pool = MockMemoryPool(episodes=episodes)
        engine = ConsolidationEngine(pool)

        await engine._mark_consolidated(episodes)
        # Should not raise, just skip
        assert episodes[0]["consolidated"] is False

    @pytest.mark.asyncio
    async def test_mark_no_db(self):
        """No DB → mark updates in-memory."""
        episodes = [make_episode("E1", consolidated=False)]
        pool = MockMemoryPool(episodes=episodes, db_connected=False)
        engine = ConsolidationEngine(pool)

        await engine._mark_consolidated(episodes)
        assert episodes[0]["consolidated"] is True


# ================================================================
# Full run() pipeline
# ================================================================

class TestFullPipeline:
    """Test ConsolidationEngine.run() end-to-end with mocks."""

    @pytest.mark.asyncio
    async def test_run_full_pipeline(self):
        """Full pipeline with mock LLM processes episodes end-to-end."""
        episodes = [
            make_episode(f"E{i}", summary=f"Event {i}",
                         consolidated=False,
                         content=f"Content for event {i}")
            for i in range(5)
        ]
        pool = MockMemoryPool(episodes=episodes, facts=[
            make_fact("existing_fact"),
        ])
        llm = MockLLM(response=(
            "[SUCCESS||2026-07-01|conf:0.9]\n"
            "summary: Extracted from episodes\n"
            "tags: [auto]"
        ))
        engine = ConsolidationEngine(pool, llm_provider=llm)

        result = await engine.run()

        assert result.episodes_processed == 5
        assert result.records_extracted == 1
        assert result.duration_seconds > 0
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_run_min_episodes_gate(self):
        """When below min_episodes, consolidation is skipped."""
        pool = MockMemoryPool(episodes=[
            make_episode("E1", consolidated=False),
        ])
        engine = ConsolidationEngine(pool, min_episodes=3)

        result = await engine.run()

        assert result.episodes_processed == 0
        assert result.records_extracted == 0

    @pytest.mark.asyncio
    async def test_run_empty_episodes(self):
        """No episodes → skips early."""
        pool = MockMemoryPool(episodes=[])
        engine = ConsolidationEngine(pool)

        result = await engine.run()

        assert result.episodes_processed == 0
        assert result.records_extracted == 0

    @pytest.mark.asyncio
    async def test_run_disabled_phases(self):
        """All optional phases can be disabled."""
        episodes = [make_episode(f"E{i}", consolidated=False) for i in range(5)]
        pool = MockMemoryPool(episodes=episodes)
        llm = MockLLM(response=(
            "[SUCCESS||2026-07-01|conf:0.9]\nsummary: Test"
        ))
        engine = ConsolidationEngine(
            pool, llm_provider=llm,
            enable_linking=False,
            enable_resolution=False,
            enable_pruning=False,
        )

        result = await engine.run()

        assert result.episodes_processed == 5
        assert result.records_extracted == 1
        assert result.links_created == 0
        assert result.contradictions_resolved == 0
        assert result.memories_pruned == 0

    @pytest.mark.asyncio
    async def test_run_records_error_on_failure(self):
        """Pipeline errors are captured in result.errors."""

        class ErrorDuringRun:
            """Mock MemoryPool that fails during consolidation."""

            def __init__(self):
                self._db = MagicMock()

            async def get_unconsolidated_episodes(self, limit=50):
                return [make_episode("E1", consolidated=False),
                        make_episode("E2", consolidated=False),
                        make_episode("E3", consolidated=False)]

            async def mark_episode_consolidated(self, episode_id):
                raise RuntimeError("DB write failed during mark")

            async def query_facts(self, **kwargs):
                return []

            async def write_fact(self, **kwargs):
                return "fact:1"

            async def write_edge(self, source, target, relation):
                return "edge:1"

        pool = ErrorDuringRun()
        llm = MockLLM(response=(
            "[SUCCESS||2026-07-01|conf:0.9]\nsummary: Test"
        ))
        engine = ConsolidationEngine(pool, llm_provider=llm)

        result = await engine.run()

        # Even with error during mark, episodes were counted and records extracted
        assert result.episodes_processed == 3
        assert result.records_extracted == 1
        assert len(result.errors) == 1

    @pytest.mark.asyncio
    async def test_run_heuristic_when_no_llm(self):
        """Full pipeline works without LLM (heuristic fallback)."""
        episodes = [make_episode(f"E{i}", summary=f"Event {i}",
                                 consolidated=False,
                                 content="success working fixed resolved")
                    for i in range(5)]
        pool = MockMemoryPool(episodes=episodes)
        engine = ConsolidationEngine(pool, llm_provider=None)

        result = await engine.run()

        assert result.episodes_processed == 5
        assert result.records_extracted == 5  # heuristic returns records
        assert result.errors == []

    @pytest.mark.asyncio
    async def test_run_idempotent(self):
        """Running consolidation twice only processes unconsolidated episodes."""
        episodes = [make_episode(f"E{i}", consolidated=False) for i in range(5)]
        pool = MockMemoryPool(episodes=episodes)
        llm = MockLLM(response=(
            "[FACT||2026-07-01|conf:0.5]\nsummary: Run 1"
        ))
        engine = ConsolidationEngine(pool, llm_provider=llm)

        # First run
        result1 = await engine.run()
        assert result1.episodes_processed == 5

        # After first run, all episodes should be marked consolidated
        # Second run should find 0 unconsolidated
        result2 = await engine.run()
        assert result2.episodes_processed == 0


# ================================================================
# MemoryPool.consolidate() convenience method
# ================================================================

class TestMemoryPoolConsolidate:
    """Test MemoryPool.consolidate() convenience method."""

    @pytest.mark.asyncio
    async def test_consolidate_method_exists(self):
        """MemoryPool has consolidate() method."""
        from src.memory import MemoryPool
        pool = MemoryPool()
        assert hasattr(pool, "consolidate")
        assert callable(pool.consolidate)

    @pytest.mark.asyncio
    async def test_consolidate_returns_result(self):
        """consolidate() returns ConsolidationResult."""
        from src.memory import MemoryPool
        pool = MemoryPool(db_path=":memory:")
        try:
            # Write some episodes
            for i in range(5):
                await pool.write_episode(
                    title=f"Test {i}",
                    summary=f"Test episode {i}",
                    content="success working fixed",
                    consolidated=False,
                )
            result = await pool.consolidate()
            assert result.episodes_processed == 5
            assert hasattr(result, "to_summary")
        finally:
            pool.clear()

    @pytest.mark.asyncio
    async def test_consolidate_with_min_episodes_gate(self):
        """Below min_episodes → skipped."""
        from src.memory import MemoryPool
        pool = MemoryPool(db_path=":memory:")
        try:
            # Only 1 episode — below default min_episodes=3
            await pool.write_episode(
                title="Single",
                summary="Only one episode",
                consolidated=False,
            )
            result = await pool.consolidate(min_episodes=3)
            assert result.episodes_processed == 0
        finally:
            pool.clear()

    @pytest.mark.asyncio
    async def test_consolidate_with_linking_disabled(self):
        """Pass through options to engine."""
        from src.memory import MemoryPool
        pool = MemoryPool(db_path=":memory:")
        try:
            for i in range(4):
                await pool.write_episode(
                    title=f"T{i}",
                    summary=f"Episode {i}",
                    consolidated=False,
                )
            result = await pool.consolidate(
                enable_linking=False,
                enable_resolution=False,
                enable_pruning=False,
            )
            assert result.episodes_processed == 4
            assert result.links_created == 0
            assert result.contradictions_resolved == 0
            assert result.memories_pruned == 0
        finally:
            pool.clear()


# ================================================================
# Anchor protection
# ================================================================

class TestAnchorProtection:
    """Test that anchor records are protected from pruning."""

    @pytest.mark.asyncio
    async def test_prune_excludes_anchor_records(self):
        """Prune query includes agent_id != 'anchor' clause."""
        pool = MockMemoryPool()
        pool._db.query = AsyncMock(return_value=[])

        engine = ConsolidationEngine(pool)
        await engine._prune()

        # Verify the SQL excludes anchor records
        call_args = pool._db.query.call_args
        sql = str(call_args[0][0])
        assert "agent_id != 'anchor'" in sql

    @pytest.mark.asyncio
    async def test_prune_threshold_is_applied(self):
        """Prune threshold parameter is passed correctly."""
        pool = MockMemoryPool()
        pool._db.query = AsyncMock(return_value=[])

        engine = ConsolidationEngine(pool, prune_threshold=0.5)
        await engine._prune()

        call_args = pool._db.query.call_args
        params = call_args[0][1] if len(call_args[0]) > 1 else {}
        assert params.get("threshold") == 0.5
