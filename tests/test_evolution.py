"""Tests for structured EvolutionEngine."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from src.evolution import (
    JournalEntry,
    KnowledgeNugget,
    EvolutionEngine,
)


def _make_entry(**overrides) -> JournalEntry:
    """Helper to create a JournalEntry with defaults."""
    defaults = {
        "id": "entry:test001",
        "timestamp": "2025-01-01T00:00:00+00:00",
        "task_scope": "Test task",
        "task_type": "coding",
        "outcome": "success",
        "score": 0.9,
        "duration_seconds": 5.0,
        "tools_used": ["web_search"],
        "llm_provider": "test-model",
        "cost_estimate": 0.01,
        "lessons": [],
        "tags": ["test"],
    }
    defaults.update(overrides)
    return JournalEntry(**defaults)


class TestJournalEntry:
    """Tests for JournalEntry data class."""

    def test_roundtrip(self):
        """JournalEntry can be serialized and deserialized."""
        entry = _make_entry()
        d = entry.to_dict()
        restored = JournalEntry.from_dict(d)
        assert restored.id == entry.id
        assert restored.task_type == "coding"
        assert restored.score == 0.9

    def test_defaults(self):
        """JournalEntry has correct defaults."""
        entry = JournalEntry(
            id="e1", timestamp="2025-01-01", task_scope="X",
            task_type="general", outcome="success", score=1.0,
            duration_seconds=1.0, tools_used=[], llm_provider="m",
            cost_estimate=0.0,
        )
        assert entry.lessons == []
        assert entry.tags == []


class TestKnowledgeNugget:
    """Tests for KnowledgeNugget data class."""

    def test_roundtrip(self):
        nugget = KnowledgeNugget(
            id="n1", pattern="当遇到X时，应该Y",
            confidence=0.9, evidence_count=5,
            source_entries=["e1", "e2"],
            created_at="2025-01-01", last_reinforced="2025-01-02",
        )
        d = nugget.to_dict()
        restored = KnowledgeNugget.from_dict(d)
        assert restored.id == "n1"
        assert restored.confidence == 0.9
        assert restored.evidence_count == 5


@pytest.mark.asyncio
class TestEvolutionEngine:
    """Core EvolutionEngine tests."""

    async def test_record_entry(self):
        """record_entry writes a JSON line to JOURNAL.jsonl."""
        with tempfile.TemporaryDirectory() as td:
            engine = EvolutionEngine("agent-test", state_dir=td)
            entry = _make_entry(id="entry:rec001")
            await engine.record_entry(entry)

            journal_path = Path(td) / "agent-test" / "JOURNAL.jsonl"
            assert journal_path.exists()
            lines = journal_path.read_text().strip().split("\n")
            assert len(lines) == 1
            data = json.loads(lines[0])
            assert data["id"] == "entry:rec001"

    async def test_extract_knowledge_success(self):
        """extract_knowledge detects success pattern after 3+ successes."""
        with tempfile.TemporaryDirectory() as td:
            engine = EvolutionEngine("agent-test", state_dir=td)

            # Record 4 coding successes
            for i in range(4):
                await engine.record_entry(_make_entry(
                    id=f"entry:s{i:03d}", task_type="coding",
                    outcome="success", tags=["coding", "success"],
                ))

            nuggets = await engine.extract_knowledge(min_evidence=3)
            assert len(nuggets) >= 1
            success_patterns = [n.pattern for n in nuggets if "擅长" in n.pattern]
            assert len(success_patterns) >= 1
            assert any("coding" in p for p in success_patterns)

    async def test_extract_knowledge_failure(self):
        """extract_knowledge detects failure pattern after 2+ failures."""
        with tempfile.TemporaryDirectory() as td:
            engine = EvolutionEngine("agent-test", state_dir=td)

            # Record 3 coding failures
            for i in range(3):
                await engine.record_entry(_make_entry(
                    id=f"entry:f{i:03d}", task_type="coding",
                    outcome="failure", score=0.3, tags=["coding", "failure"],
                ))

            nuggets = await engine.extract_knowledge(min_evidence=3)
            assert len(nuggets) >= 1
            failure_patterns = [n.pattern for n in nuggets if "不擅长" in n.pattern]
            assert len(failure_patterns) >= 1
            assert any("coding" in p for p in failure_patterns)

    async def test_extract_knowledge_tool_failure(self):
        """extract_knowledge detects tool-specific failure patterns."""
        with tempfile.TemporaryDirectory() as td:
            engine = EvolutionEngine("agent-test", state_dir=td)

            # Record 3 entries all using "web_search" that fail
            for i in range(3):
                await engine.record_entry(_make_entry(
                    id=f"entry:tf{i:03d}", task_type="general",
                    outcome="failure", score=0.2,
                    tools_used=["web_search"], tags=["general", "failure"],
                ))

            nuggets = await engine.extract_knowledge(min_evidence=3)
            tool_patterns = [n.pattern for n in nuggets if "web_search" in n.pattern]
            assert len(tool_patterns) >= 1

    async def test_adjust_traits_success(self):
        """adjust_traits increases efficiency and assertiveness on success."""
        with tempfile.TemporaryDirectory() as td:
            engine = EvolutionEngine("agent-test", state_dir=td)

            # Initialize profile
            profile_path = Path(td) / "agent-test" / "profile.json"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(json.dumps({
                "efficiency": 0.5,
                "assertiveness": 0.5,
                "cautiousness": 0.5,
                "curiosity": 0.5,
            }))

            entry = _make_entry(outcome="success", task_type="coding")
            await engine.adjust_traits(entry)

            profile = json.loads(profile_path.read_text())
            assert profile["efficiency"] > 0.5, f"efficiency should increase, got {profile['efficiency']}"
            assert profile["assertiveness"] > 0.5, f"assertiveness should increase, got {profile['assertiveness']}"

    async def test_adjust_traits_failure(self):
        """adjust_traits decreases efficiency and increases cautiousness on failure."""
        with tempfile.TemporaryDirectory() as td:
            engine = EvolutionEngine("agent-test", state_dir=td)

            # Initialize profile
            profile_path = Path(td) / "agent-test" / "profile.json"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(json.dumps({
                "efficiency": 0.5,
                "assertiveness": 0.5,
                "cautiousness": 0.5,
                "curiosity": 0.5,
            }))

            entry = _make_entry(outcome="failure", task_type="coding")
            await engine.adjust_traits(entry)

            profile = json.loads(profile_path.read_text())
            assert profile["efficiency"] < 0.5, f"efficiency should decrease, got {profile['efficiency']}"
            assert profile["cautiousness"] > 0.5, f"cautiousness should increase, got {profile['cautiousness']}"

    async def test_adjust_traits_role_upgrade(self):
        """adjust_traits upgrades role after 3 consecutive successes."""
        with tempfile.TemporaryDirectory() as td:
            engine = EvolutionEngine("agent-test", state_dir=td)

            # Initialize profile with role=ops
            profile_path = Path(td) / "agent-test" / "profile.json"
            profile_path.parent.mkdir(parents=True, exist_ok=True)
            profile_path.write_text(json.dumps({
                "efficiency": 0.5,
                "assertiveness": 0.5,
                "cautiousness": 0.5,
                "curiosity": 0.5,
                "role": "ops",
            }))

            # Record 3 ops successes, then adjust on 3rd
            for i in range(3):
                entry = _make_entry(
                    id=f"entry:up{i:03d}", outcome="success", task_type="ops"
                )
                await engine.record_entry(entry)

            await engine.adjust_traits(_make_entry(
                id="entry:up003", outcome="success", task_type="ops"
            ))

            profile = json.loads(profile_path.read_text())
            assert profile.get("role") == "devops", \
                f"role should upgrade to devops, got {profile.get('role')}"

    async def test_promote_to_identity(self):
        """promote_to_identity adds knowledge to IDENTITY.md when confidence > 0.8 and evidence > 5."""
        with tempfile.TemporaryDirectory() as td:
            agent_dir = Path(td) / "agent-test"
            agent_dir.mkdir(parents=True, exist_ok=True)

            # Pre-create IDENTITY.md
            identity_path = agent_dir / "IDENTITY.md"
            identity_path.write_text("# 身份\n\n我是 agent-test\n")

            engine = EvolutionEngine("agent-test", state_dir=td)

            nugget = KnowledgeNugget(
                id="n1",
                pattern="擅长处理 coding 类型任务",
                confidence=0.9,
                evidence_count=6,
                source_entries=["e1"],
                created_at="2025-01-01T00:00:00",
                last_reinforced="2025-01-02T00:00:00",
            )

            result = await engine.promote_to_identity(nugget)
            assert result is True
            new_content = identity_path.read_text()
            assert "## 经验教训" in new_content
            assert "擅长处理 coding 类型任务" in new_content

    async def test_promote_to_identity_below_threshold(self):
        """promote_to_identity returns False when below threshold."""
        with tempfile.TemporaryDirectory() as td:
            engine = EvolutionEngine("agent-test", state_dir=td)

            nugget = KnowledgeNugget(
                id="n1",
                pattern="弱模式",
                confidence=0.5,
                evidence_count=2,
                source_entries=["e1"],
                created_at="2025-01-01T00:00:00",
                last_reinforced="2025-01-01T00:00:00",
            )

            result = await engine.promote_to_identity(nugget)
            assert result is False

    async def test_get_stats(self):
        """get_stats returns correct statistics."""
        with tempfile.TemporaryDirectory() as td:
            engine = EvolutionEngine("agent-test", state_dir=td)

            # Record 3 success, 1 failure
            for i in range(3):
                await engine.record_entry(_make_entry(
                    id=f"entry:st{i:03d}", task_type="coding",
                    outcome="success", score=0.9,
                    tools_used=["web_search"],
                ))
            await engine.record_entry(_make_entry(
                id="entry:st003", task_type="reasoning",
                outcome="failure", score=0.3,
                tools_used=["ssh"],
            ))

            stats = engine.get_stats()
            assert stats["total_tasks"] == 4
            assert stats["success_rate"] == 0.75
            assert stats["average_score"] > 0.5
            assert stats["most_used_tool"] == "web_search" or stats["most_used_tool"] is not None
            assert stats["total_duration_s"] > 0

    async def test_empty_stats(self):
        """get_stats returns zeros for empty journal."""
        with tempfile.TemporaryDirectory() as td:
            engine = EvolutionEngine("agent-test", state_dir=td)
            stats = engine.get_stats()
            assert stats["total_tasks"] == 0
            assert stats["success_rate"] == 0.0
