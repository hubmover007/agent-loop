"""Tests for Agent Forking system."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from src.agent_fork import AgentForker, ForkConfig


def _setup_agent(state_dir: str, agent_id: str,
                 with_soul: bool = True,
                 with_knowledge: bool = True,
                 with_journal: bool = False) -> None:
    """Set up a minimal agent state directory."""
    agent_dir = Path(state_dir) / agent_id
    agent_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "agent_id": agent_id,
        "created_at": "2025-01-01T00:00:00+00:00",
        "total_tasks": 5,
        "success_rate": 0.8,
        "personality": "executor",
        "role": "coder",
    }
    if with_soul:
        (agent_dir / "IDENTITY.md").write_text("# Ident\n\nI am " + agent_id)
        (agent_dir / "ROLE.md").write_text("# Role\n\nCoder")
        (agent_dir / "JOURNAL.md").write_text("# Journal\n\nEntry 1\n")
        (agent_dir / "profile.json").write_text(json.dumps({
            "efficiency": 0.7, "assertiveness": 0.6,
            "cautiousness": 0.4, "curiosity": 0.8,
        }))
        (agent_dir / "meta.json").write_text(json.dumps(meta))

    if with_knowledge:
        (agent_dir / "KNOWLEDGE.json").write_text(json.dumps([
            {"id": "n1", "pattern": "擅长 coding", "confidence": 0.9,
             "evidence_count": 5, "source_entries": ["e1"],
             "created_at": "2025-01-01", "last_reinforced": "2025-01-01"},
        ]))

    if with_journal:
        (agent_dir / "JOURNAL.jsonl").write_text(
            json.dumps({"id": "entry:j1", "task_type": "coding"}) + "\n"
        )


@pytest.mark.asyncio
class TestAgentForker:
    """Core AgentForker tests."""

    async def test_fork_creates_child(self):
        """fork creates a child agent with its own directory."""
        with tempfile.TemporaryDirectory() as td:
            _setup_agent(td, "agent:parent")
            forker = AgentForker(state_dir=td)

            child_id = await forker.fork(ForkConfig(
                parent_id="agent:parent",
                fork_reason="parallel_task",
            ))

            assert child_id.startswith("agent:fork:")
            child_dir = Path(td) / child_id
            assert child_dir.exists()

    async def test_inherit_soul(self):
        """Child inherits soul files from parent."""
        with tempfile.TemporaryDirectory() as td:
            _setup_agent(td, "agent:parent")
            forker = AgentForker(state_dir=td)

            child_id = await forker.fork(ForkConfig(
                parent_id="agent:parent",
                inherit_soul=True,
                inherit_knowledge=False,
                inherit_journal=False,
            ))

            child_dir = Path(td) / child_id
            assert (child_dir / "IDENTITY.md").exists()
            assert (child_dir / "ROLE.md").exists()
            assert (child_dir / "JOURNAL.md").exists()
            assert (child_dir / "profile.json").exists()

            # Verify content was copied
            ident = (child_dir / "IDENTITY.md").read_text()
            assert "agent:parent" in ident

    async def test_inherit_knowledge(self):
        """Child inherits KNOWLEDGE.json from parent."""
        with tempfile.TemporaryDirectory() as td:
            _setup_agent(td, "agent:parent", with_knowledge=True)
            forker = AgentForker(state_dir=td)

            child_id = await forker.fork(ForkConfig(
                parent_id="agent:parent",
                inherit_soul=False,
                inherit_knowledge=True,
                inherit_journal=False,
            ))

            child_dir = Path(td) / child_id
            assert (child_dir / "KNOWLEDGE.json").exists()
            knowledge = json.loads((child_dir / "KNOWLEDGE.json").read_text())
            assert len(knowledge) == 1
            assert knowledge[0]["id"] == "n1"

    async def test_inherit_journal(self):
        """Child inherits JOURNAL.jsonl when configured."""
        with tempfile.TemporaryDirectory() as td:
            _setup_agent(td, "agent:parent", with_journal=True)
            forker = AgentForker(state_dir=td)

            child_id = await forker.fork(ForkConfig(
                parent_id="agent:parent",
                inherit_soul=False,
                inherit_knowledge=False,
                inherit_journal=True,
            ))

            child_dir = Path(td) / child_id
            assert (child_dir / "JOURNAL.jsonl").exists()

    async def test_merge_back(self):
        """merge_back merges child knowledge into parent."""
        with tempfile.TemporaryDirectory() as td:
            _setup_agent(td, "agent:parent", with_knowledge=True)
            forker = AgentForker(state_dir=td)

            child_id = await forker.fork(ForkConfig(
                parent_id="agent:parent",
                inherit_knowledge=True,
            ))

            # Child does some work, gains new knowledge
            child_dir = Path(td) / child_id
            child_dir.mkdir(parents=True, exist_ok=True)
            knowledge = json.loads((child_dir / "KNOWLEDGE.json").read_text())
            knowledge.append({
                "id": "n2", "pattern": "子Agent發現的模式",
                "confidence": 0.85, "evidence_count": 4,
                "source_entries": ["e-child"], "created_at": "2025-02-01",
                "last_reinforced": "2025-02-01",
            })
            (child_dir / "KNOWLEDGE.json").write_text(json.dumps(knowledge))

            # Merge back
            await forker.merge_back(child_id, "agent:parent")

            # Parent should now have both nuggets
            parent_knowledge = json.loads(
                (Path(td) / "agent:parent" / "KNOWLEDGE.json").read_text()
            )
            assert len(parent_knowledge) == 2

            # Child should be marked merged
            child_meta = json.loads(
                (Path(td) / child_id / "meta.json").read_text()
            )
            assert child_meta["merged"] is True
            assert child_meta["merged_into"] == "agent:parent"

    async def test_family_tree(self):
        """get_family_tree returns parent, children, and siblings."""
        with tempfile.TemporaryDirectory() as td:
            _setup_agent(td, "agent:parent")
            forker = AgentForker(state_dir=td)

            child1 = await forker.fork(ForkConfig(
                parent_id="agent:parent",
                fork_reason="parallel_task",
            ))
            child2 = await forker.fork(ForkConfig(
                parent_id="agent:parent",
                fork_reason="experiment",
            ))

            # Check child1's family tree
            tree = forker.get_family_tree(child1)
            assert tree["agent_id"] == child1
            assert tree["parent_id"] == "agent:parent"
            assert len(tree["siblings"]) == 1
            assert tree["siblings"][0]["agent_id"] == child2
            assert tree["merged"] is False

            # Check parent's family tree
            parent_tree = forker.get_family_tree("agent:parent")
            assert parent_tree["agent_id"] == "agent:parent"
            assert parent_tree["parent_id"] is None
            assert len(parent_tree["children"]) == 2

    async def test_fork_with_no_parent_dir(self):
        """Forking from non-existent parent raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as td:
            forker = AgentForker(state_dir=td)
            with pytest.raises(FileNotFoundError):
                await forker.fork(ForkConfig(
                    parent_id="agent:nonexistent",
                ))

    async def test_fork_with_overrides(self):
        """Fork applies personality and role overrides."""
        with tempfile.TemporaryDirectory() as td:
            _setup_agent(td, "agent:parent")
            forker = AgentForker(state_dir=td)

            child_id = await forker.fork(ForkConfig(
                parent_id="agent:parent",
                personality_override="analyst",
                role_override="researcher",
            ))

            child_dir = Path(td) / child_id
            profile = json.loads((child_dir / "profile.json").read_text())
            assert profile.get("personality") == "analyst"
            assert profile.get("role") == "researcher"

            role_md = (child_dir / "ROLE.md").read_text()
            assert "researcher" in role_md

    async def test_dont_inherit_soul(self):
        """With inherit_soul=False, no soul files are copied."""
        with tempfile.TemporaryDirectory() as td:
            _setup_agent(td, "agent:parent")
            forker = AgentForker(state_dir=td)

            child_id = await forker.fork(ForkConfig(
                parent_id="agent:parent",
                inherit_soul=False,
                inherit_knowledge=False,
                inherit_journal=False,
            ))

            child_dir = Path(td) / child_id
            # Soul files should NOT exist (only meta.json is always written)
            assert not (child_dir / "IDENTITY.md").exists()
            assert not (child_dir / "JOURNAL.md").exists()
            # meta.json is always written by fork
            assert (child_dir / "meta.json").exists()
