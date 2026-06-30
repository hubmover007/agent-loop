"""Tests for PersistenceManager — cross-session snapshot/restore/rollback."""

import asyncio
import json
import pytest
import tempfile
from pathlib import Path

from src.persistence import PersistenceManager, Snapshot


@pytest.fixture
def pm(tmp_path):
    """PersistenceManager with tmp directories."""
    state_dir = tmp_path / "state"
    snap_dir = tmp_path / "state" / "snapshots"
    state_dir.mkdir(parents=True, exist_ok=True)
    return PersistenceManager(state_dir=str(state_dir), snapshot_dir=str(snap_dir))


@pytest.fixture
def agent_state(pm):
    """Create a fake agent state directory with content."""
    agent_dir = Path(pm.state_dir) / "agents" / "agent-test-001"
    agent_dir.mkdir(parents=True, exist_ok=True)
    
    # Write IDENTITY.md
    (agent_dir / "IDENTITY.md").write_text("# Test Agent\nI am a test agent.")
    # Write profile.json
    (agent_dir / "profile.json").write_text(json.dumps({
        "efficiency": 0.5,
        "confidence": 0.7,
    }))
    # Write JOURNAL.jsonl
    (agent_dir / "JOURNAL.jsonl").write_text(
        '{"id":"1","timestamp":"2026-01-01T00:00:00Z","task_scope":"test","outcome":"success"}\n'
        '{"id":"2","timestamp":"2026-01-01T01:00:00Z","task_scope":"test2","outcome":"success"}\n'
    )
    # Write KNOWLEDGE.json
    (agent_dir / "KNOWLEDGE.json").write_text(json.dumps({
        "nuggets": [
            {"id": "k1", "pattern": "test pattern", "confidence": 0.8}
        ]
    }))
    
    return "agent-test-001"


# ============================================================
# Tests
# ============================================================


class TestPersistenceManager:
    
    @pytest.mark.asyncio
    async def test_snapshot_and_restore(self, pm, agent_state):
        """Snapshot creates a copy; restore brings it back."""
        # Create snapshot
        snap = await pm.snapshot(agent_state)
        assert snap.agent_id == agent_state
        assert snap.journal_entries == 2
        assert snap.knowledge_count == 1
        assert "efficiency" in snap.profile
        
        # Corrupt the original
        agent_dir = Path(pm.state_dir) / "agents" / agent_state
        (agent_dir / "IDENTITY.md").write_text("# CORRUPTED")
        (agent_dir / "JOURNAL.jsonl").write_text("")
        
        # Restore
        ok = await pm.restore(agent_state)
        assert ok is True
        
        # Verify restored content
        assert (agent_dir / "IDENTITY.md").read_text() == "# Test Agent\nI am a test agent."
        assert (agent_dir / "JOURNAL.jsonl").read_text().count("\n") == 2
    
    @pytest.mark.asyncio
    async def test_list_snapshots(self, pm, agent_state):
        """list_snapshots returns all snapshots for an agent."""
        await pm.snapshot(agent_state)
        await asyncio.sleep(0.05)  # ensure different timestamps
        await pm.snapshot(agent_state)
        
        snaps = pm.list_snapshots(agent_state)
        assert len(snaps) == 2
        # Latest first (descending by timestamp)
        assert snaps[0].timestamp >= snaps[1].timestamp
    
    @pytest.mark.asyncio
    async def test_rollback(self, pm, agent_state):
        """Rollback reverts to a specific snapshot."""
        # First snapshot
        snap1 = await pm.snapshot(agent_state)
        await asyncio.sleep(0.05)
        
        # Modify state
        agent_dir = Path(pm.state_dir) / "agents" / agent_state
        (agent_dir / "IDENTITY.md").write_text("# Changed")
        await pm.snapshot(agent_state)
        
        # Rollback to first
        ok = await pm.rollback(agent_state, snap1.timestamp)
        assert ok is True
        
        # Should have original content
        assert (agent_dir / "IDENTITY.md").read_text() == "# Test Agent\nI am a test agent."
    
    @pytest.mark.asyncio
    async def test_incremental_save(self, pm, agent_state):
        """Incremental save only persists changed files."""
        # Initial snapshot
        await pm.snapshot(agent_state)
        
        # Change one file
        agent_dir = Path(pm.state_dir) / "agents" / agent_state
        (agent_dir / "IDENTITY.md").write_text("# Updated")
        
        # Incremental save
        await pm.incremental_save(agent_state, ["IDENTITY.md"])
        
        # Restore latest and verify
        await pm.restore(agent_state)
        assert (agent_dir / "IDENTITY.md").read_text() == "# Updated"
    
    @pytest.mark.asyncio
    async def test_restore_latest(self, pm, agent_state):
        """Restore without timestamp uses the latest snapshot."""
        # Snapshot 1
        await pm.snapshot(agent_state)
        await asyncio.sleep(0.05)
        
        # Modify and snapshot 2
        agent_dir = Path(pm.state_dir) / "agents" / agent_state
        (agent_dir / "IDENTITY.md").write_text("# Version 2")
        snap2 = await pm.snapshot(agent_state)
        
        # Modify again
        (agent_dir / "IDENTITY.md").write_text("# Version 3")
        
        # Restore latest (should be snap2 = "Version 2")
        ok = await pm.restore(agent_state)
        assert ok is True
        assert (agent_dir / "IDENTITY.md").read_text() == "# Version 2"
    
    @pytest.mark.asyncio
    async def test_restore_no_snapshots(self, pm, agent_state):
        """Restore with no snapshots returns False."""
        ok = await pm.restore(agent_state)
        assert ok is False
    
    @pytest.mark.asyncio
    async def test_snapshot_metadata(self, pm, agent_state):
        """Snapshot metadata is correct."""
        snap = await pm.snapshot(agent_state)
        
        assert snap.agent_id == agent_state
        assert snap.timestamp  # non-empty
        assert snap.soul_md5  # has hash
        assert snap.journal_entries == 2
        assert snap.knowledge_count == 1
        assert snap.profile == {"efficiency": 0.5, "confidence": 0.7}
        assert snap.state_version == 1
