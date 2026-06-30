"""Cross-session persistence for Agent state, memory, and evolution data.

Provides snapshot / restore / rollback / incremental-save for Agent state
so that Agent lifecycles survive process restarts.

Key concepts:
  - Snapshot: a point-in-time copy of an Agent's state directory
  - Restore: bring back a previous snapshot (default: latest)
  - Rollback: revert to a specific historical snapshot
  - Incremental save: only persist files that changed since last snapshot
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================
# Snapshot
# ============================================================


@dataclass
class Snapshot:
    """Metadata for an Agent state snapshot.

    A snapshot is a point-in-time copy of ``state/agents/{id}/`` directory
    stored under ``state/snapshots/{id}/{timestamp}/``.
    """
    agent_id: str
    timestamp: str          # ISO 8601, used as the directory name
    soul_md5: str           # MD5 hash of IDENTITY.md + ROLE.md
    journal_entries: int    # Lines in JOURNAL.jsonl
    knowledge_count: int    # Items in KNOWLEDGE.json
    profile: dict           # Contents of profile.json
    cost_total: float       # Cumulative cost in USD
    state_version: int = 1  # Snapshot format version

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "timestamp": self.timestamp,
            "soul_md5": self.soul_md5,
            "journal_entries": self.journal_entries,
            "knowledge_count": self.knowledge_count,
            "profile": self.profile,
            "cost_total": self.cost_total,
            "state_version": self.state_version,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Snapshot":
        return cls(**{k: d.get(k) for k in [
            "agent_id", "timestamp", "soul_md5",
            "journal_entries", "knowledge_count",
            "profile", "cost_total", "state_version",
        ]})


# ============================================================
# PersistenceManager
# ============================================================


class PersistenceManager:
    """Cross-session persistence manager for Agent state.

    Capabilities:
      1. Periodic snapshots of Agent state directories
      2. Restore from latest (or specified) snapshot on startup
      3. Rollback to any historical snapshot
      4. Incremental save — only persist changed files
    """

    def __init__(self, state_dir: str = "state",
                 snapshot_dir: str = "state/snapshots"):
        self.state_dir = Path(state_dir)
        self.snapshot_dir = Path(snapshot_dir)

    # ── Agent state directory ─────────────────────────────────────

    def _agent_dir(self, agent_id: str) -> Path:
        return self.state_dir / "agents" / agent_id

    def _snapshot_agent_dir(self, agent_id: str, timestamp: str) -> Path:
        return self.snapshot_dir / agent_id / timestamp

    # ── Snapshot ──────────────────────────────────────────────────

    async def snapshot(self, agent_id: str) -> Snapshot:
        """Create a full snapshot of an Agent's state directory.

        Steps:
          1. Determine timestamp
          2. Copy ``state/agents/{id}/`` → ``state/snapshots/{id}/{ts}/``
          3. Build and write ``meta.json`` with snapshot metadata
          4. Return Snapshot metadata

        Returns:
            Snapshot metadata. If the agent directory doesn't exist, returns
            a minimal Snapshot with zero counts.
        """
        now = datetime.now(timezone.utc)
        timestamp = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"

        snap_dir = self._snapshot_agent_dir(agent_id, timestamp)
        agent_dir = self._agent_dir(agent_id)

        # Compute metadata from source
        soul_md5 = ""
        journal_entries = 0
        knowledge_count = 0
        profile: dict = {}
        cost_total = 0.0

        if agent_dir.exists():
            # Copy directory
            snap_dir.parent.mkdir(parents=True, exist_ok=True)
            if snap_dir.exists():
                shutil.rmtree(str(snap_dir))
            shutil.copytree(str(agent_dir), str(snap_dir))

            # Compute metadata
            soul_md5 = await self._compute_soul_md5(agent_id)
            journal_entries = self._count_journal_entries(snap_dir)
            knowledge_count = self._count_knowledge(snap_dir)
            profile = self._load_profile(snap_dir)
            cost_total = self._load_cost_total(snap_dir)

        else:
            # No state yet — create minimal snapshot
            snap_dir.parent.mkdir(parents=True, exist_ok=True)
            snap_dir.mkdir(parents=True, exist_ok=True)

        # Write meta.json
        snapshot = Snapshot(
            agent_id=agent_id,
            timestamp=timestamp,
            soul_md5=soul_md5,
            journal_entries=journal_entries,
            knowledge_count=knowledge_count,
            profile=profile,
            cost_total=cost_total,
        )
        meta_path = snap_dir / "meta.json"
        meta_path.write_text(json.dumps(snapshot.to_dict(), indent=2))

        logger.info("PersistenceManager: snapshot created for '%s' @ %s", agent_id, timestamp)
        return snapshot

    # ── Restore ───────────────────────────────────────────────────

    async def restore(self, agent_id: str,
                      snapshot_timestamp: str | None = None) -> bool:
        """Restore agent state from a snapshot.

        If ``snapshot_timestamp`` is None, restores from the latest snapshot.
        If no snapshots exist, returns False.

        Returns True on success.
        """
        timestamp = snapshot_timestamp
        if timestamp is None:
            # Find latest
            snapshots = self.list_snapshots(agent_id)
            if not snapshots:
                logger.debug("PersistenceManager: no snapshots for '%s'", agent_id)
                return False
            timestamp = snapshots[0].timestamp

        snap_dir = self._snapshot_agent_dir(agent_id, timestamp)
        if not snap_dir.exists():
            logger.warning("PersistenceManager: snapshot '%s/%s' not found",
                          agent_id, timestamp)
            return False

        agent_dir = self._agent_dir(agent_id)
        # Clear current state
        if agent_dir.exists():
            shutil.rmtree(str(agent_dir))
        agent_dir.mkdir(parents=True, exist_ok=True)

        # Copy snapshot back (skip meta.json — it's snapshot metadata only)
        for item in snap_dir.iterdir():
            if item.name == "meta.json":
                continue
            dest = agent_dir / item.name
            if item.is_dir():
                shutil.copytree(str(item), str(dest))
            else:
                shutil.copy2(str(item), str(dest))

        logger.info("PersistenceManager: restored '%s' from snapshot %s", agent_id, timestamp)
        return True

    # ── List ──────────────────────────────────────────────────────

    def list_snapshots(self, agent_id: str) -> list[Snapshot]:
        """List all snapshots for an agent, newest first."""
        base = self.snapshot_dir / agent_id
        if not base.exists():
            return []

        result: list[Snapshot] = []
        for ts_dir in sorted(base.iterdir(), reverse=True):
            if not ts_dir.is_dir():
                continue
            meta_path = ts_dir / "meta.json"
            if meta_path.exists():
                try:
                    data = json.loads(meta_path.read_text())
                    result.append(Snapshot.from_dict(data))
                except Exception as e:
                    logger.warning("PersistenceManager: bad meta.json in %s: %s", ts_dir, e)
        return result

    # ── Rollback ──────────────────────────────────────────────────

    async def rollback(self, agent_id: str, timestamp: str) -> bool:
        """Rollback to a specific historical snapshot.

        This is a convenience wrapper around ``restore`` with an explicit timestamp.
        """
        return await self.restore(agent_id, snapshot_timestamp=timestamp)

    # ── Incremental save ──────────────────────────────────────────

    async def incremental_save(self, agent_id: str,
                               changed_files: list[str]) -> None:
        """Incrementally save only the files that changed since last snapshot.

        Args:
            agent_id: Agent identifier.
            changed_files: List of file paths relative to the agent directory
                           that should be persisted.
        """
        agent_dir = self._agent_dir(agent_id)
        if not agent_dir.exists():
            logger.debug("PersistenceManager: agent dir '%s' not found, skipping incr save", agent_id)
            return

        # Use latest snapshot directory as base
        snapshots = self.list_snapshots(agent_id)
        if not snapshots:
            # No previous snapshot — do a full snapshot instead
            await self.snapshot(agent_id)
            return

        latest_ts = snapshots[0].timestamp
        base_dir = self._snapshot_agent_dir(agent_id, latest_ts)

        # Create a new incremental snapshot directory
        now = datetime.now(timezone.utc)
        new_ts = now.strftime("%Y%m%dT%H%M%S") + f"{now.microsecond // 1000:03d}Z"
        new_dir = self._snapshot_agent_dir(agent_id, new_ts)
        new_dir.parent.mkdir(parents=True, exist_ok=True)
        new_dir.mkdir(parents=True, exist_ok=True)

        # Copy unchanged files from base snapshot via hard link, copy changed ones
        saved = 0
        for fname in changed_files:
            src = agent_dir / fname
            dst = new_dir / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.exists():
                if src.is_file():
                    shutil.copy2(str(src), str(dst))
                else:
                    shutil.copytree(str(src), str(dst))
                saved += 1

        # For unchanged files, copy from base snapshot if they exist there
        if base_dir.exists():
            for item in agent_dir.iterdir():
                if item.name in changed_files:
                    continue  # Already handled
                dest = new_dir / item.name
                if dest.exists():
                    continue  # Don't overwrite
                base_item = base_dir / item.name
                if base_item.exists():
                    if base_item.is_dir():
                        shutil.copytree(str(base_item), str(dest))
                    else:
                        shutil.copy2(str(base_item), str(dest))
                else:
                    if item.is_dir():
                        shutil.copytree(str(item), str(dest))
                    else:
                        shutil.copy2(str(item), str(dest))

        # Write meta.json
        soul_md5 = await self._compute_soul_md5(agent_id)
        journal_entries = self._count_journal_entries(new_dir)
        knowledge_count = self._count_knowledge(new_dir)
        profile = self._load_profile(new_dir)
        cost_total = self._load_cost_total(new_dir)

        snapshot = Snapshot(
            agent_id=agent_id,
            timestamp=new_ts,
            soul_md5=soul_md5,
            journal_entries=journal_entries,
            knowledge_count=knowledge_count,
            profile=profile,
            cost_total=cost_total,
        )
        meta_path = new_dir / "meta.json"
        meta_path.write_text(json.dumps(snapshot.to_dict(), indent=2))

        logger.info("PersistenceManager: incremental save for '%s': %d files changed", agent_id, saved)

    # ── Internal helpers ──────────────────────────────────────────

    async def _compute_soul_md5(self, agent_id: str) -> str:
        """Compute MD5 hash of IDENTITY.md + ROLE.md (the Agent's soul)."""
        agent_dir = self._agent_dir(agent_id)
        m = hashlib.md5()
        for fname in ("IDENTITY.md", "ROLE.md"):
            fpath = agent_dir / fname
            if fpath.exists():
                m.update(fpath.read_bytes())
        return m.hexdigest()

    @staticmethod
    def _count_journal_entries(dir_path: Path) -> int:
        """Count lines in JOURNAL.jsonl."""
        jpath = dir_path / "JOURNAL.jsonl"
        if not jpath.exists():
            return 0
        try:
            with open(jpath) as f:
                return sum(1 for _ in f)
        except Exception:
            return 0

    @staticmethod
    def _count_knowledge(dir_path: Path) -> int:
        """Count items in KNOWLEDGE.json."""
        kpath = dir_path / "KNOWLEDGE.json"
        if not kpath.exists():
            return 0
        try:
            data = json.loads(kpath.read_text())
            if isinstance(data, list):
                return len(data)
            if isinstance(data, dict):
                return len(data)
            return 0
        except Exception:
            return 0

    @staticmethod
    def _load_profile(dir_path: Path) -> dict:
        """Load profile.json content."""
        ppath = dir_path / "profile.json"
        if not ppath.exists():
            return {}
        try:
            return json.loads(ppath.read_text())
        except Exception:
            return {}

    @staticmethod
    def _load_cost_total(dir_path: Path) -> float:
        """Load cost_total from profile.json."""
        profile = PersistenceManager._load_profile(dir_path)
        return float(profile.get("cost_total", 0.0))

    def _md5_file(self, path: Path) -> str:
        """Compute MD5 hash of a single file."""
        if not path.exists():
            return ""
        m = hashlib.md5()
        m.update(path.read_bytes())
        return m.hexdigest()
