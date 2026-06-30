"""Agent Forking System — clone agents for parallel tasks.

An agent can fork into child agents that inherit the parent's soul and
knowledge. After completing parallel work, children merge their knowledge
back into the parent.

Key concepts:
  - ForkConfig: defines what to inherit and any overrides
  - AgentForker: orchestrates fork, merge, and family tree queries
  - meta.json: tracks fork relationships (parent_id, fork_reason, merged status)
"""

from __future__ import annotations

import json
import logging
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ForkConfig:
    """Configuration for forking a new agent.

    Attributes:
        parent_id: The agent being forked from
        fork_reason: "parallel_task" | "experiment" | "specialization"
        inherit_soul: Copy IDENTITY.md, ROLE.md, JOURNAL.md from parent
        inherit_knowledge: Copy KNOWLEDGE.json from parent
        inherit_journal: Copy JOURNAL.jsonl from parent
        personality_override: Override personality card name
        role_override: Override role card name
    """

    parent_id: str
    fork_reason: str = "parallel_task"
    inherit_soul: bool = True
    inherit_knowledge: bool = True
    inherit_journal: bool = False
    personality_override: str | None = None
    role_override: str | None = None


class AgentForker:
    """Agent cloning and merge system.

    Usage:
        forker = AgentForker()
        child_id = await forker.fork(ForkConfig(
            parent_id="agent:abc",
            fork_reason="parallel_task",
        ))
        # ... child completes work ...
        await forker.merge_back(child_id, "agent:abc")
        tree = forker.get_family_tree("agent:abc")
    """

    # Files to copy when inherit_soul=True
    SOUL_FILES = ["IDENTITY.md", "ROLE.md", "JOURNAL.md", "profile.json", "meta.json"]

    # Files to copy when inherit_knowledge=True
    KNOWLEDGE_FILES = ["KNOWLEDGE.json"]

    # Files to copy when inherit_journal=True
    JOURNAL_FILES = ["JOURNAL.jsonl"]

    def __init__(self, state_dir: str = "state/agents"):
        self._state_dir = Path(state_dir)

    def _agent_dir(self, agent_id: str) -> Path:
        """Get the state directory for an agent."""
        return self._state_dir / agent_id

    def _read_json(self, path: Path) -> dict:
        """Read a JSON file, return empty dict on failure."""
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, IOError) as e:
            logger.warning("AgentForker: failed to read %s: %s", path, e)
            return {}

    def _write_json(self, path: Path, data: dict) -> None:
        """Write data as JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    # ── Fork ─────────────────────────────────────────────────────

    async def fork(self, config: ForkConfig) -> str:
        """Fork a new agent from the parent.

        Steps:
          1. Generate a new child agent_id
          2. Copy parent's private state files per config
          3. Apply personality/role overrides
          4. Write child's meta.json with fork relationship
          5. Return child agent_id

        Args:
            config: ForkConfig specifying what to inherit

        Returns:
            The new child agent's ID string
        """
        parent_dir = self._agent_dir(config.parent_id)
        if not parent_dir.exists():
            raise FileNotFoundError(f"Parent agent directory not found: {parent_dir}")

        child_id = f"agent:fork:{uuid.uuid4().hex[:8]}"
        child_dir = self._agent_dir(child_id)
        child_dir.mkdir(parents=True, exist_ok=True)

        files_copied: list[str] = []

        # Copy soul files (IDENTITY.md, ROLE.md, JOURNAL.md, profile.json, meta.json)
        if config.inherit_soul:
            for fname in self.SOUL_FILES:
                src = parent_dir / fname
                dst = child_dir / fname
                if src.exists():
                    shutil.copy2(str(src), str(dst))
                    files_copied.append(fname)

        # Copy knowledge files (KNOWLEDGE.json)
        if config.inherit_knowledge:
            for fname in self.KNOWLEDGE_FILES:
                src = parent_dir / fname
                dst = child_dir / fname
                if src.exists():
                    shutil.copy2(str(src), str(dst))
                    files_copied.append(fname)

        # Copy journal files (JOURNAL.jsonl)
        if config.inherit_journal:
            for fname in self.JOURNAL_FILES:
                src = parent_dir / fname
                dst = child_dir / fname
                if src.exists():
                    shutil.copy2(str(src), str(dst))
                    files_copied.append(fname)

        # Apply personality override
        if config.personality_override:
            profile = self._read_json(child_dir / "profile.json")
            profile["personality"] = config.personality_override
            self._write_json(child_dir / "profile.json", profile)

            # Update ROLE.md if it exists
            role_path = child_dir / "ROLE.md"
            if role_path.exists():
                content = role_path.read_text()
                content = content.replace(
                    f"个性类型：{config.personality_override}",
                    ""  # handled in IDENTITY.md instead
                )
                role_path.write_text(content)

        # Apply role override
        if config.role_override:
            profile = self._read_json(child_dir / "profile.json")
            profile["role"] = config.role_override
            self._write_json(child_dir / "profile.json", profile)

            # Regenerate ROLE.md
            role_path = child_dir / "ROLE.md"
            role_path.write_text(f"# 角色\n\n当前角色：{config.role_override}\n")

        # Write child's meta.json with fork relationship
        now = datetime.now(timezone.utc).isoformat()
        child_meta = self._read_json(child_dir / "meta.json")
        child_meta.update({
            "agent_id": child_id,
            "parent_id": config.parent_id,
            "fork_reason": config.fork_reason,
            "forked_at": now,
            "merged": False,
            "merged_at": None,
            "personality_override": config.personality_override,
            "role_override": config.role_override,
            "files_inherited": files_copied,
        })
        self._write_json(child_dir / "meta.json", child_meta)

        # Update parent's meta.json to record child
        parent_meta_path = parent_dir / "meta.json"
        if parent_meta_path.exists():
            parent_meta = self._read_json(parent_meta_path)
            children = parent_meta.get("children", [])
            if child_id not in children:
                children.append(child_id)
            parent_meta["children"] = children
            self._write_json(parent_meta_path, parent_meta)

        logger.info("AgentForker: forked %s → %s (reason: %s, inherited: %s)",
                    config.parent_id, child_id, config.fork_reason, files_copied)
        return child_id

    # ── Merge Back ───────────────────────────────────────────────

    async def merge_back(self, child_id: str, parent_id: str) -> None:
        """Merge child agent's knowledge back into parent.

        Steps:
          1. Read child's KNOWLEDGE.json, merge into parent's
          2. Append child's last few JOURNAL.jsonl entries into parent's
          3. Mark child as "merged" in its meta.json

        Args:
            child_id: The child agent to merge from
            parent_id: The parent agent to merge into
        """
        child_dir = self._agent_dir(child_id)
        parent_dir = self._agent_dir(parent_id)

        if not child_dir.exists():
            logger.warning("AgentForker: child %s not found for merge", child_id)
            return

        parent_dir.mkdir(parents=True, exist_ok=True)

        # Merge KNOWLEDGE.json
        child_knowledge = self._read_json(child_dir / "KNOWLEDGE.json")
        parent_knowledge = self._read_json(parent_dir / "KNOWLEDGE.json")

        if child_knowledge:
            # Merge by id, preferring child's newer entries
            existing_ids = {n.get("id") for n in parent_knowledge if isinstance(parent_knowledge, list)}
            if isinstance(child_knowledge, list):
                new_entries = [n for n in child_knowledge if n.get("id") not in existing_ids]
                parent_knowledge_list = parent_knowledge if isinstance(parent_knowledge, list) else []
                merged = parent_knowledge_list + new_entries
            else:
                merged = parent_knowledge

            self._write_json(parent_dir / "KNOWLEDGE.json", merged)
            logger.debug("AgentForker: merged KNOWLEDGE.json %s → %s (+%d entries)",
                        child_id, parent_id, len(new_entries) if isinstance(child_knowledge, list) else 0)

        # Append child's JOURNAL.jsonl (last 10 entries) to parent
        child_journal = child_dir / "JOURNAL.jsonl"
        parent_journal = parent_dir / "JOURNAL.jsonl"

        if child_journal.exists():
            try:
                lines = child_journal.read_text().strip().split("\n")
                # Take last 10 non-empty lines
                recent = [l for l in lines if l.strip()][-10:]
                with open(parent_journal, "a") as f:
                    for line in recent:
                        f.write(line + "\n")
                logger.debug("AgentForker: appended %d journal entries %s → %s",
                            len(recent), child_id, parent_id)
            except Exception as e:
                logger.warning("AgentForker: failed to merge journal: %s", e)

        # Mark child as merged
        child_meta = self._read_json(child_dir / "meta.json")
        child_meta["merged"] = True
        child_meta["merged_at"] = datetime.now(timezone.utc).isoformat()
        child_meta["merged_into"] = parent_id
        self._write_json(child_dir / "meta.json", child_meta)

        logger.info("AgentForker: merged %s → %s", child_id, parent_id)

    # ── Family Tree ──────────────────────────────────────────────

    def get_family_tree(self, agent_id: str) -> dict:
        """Get the family tree for an agent (parent, children, siblings).

        Returns a dict:
            {
                "agent_id": "...",
                "parent_id": "..." | None,
                "children": [...],
                "siblings": [...],
                "fork_reason": "...",
                "merged": bool,
            }
        """
        agent_dir = self._agent_dir(agent_id)
        meta = self._read_json(agent_dir / "meta.json")

        parent_id = meta.get("parent_id")

        # Find siblings (other children of the same parent)
        siblings: list[dict] = []
        if parent_id:
            parent_dir = self._agent_dir(parent_id)
            parent_meta = self._read_json(parent_dir / "meta.json")
            for child in parent_meta.get("children", []):
                if child != agent_id:
                    child_meta = self._read_json(self._agent_dir(child) / "meta.json")
                    siblings.append({
                        "agent_id": child,
                        "merged": child_meta.get("merged", False),
                        "fork_reason": child_meta.get("fork_reason", ""),
                    })

        # Collect children
        children: list[dict] = []
        for child in meta.get("children", []):
            child_meta = self._read_json(self._agent_dir(child) / "meta.json")
            children.append({
                "agent_id": child,
                "merged": child_meta.get("merged", False),
                "fork_reason": child_meta.get("fork_reason", ""),
            })

        return {
            "agent_id": agent_id,
            "parent_id": parent_id,
            "children": children,
            "siblings": siblings,
            "fork_reason": meta.get("fork_reason", ""),
            "merged": meta.get("merged", False),
        }
