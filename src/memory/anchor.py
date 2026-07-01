"""Anchor Layer — stable key-fact storage.

Anchor files are Markdown files in state/anchors/ containing
project metadata, system config, and relationship maps.
They are:
  - NOT subject to consolidation (stable data)
  - Overwritten on change (not appended)
  - Synced to SurrealDB fact table (entity type, agent_id="anchor")
  - Directly readable by agents (precise lookup, O(1))
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

ANCHOR_DIR = "state/anchors"
ANCHOR_AGENT_ID = "anchor"


@dataclass
class AnchorEntry:
    """A single anchor entry (key-value pair in a markdown file)."""
    key: str          # e.g. "github_repo"
    value: str        # e.g. "https://github.com/xxx/agent-loop"
    category: str = ""  # e.g. "repository", "server", "owner"


@dataclass
class AnchorFile:
    """An anchor markdown file."""
    name: str                    # file name without .md
    title: str                   # human-readable title
    entries: list[AnchorEntry] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Render as markdown."""
        lines = [f"# {self.title}", ""]
        # Group by category
        by_cat: dict[str, list[AnchorEntry]] = {}
        for e in self.entries:
            by_cat.setdefault(e.category, []).append(e)

        for cat, entries in by_cat.items():
            # Always emit category heading for roundtrip fidelity.
            # Empty-category entries get ## General.
            label = cat if cat else "General"
            lines.append(f"## {label}")
            for e in entries:
                lines.append(f"- **{e.key}**: {e.value}")
            lines.append("")

        return "\n".join(lines)

    @classmethod
    def from_markdown(cls, content: str, name: str) -> "AnchorFile":
        """Parse markdown back to AnchorFile."""
        lines = content.strip().split("\n")
        title = ""
        entries = []
        current_cat = ""

        for line in lines:
            line = line.strip()
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("## "):
                current_cat = line[3:].strip()
            elif line.startswith("- **"):
                # Parse: - **key**: value
                try:
                    rest = line[4:]  # after "- **"
                    key_end = rest.index("**")
                    key = rest[:key_end]
                    value = rest[key_end + 2:].lstrip(": ").strip()
                    entries.append(AnchorEntry(key=key, value=value, category=current_cat))
                except (ValueError, IndexError):
                    pass

        return cls(name=name, title=title, entries=entries)


class AnchorManager:
    """Manages anchor files and syncs to SurrealDB.

    Features:
    - Read/write anchor markdown files
    - Sync to fact table (entity type, agent_id="anchor")
    - Precise lookup by key
    - List all anchors
    - Auto-discover anchor files in directory
    """

    def __init__(self, base_dir: str = ANCHOR_DIR, memory_pool=None):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.memory = memory_pool

    def _anchor_path(self, name: str) -> Path:
        """Get path for anchor file."""
        if not name.endswith(".md"):
            name = name + ".md"
        return self.base_dir / name

    def write_anchor(self, name: str, title: str, entries: list[AnchorEntry]) -> str:
        """Write an anchor file. Overwrites if exists.

        Args:
            name: File name (without .md)
            title: Human-readable title
            entries: List of key-value entries

        Returns:
            Path to the written file
        """
        anchor = AnchorFile(name=name, title=title, entries=entries)
        path = self._anchor_path(name)
        path.write_text(anchor.to_markdown(), encoding="utf-8")
        logger.info("AnchorManager: wrote %s (%d entries)", name, len(entries))
        return str(path)

    def read_anchor(self, name: str) -> AnchorFile | None:
        """Read an anchor file.

        Returns None if not found.
        """
        path = self._anchor_path(name)
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        return AnchorFile.from_markdown(content, name)

    def list_anchors(self) -> list[str]:
        """List all anchor file names (without .md)."""
        return sorted([f.stem for f in self.base_dir.glob("*.md")])

    def lookup(self, name: str, key: str) -> str | None:
        """Precise lookup: get value for a specific key in an anchor file.

        O(1) file read + O(n) scan (n = entries in file, usually <20).
        """
        anchor = self.read_anchor(name)
        if not anchor:
            return None
        for entry in anchor.entries:
            if entry.key == key:
                return entry.value
        return None

    def get_all_entries(self) -> list[tuple[str, AnchorEntry]]:
        """Get all entries from all anchor files, with their anchor name."""
        result = []
        for name in self.list_anchors():
            anchor = self.read_anchor(name)
            if anchor:
                for entry in anchor.entries:
                    result.append((name, entry))
        return result

    async def sync_to_db(self, name: str | None = None) -> int:
        """Sync anchor file(s) to SurrealDB fact table.

        Writes each entry as a fact:
          - fact_type: "entity"
          - name: "{anchor_name}.{key}"
          - value: the value
          - agent_id: "anchor"

        Args:
            name: Specific anchor to sync, or None for all

        Returns:
            Number of facts synced
        """
        if not self.memory or not self.memory._db:
            logger.debug("AnchorManager: no DB connection, skipping sync")
            return 0

        names = [name] if name else self.list_anchors()
        count = 0

        for n in names:
            anchor = self.read_anchor(n)
            if not anchor:
                continue

            for entry in anchor.entries:
                try:
                    await self.memory.write_fact(
                        fact_type="entity",
                        name=f"{n}.{entry.key}",
                        value=entry.value,
                        agent_id=ANCHOR_AGENT_ID,
                        upsert=True,
                    )
                    count += 1
                except Exception as e:
                    logger.warning("Anchor sync %s.%s failed: %s", n, entry.key, e)

        logger.info("AnchorManager: synced %d facts from %d files", count, len(names))
        return count

    async def load_from_db(self, name: str) -> AnchorFile | None:
        """Load anchor from SurrealDB fact table (fallback if file missing)."""
        if not self.memory or not self.memory._db:
            return None

        try:
            facts = await self.memory.query_facts(
                agent_id=ANCHOR_AGENT_ID,
                limit=1000,
            )
            entries = []
            for fact in facts:
                fact_name = fact.get("name", "")
                if fact_name.startswith(f"{name}."):
                    key = fact_name[len(name) + 1:]
                    value = fact.get("value", "")
                    entries.append(AnchorEntry(key=key, value=str(value)))

            if not entries:
                return None

            return AnchorFile(
                name=name,
                title=name.replace("_", " ").title(),
                entries=entries,
            )
        except Exception as e:
            logger.warning("Anchor load_from_db %s failed: %s", name, e)
            return None

    async def delete_anchor_from_db(self, name: str) -> int:
        """Delete all facts for a given anchor from DB."""
        if not self.memory or not self.memory._db:
            return 0
        try:
            result = await self.memory._db.query(
                "DELETE FROM fact WHERE agent_id = $agent_id AND "
                "name CONTAINS $prefix",
                {"agent_id": ANCHOR_AGENT_ID, "prefix": f"{name}."},
            )
            count = len(result) if isinstance(result, list) else 0
            logger.info("AnchorManager: deleted %d facts for %s", count, name)
            return count
        except Exception as e:
            logger.warning("Anchor delete_from_db %s failed: %s", name, e)
            return 0
