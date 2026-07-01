"""Project-based workspace management.

Each project = a folder on disk. All agents working on the same project
share the same workspace directory (like Cursor/Claude Code).

Directory structure:
    ~/projects/my-app/
    ├── .agent_loop/           # agent metadata
    │   ├── project_card.md    # auto-generated project card (Layer 1)
    │   ├── sessions/          # session summaries
    │   ├── ammo/              # ammo box caches per task
    │   └── temp/              # intermediate findings
    ├── src/
    ├── docs/
    └── ...
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Project:
    """A project workspace rooted at a filesystem directory.

    All agents working on this project share:
      - File system: direct read/write under `root`
      - Memory: MemoryPool filtered by `project_id`
      - Project Card: auto-injected pinned context
    """

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        if not self.root.exists():
            self.root.mkdir(parents=True, exist_ok=True)
        self.agent_dir = self.root / ".agent_loop"
        self.agent_dir.mkdir(exist_ok=True)
        (self.agent_dir / "sessions").mkdir(exist_ok=True)
        (self.agent_dir / "ammo").mkdir(exist_ok=True)
        (self.agent_dir / "temp").mkdir(exist_ok=True)

    @property
    def project_id(self) -> str:
        """Project ID = directory name."""
        return self.root.name

    @property
    def card_path(self) -> Path:
        return self.agent_dir / "project_card.json"

    @property
    def workspace(self) -> Path:
        """The workspace directory (same as root)."""
        return self.root

    # ── Project Card (Layer 1: auto-injected) ────────────────────

    def load_card(self) -> "ProjectCard":
        """Load or create the project card."""
        card = ProjectCard(self)
        card.load()
        return card

    def save_card(self, card: "ProjectCard") -> None:
        card.save()

    # ── Session summaries (Layer 2: recent context) ──────────────

    def save_session_summary(self, session_id: str, summary: str,
                             user_input: str, output: str) -> None:
        """Save a session summary for recent context."""
        path = self.agent_dir / "sessions" / f"{session_id[:8]}.json"
        data = {
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary[:500],  # cap at 500 chars
            "user_input": user_input[:200],
            "output": output[:200],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        # Keep only recent 10 sessions
        sessions = sorted(
            (self.agent_dir / "sessions").glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in sessions[10:]:
            old.unlink(missing_ok=True)

    def load_recent_sessions(self, limit: int = 3) -> list[dict]:
        """Load recent session summaries."""
        sessions_dir = self.agent_dir / "sessions"
        if not sessions_dir.exists():
            return []
        sessions = sorted(
            sessions_dir.glob("*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        results = []
        for p in sessions:
            try:
                results.append(json.loads(p.read_text(encoding="utf-8")))
            except Exception:
                continue
        return results

    # ── Intermediate findings (shared between workers) ───────────

    def write_finding(self, task_id: str, finding: str) -> Path:
        """Write a finding that other workers can read."""
        path = self.agent_dir / "temp" / f"finding_{task_id[:8]}.md"
        path.write_text(finding, encoding="utf-8")
        return path

    def read_findings(self, exclude_task_id: str | None = None) -> list[dict]:
        """Read findings from other workers."""
        temp_dir = self.agent_dir / "temp"
        if not temp_dir.exists():
            return []
        findings = []
        for p in sorted(temp_dir.glob("finding_*.md"), key=lambda x: x.stat().st_mtime, reverse=True):
            tid = p.stem.replace("finding_", "")
            if exclude_task_id and tid == exclude_task_id[:8]:
                continue
            try:
                content = p.read_text(encoding="utf-8")
                findings.append({"task_id": tid, "content": content[:1000]})
            except Exception:
                continue
        return findings[:10]  # cap at 10 findings

    def cleanup_findings(self) -> None:
        """Clean up old findings (call after collect phase)."""
        temp_dir = self.agent_dir / "temp"
        if temp_dir.exists():
            for p in temp_dir.glob("finding_*.md"):
                p.unlink(missing_ok=True)

    # ── Project info for card auto-generation ────────────────────

    def detect_info(self) -> dict[str, str]:
        """Auto-detect project info from files."""
        info = {"name": self.project_id, "description": "", "tech_stack": ""}

        # Try README
        for readme in ["README.md", "readme.md", "README.rst", "README.txt"]:
            p = self.root / readme
            if p.exists():
                try:
                    text = p.read_text(encoding="utf-8")[:2000]
                    # Extract first paragraph as description
                    lines = text.split("\n")
                    desc_lines = []
                    for line in lines[1:]:  # skip title
                        if line.strip():
                            desc_lines.append(line.strip())
                        elif desc_lines:
                            break
                    info["description"] = " ".join(desc_lines)[:300]
                    break
                except Exception:
                    continue

        # Try package.json / pyproject.toml / Cargo.toml
        pkg = self.root / "package.json"
        if pkg.exists():
            try:
                data = json.loads(pkg.read_text(encoding="utf-8"))
                info["name"] = data.get("name", info["name"])
                deps = list(data.get("dependencies", {}).keys())[:5]
                info["tech_stack"] = ", ".join(deps) if deps else "Node.js"
            except Exception:
                pass

        pyproject = self.root / "pyproject.toml"
        if pyproject.exists():
            try:
                text = pyproject.read_text(encoding="utf-8")
                info["tech_stack"] = "Python"
                if "fastapi" in text.lower():
                    info["tech_stack"] += ", FastAPI"
                if "django" in text.lower():
                    info["tech_stack"] += ", Django"
            except Exception:
                pass

        if not info["description"]:
            info["description"] = f"Project at {self.root}"

        return info


class ProjectCard:
    """Layer 1: auto-injected pinned context (~300 tokens).

    Loaded at session start, contains:
      - Project name, description, tech stack
      - Current status
      - Recent session summaries (1-2 sentences each)
    """

    def __init__(self, project: Project):
        self.project = project
        self.name: str = project.project_id
        self.description: str = ""
        self.tech_stack: str = ""
        self.status: str = "active"
        self.recent_sessions: list[dict] = []
        self.updated_at: str = ""

    def load(self) -> None:
        """Load from card_path or auto-detect."""
        if self.project.card_path.exists():
            try:
                data = json.loads(self.project.card_path.read_text(encoding="utf-8"))
                self.name = data.get("name", self.name)
                self.description = data.get("description", "")
                self.tech_stack = data.get("tech_stack", "")
                self.status = data.get("status", "active")
                self.recent_sessions = data.get("recent_sessions", [])
                self.updated_at = data.get("updated_at", "")
                return
            except Exception:
                pass
        # Auto-detect on first load
        info = self.project.detect_info()
        self.name = info["name"]
        self.description = info["description"]
        self.tech_stack = info["tech_stack"]
        self.recent_sessions = self.project.load_recent_sessions(limit=3)
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.save()

    def save(self) -> None:
        """Save card to disk."""
        data = {
            "name": self.name,
            "description": self.description,
            "tech_stack": self.tech_stack,
            "status": self.status,
            "recent_sessions": self.recent_sessions,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.project.card_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def update_session(self, session_id: str, summary: str,
                       user_input: str, output: str) -> None:
        """Add a session to recent list (keep last 3)."""
        self.recent_sessions.append({
            "session_id": session_id[:8],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": summary[:200],
        })
        self.recent_sessions = self.recent_sessions[-3:]
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.save()

    def to_context(self) -> str:
        """Render to ~300 token string for context injection."""
        parts = [f"# Project: {self.name}"]
        if self.description:
            parts.append(f"## 描述\n{self.description}")
        if self.tech_stack:
            parts.append(f"## 技术栈\n{self.tech_stack}")
        if self.recent_sessions:
            parts.append("## 最近会话")
            for s in self.recent_sessions[-3:]:
                parts.append(f"- {s.get('summary', '')[:100]}")
        return "\n".join(parts)
