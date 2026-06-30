"""Agent Soul — managed collection of md files that define an agent's personality.

Loading order (priority: shared > cards > private):
  1. shared/SAFETY.md        (铁律, 不可写)
  2. shared/CONSTRAINTS.md   (硬约束, 不可写)
  3. shared/PRINCIPLES.md    (行为准则, 不可写)
  4. cards/personalities/{type}.md  (卡片模板, 只读引用)
  5. cards/roles/{type}.md         (角色卡片, 只读引用)
  6. private/IDENTITY.md           (可写, Loop Engine 可进化)
  7. private/ROLE.md               (可写, Loop Engine 可进化)
  8. private/JOURNAL.md            (追加写, Loop Engine 进化)

Priority rule:
  SAFETY > CONSTRAINTS > PRINCIPLES > role > personality
  SAFETY.md has absolute priority — any conflict, SAFETY wins.
  Same-layer conflicts: append, don't override, mark ⚠️ 冲突待确认.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Default root for agent soul config
DEFAULT_SOUL_ROOT = Path("config/agents")
DEFAULT_STATE_ROOT = Path("state/agents")


class AgentSoul:
    """Managed collection of md files that define an agent's personality.

    Provides:
    - build_system_prompt(): merge all layers into a system prompt
    - evolve(): append reflection to JOURNAL.md
    - update_role(): change role card
    - update_identity_trait(): adjust profile.json numeric traits
    """

    def __init__(self, agent_id: str, personality: str = "executor", role: str = "coder",
                 soul_root: str | Path = DEFAULT_SOUL_ROOT,
                 state_root: str | Path = DEFAULT_STATE_ROOT):
        self.agent_id = agent_id
        self.personality = personality
        self.role = role
        self._soul_root = Path(soul_root)
        self._state_root = Path(state_root)
        self._private_dir = self._state_root / agent_id

        # Lazy-loaded caches
        self._identity_content: str | None = None
        self._role_content: str | None = None
        self._journal_content: str | None = None
        self._profile: dict[str, Any] | None = None
        self._meta: dict[str, Any] | None = None

        # Initialize private directory if needed
        self._ensure_private_dir()

    def _ensure_private_dir(self) -> None:
        """Create agent's private state directory with default files."""
        self._private_dir.mkdir(parents=True, exist_ok=True)

        # IDENTITY.md
        identity_path = self._private_dir / "IDENTITY.md"
        if not identity_path.exists():
            identity_path.write_text(
                f"# 身份\n\n"
                f"我是 {self.agent_id}，"
                f"个性类型：{self.personality}，"
                f"角色：{self.role}。\n"
            )

        # ROLE.md
        role_path = self._private_dir / "ROLE.md"
        if not role_path.exists():
            role_path.write_text(f"# 角色\n\n当前角色：{self.role}\n")

        # JOURNAL.md
        journal_path = self._private_dir / "JOURNAL.md"
        if not journal_path.exists():
            journal_path.write_text(f"# 执行日志\n\nAgent {self.agent_id} 创建于 {datetime.now(timezone.utc).isoformat()}\n\n")

        # profile.json
        profile_path = self._private_dir / "profile.json"
        if not profile_path.exists():
            profile_path.write_text(json.dumps({
                "curiosity": 0.5,
                "cautiousness": 0.5,
                "assertiveness": 0.5,
                "creativity": 0.5,
                "efficiency": 0.5,
            }, indent=2))

        # meta.json
        meta_path = self._private_dir / "meta.json"
        if not meta_path.exists():
            meta_path.write_text(json.dumps({
                "agent_id": self.agent_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "total_tasks": 0,
                "success_rate": 0.0,
                "personality": self.personality,
                "role": self.role,
            }, indent=2))

    # ── File reading helpers ───────────────────────────────────────

    def _read_md(self, relative_path: str) -> str:
        """Read an md file, return empty string if missing."""
        full = self._soul_root / relative_path
        if full.exists():
            return full.read_text()
        return ""

    def _read_private_md(self, filename: str) -> str:
        """Read a private md file, return empty string if missing."""
        full = self._private_dir / filename
        if full.exists():
            return full.read_text()
        return ""

    def _read_private_json(self, filename: str) -> dict:
        """Read a private JSON file, return empty dict if missing."""
        full = self._private_dir / filename
        if full.exists():
            try:
                return json.loads(full.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    # ── System prompt building ─────────────────────────────────────

    def build_system_prompt(self) -> str:
        """Merge all layers into a system prompt.

        SAFETY.md MUST appear first.
        Same-layer conflicts: append, don't override.
        Cross-layer conflicts: higher priority wins (already ordered).
        """
        sections: list[str] = []

        # Layer 1-3: Shared constitution (highest priority)
        safety = self._read_md("shared/SAFETY.md")
        constraints = self._read_md("shared/CONSTRAINTS.md")
        principles = self._read_md("shared/PRINCIPLES.md")

        # SAFETY must be first
        if safety.strip():
            sections.append(safety.strip())
        if constraints.strip():
            sections.append(constraints.strip())
        if principles.strip():
            sections.append(principles.strip())

        # Layer 4-5: Card templates (personality + role)
        personality_md = self._read_md(f"cards/personalities/{self.personality}.md")
        role_md = self._read_md(f"cards/roles/{self.role}.md")

        if personality_md.strip():
            sections.append(personality_md.strip())
        if role_md.strip():
            sections.append(role_md.strip())

        # Layer 6-8: Private state (evolvable)
        identity = self._read_private_md("IDENTITY.md")
        role_state = self._read_private_md("ROLE.md")
        journal = self._read_private_md("JOURNAL.md")

        if identity.strip():
            sections.append(identity.strip())
        if role_state.strip():
            sections.append(role_state.strip())
        if journal.strip():
            # Limit journal to last 2000 chars to avoid blowing up prompt
            journal_trimmed = journal.strip()
            if len(journal_trimmed) > 2000:
                journal_trimmed = "...(earlier entries truncated)...\n\n" + journal_trimmed[-2000:]
            sections.append(journal_trimmed)

        return "\n\n---\n\n".join(sections)

    # ── Evolution ──────────────────────────────────────────────────

    async def evolve(self, journal_entry: str) -> None:
        """Loop Engine calls this — append reflection to JOURNAL.md."""
        journal_path = self._private_dir / "JOURNAL.md"
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = f"\n## {timestamp}\n{journal_entry}\n"
        with open(journal_path, "a") as f:
            f.write(entry)
        logger.debug("AgentSoul[%s]: evolved — wrote journal entry", self.agent_id)

    async def update_role(self, new_role: str) -> None:
        """Upgrade role (e.g., coder → architect)."""
        self.role = new_role
        role_path = self._private_dir / "ROLE.md"
        role_path.write_text(f"# 角色\n\n当前角色：{new_role}\n")
        # Update meta.json
        meta = self._read_private_json("meta.json")
        meta["role"] = new_role
        meta["role_upgraded_at"] = datetime.now(timezone.utc).isoformat()
        (self._private_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        logger.info("AgentSoul[%s]: role upgraded → %s", self.agent_id, new_role)

    async def update_identity_trait(self, key: str, delta: float) -> None:
        """Adjust a personality trait in profile.json by delta."""
        profile_path = self._private_dir / "profile.json"
        profile = self._read_private_json("profile.json")
        current = profile.get(key, 0.5)
        new_value = max(0.0, min(1.0, current + delta))
        profile[key] = round(new_value, 2)
        profile_path.write_text(json.dumps(profile, indent=2))
        logger.debug("AgentSoul[%s]: trait '%s' adjusted %.2f → %.2f",
                     self.agent_id, key, current, new_value)

    async def record_task(self, success: bool) -> None:
        """Record task completion, update meta.json stats."""
        meta = self._read_private_json("meta.json")
        meta["total_tasks"] = meta.get("total_tasks", 0) + 1
        success_count = meta.get("success_count", 0)
        if success:
            success_count += 1
            meta["success_count"] = success_count
        total = meta["total_tasks"]
        meta["success_rate"] = round(success_count / total, 3) if total > 0 else 0.0
        (self._private_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    # ── Self-modification ────────────────────────────────────────────

    async def self_modify(self, file_type: str, new_content: str,
                         permissions: Any = None) -> bool:
        """Agent autonomously modifies its own files.

        Full audit trail:
        1. Permission check: permissions.can_modify_file(file_type)
        2. Read old content
        3. Generate diff-like summary
        4. Write audit log to JOURNAL.md
        5. Apply new content

        Returns True on success.
        """
        # Permission check
        if permissions is not None:
            if not permissions.can_modify_file(file_type):
                logger.warning(
                    "AgentSoul[%s]: self_modify rejected — insufficient permissions for '%s'",
                    self.agent_id, file_type,
                )
                return False

        # Map file_type to actual file
        file_map = {
            "identity": "IDENTITY.md",
            "role": "ROLE.md",
            "journal": "JOURNAL.md",
            "knowledge": "KNOWLEDGE.md",
            "profile": "profile.json",
        }

        filename = file_map.get(file_type)
        if not filename:
            logger.warning(
                "AgentSoul[%s]: unknown file_type '%s'",
                self.agent_id, file_type,
            )
            return False

        file_path = self._private_dir / filename

        # Read old content
        old_content = ""
        if file_path.exists():
            old_content = file_path.read_text()

        # Generate summary of change
        old_len = len(old_content)
        new_len = len(new_content)
        diff_summary = (
            f"Self-modification: {file_type} ({filename})\n"
            f"  Old size: {old_len} chars → New size: {new_len} chars\n"
            f"  Change: {'+' if new_len > old_len else '-'}{abs(new_len - old_len)} chars\n"
        )

        # Write audit log to JOURNAL
        timestamp = datetime.now(timezone.utc).isoformat()
        audit_entry = f"\n## {timestamp}\n### 🔄 Self-Modify: {file_type}\n{diff_summary}\n"
        journal_path = self._private_dir / "JOURNAL.md"
        with open(journal_path, "a") as f:
            f.write(audit_entry)

        # Apply new content
        file_path.write_text(new_content)
        logger.info(
            "AgentSoul[%s]: self-modified '%s' (%d → %d chars)",
            self.agent_id, file_type, old_len, new_len,
        )
        return True

    async def request_safety_change(self, suggestion: str) -> None:
        """Agent suggests a change to SAFETY.md (cannot directly modify).

        The suggestion is written to JOURNAL.md for human review.
        SAFETY.md has absolute priority and cannot be modified by agents.
        """
        timestamp = datetime.now(timezone.utc).isoformat()
        entry = (
            f"\n## {timestamp}\n"
            f"### ⚠️ SAFETY Change Suggestion\n"
            f"{suggestion}\n"
            f"_Note: This suggestion was recorded to JOURNAL. "
            f"SAFETY.md can only be modified by a human._\n"
        )
        journal_path = self._private_dir / "JOURNAL.md"
        with open(journal_path, "a") as f:
            f.write(entry)
        logger.info(
            "AgentSoul[%s]: recorded SAFETY change suggestion",
            self.agent_id,
        )

    def get_meta(self) -> dict:
        """Get agent metadata."""
        return self._read_private_json("meta.json")

    @property
    def identity_content(self) -> str:
        if self._identity_content is None:
            self._identity_content = self._read_private_md("IDENTITY.md")
        return self._identity_content

    @property
    def role_content(self) -> str:
        if self._role_content is None:
            self._role_content = self._read_private_md("ROLE.md")
        return self._role_content


class SoulBuilder:
    """Fluently construct an AgentSoul from cards.

    Usage:
        soul = SoulBuilder(agent_id="agent-abc")
            .with_personality("executor")
            .with_role("coder")
            .build()
    """

    def __init__(self, agent_id: str,
                 soul_root: str | Path = DEFAULT_SOUL_ROOT,
                 state_root: str | Path = DEFAULT_STATE_ROOT):
        self.agent_id = agent_id
        self._personality = "executor"
        self._role = "coder"
        self._soul_root = Path(soul_root)
        self._state_root = Path(state_root)

    def with_personality(self, name: str) -> "SoulBuilder":
        self._personality = name
        return self

    def with_role(self, name: str) -> "SoulBuilder":
        self._role = name
        return self

    def build(self) -> AgentSoul:
        return AgentSoul(
            agent_id=self.agent_id,
            personality=self._personality,
            role=self._role,
            soul_root=self._soul_root,
            state_root=self._state_root,
        )
