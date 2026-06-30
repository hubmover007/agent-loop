"""Structured Evolution Engine — upgrades the EVOLVE phase from raw string
appends to structured knowledge extraction + automatic trait adjustment.

Workflow:
  1. JournalEntry — structured log of each task execution
  2. KnowledgeNugget — extracted pattern from multiple JournalEntry records
  3. EvolutionEngine — orchestrates recording, extraction, adjustment, promotion
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Data Classes ────────────────────────────────────────────────────

@dataclass
class JournalEntry:
    """Structured log entry for a single task execution.

    Replaces raw string appends to JOURNAL.md with machine-readable JSONL.
    """
    id: str
    timestamp: str           # ISO 8601
    task_scope: str
    task_type: str           # "coding" | "reasoning" | "ops" | "general"
    outcome: str             # "success" | "failure" | "partial"
    score: float             # 0.0 – 1.0
    duration_seconds: float
    tools_used: list[str]
    llm_provider: str
    cost_estimate: float
    lessons: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "task_scope": self.task_scope,
            "task_type": self.task_type,
            "outcome": self.outcome,
            "score": self.score,
            "duration_seconds": self.duration_seconds,
            "tools_used": self.tools_used,
            "llm_provider": self.llm_provider,
            "cost_estimate": self.cost_estimate,
            "lessons": self.lessons,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "JournalEntry":
        return cls(**d)


@dataclass
class KnowledgeNugget:
    """Structured knowledge extracted from multiple JournalEntry records.

    Represents a learned pattern like "当遇到 X 时，应该 Y".
    Confidence increases with more supporting evidence.
    """

    id: str
    pattern: str            # "当遇到 X 时，应该 Y"
    confidence: float       # 0.0 – 1.0
    evidence_count: int     # number of supporting journal entries
    source_entries: list[str]  # associated JournalEntry.id values
    created_at: str
    last_reinforced: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pattern": self.pattern,
            "confidence": self.confidence,
            "evidence_count": self.evidence_count,
            "source_entries": self.source_entries,
            "created_at": self.created_at,
            "last_reinforced": self.last_reinforced,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeNugget":
        return cls(**d)


# ── Evolution Engine ────────────────────────────────────────────────

class EvolutionEngine:
    """Structured evolution engine.

    Responsibilities:
      1. Record JournalEntry as JSONL
      2. Extract KnowledgeNuggets from multiple entries
      3. Adjust agent profile.json traits based on outcomes
      4. Promote high-confidence knowledge to IDENTITY.md
    """

    ROLE_UPGRADE_MAP: dict[str, str] = {
        "coder": "senior_coder",
        "researcher": "senior_researcher",
        "ops": "devops",
    }

    def __init__(self, agent_id: str, state_dir: str = "state/agents"):
        self.agent_id = agent_id
        self.journal_path = Path(state_dir) / agent_id / "JOURNAL.jsonl"
        self.knowledge_path = Path(state_dir) / agent_id / "KNOWLEDGE.json"
        self.profile_path = Path(state_dir) / agent_id / "profile.json"
        self.identity_path = Path(state_dir) / agent_id / "IDENTITY.md"

        # Ensure parent directory exists
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)

        # Lazy-loaded caches
        self._entries: list[JournalEntry] | None = None
        self._nuggets: list[KnowledgeNugget] | None = None

    # ── Recording ────────────────────────────────────────────────

    async def record_entry(self, entry: JournalEntry) -> None:
        """Append a structured JournalEntry as one JSON line to JOURNAL.jsonl."""
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(entry.to_dict(), ensure_ascii=False)
        with open(self.journal_path, "a") as f:
            f.write(line + "\n")
        # Invalidate cache
        self._entries = None
        logger.debug("EvolutionEngine[%s]: recorded entry %s", self.agent_id, entry.id)

    # ── Knowledge Extraction ─────────────────────────────────────

    def _load_entries(self) -> list[JournalEntry]:
        """Load all journal entries from JSONL."""
        if self._entries is not None:
            return self._entries
        entries: list[JournalEntry] = []
        if self.journal_path.exists():
            for line in self.journal_path.read_text().strip().split("\n"):
                if line.strip():
                    try:
                        entries.append(JournalEntry.from_dict(json.loads(line)))
                    except (json.JSONDecodeError, TypeError) as e:
                        logger.warning("EvolutionEngine[%s]: corrupt journal line: %s", self.agent_id, e)
        self._entries = entries
        return entries

    def _load_nuggets(self) -> list[KnowledgeNugget]:
        """Load existing knowledge nuggets from JSON."""
        if self._nuggets is not None:
            return self._nuggets
        nuggets: list[KnowledgeNugget] = []
        if self.knowledge_path.exists():
            try:
                data = json.loads(self.knowledge_path.read_text())
                for d in data:
                    nuggets.append(KnowledgeNugget.from_dict(d))
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning("EvolutionEngine[%s]: corrupt KNOWLEDGE.json: %s", self.agent_id, e)
        self._nuggets = nuggets
        return nuggets

    async def extract_knowledge(self, min_evidence: int = 3) -> list[KnowledgeNugget]:
        """Extract knowledge nuggets from journal entries.

        Rules:
          - Same task_type + "success" ≥ min_evidence → "擅长 {task_type}" nugget
          - Same task_type + "failure" ≥ 2 → "不擅长 {task_type}" nugget
          - Same tool appearing in failures repeatedly → "工具 {tool} 需要检查"
        """
        entries = self._load_entries()
        if len(entries) < min_evidence:
            return self._load_nuggets()

        now = datetime.now(timezone.utc).isoformat()
        new_nuggets: list[KnowledgeNugget] = []

        # Group by task_type
        by_type: dict[str, list[JournalEntry]] = {}
        for e in entries:
            by_type.setdefault(e.task_type, []).append(e)

        for task_type, group in by_type.items():
            # ── Success pattern ──
            successes = [e for e in group if e.outcome == "success"]
            if len(successes) >= min_evidence:
                pattern = f"擅长处理 {task_type} 类型任务"
                # Check if this nugget already exists
                existing = [n for n in self._load_nuggets() if n.pattern == pattern]
                if existing:
                    existing[0].evidence_count = len(successes)
                    existing[0].confidence = min(1.0, 0.5 + len(successes) * 0.1)
                    existing[0].last_reinforced = now
                    existing[0].source_entries = [e.id for e in successes]
                else:
                    new_nuggets.append(KnowledgeNugget(
                        id=f"nugget:{uuid.uuid4().hex[:8]}",
                        pattern=pattern,
                        confidence=min(1.0, 0.5 + len(successes) * 0.1),
                        evidence_count=len(successes),
                        source_entries=[e.id for e in successes],
                        created_at=now,
                        last_reinforced=now,
                    ))

            # ── Failure pattern ──
            failures = [e for e in group if e.outcome == "failure"]
            if len(failures) >= 2:
                pattern = f"不擅长处理 {task_type} 类型任务"
                existing = [n for n in self._load_nuggets() if n.pattern == pattern]
                if existing:
                    existing[0].evidence_count = len(failures)
                    existing[0].confidence = min(1.0, 0.5 + len(failures) * 0.15)
                    existing[0].last_reinforced = now
                    existing[0].source_entries = [e.id for e in failures]
                else:
                    new_nuggets.append(KnowledgeNugget(
                        id=f"nugget:{uuid.uuid4().hex[:8]}",
                        pattern=pattern,
                        confidence=min(1.0, 0.5 + len(failures) * 0.15),
                        evidence_count=len(failures),
                        source_entries=[e.id for e in failures],
                        created_at=now,
                        last_reinforced=now,
                    ))

        # ── Tool-specific failure pattern ──
        tool_fails: dict[str, list[JournalEntry]] = {}
        for e in entries:
            if e.outcome == "failure":
                for tool in e.tools_used:
                    tool_fails.setdefault(tool, []).append(e)

        for tool, group in tool_fails.items():
            if len(group) >= min_evidence:
                pattern = f"工具 '{tool}' 频繁失败，需要检查参数或环境"
                existing = [n for n in self._load_nuggets() if n.pattern == pattern]
                if existing:
                    existing[0].evidence_count = len(group)
                    existing[0].confidence = min(1.0, 0.5 + len(group) * 0.1)
                    existing[0].last_reinforced = now
                    existing[0].source_entries = [e.id for e in group]
                else:
                    new_nuggets.append(KnowledgeNugget(
                        id=f"nugget:{uuid.uuid4().hex[:8]}",
                        pattern=pattern,
                        confidence=min(1.0, 0.5 + len(group) * 0.1),
                        evidence_count=len(group),
                        source_entries=[e.id for e in group],
                        created_at=now,
                        last_reinforced=now,
                    ))

        # Merge new nuggets into existing, save
        all_nuggets = self._load_nuggets() + new_nuggets
        # Deduplicate by pattern (keep highest confidence)
        seen: dict[str, KnowledgeNugget] = {}
        for n in all_nuggets:
            if n.pattern not in seen or n.confidence > seen[n.pattern].confidence:
                seen[n.pattern] = n
        deduped = list(seen.values())

        self._nuggets = deduped
        self.knowledge_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge_path.write_text(
            json.dumps([n.to_dict() for n in deduped], ensure_ascii=False, indent=2)
        )

        logger.info("EvolutionEngine[%s]: extracted %d nuggets (total: %d)",
                     self.agent_id, len(new_nuggets), len(deduped))
        return deduped

    # ── Trait Adjustment ─────────────────────────────────────────

    async def adjust_traits(self, entry: JournalEntry) -> None:
        """Adjust profile.json traits based on the outcome of this entry.

        Success → efficiency +0.02, confidence +0.01
        Failure → efficiency -0.03, caution +0.02
        """
        profile = {}
        if self.profile_path.exists():
            try:
                profile = json.loads(self.profile_path.read_text())
            except json.JSONDecodeError:
                pass

        if entry.outcome == "success":
            profile["efficiency"] = min(1.0, round(profile.get("efficiency", 0.5) + 0.02, 2))
            # "confidence" is mapped to assertiveness in the profile schema
            profile["assertiveness"] = min(1.0, round(profile.get("assertiveness", 0.5) + 0.01, 2))
            logger.debug("EvolutionEngine[%s]: success → efficiency +0.02, assertiveness +0.01",
                         self.agent_id)
        elif entry.outcome == "failure":
            profile["efficiency"] = max(0.0, round(profile.get("efficiency", 0.5) - 0.03, 2))
            profile["cautiousness"] = min(1.0, round(profile.get("cautiousness", 0.5) + 0.02, 2))
            logger.debug("EvolutionEngine[%s]: failure → efficiency -0.03, cautiousness +0.02",
                         self.agent_id)

        # ── Role upgrade on consecutive successes ──
        entries = self._load_entries()
        recent = [e for e in entries if e.task_type == entry.task_type]
        consecutive_successes = 0
        for e in reversed(recent):
            if e.outcome == "success":
                consecutive_successes += 1
            else:
                break

        if consecutive_successes >= 3 and entry.task_type in self.ROLE_UPGRADE_MAP:
            # Check current role (from ROLE.md or profile.json)
            current_role = profile.get("role", entry.task_type)
            upgraded = self.ROLE_UPGRADE_MAP.get(current_role, current_role)
            profile["role"] = upgraded
            profile["curiosity"] = min(1.0, round(profile.get("curiosity", 0.5) + 0.05, 2))
            logger.info("EvolutionEngine[%s]: role upgraded %s → %s (consecutive: %d)",
                        self.agent_id, current_role, upgraded, consecutive_successes)

        # ── Confidence penalty on consecutive failures ──
        consecutive_failures = 0
        for e in reversed(recent):
            if e.outcome == "failure":
                consecutive_failures += 1
            else:
                break

        if consecutive_failures >= 3:
            profile["assertiveness"] = max(0.0, round(profile.get("assertiveness", 0.5) - 0.1, 2))
            logger.info("EvolutionEngine[%s]: confidence -0.10 (consecutive failures: %d)",
                        self.agent_id, consecutive_failures)

        self.profile_path.parent.mkdir(parents=True, exist_ok=True)
        self.profile_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2))

    # ── Knowledge Promotion ──────────────────────────────────────

    async def promote_to_identity(self, nugget: KnowledgeNugget) -> bool:
        """Promote high-confidence knowledge to IDENTITY.md.

        Trigger: confidence > 0.8 AND evidence_count > 5

        Returns True if promotion happened.
        """
        if nugget.confidence <= 0.8 or nugget.evidence_count <= 5:
            return False

        self.identity_path.parent.mkdir(parents=True, exist_ok=True)

        existing = ""
        if self.identity_path.exists():
            existing = self.identity_path.read_text()

        # Append or update "## 经验教训" section
        section_header = "## 经验教训"
        new_line = f"- [{nugget.evidence_count}次验证] {nugget.pattern}"

        if section_header in existing:
            # Update existing section — append if not already present
            if nugget.pattern not in existing:
                idx = existing.index(section_header) + len(section_header)
                after_header = existing[idx:]
                # Insert after section header, before next section
                next_section = after_header.find("\n## ")
                if next_section != -1:
                    existing = (
                        existing[:idx + next_section]
                        + f"\n{new_line}"
                        + existing[idx + next_section:]
                    )
                else:
                    existing = existing + f"\n{new_line}"
        else:
            existing = existing.rstrip() + f"\n\n{section_header}\n{new_line}\n"

        self.identity_path.write_text(existing)
        logger.info("EvolutionEngine[%s]: promoted knowledge to IDENTITY.md: %s",
                     self.agent_id, nugget.pattern[:60])
        return True

    # ── Statistics ───────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return evolution statistics for this agent."""
        entries = self._load_entries()
        nuggets = self._load_nuggets()

        total = len(entries)
        if total == 0:
            return {
                "total_tasks": 0,
                "success_rate": 0.0,
                "average_score": 0.0,
                "most_used_tool": None,
                "strongest_domain": None,
                "total_nuggets": 0,
                "total_duration_s": 0.0,
            }

        successes = sum(1 for e in entries if e.outcome == "success")
        avg_score = sum(e.score for e in entries) / total

        # Most used tool
        from collections import Counter
        tool_counter = Counter()
        for e in entries:
            tool_counter.update(e.tools_used)
        most_used_tool = tool_counter.most_common(1)[0] if tool_counter else None

        # Strongest domain (by success rate)
        domain_stats: dict[str, tuple[int, int]] = {}  # task_type → (successes, total)
        for e in entries:
            prev = domain_stats.get(e.task_type, (0, 0))
            domain_stats[e.task_type] = (
                prev[0] + (1 if e.outcome == "success" else 0),
                prev[1] + 1,
            )

        strongest_domain = None
        best_rate = 0.0
        for domain, (s, t) in domain_stats.items():
            if t >= 2:  # require at least 2 tasks
                rate = s / t
                if rate > best_rate:
                    best_rate = rate
                    strongest_domain = domain

        total_duration = sum(e.duration_seconds for e in entries)

        return {
            "total_tasks": total,
            "success_rate": round(successes / total, 3),
            "average_score": round(avg_score, 2),
            "most_used_tool": most_used_tool[0] if most_used_tool else None,
            "strongest_domain": strongest_domain,
            "total_nuggets": len(nuggets),
            "total_duration_s": round(total_duration, 1),
        }
