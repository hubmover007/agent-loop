"""Unified memory record schema.

All memory types (fact/episode/project) share common fields
for consistent LLM interaction. This module defines:
  - MemoryType: success/failure/lesson/fact/procedure/pattern
  - MemoryRecord: dataclass with unified fields
  - render_for_llm(): compact text rendering for LLM prompts
  - parse_llm_extraction(): parse LLM output back to MemoryRecord
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MemoryType(Enum):
    SUCCESS = "success"
    FAILURE = "failure"
    LESSON = "lesson"
    FACT = "fact"
    PROCEDURE = "procedure"
    PATTERN = "pattern"


@dataclass
class MemoryRecord:
    """Unified memory record. All types share this schema.

    Required fields (3): id, type, summary
    Everything else is optional. This keeps it flexible but structured.
    """

    # ── Required ──
    id: str = ""
    type: MemoryType = MemoryType.FACT
    summary: str = ""  # ≤100 chars, the core info LLM sees first

    # ── Core optional (high frequency) ──
    project: str = ""
    trigger: str = ""       # trigger condition / applicable scenario
    action: str = ""        # what was done
    outcome: str = ""       # result description
    lesson: str = ""        # lesson learned
    confidence: float = 0.5  # 0-1, LLM-assessed confidence

    # ── Relations (graph edges) ──
    related_ids: list[str] = field(default_factory=list)
    related_type: str = ""  # similar|contrast|cause|followup

    # ── Context (on demand) ──
    context: dict[str, Any] = field(default_factory=dict)
    steps: list[str] = field(default_factory=list)
    error: str = ""         # failure reason (failure only)
    fix: str = ""           # correct approach (failure → next time)

    # ── Metadata (auto) ──
    agent_id: str = "shared"
    created_at: str = ""    # ISO date
    updated_at: str = ""
    access_count: int = 0

    # ── Extension ──
    tags: list[str] = field(default_factory=list)

    def to_compact_text(self) -> str:
        """Render as compact text block for LLM consumption.

        Format:
        [TYPE|project|date|conf:x.x]
        summary: one line
        trigger: ...
        action: ...
        outcome: ...
        lesson: ...
        error: ...     (failure only)
        fix: ...       (failure only)
        tags: [t1, t2]

        Empty fields are omitted to save tokens.
        """
        date_str = self.created_at[:10] if self.created_at else ""
        lines = [f"[{self.type.value.upper()}|{self.project}|{date_str}|conf:{self.confidence:.1f}]"]

        if self.summary:
            lines.append(f"summary: {self.summary}")
        if self.trigger:
            lines.append(f"trigger: {self.trigger}")
        if self.action:
            lines.append(f"action: {self.action}")
        if self.outcome:
            lines.append(f"outcome: {self.outcome}")
        if self.lesson:
            lines.append(f"lesson: {self.lesson}")
        if self.error:
            lines.append(f"error: {self.error}")
        if self.fix:
            lines.append(f"fix: {self.fix}")
        if self.steps:
            lines.append(f"steps: {', '.join(self.steps)}")
        if self.related_ids:
            lines.append(f"related: {self.related_ids}")
        if self.tags:
            lines.append(f"tags: [{', '.join(self.tags)}]")

        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        """Create from dict (e.g., from SurrealDB row).

        Compatible with old-style records where summary may come from
        ``title`` (episode) or ``name`` (fact).
        """
        type_str = data.get("type", "fact")
        try:
            mem_type = MemoryType(type_str)
        except ValueError:
            mem_type = MemoryType.FACT

        return cls(
            id=str(data.get("id", "")),
            type=mem_type,
            summary=data.get("summary", data.get("title", data.get("name", ""))),
            project=data.get("project", ""),
            trigger=data.get("trigger", ""),
            action=data.get("action", ""),
            outcome=data.get("outcome", ""),
            lesson=data.get("lesson", ""),
            confidence=float(data.get("confidence", 0.5)),
            related_ids=data.get("related_ids", []),
            related_type=data.get("related_type", ""),
            context=data.get("context", {}),
            steps=data.get("steps", []),
            error=data.get("error", ""),
            fix=data.get("fix", ""),
            agent_id=data.get("agent_id", "shared"),
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
            access_count=int(data.get("access_count", 0)),
            tags=data.get("tags", []),
        )


def render_for_llm(records: list[MemoryRecord], max_tokens: int = 2000) -> str:
    """Render multiple records as LLM-friendly context block.

    Joins records with '---' separator. Truncates at max_tokens.
    """
    blocks = []
    token_est = 0
    for r in records:
        block = r.to_compact_text()
        # Rough token estimate: 1 token ≈ 4 chars
        token_est += len(block) // 4
        if token_est > max_tokens:
            break
        blocks.append(block)

    return "\n---\n".join(blocks) if blocks else ""


def parse_llm_extraction(text: str) -> MemoryRecord | None:
    """Parse LLM extraction output into MemoryRecord.

    Expected format::

        [TYPE|project|date|conf:x.x]
        summary: ...
        trigger: ...
        ...

    Returns None if the text can't be parsed.
    """
    if not text.strip():
        return None

    lines = text.strip().split("\n")

    # Parse header line
    header = lines[0]
    if not header.startswith("["):
        return None

    # Extract type from header: [TYPE|project|date|conf:x.x]
    header_inner = header.strip("[]")
    parts = header_inner.split("|")
    if len(parts) < 4:
        return None

    type_str = parts[0].lower()
    try:
        mem_type = MemoryType(type_str)
    except ValueError:
        mem_type = MemoryType.FACT

    project = parts[1] if len(parts) > 1 else ""
    date_str = parts[2] if len(parts) > 2 else ""

    # Parse confidence
    confidence = 0.5
    if len(parts) > 3 and parts[3].startswith("conf:"):
        try:
            confidence = float(parts[3][5:])
        except ValueError:
            pass

    # Parse key-value lines
    fields: dict[str, Any] = {}
    for line in lines[1:]:
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()

            if key == "tags":
                # Parse [t1, t2] format
                value_clean = value.strip("[]")
                fields["tags"] = [t.strip() for t in value_clean.split(",") if t.strip()]
            elif key == "related":
                # Parse list format
                value_clean = value.strip("[]")
                fields["related_ids"] = [r.strip() for r in value_clean.split(",") if r.strip()]
            elif key == "steps":
                # Parse list format (with or without brackets)
                value_clean = value.strip("[]")
                fields["steps"] = [s.strip() for s in value_clean.split(",") if s.strip()]
            else:
                fields[key] = value

    return MemoryRecord(
        type=mem_type,
        project=project,
        created_at=date_str,
        confidence=confidence,
        summary=fields.get("summary", ""),
        trigger=fields.get("trigger", ""),
        action=fields.get("action", ""),
        outcome=fields.get("outcome", ""),
        lesson=fields.get("lesson", ""),
        error=fields.get("error", ""),
        fix=fields.get("fix", ""),
        tags=fields.get("tags", []),
        related_ids=fields.get("related_ids", []),
        steps=fields.get("steps", []),
    )


# ── LLM Prompt Templates ──

EXTRACTION_PROMPT = """\
Extract memory records from the following episode. Use this exact format:

[TYPE|project|date|conf:x.x]
summary: one-line summary (≤100 chars)
trigger: trigger condition
action: what was done
outcome: result
lesson: lesson learned (required for failure)
error: failure reason (failure only)
fix: correct approach for next time (failure only)
tags: [tag1, tag2]

Rules:
- TYPE: success | failure | lesson | fact | procedure
- success and failure use the SAME format, only type differs
- failure MUST include error + fix
- summary ≤ 100 characters
- Omit empty fields (don't write "trigger: " with nothing after)
- For success, record what worked and why
- For failure, record what failed, why, and the correct approach

Episode:
{episode_content}

Extracted records:
"""

CONSOLIDATION_PROMPT = """\
You are a memory consolidation engine. Review the following episodes and extract
structured memory records. For each significant event, create a record using the
format below.

Guidelines:
1. SUCCESS: record what worked, the approach, and why it succeeded
2. FAILURE: record what failed, the root cause, and the correct approach
3. LESSON: extract general principles from multiple episodes
4. Link related records: if a failure was later resolved by a success, note the
   relationship
5. Resolve contradictions: if new info contradicts old info, mark confidence
   accordingly

Episodes to consolidate:
{episodes_text}

Existing memories for contradiction check:
{existing_memories_text}

Extracted records (one per block, separated by ---):
"""
