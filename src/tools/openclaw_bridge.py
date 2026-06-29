"""OpenClaw Skill Bridge — integrate OpenClaw skills as Agent-Loop tools.

OpenClaw has a rich skill system (42+ skills) that can be leveraged
by Agent-Loop agents. This module bridges OpenClaw skills into the
Agent-Loop tool system, allowing agents to invoke skills like:

  - feishu-doc: create/read/update Feishu documents
  - feishu-sheet: create/read Feishu spreadsheets
  - feishu-task: manage Feishu tasks
  - feishu-calendar: manage calendar events
  - lark-im: send/search Feishu messages
  - web_search: Brave web search
  - web_fetch: fetch web content
  - memory_search: search OpenClaw memory
  - exec: run shell commands
  - tts: text-to-speech
  - image_generate: generate images
  - etc.

The bridge dynamically wraps OpenClaw skills as Agent-Loop ToolInterface
instances, so they can be registered in ToolRegistry and used by any
agent through the normal tool execution pipeline.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..core import ToolResult, ToolResultStatus
from .base import ToolInterface

logger = logging.getLogger(__name__)


class OpenClawSkillTool(ToolInterface):
    """Wrap an OpenClaw skill as an Agent-Loop tool.

    OpenClaw skills are invoked via `openclaw skill <name> <args>`
    or through the OpenClaw API. This tool provides a unified interface
    for agents to call any OpenClaw skill.
    """

    def __init__(self, skill_name: str, skill_description: str = "",
                 openclaw_bin: str = "openclaw"):
        self._name = f"oc_{skill_name.replace('-', '_')}"
        self._description = skill_description or f"OpenClaw skill: {skill_name}"
        self.skill_name = skill_name
        self.openclaw_bin = openclaw_bin

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    async def execute(self, **kwargs) -> ToolResult:
        """Execute the OpenClaw skill.

        Args:
            action: skill action (e.g. "send", "search", "create")
            **kwargs: skill-specific parameters
        """
        action = kwargs.pop("action", "")
        args = kwargs

        try:
            # Build command
            cmd_parts = [self.openclaw_bin, "skill", self.skill_name]
            if action:
                cmd_parts.append(action)

            # Pass args as JSON via stdin for complex parameters
            args_json = json.dumps(args)

            proc = await asyncio.create_subprocess_exec(
                *cmd_parts,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(args_json.encode()),
                timeout=120.0,
            )

            output = stdout.decode("utf-8", errors="replace").strip()
            error = stderr.decode("utf-8", errors="replace").strip()

            if proc.returncode == 0:
                # Try to parse as JSON, fallback to raw text
                try:
                    data = json.loads(output)
                except json.JSONDecodeError:
                    data = {"output": output}

                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    data=data,
                )
            else:
                return ToolResult(
                    status=ToolResultStatus.TRANSIENT_ERROR,
                    data={"output": output},
                    error=error or f"Exit code {proc.returncode}",
                )

        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolResultStatus.TRANSIENT_ERROR,
                error="OpenClaw skill timed out after 120s"
            )
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=f"OpenClaw skill error: {e}"
            )


# ============================================================
# Skill Registry — known OpenClaw skills
# ============================================================

KNOWN_SKILLS = {
    # Feishu / Lark skills
    "feishu-doc": "Create, read, update Feishu cloud documents",
    "feishu-sheet": "Create and operate Feishu spreadsheets",
    "feishu-task": "Manage Feishu tasks and task lists",
    "feishu-calendar": "Manage Feishu calendar events",
    "feishu-wiki": "Navigate Feishu knowledge base",
    "feishu-drive": "Manage Feishu cloud storage files",
    "lark-im": "Send and search Feishu IM messages",
    "lark-mail": "Compose, send, read emails via Feishu",
    "lark-vc": "Query Feishu video conference records",
    "lark-contact": "Search Feishu contacts and org structure",

    # Web and search
    "web_search": "Search the web via Brave Search API",
    "web_fetch": "Fetch and extract content from URLs",

    # System and ops
    "memory_search": "Search OpenClaw memory (MEMORY.md + memory/*.md)",
    "exec": "Execute shell commands",
    "tts": "Convert text to speech",
    "image_generate": "Generate images with AI",
    "video_generate": "Generate videos with AI",
    "music_generate": "Generate music with AI",

    # File and document
    "officecli": "Create and modify Office documents (.docx, .xlsx, .pptx)",

    # Development
    "dvcode": "Code development and file generation",
    "api-dev": "Scaffold and test REST/GraphQL APIs",
    "database": "Database management and queries",
}


def register_openclaw_skills(registry: ToolRegistry,
                             skill_names: list[str] | None = None,
                             openclaw_bin: str = "openclaw") -> int:
    """Register OpenClaw skills as Agent-Loop tools.

    Args:
        registry: The ToolRegistry to register into
        skill_names: List of skill names to register (default: all known)
        openclaw_bin: Path to the openclaw binary

    Returns:
        Number of skills registered
    """
    skills = skill_names or list(KNOWN_SKILLS.keys())
    count = 0

    for skill_name in skills:
        if skill_name in KNOWN_SKILLS:
            tool = OpenClawSkillTool(
                skill_name=skill_name,
                skill_description=KNOWN_SKILLS[skill_name],
                openclaw_bin=openclaw_bin,
            )
            registry.register(tool)
            count += 1

    logger.info("Registered %d OpenClaw skills as tools", count)
    return count
