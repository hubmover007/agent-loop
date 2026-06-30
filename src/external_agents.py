"""External Agent Bridge — dispatch tasks to external AI agents.

Supports dispatching tasks to external coding agents via:
  1. OpenClaw ACP (Agent Client Protocol) — for Codex, Claude Code, Gemini CLI, etc.
  2. Direct CLI invocation — fallback when ACP not available
  3. OpenClaw sessions_spawn — for OpenClaw-native sub-agents

Supported external agents:
  - Codex (OpenAI) — coding, debugging, refactoring
  - Claude Code (Anthropic) — coding, analysis, agentic tasks
  - Gemini CLI (Google) — coding, multimodal
  - OpenCode — open-source coding agent
  - OpenClaw sub-agents — native OpenClaw workers
  - Hermes Agent — general-purpose

Architecture:
  AgentManagerAgent.assign(task)
    → if task requires external agent:
        ExternalAgentBridge.dispatch(task, agent_type)
    → else:
        internal WorkerPool

The bridge uses ACP as the primary protocol (via OpenClaw's sessions_spawn
with runtime="acp"), falling back to direct CLI invocation.

CC Switch is a CLI switcher (not an orchestrator) — it manages which
coding CLI is active. Agent-Loop doesn't need CC Switch itself, but
can detect which agents are installed and available.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .core import TaskStatus, TaskResult, AgentRole, AgentStatus

logger = logging.getLogger(__name__)


# ============================================================
# External Agent Types
# ============================================================

@dataclass
class ExternalAgentConfig:
    """Configuration for an external agent."""
    agent_type: str           # "codex" | "claude" | "gemini" | "opencode" | "openclaw"
    cli_name: str             # CLI binary name: "codex", "claude", "gemini"
    description: str = ""
    available: bool = False   # Whether CLI is installed
    acp_enabled: bool = False # Whether ACP is available
    max_concurrent: int = 4   # Max concurrent sessions
    default_timeout: int = 300  # Default task timeout in seconds
    capabilities: list[str] = field(default_factory=list)  # What it can do


# Known external agents and their configurations
KNOWN_AGENTS: dict[str, ExternalAgentConfig] = {
    "codex": ExternalAgentConfig(
        agent_type="codex",
        cli_name="codex",
        description="OpenAI Codex — coding, debugging, refactoring",
        capabilities=["code", "debug", "refactor", "test", "document"],
    ),
    "claude": ExternalAgentConfig(
        agent_type="claude",
        cli_name="claude",
        description="Claude Code — coding, analysis, agentic tasks",
        capabilities=["code", "debug", "analyze", "plan", "execute", "document"],
    ),
    "gemini": ExternalAgentConfig(
        agent_type="gemini",
        cli_name="gemini",
        description="Google Gemini CLI — coding, multimodal",
        capabilities=["code", "multimodal", "analyze"],
    ),
    "opencode": ExternalAgentConfig(
        agent_type="opencode",
        cli_name="opencode",
        description="OpenCode — open-source coding agent",
        capabilities=["code", "debug", "refactor"],
    ),
    "openclaw": ExternalAgentConfig(
        agent_type="openclaw",
        cli_name="openclaw",
        description="OpenClaw sub-agent — native worker",
        capabilities=["code", "exec", "web", "memory", "feishu", "task"],
    ),
    "kimi": ExternalAgentConfig(
        agent_type="kimi",
        cli_name="kimi",
        description="Kimi CLI — coding assistant",
        capabilities=["code", "analyze"],
    ),
}


# ============================================================
# Agent Availability Detection
# ============================================================

def detect_available_agents() -> dict[str, bool]:
    """Detect which external agent CLIs are installed.

    Checks PATH for each known CLI binary.
    Returns dict of agent_type → available.
    """
    available = {}
    for agent_type, config in KNOWN_AGENTS.items():
        # Check if CLI binary exists in PATH
        cli_path = shutil.which(config.cli_name)
        available[agent_type] = cli_path is not None
        config.available = cli_path is not None

        if cli_path:
            logger.debug("External agent %s available: %s", agent_type, cli_path)
        else:
            logger.debug("External agent %s not found", agent_type)

    return available


def detect_acp_available() -> bool:
    """Check if OpenClaw ACP is available.

    ACP is available when:
    1. openclaw binary is installed
    2. ACP plugin is enabled (check openclaw.json)
    """
    # Check for openclaw binary
    if not shutil.which("openclaw"):
        return False

    # Check for ACP config (simplified — real check would parse openclaw.json)
    # For now, assume ACP is available if openclaw is installed
    return True


# ============================================================
# External Agent Bridge
# ============================================================

class ExternalAgentBridge:
    """Bridge to dispatch tasks to external AI agents.

    Primary method: ACP (via OpenClaw sessions_spawn with runtime="acp")
    Fallback: Direct CLI invocation

    Usage:
        bridge = ExternalAgentBridge()

        # Check what's available
        available = bridge.list_available()

        # Dispatch a task to Codex
        result = await bridge.dispatch(
            agent_type="codex",
            task="Fix the failing tests in auth.py",
            workspace="/path/to/repo",
        )

        # Dispatch to Claude Code
        result = await bridge.dispatch(
            agent_type="claude",
            task="Refactor the database layer",
            workspace="/path/to/repo",
        )
    """

    def __init__(self, openclaw_bin: str = "openclaw"):
        self.openclaw_bin = openclaw_bin
        self._available_agents: dict[str, bool] = {}
        self._acp_available: bool = False
        self._active_sessions: dict[str, dict] = {}  # session_id → metadata

        # Detect available agents
        self._detect()

    def _detect(self) -> None:
        """Detect available agents and ACP support."""
        self._available_agents = detect_available_agents()
        self._acp_available = detect_acp_available()

        logger.info("External agents available: %s", {
            k: v for k, v in self._available_agents.items() if v
        })
        logger.info("ACP available: %s", self._acp_available)

    def list_available(self) -> dict[str, ExternalAgentConfig]:
        """List all available external agents with their configs."""
        return {
            agent_type: config
            for agent_type, config in KNOWN_AGENTS.items()
            if self._available_agents.get(agent_type, False)
        }

    def is_available(self, agent_type: str) -> bool:
        """Check if a specific agent type is available."""
        return self._available_agents.get(agent_type, False)

    def recommend_agent(self, task_scope: str,
                        required_capabilities: list[str] | None = None) -> str | None:
        """Recommend the best external agent for a task.

        Uses simple capability matching:
        1. Filter agents by required capabilities
        2. Prefer agents with more capabilities
        3. Prefer ACP-enabled agents
        """
        caps = required_capabilities or []

        candidates = []
        for agent_type, config in self.list_available().items():
            # Check if agent has all required capabilities
            if caps and not all(c in config.capabilities for c in caps):
                continue
            candidates.append((agent_type, config))

        if not candidates:
            return None

        # Sort by: ACP-enabled first, then by number of capabilities
        candidates.sort(
            key=lambda x: (self._acp_available, len(x[1].capabilities)),
            reverse=True
        )

        return candidates[0][0]

    # ============================================================
    # Dispatch (primary entry point)
    # ============================================================

    async def dispatch(self, agent_type: str, task: str,
                       workspace: str | None = None,
                       timeout: int = 300,
                       mode: str = "run") -> dict:
        """Dispatch a task to an external agent.

        Args:
            agent_type: "codex" | "claude" | "gemini" | "opencode" | "openclaw"
            task: Task description/prompt
            workspace: Working directory (default: current dir)
            timeout: Task timeout in seconds
            mode: "run" (one-shot) or "session" (persistent)

        Returns:
            Dict with: agent_type, status, output, session_id, error
        """
        if not self.is_available(agent_type):
            return {
                "agent_type": agent_type,
                "status": "failed",
                "error": f"Agent {agent_type} not available",
                "output": "",
            }

        # Primary path: ACP via OpenClaw
        if self._acp_available and agent_type != "openclaw":
            return await self._dispatch_via_acp(
                agent_type=agent_type,
                task=task,
                workspace=workspace,
                timeout=timeout,
                mode=mode,
            )

        # Fallback: direct CLI invocation
        return await self._dispatch_via_cli(
            agent_type=agent_type,
            task=task,
            workspace=workspace,
            timeout=timeout,
        )

    # ============================================================
    # ACP dispatch (via OpenClaw sessions_spawn)
    # ============================================================

    async def _dispatch_via_acp(self, agent_type: str, task: str,
                                workspace: str | None,
                                timeout: int, mode: str) -> dict:
        """Dispatch task via OpenClaw ACP (sessions_spawn runtime=acp).

        Uses `openclaw skill sessions_spawn` with:
          runtime: "acp"
          agentId: agent_type
          task: task description
          mode: "run" (one-shot) or "session" (persistent)
        """
        import uuid

        session_id = f"acp:{agent_type}:{uuid.uuid4().hex[:8]}"

        # Build openclaw command
        # openclaw sessions spawn --runtime acp --agent <agent_type> --mode <mode> --task <task>
        cmd = [
            self.openclaw_bin,
            "sessions", "spawn",
            "--runtime", "acp",
            "--agent", agent_type,
            "--mode", mode,
            "--task", task,
        ]

        if workspace:
            cmd.extend(["--cwd", workspace])

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            output = stdout.decode("utf-8", errors="replace").strip()
            error = stderr.decode("utf-8", errors="replace").strip()

            result = {
                "agent_type": agent_type,
                "dispatch_method": "acp",
                "session_id": session_id,
                "status": "done" if proc.returncode == 0 else "failed",
                "output": output,
                "error": error if proc.returncode != 0 else "",
                "exit_code": proc.returncode,
            }

            self._active_sessions[session_id] = {
                "agent_type": agent_type,
                "task": task,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "status": result["status"],
            }

            logger.info("ACP dispatch %s → %s (status=%s)",
                       session_id, agent_type, result["status"])
            return result

        except asyncio.TimeoutError:
            return {
                "agent_type": agent_type,
                "dispatch_method": "acp",
                "status": "timeout",
                "error": f"ACP dispatch timed out after {timeout}s",
                "output": "",
            }
        except Exception as e:
            return {
                "agent_type": agent_type,
                "dispatch_method": "acp",
                "status": "failed",
                "error": f"ACP error: {e}",
                "output": "",
            }

    # ============================================================
    # CLI dispatch (fallback)
    # ============================================================

    async def _dispatch_via_cli(self, agent_type: str, task: str,
                                workspace: str | None,
                                timeout: int) -> dict:
        """Dispatch task via direct CLI invocation.

        Each agent CLI has its own invocation pattern:
          - codex: `codex --task "..."` or `codex` + stdin
          - claude: `claude --print "..."` or `claude` + stdin
          - gemini: `gemini --prompt "..."`
          - opencode: `opencode --task "..."`
        """
        config = KNOWN_AGENTS.get(agent_type)
        if not config:
            return {"agent_type": agent_type, "status": "failed",
                    "error": "Unknown agent type", "output": ""}

        # Build CLI command based on agent type
        cmd = self._build_cli_command(agent_type, config.cli_name, task)

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workspace,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            output = stdout.decode("utf-8", errors="replace").strip()
            error = stderr.decode("utf-8", errors="replace").strip()

            return {
                "agent_type": agent_type,
                "dispatch_method": "cli",
                "status": "done" if proc.returncode == 0 else "failed",
                "output": output,
                "error": error if proc.returncode != 0 else "",
                "exit_code": proc.returncode,
            }

        except asyncio.TimeoutError:
            return {
                "agent_type": agent_type,
                "dispatch_method": "cli",
                "status": "timeout",
                "error": f"CLI timed out after {timeout}s",
                "output": "",
            }
        except Exception as e:
            return {
                "agent_type": agent_type,
                "dispatch_method": "cli",
                "status": "failed",
                "error": f"CLI error: {e}",
                "output": "",
            }

    @staticmethod
    def _build_cli_command(agent_type: str, cli_name: str, task: str) -> list[str]:
        """Build CLI command for each agent type.

        Each external agent has a different CLI interface:
        """
        if agent_type == "codex":
            # Codex CLI: codex --task "..."
            return [cli_name, "--task", task]

        elif agent_type == "claude":
            # Claude Code: claude --print "..."
            return [cli_name, "--print", task]

        elif agent_type == "gemini":
            # Gemini CLI: gemini --prompt "..."
            return [cli_name, "--prompt", task]

        elif agent_type == "opencode":
            # OpenCode: opencode --task "..."
            return [cli_name, "--task", task]

        elif agent_type == "openclaw":
            # OpenClaw: openclaw chat --message "..."
            return [cli_name, "chat", "--message", task]

        else:
            # Generic fallback: pipe task via stdin
            return [cli_name]

    # ============================================================
    # Session management
    # ============================================================

    def list_active_sessions(self) -> dict[str, dict]:
        """List all active external agent sessions."""
        return dict(self._active_sessions)

    async def cancel_session(self, session_id: str) -> bool:
        """Cancel an active external agent session."""
        if session_id not in self._active_sessions:
            return False

        session = self._active_sessions[session_id]
        agent_type = session.get("agent_type", "")

        # For ACP sessions, use openclaw to cancel
        if session_id.startswith("acp:") and shutil.which("openclaw"):
            try:
                proc = await asyncio.create_subprocess_exec(
                    self.openclaw_bin, "sessions", "kill",
                    "--session-id", session_id,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                await proc.communicate()
            except Exception:
                pass

        self._active_sessions.pop(session_id, None)
        logger.info("Cancelled external session %s", session_id)
        return True

    # ============================================================
    # Stats
    # ============================================================

    def stats(self) -> dict:
        """Get bridge statistics."""
        return {
            "available_agents": [k for k, v in self._available_agents.items() if v],
            "acp_available": self._acp_available,
            "active_sessions": len(self._active_sessions),
            "known_agents": list(KNOWN_AGENTS.keys()),
        }
