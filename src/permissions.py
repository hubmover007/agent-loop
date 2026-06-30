"""Permission system — capability-based + resource-scoped access control.

Design (Plan C):
  - Role templates define baseline permissions (trust_level + resource scopes)
  - Agent evolution can request elevation (requires human approval)
  - High-risk operations always require confirmation

Usage:
    checker = PermissionChecker("config/permissions.json")
    perms = checker.get_permissions("agent-1", "coder")

    if perms.can_read("state/agents/agent-1/IDENTITY.md"):
        ...

    if perms.can_execute("git status"):
        ...

    if perms.can_modify_file("identity"):
        ...
"""

from __future__ import annotations

import fnmatch
import json
import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AgentPermissions:
    """Agent permission configuration loaded from a role template.

    Provides resource-scoped checks for filesystem, network, shell,
    agent operations, and self-modification.
    """

    def __init__(
        self,
        template_name: str,
        agent_id: str,
        config_path: str | Path = "config/permissions.json",
    ):
        self._template_name = template_name
        self._agent_id = agent_id
        self._config_path = Path(config_path)
        self._config: dict[str, Any] = {}
        self._template: dict[str, Any] = {}
        self._trust_level: str = "untrusted"
        self._fs_read: list[str] = []
        self._fs_write: list[str] = []
        self._fs_blocked: list[str] = []
        self._net_allowed: list[str] = []
        self._net_blocked: list[str] = []
        self._shell_allowed: bool = False
        self._shell_allowed_cmds: list[str] = []
        self._shell_blocked_cmds: list[str] = []
        self._shell_timeout: int = 10
        self._agent_ops: dict[str, bool] = {}
        self._self_mod: dict[str, bool] = {}
        self._elevation_cfg: dict[str, Any] = {}
        self._call_times: list[float] = []  # Rate limit tracking
        self._load()

    def reload(self) -> None:
        """Reload the configuration from disk."""
        self._load()

    def _load(self) -> None:
        """Load permission config from JSON and resolve template."""
        if not self._config_path.exists():
            logger.warning(
                "Permissions config not found: %s, using empty defaults",
                self._config_path,
            )
            return

        with open(self._config_path) as f:
            self._config = json.load(f)

        self._template = (
            self._config.get("templates", {}).get(self._template_name, {})
        )
        self._elevation_cfg = self._config.get("elevation", {})

        if not self._template:
            logger.warning(
                "Permission template '%s' not found, using empty defaults",
                self._template_name,
            )
            return

        self._trust_level = self._template.get("trust_level", "untrusted")

        # Filesystem
        fs = self._template.get("filesystem", {})
        self._fs_read = self._resolve_paths(fs.get("read_paths", []))
        self._fs_write = self._resolve_paths(fs.get("write_paths", []))
        self._fs_blocked = self._resolve_paths(fs.get("blocked_paths", []))

        # Network
        net = self._template.get("network", {})
        self._net_allowed = net.get("allowed_hosts", [])
        self._net_blocked = net.get("blocked_hosts", [])

        # Shell
        shell = self._template.get("shell", {})
        self._shell_allowed = shell.get("allowed", False)
        self._shell_allowed_cmds = shell.get("allowed_commands", [])
        self._shell_blocked_cmds = shell.get("blocked_commands", [])
        self._shell_timeout = shell.get("timeout_seconds", 10)

        # Agent ops
        self._agent_ops = self._template.get("agent_ops", {})

        # Self modification
        self._self_mod = self._template.get("self_modification", {})

    def _resolve_paths(self, paths: list[str]) -> list[str]:
        """Resolve {agent_id} placeholder in path patterns."""
        resolved = []
        for p in paths:
            resolved.append(p.replace("{agent_id}", self._agent_id))
        return resolved

    @property
    def trust_level(self) -> str:
        return self._trust_level

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def template_name(self) -> str:
        return self._template_name

    # ── Filesystem ──────────────────────────────────────────────────

    def _match_path(self, path: str, patterns: list[str]) -> bool:
        """Check if path matches any fnmatch pattern.

        Normalizes path to relative and handles glob-style patterns.
        """
        import os
        normalized = os.path.normpath(str(path))
        for pattern in patterns:
            if fnmatch.fnmatch(normalized, pattern):
                return True
        return False

    def can_read(self, path: str) -> bool:
        """Check if agent can read a file path."""
        # Blocked paths always denied
        if self._match_path(path, self._fs_blocked):
            return False
        # Must match at least one allowed read path
        return self._match_path(path, self._fs_read)

    def can_write(self, path: str) -> bool:
        """Check if agent can write to a file path."""
        # Blocked paths always denied
        if self._match_path(path, self._fs_blocked):
            return False
        # Must match at least one allowed write path
        return self._match_path(path, self._fs_write)

    # ── Network ─────────────────────────────────────────────────────

    def can_access_host(self, host: str) -> bool:
        """Check if agent can access a network host."""
        # Check blocked hosts first
        if "*" in self._net_blocked:
            return self._match_host(host, self._net_allowed)
        for pattern in self._net_blocked:
            if fnmatch.fnmatch(host, pattern):
                return False
        # Check allowed hosts
        if "*" in self._net_allowed:
            return True
        return self._match_host(host, self._net_allowed)

    def _match_host(self, host: str, patterns: list[str]) -> bool:
        for pattern in patterns:
            if fnmatch.fnmatch(host, pattern):
                return True
        return False

    # ── Shell ───────────────────────────────────────────────────────

    def can_execute(self, command: str) -> bool:
        """Check if agent can execute a shell command."""
        if not self._shell_allowed:
            return False

        cmd_parts = command.strip().split()
        if not cmd_parts:
            return False

        base_cmd = cmd_parts[0]
        full_cmd = command.strip()

        # Check blocked commands (exact matches and partial)
        for blocked in self._shell_blocked_cmds:
            if full_cmd.startswith(blocked) or base_cmd == blocked.split()[0]:
                return False

        # Check allowed whitelist
        for allowed in self._shell_allowed_cmds:
            if base_cmd.startswith(allowed) or allowed.startswith(base_cmd):
                return True

        return False

    @property
    def shell_timeout(self) -> int:
        return self._shell_timeout

    # ── Agent Operations ────────────────────────────────────────────

    def can_create_agents(self) -> bool:
        return self._agent_ops.get("can_create_agents", False)

    def can_destroy_agents(self) -> bool:
        return self._agent_ops.get("can_destroy_agents", False)

    def can_modify_shared(self) -> bool:
        return self._agent_ops.get("can_modify_shared", False)

    def can_modify_own_soul(self) -> bool:
        return self._agent_ops.get("can_modify_own_soul", False)

    def can_modify_other_agents(self) -> bool:
        return self._agent_ops.get("can_modify_other_agents", False)

    # ── Self Modification ───────────────────────────────────────────

    def can_modify_file(self, file_type: str) -> bool:
        """Check if agent can self-modify a file type.

        file_type: identity / role / journal / knowledge / profile / shared
        """
        key_map = {
            "identity": "can_modify_identity",
            "role": "can_modify_role",
            "journal": "can_append_journal",
            "knowledge": "can_modify_knowledge",
            "profile": "can_modify_profile",
            "shared": "can_modify_shared",
        }
        key = key_map.get(file_type, "can_append_journal")
        return self._self_mod.get(key, False)

    # ── Elevation ───────────────────────────────────────────────────

    def check_rate_limit(self) -> bool:
        """Check if the agent has exceeded its rate limit.

        1. Clean up timestamps older than 60 seconds
        2. Check if current minute's call count exceeds max
        3. Record this call timestamp if within limit

        Returns True if the call is allowed, False if rate limited.
        """
        now = time.time()
        self._call_times = [t for t in self._call_times if now - t < 60]
        max_per_min = (
            self._template.get("rate_limit", {}).get("max_calls_per_minute", 999999)
        )
        if len(self._call_times) >= max_per_min:
            return False
        self._call_times.append(now)
        return True

    def request_elevation(self, reason: str) -> bool:
        """Request permission elevation (requires InteractionHub approval).

        Returns True if elevation is theoretically allowed (gate is open),
        False if elevation is completely disabled.

        The actual elevation requires human approval through InteractionHub.
        """
        requires_approval = self._elevation_cfg.get("requires_approval", True)
        if not requires_approval:
            return False
        return True

    def get_elevation_requirements(self) -> dict:
        """Get the elevation requirements from config."""
        return {
            "requires_approval": self._elevation_cfg.get("requires_approval", True),
            "min_tasks": self._elevation_cfg.get("min_tasks_for_elevation", 10),
            "min_success_rate": self._elevation_cfg.get("min_success_rate", 0.7),
        }


class PermissionChecker:
    """Global permission checker held by AgentManager.

    Maintains a cache of AgentPermissions instances and provides
    a unified check_operation() entry point for all permission checks.
    """

    def __init__(self, config_path: str | Path = "config/permissions.json"):
        self._config_path = Path(config_path)
        self._cache: dict[str, AgentPermissions] = {}

    def reload(self) -> None:
        """Reload all cached AgentPermissions from the config file."""
        for perms in self._cache.values():
            perms.reload()
        logger.info("PermissionChecker: reloaded %d cached permissions", len(self._cache))

    def get_permissions(
        self, agent_id: str, template: str
    ) -> AgentPermissions:
        """Get or create AgentPermissions for an agent from a template."""
        cache_key = f"{agent_id}:{template}"
        if cache_key not in self._cache:
            self._cache[cache_key] = AgentPermissions(
                template_name=template,
                agent_id=agent_id,
                config_path=self._config_path,
            )
        return self._cache[cache_key]

    def check_operation(
        self, agent_id: str, operation: str, **kwargs
    ) -> bool:
        """Unified permission check entry point.

        Args:
            agent_id: Agent ID
            operation: Operation type:
                - "read_file" (needs path)
                - "write_file" (needs path)
                - "execute" (needs command)
                - "access_host" (needs host)
                - "modify_file" (needs file_type)
                - "create_agent" / "destroy_agent" / "modify_shared" / "modify_own_soul"
            **kwargs: operation-specific parameters
        """
        # Resolve template from agent config or default
        template = kwargs.pop("_template", "coder")
        perms = self.get_permissions(agent_id, template)

        handlers = {
            "read_file": lambda: perms.can_read(kwargs.get("path", "")),
            "write_file": lambda: perms.can_write(kwargs.get("path", "")),
            "execute": lambda: perms.can_execute(kwargs.get("command", "")),
            "access_host": lambda: perms.can_access_host(
                kwargs.get("host", "")
            ),
            "modify_file": lambda: perms.can_modify_file(
                kwargs.get("file_type", "")
            ),
            "create_agent": lambda: perms.can_create_agents(),
            "destroy_agent": lambda: perms.can_destroy_agents(),
            "modify_shared": lambda: perms.can_modify_shared(),
            "modify_own_soul": lambda: perms.can_modify_own_soul(),
        }

        handler = handlers.get(operation)
        if handler:
            return handler()

        logger.warning("Unknown operation: %s", operation)
        return False
