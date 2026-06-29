"""Tool base interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..core import ToolResult


class ToolInterface(ABC):
    """Base class for all tools."""

    name: str = "base"
    description: str = "Base tool"

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given parameters."""
        ...


class ToolRegistry:
    """Registry of available tools."""

    def __init__(self):
        self._tools: dict[str, ToolInterface] = {}

    def register(self, tool: ToolInterface) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        self._tools.pop(name, None)

    def get(self, name: str) -> ToolInterface | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_allowed(self, allowed: list[str]) -> list[ToolInterface]:
        """Get tools matching the allowed whitelist."""
        if not allowed:
            return list(self._tools.values())
        return [t for name, t in self._tools.items() if name in allowed]

    @property
    def tool_names(self) -> list[str]:
        """List all registered tool names."""
        return list(self._tools.keys())

    @property
    def all(self) -> dict[str, ToolInterface]:
        """Get all tools as a dict (for iteration)."""
        return dict(self._tools)

    def register_defaults(self, **kwargs) -> None:
        """Register all built-in tools with optional configuration.

        Args:
            brave_api_key: API key for Brave Search (WebTool)
            ssh_known_hosts: dict of known SSH hosts
            code_workspace: directory for CodeTool workspace
            enable_openclaw: register OpenClaw skills as tools (default: True)
        """
        from .ssh import SSHTool
        from .web import WebTool, CodeTool

        # SSH tool
        ssh_hosts = kwargs.get("ssh_known_hosts", {})
        self.register(SSHTool(known_hosts=ssh_hosts))

        # Web tool
        brave_key = kwargs.get("brave_api_key")
        self.register(WebTool(brave_api_key=brave_key))

        # Code tool
        code_ws = kwargs.get("code_workspace", "/tmp/agent_loop/workspace")
        self.register(CodeTool(workspace_dir=code_ws))

        # OpenClaw skill bridge (integrates OpenClaw's 42+ skills)
        if kwargs.get("enable_openclaw", True):
            from .openclaw_bridge import register_openclaw_skills
            register_openclaw_skills(self)

        # Auto-register any future tools here
        logger = __import__("logging").getLogger(__name__)
        logger.info("Registered %d default tools: %s",
                    len(self._tools), list(self._tools.keys()))
