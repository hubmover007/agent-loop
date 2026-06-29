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
