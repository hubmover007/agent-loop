"""Tests for ToolRegistry and tools."""

import pytest
from src.tools.base import ToolRegistry, ToolInterface
from src.core import ToolResult, ToolResultStatus


def test_tool_registry_register():
    """Test tool registration."""
    registry = ToolRegistry()
    assert registry.tool_names == []


def test_tool_registry_defaults():
    """Test default tool registration."""
    registry = ToolRegistry()
    registry.register_defaults()

    assert "ssh" in registry.tool_names
    assert "web" in registry.tool_names
    assert "code" in registry.tool_names


def test_tool_registry_list_allowed():
    """Test tool whitelist filtering."""
    registry = ToolRegistry()
    registry.register_defaults()

    # Get only ssh
    tools = registry.list_allowed(["ssh"])
    assert len(tools) == 1
    assert tools[0].name == "ssh"

    # Empty allowed = all tools
    all_tools = registry.list_allowed([])
    assert len(all_tools) == 3
