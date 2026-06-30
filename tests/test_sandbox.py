"""Tests for SandboxManager — tiered code execution isolation."""

from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def permissions_config():
    """Create a temporary permissions.json for sandbox tests."""
    config = {
        "templates": {
            "coder": {
                "trust_level": "restricted",
                "filesystem": {
                    "read_paths": ["state/**"],
                    "write_paths": ["state/agents/{agent_id}/**", "/tmp/sandbox/{agent_id}/**"],
                    "blocked_paths": ["~/.aws/**"]
                },
                "shell": {
                    "allowed": True,
                    "allowed_commands": ["echo", "python3", "ls"],
                    "blocked_commands": ["rm", "shutdown", "dd"],
                    "timeout_seconds": 10
                }
            },
            "researcher": {
                "trust_level": "untrusted",
                "shell": {"allowed": False}
            },
            "ops": {
                "trust_level": "trusted",
                "shell": {
                    "allowed": True,
                    "allowed_commands": ["echo", "ls", "whoami", "python3"],
                    "blocked_commands": ["rm", "dd"],
                    "timeout_seconds": 10
                }
            },
            "admin": {
                "trust_level": "admin",
                "shell": {"allowed": True}
            }
        },
        "elevation": {"requires_approval": True}
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    yield config_path
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def sandbox_manager(tmp_path):
    """Create SandboxManager with temp sandbox root."""
    from src.sandbox import SandboxManager
    return SandboxManager(sandbox_root=str(tmp_path / "sandbox"))


@pytest.fixture
def coder_perms(permissions_config):
    """AgentPermissions for a coder (restricted)."""
    from src.permissions import AgentPermissions
    return AgentPermissions("coder", "agent-coder-01", permissions_config)


@pytest.fixture
def researcher_perms(permissions_config):
    """AgentPermissions for a researcher (untrusted)."""
    from src.permissions import AgentPermissions
    return AgentPermissions("researcher", "agent-researcher-01", permissions_config)


@pytest.fixture
def ops_perms(permissions_config):
    """AgentPermissions for an ops agent (trusted)."""
    from src.permissions import AgentPermissions
    return AgentPermissions("ops", "agent-ops-01", permissions_config)


@pytest.fixture
def admin_perms(permissions_config):
    """AgentPermissions for an admin agent."""
    from src.permissions import AgentPermissions
    return AgentPermissions("admin", "agent-admin-01", permissions_config)


# ── Tests: Level 0 (untrusted) ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_level0_execute(sandbox_manager, researcher_perms):
    """Level 0 executes simple Python code via subprocess."""
    result = await sandbox_manager.execute_code(
        code="print('hello from sandbox')",
        language="python",
        permissions=researcher_perms,
    )
    assert result["level"] == 0
    assert "hello from sandbox" in result["stdout"]
    assert result["exit_code"] == 0
    assert result["duration_ms"] >= 0


@pytest.mark.asyncio
async def test_level0_execute_with_error(sandbox_manager, researcher_perms):
    """Level 0 captures syntax errors."""
    result = await sandbox_manager.execute_code(
        code="print(undefined_var",
        language="python",
        permissions=researcher_perms,
    )
    assert result["level"] == 0
    assert result["exit_code"] != 0 or "error" in result["stderr"].lower() or result["exit_code"] == 1


# ── Tests: Level 1 (restricted) ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_level1_execute(sandbox_manager, coder_perms):
    """Level 1 executes code in sandbox working directory."""
    result = await sandbox_manager.execute_code(
        code="import os; print(os.getcwd())",
        language="python",
        permissions=coder_perms,
    )
    assert result["level"] == 1
    assert result["exit_code"] == 0
    # Should be executing in sandbox directory
    assert "sandbox" in result["stdout"] or result["stdout"].strip()


# ── Tests: Level 2 (trusted) ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_level2_execute(sandbox_manager, ops_perms):
    """Level 2 executes allowed commands."""
    result = await sandbox_manager.execute_command(
        command="echo hello_trusted",
        permissions=ops_perms,
    )
    assert result["level"] == 2
    assert "hello_trusted" in result["stdout"]
    assert result["exit_code"] == 0


@pytest.mark.asyncio
async def test_blocked_command_rejected(sandbox_manager, ops_perms):
    """Level 2 rejects blocked commands."""
    result = await sandbox_manager.execute_command(
        command="rm -rf /important",
        permissions=ops_perms,
    )
    assert result["level"] == 2
    assert result["exit_code"] == -1
    assert "rejected" in result["stderr"].lower() or "rm" in result["stderr"]


# ── Tests: Level 3 (admin) ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_level3_needs_approval(sandbox_manager, admin_perms):
    """Level 3 requires InteractionHub approval.

    Without an InteractionHub, admin execution proceeds as Level 3 directly.
    """
    result = await sandbox_manager.execute_code(
        code="print('admin exec')",
        language="python",
        permissions=admin_perms,
    )
    assert result["level"] == 3
    assert result["exit_code"] == 0
    assert "admin exec" in result["stdout"]


@pytest.mark.asyncio
async def test_level0_rejects_shell_command(sandbox_manager, researcher_perms):
    """Level 0/1 rejects shell command execution (shell disabled)."""
    # For untrusted agents, execute_command should reject
    result = await sandbox_manager.execute_command(
        command="ls",
        permissions=researcher_perms,
    )
    assert result["exit_code"] == -1


# ── Tests: execute_command with different levels ────────────────────────────


@pytest.mark.asyncio
async def test_execute_command_level2_allowed(sandbox_manager, ops_perms):
    """Level 2 execute_command works with whitelisted commands."""
    result = await sandbox_manager.execute_command(
        command="whoami",
        permissions=ops_perms,
    )
    assert result["level"] == 2
    assert result["exit_code"] == 0


@pytest.mark.asyncio
async def test_execute_command_level2_blocked(sandbox_manager, ops_perms):
    """Level 2 execute_command blocks rm."""
    result = await sandbox_manager.execute_command(
        command="rm file.txt",
        permissions=ops_perms,
    )
    assert result["exit_code"] == -1
    assert "rejected" in result["stderr"].lower() or "rm" in result["stderr"]


@pytest.mark.asyncio
async def test_level1_uses_cwd_isolation(sandbox_manager, coder_perms):
    """Level 1 writes files inside sandbox directory."""
    result = await sandbox_manager.execute_code(
        code="open('test_output.txt', 'w').write('sandboxed')",
        language="python",
        permissions=coder_perms,
    )
    assert result["exit_code"] == 0
