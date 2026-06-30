"""Tests for PermissionChecker + AgentPermissions — capability-based access control."""

from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def permissions_config():
    """Create a temporary permissions.json config file."""
    config = {
        "templates": {
            "coder": {
                "trust_level": "restricted",
                "filesystem": {
                    "read_paths": ["state/**", "config/**", "*.md"],
                    "write_paths": ["state/agents/{agent_id}/**", "/tmp/sandbox/{agent_id}/**"],
                    "blocked_paths": ["~/.aws/**", "~/.openclaw/.env", "/etc/shadow", "~/.ssh/**"]
                },
                "network": {
                    "allowed_hosts": ["pypi.org", "api.github.com"],
                    "blocked_hosts": ["*"]
                },
                "shell": {
                    "allowed": True,
                    "allowed_commands": ["git", "python3", "pytest", "pip"],
                    "blocked_commands": ["rm", "sudo", "dd", "mkfs", "shutdown"],
                    "timeout_seconds": 30
                },
                "agent_ops": {
                    "can_create_agents": False,
                    "can_destroy_agents": False,
                    "can_modify_shared": False,
                    "can_modify_own_soul": True,
                    "can_modify_other_agents": False
                },
                "self_modification": {
                    "can_modify_identity": True,
                    "can_modify_role": True,
                    "can_append_journal": True,
                    "can_modify_knowledge": True,
                    "can_modify_profile": False
                }
            },
            "researcher": {
                "trust_level": "untrusted",
                "filesystem": {
                    "read_paths": ["state/**", "config/**"],
                    "write_paths": ["state/agents/{agent_id}/**"],
                    "blocked_paths": ["~/.aws/**", "~/.openclaw/.env"]
                },
                "network": {
                    "allowed_hosts": ["*"],
                    "blocked_hosts": []
                },
                "shell": {
                    "allowed": False
                },
                "agent_ops": {
                    "can_modify_own_soul": True,
                    "can_modify_shared": False
                }
            },
            "admin": {
                "trust_level": "admin",
                "filesystem": {"read_paths": ["**"], "write_paths": ["**"]},
                "network": {"allowed_hosts": ["*"]},
                "shell": {"allowed": True},
                "agent_ops": {
                    "can_create_agents": True,
                    "can_destroy_agents": True,
                    "can_modify_shared": True,
                    "can_modify_own_soul": True
                }
            }
        },
        "elevation": {
            "requires_approval": True,
            "min_tasks_for_elevation": 10,
            "min_success_rate": 0.7
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name

    yield config_path
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def perms(permissions_config):
    """Create AgentPermissions for a coder agent."""
    from src.permissions import AgentPermissions
    return AgentPermissions(
        template_name="coder",
        agent_id="agent-test-001",
        config_path=permissions_config,
    )


@pytest.fixture
def perms_admin(permissions_config):
    """Create AgentPermissions for an admin agent."""
    from src.permissions import AgentPermissions
    return AgentPermissions(
        template_name="admin",
        agent_id="agent-admin",
        config_path=permissions_config,
    )


# ── Tests: loading ──────────────────────────────────────────────────────────


def test_load_template(perms):
    """Permission template loads correctly."""
    assert perms.template_name == "coder"
    assert perms.trust_level == "restricted"
    assert perms.agent_id == "agent-test-001"


def test_load_defaults_for_missing_template(permissions_config):
    """Unknown template defaults to empty permissions."""
    from src.permissions import AgentPermissions
    perms = AgentPermissions(
        template_name="nonexistent",
        agent_id="bad-agent",
        config_path=permissions_config,
    )
    assert perms.trust_level == "untrusted"
    # Should still have default empty lists
    assert not perms.can_execute("git")
    assert not perms.can_read("anything")


# ── Tests: filesystem ───────────────────────────────────────────────────────


def test_can_read_allowed(perms):
    """Agent can read files in allowed paths."""
    assert perms.can_read("state/agents/agent-test-001/IDENTITY.md")
    assert perms.can_read("config/agents/shared/SAFETY.md")
    assert perms.can_read("README.md")


def test_can_read_blocked(perms):
    """Agent cannot read blocked paths."""
    assert not perms.can_read("~/.aws/credentials")
    assert not perms.can_read("~/.openclaw/.env")
    assert not perms.can_read("/etc/shadow")


def test_can_write_sandbox(perms):
    """Agent can write within its sandbox directory."""
    assert perms.can_write("state/agents/agent-test-001/IDENTITY.md")
    assert perms.can_write("/tmp/sandbox/agent-test-001/output.py")
    assert perms.can_write("state/agents/agent-test-001/JOURNAL.md")


def test_can_write_blocked(perms):
    """Agent cannot write to blocked paths."""
    assert not perms.can_write("~/.aws/credentials")
    assert not perms.can_write("~/.openclaw/.env")


def test_admin_can_write_everything(perms_admin):
    """Admin agent can write anywhere."""
    assert perms_admin.can_write("~/.aws/credentials")
    assert perms_admin.can_write("/etc/shadow")
    assert perms_admin.can_write("anything")


# ── Tests: shell ────────────────────────────────────────────────────────────


def test_can_execute_allowed(perms):
    """Agent can execute allowed commands."""
    assert perms.can_execute("git status")
    assert perms.can_execute("python3 script.py")
    assert perms.can_execute("pytest tests/")
    assert perms.can_execute("pip install requests")


def test_can_execute_blocked(perms):
    """Agent cannot execute blocked commands."""
    assert not perms.can_execute("rm -rf /")
    assert not perms.can_execute("sudo reboot")
    assert not perms.can_execute("dd if=/dev/zero of=/dev/sda")
    assert not perms.can_execute("shutdown now")


def test_researcher_cannot_execute_shell(permissions_config):
    """Researcher template has shell disabled."""
    from src.permissions import AgentPermissions
    perms = AgentPermissions("researcher", "agent-r", permissions_config)
    assert not perms.can_execute("git status")
    assert not perms.can_execute("echo hello")


# ── Tests: self modification ────────────────────────────────────────────────


def test_can_modify_file_identity(perms):
    """Agent can modify its own IDENTITY."""
    assert perms.can_modify_file("identity")


def test_can_modify_file_profile(perms):
    """Agent cannot modify its own profile."""
    assert not perms.can_modify_file("profile")


def test_can_modify_file_journal(perms):
    """Agent can append to journal."""
    assert perms.can_modify_file("journal")


def test_can_modify_file_knowledge(perms):
    """Agent can modify its own knowledge."""
    assert perms.can_modify_file("knowledge")


# ── Tests: elevation ────────────────────────────────────────────────────────


def test_request_elevation(perms):
    """Agent can request elevation (gate is open)."""
    assert perms.request_elevation("need more permissions for deployment") is True

    reqs = perms.get_elevation_requirements()
    assert reqs["requires_approval"] is True
    assert reqs["min_tasks"] == 10
    assert reqs["min_success_rate"] == 0.7


# ── Tests: network ──────────────────────────────────────────────────────────


def test_can_access_allowed_host(perms):
    """Agent can access allowed network hosts."""
    assert perms.can_access_host("pypi.org")
    assert perms.can_access_host("api.github.com")


def test_can_access_blocked_host(perms):
    """Agent cannot access non-whitelisted hosts (blocked by *)."""
    # coder has blocked_hosts=["*"], so only whitelist is allowed
    assert not perms.can_access_host("google.com")
    assert not perms.can_access_host("evil.com")


# ── Tests: PermissionChecker ────────────────────────────────────────────────


def test_permission_checker_get_perms(permissions_config):
    """PermissionChecker returns cached AgentPermissions."""
    from src.permissions import PermissionChecker
    checker = PermissionChecker(config_path=permissions_config)
    perms1 = checker.get_permissions("agent-1", "coder")
    perms2 = checker.get_permissions("agent-1", "coder")
    assert perms1 is perms2  # Same instance (cached)


def test_permission_checker_check_operation(permissions_config):
    """PermissionChecker.check_operation routes correctly."""
    from src.permissions import PermissionChecker
    checker = PermissionChecker(config_path=permissions_config)

    assert checker.check_operation("agent-1", "execute", command="git status")
    assert not checker.check_operation("agent-1", "execute", command="rm -rf /")
    assert checker.check_operation("agent-1", "read_file", path="state/test")
    assert not checker.check_operation("agent-1", "create_agent")


def test_check_operation_unknown_operation(permissions_config):
    """Unknown operation returns False."""
    from src.permissions import PermissionChecker
    checker = PermissionChecker(config_path=permissions_config)
    assert not checker.check_operation("agent-1", "fly_to_moon")
