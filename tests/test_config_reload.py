"""Tests for P2 config hot-reload (AgentManagerAgent + PermissionChecker)."""

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from src.permissions import AgentPermissions, PermissionChecker
from src.system_agents import AgentManagerAgent


class TestPermissionCheckerReload:
    """Tests for PermissionChecker.reload()."""

    def test_reload_on_file_change(self):
        """Permission is reloaded when the config file is modified."""
        # Create a temp config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "templates": {
                    "coder": {
                        "trust_level": "trusted",
                        "filesystem": {"read_paths": ["/tmp/old/**"], "write_paths": [], "blocked_paths": []},
                        "network": {"allowed_hosts": [], "blocked_hosts": []},
                        "shell": {"allowed": False, "allowed_commands": [], "blocked_commands": [], "timeout_seconds": 10},
                        "agent_ops": {},
                        "self_modification": {},
                    }
                },
                "elevation": {},
            }
            json.dump(config, f)
            temp_path = f.name

        try:
            checker = PermissionChecker(temp_path)
            perms = checker.get_permissions("agent-1", "coder")

            # Initially can read /tmp/old/
            assert perms.can_read("/tmp/old/file.txt") is True
            assert perms.can_read("/tmp/new/file.txt") is False

            # Modify the config file
            time.sleep(0.1)  # Ensure mtime differs
            config["templates"]["coder"]["filesystem"]["read_paths"] = ["/tmp/new/**"]
            with open(temp_path, "w") as f:
                json.dump(config, f)

            # Reload
            checker.reload()

            # Create fresh perms to check (cache is cleared by reload)
            perms2 = checker.get_permissions("agent-1", "coder")
            assert perms2.can_read("/tmp/new/file.txt") is True
            assert perms2.can_read("/tmp/old/file.txt") is False
        finally:
            os.unlink(temp_path)

    def test_no_reload_unchanged(self):
        """Reload without changes keeps same behavior."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "templates": {
                    "coder": {
                        "trust_level": "trusted",
                        "filesystem": {"read_paths": ["/tmp/test/**"], "write_paths": [], "blocked_paths": []},
                        "network": {"allowed_hosts": [], "blocked_hosts": []},
                        "shell": {"allowed": False, "allowed_commands": [], "blocked_commands": [], "timeout_seconds": 10},
                        "agent_ops": {},
                        "self_modification": {},
                    }
                },
                "elevation": {},
            }
            json.dump(config, f)
            temp_path = f.name

        try:
            checker = PermissionChecker(temp_path)
            perms = checker.get_permissions("agent-1", "coder")
            assert perms.can_read("/tmp/test/file.txt") is True

            checker.reload()
            perms2 = checker.get_permissions("agent-1", "coder")
            assert perms2.can_read("/tmp/test/file.txt") is True
        finally:
            os.unlink(temp_path)

    def test_config_reload_check(self):
        """AgentManagerAgent._check_config_reload detects file changes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"templates": {}, "elevation": {}}, f)
            temp_path = f.name

        try:
            manager = type('M', (), {
                '_config_mtime': 0,
                '_config_path': temp_path,
                '_check_config_reload': AgentManagerAgent._check_config_reload,
            })()

            # First call: should detect change (mtime > 0)
            assert manager._check_config_reload() is True

            # Second call: no change
            assert manager._check_config_reload() is False

            # Touch the file
            time.sleep(0.1)
            with open(temp_path, "a") as f:
                f.write(" ")

            # Should detect change again
            assert manager._check_config_reload() is True
        finally:
            os.unlink(temp_path)

    def test_permission_checker_reload_clears_cache(self):
        """reload() on PermissionChecker calls reload on each cached AgentPermissions."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            config = {
                "templates": {
                    "coder": {
                        "trust_level": "trusted",
                        "filesystem": {"read_paths": ["/tmp/v1/**"], "write_paths": [], "blocked_paths": []},
                        "network": {"allowed_hosts": [], "blocked_hosts": []},
                        "shell": {"allowed": False, "allowed_commands": [], "blocked_commands": [], "timeout_seconds": 10},
                        "agent_ops": {},
                        "self_modification": {},
                    }
                },
                "elevation": {},
            }
            json.dump(config, f)
            temp_path = f.name

        try:
            checker = PermissionChecker(temp_path)
            perms = checker.get_permissions("agent-1", "coder")
            assert perms.can_read("/tmp/v1/file.txt") is True

            # Update file
            time.sleep(0.1)
            config["templates"]["coder"]["filesystem"]["read_paths"] = ["/tmp/v2/**"]
            with open(temp_path, "w") as f:
                json.dump(config, f)

            checker.reload()
            perms2 = checker.get_permissions("agent-1", "coder")
            assert perms2.can_read("/tmp/v2/file.txt") is True
        finally:
            os.unlink(temp_path)
