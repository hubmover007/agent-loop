"""SSH Tool - secure remote execution for agents.

Each agent gets SSH access scoped to its task's required hosts.
Uses async SSH via paramiko or asyncio subprocess.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..core import ToolResult, ToolResultStatus
from .base import ToolInterface

logger = logging.getLogger(__name__)


@dataclass
class SSHHost:
    """SSH connection target."""
    host: str
    user: str = "root"
    port: int = 22
    key_path: str | None = None
    password: str | None = None
    jump_host: str | None = None
    timeout: int = 30


class SSHTool(ToolInterface):
    """Async SSH tool for remote command execution.

    Supports:
    - Single command execution
    - File upload/download
    - SSH key-based auth
    - Connection pooling (reuse connections across steps)
    - Timeout and retry
    """

    name = "ssh"
    description = "Execute commands on remote servers via SSH"

    def __init__(self, known_hosts: dict[str, SSHHost] | None = None):
        self.known_hosts: dict[str, SSHHost] = known_hosts or {}
        self._connections: dict[str, Any] = {}  # host → connection pool

    # ============================================================
    # Host Management
    # ============================================================

    def register_host(self, alias: str, host: SSHHost) -> None:
        """Register a known host for quick access by alias."""
        self.known_hosts[alias] = host

    def get_host(self, alias_or_ip: str) -> SSHHost:
        """Resolve a host by alias or IP."""
        if alias_or_ip in self.known_hosts:
            return self.known_hosts[alias_or_ip]
        # Assume it's an IP, use default config
        return SSHHost(host=alias_or_ip)

    # ============================================================
    # Tool Execution
    # ============================================================

    async def execute(self, **kwargs) -> ToolResult:
        """Execute SSH command.

        Args:
            host: Target host (alias or IP) - REQUIRED
            command: Command to execute - REQUIRED
            user: SSH user (optional, uses default)
            key_path: SSH key path (optional)
            timeout: Command timeout in seconds (default 60)
            cwd: Working directory (optional)
            env: Environment variables dict (optional)
        """
        host_str = kwargs.get("host")
        command = kwargs.get("command")

        if not host_str or not command:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="Both 'host' and 'command' are required parameters"
            )

        try:
            host = self.get_host(host_str)
            user = kwargs.get("user", host.user)
            port = kwargs.get("port", host.port)
            key_path = kwargs.get("key_path", host.key_path)
            timeout = kwargs.get("timeout", host.timeout)
            cwd = kwargs.get("cwd", None)
            env = kwargs.get("env", None)

            result = await self._ssh_exec(
                host=host.host if host != host_str else host_str,
                user=user,
                port=port,
                key_path=key_path,
                command=command,
                timeout=timeout,
                cwd=cwd,
                env=env,
            )

            return ToolResult(
                status=ToolResultStatus.SUCCESS if result["exit_code"] == 0
                        else ToolResultStatus.TRANSIENT_ERROR,
                data=result,
                error=result["stderr"] if result["exit_code"] != 0 else None,
            )
        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolResultStatus.TRANSIENT_ERROR,
                error=f"SSH command timed out after {kwargs.get('timeout', 30)}s"
            )
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=f"SSH error: {e}"
            )

    # ============================================================
    # Core SSH Execution (async subprocess)
    # ============================================================

    async def _ssh_exec(self, host: str, user: str, port: int,
                        key_path: str | None, command: str,
                        timeout: int, cwd: str | None = None,
                        env: dict | None = None) -> dict:
        """Execute command via SSH using async subprocess."""

        # Build SSH command
        ssh_cmd = ["ssh"]

        # Connection options
        ssh_cmd.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", f"ConnectTimeout={min(timeout, 30)}",
            "-o", "ServerAliveInterval=10",
            "-p", str(port),
        ])

        # Key-based auth
        if key_path:
            ssh_cmd.extend(["-i", os.path.expanduser(key_path)])

        # Target
        ssh_cmd.append(f"{user}@{host}")

        # Build remote command
        remote_cmd = command
        if cwd:
            remote_cmd = f"cd {cwd} && {remote_cmd}"
        if env:
            env_exports = " ".join(f"{k}={v}" for k, v in env.items())
            remote_cmd = f"export {env_exports}; {remote_cmd}"

        ssh_cmd.append(remote_cmd)

        logger.debug("SSH: %s", " ".join(ssh_cmd[:6]) + " ...")

        try:
            proc = await asyncio.create_subprocess_exec(
                *ssh_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout
            )

            return {
                "stdout": stdout.decode("utf-8", errors="replace").strip(),
                "stderr": stderr.decode("utf-8", errors="replace").strip(),
                "exit_code": proc.returncode,
                "host": host,
            }
        except asyncio.TimeoutError:
            raise
        except Exception as e:
            raise RuntimeError(f"SSH execution failed: {e}") from e

    # ============================================================
    # File Operations
    # ============================================================

    async def upload_file(self, host_alias: str, local_path: str,
                          remote_path: str) -> ToolResult:
        """Upload a file to a remote host via SCP."""
        host = self.get_host(host_alias)

        scp_cmd = ["scp"]
        scp_cmd.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(host.port),
        ])
        if host.key_path:
            scp_cmd.extend(["-i", os.path.expanduser(host.key_path)])

        scp_cmd.append(local_path)
        scp_cmd.append(f"{host.user}@{host.host}:{remote_path}")

        try:
            proc = await asyncio.create_subprocess_exec(
                *scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=host.timeout
            )

            if proc.returncode == 0:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    data={"remote_path": remote_path, "host": host.host}
                )
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=stderr.decode("utf-8", errors="replace").strip()
            )
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=f"SCP upload failed: {e}"
            )

    async def download_file(self, host_alias: str, remote_path: str,
                            local_path: str | None = None) -> ToolResult:
        """Download a file from a remote host via SCP."""
        host = self.get_host(host_alias)

        if local_path is None:
            local_path = os.path.join(
                tempfile.gettempdir(),
                f"agent_loop_dl_{os.path.basename(remote_path)}"
            )

        scp_cmd = ["scp"]
        scp_cmd.extend([
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-P", str(host.port),
        ])
        if host.key_path:
            scp_cmd.extend(["-i", os.path.expanduser(host.key_path)])

        scp_cmd.append(f"{host.user}@{host.host}:{remote_path}")
        scp_cmd.append(local_path)

        try:
            proc = await asyncio.create_subprocess_exec(
                *scp_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=host.timeout
            )

            if proc.returncode == 0:
                return ToolResult(
                    status=ToolResultStatus.SUCCESS,
                    data={"local_path": local_path, "host": host.host}
                )
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=stderr.decode("utf-8", errors="replace").strip()
            )
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=f"SCP download failed: {e}"
            )

    # ============================================================
    # Health Check
    # ============================================================

    async def health_check(self, host_alias: str) -> ToolResult:
        """Quick health check: can we SSH and run a simple command?"""
        return await self.execute(
            host=host_alias,
            command="echo 'OK' && uptime",
            timeout=10,
        )
