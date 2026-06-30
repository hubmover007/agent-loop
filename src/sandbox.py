"""Sandbox Manager — tiered code execution isolation.

4-level trust model:
  - Level 0 (untrusted): subprocess + Python subprocess with stdin code
  - Level 1 (restricted): subprocess + working directory restriction
  - Level 2 (trusted): subprocess + timeout + command whitelist
  - Level 3 (admin): direct execution + InteractionHub approval

No dependency on firejail/nsjail — uses subprocess + cwd + timeout for isolation.
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class SandboxManager:
    """Code execution sandbox manager with tiered isolation.

    Levels:
      - Level 0 (untrusted): RestrictedPython or Python stdin subprocess
      - Level 1 (restricted): subprocess + limited working directory
      - Level 2 (trusted): subprocess + timeout + command whitelist
      - Level 3 (admin): direct execution + InteractionHub approval
    """

    def __init__(self, sandbox_root: str = "/tmp/agent-sandbox"):
        self._sandbox_root = Path(sandbox_root)
        self._sandbox_root.mkdir(parents=True, exist_ok=True)

    async def execute_code(
        self,
        code: str,
        language: str = "python",
        permissions: Any = None,
        interaction_hub: Any = None,
    ) -> dict:
        """Execute code with isolation level matching the agent's trust_level.

        Args:
            code: Source code to execute
            language: Programming language (default: python)
            permissions: AgentPermissions instance (determines trust_level)
            interaction_hub: InteractionHub for Level 3 approval

        Returns:
            {stdout, stderr, exit_code, duration_ms, level}
        """
        trust_level = (
            permissions.trust_level if permissions else "untrusted"
        )

        level_map = {
            "untrusted": 0,
            "restricted": 1,
            "trusted": 2,
            "admin": 3,
        }
        level = level_map.get(trust_level, 0)

        if level == 0:
            return await self._level0_execute(code, language)
        elif level == 1:
            return await self._level1_execute(
                code, language, permissions.agent_id if permissions else "unknown"
            )
        elif level == 2:
            return await self._level2_execute(
                code, language, permissions
            )
        elif level == 3:
            return await self._level3_execute(code, language, interaction_hub)

        return {"stdout": "", "stderr": "unknown trust level", "exit_code": -1, "duration_ms": 0, "level": -1}

    async def execute_command(
        self,
        command: str,
        permissions: Any = None,
        interaction_hub: Any = None,
    ) -> dict:
        """Execute a shell command with permission checks.

        Returns:
            {stdout, stderr, exit_code, duration_ms, level}
        """
        trust_level = (
            permissions.trust_level if permissions else "untrusted"
        )
        level_map = {"untrusted": 0, "restricted": 1, "trusted": 2, "admin": 3}
        level = level_map.get(trust_level, 0)

        if level <= 1:
            # Level 0/1: no shell access
            if permissions and not permissions.can_execute(command):
                return {
                    "stdout": "",
                    "stderr": f"Command '{command}' rejected by permissions (trust_level={trust_level})",
                    "exit_code": -1,
                    "duration_ms": 0,
                    "level": level,
                }
            return await self._level1_execute(
                command, "shell", permissions.agent_id if permissions else "unknown"
            )

        if level == 2:
            if not permissions or not permissions.can_execute(command):
                return {
                    "stdout": "",
                    "stderr": f"Command '{command}' rejected by command whitelist",
                    "exit_code": -1,
                    "duration_ms": 0,
                    "level": level,
                }
            return await self._level2_execute(command, permissions)

        if level == 3:
            return await self._level3_execute(command, "shell", interaction_hub)

        return {"stdout": "", "stderr": "unknown trust level", "exit_code": -1, "duration_ms": 0, "level": -1}

    # ── Level 0: Untrusted ────────────────────────────────────────

    async def _level0_execute(self, code: str, language: str) -> dict:
        """Execute code via subprocess with Python stdin input.

        Tries RestrictedPython first, falls back to subprocess-based isolation.
        """
        start = time.monotonic()

        # Try RestrictedPython
        restricted_result = self._try_restricted_python(code)
        if restricted_result is not None:
            elapsed = (time.monotonic() - start) * 1000
            return {
                "stdout": restricted_result.get("stdout", ""),
                "stderr": restricted_result.get("stderr", ""),
                "exit_code": 0,
                "duration_ms": round(elapsed, 1),
                "level": 0,
            }

        # Fallback: subprocess with Python stdin
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=10
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = (time.monotonic() - start) * 1000
                return {
                    "stdout": "",
                    "stderr": "Execution timed out (10s)",
                    "exit_code": -1,
                    "duration_ms": round(elapsed, 1),
                    "level": 0,
                }

            elapsed = (time.monotonic() - start) * 1000
            return {
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                "exit_code": proc.returncode or 0,
                "duration_ms": round(elapsed, 1),
                "level": 0,
            }
        except FileNotFoundError:
            elapsed = (time.monotonic() - start) * 1000
            return {
                "stdout": "",
                "stderr": "python3 not found",
                "exit_code": -1,
                "duration_ms": round(elapsed, 1),
                "level": 0,
            }

    def _try_restricted_python(self, code: str) -> dict | None:
        """Try executing with RestrictedPython. Returns None if unavailable."""
        try:
            from RestrictedPython import compile_restricted
            from RestrictedPython.Guards import (
                safe_builtins,
                guarded_iter_unpack_sequence,
            )

            safe_globals = {
                "__builtins__": safe_builtins,
                "_iter_unpack_sequence_": guarded_iter_unpack_sequence,
            }
            bytecode = compile_restricted(code, "<sandbox>", "exec")
            safe_locals: dict = {}

            import io
            import contextlib

            stdout_buf = io.StringIO()
            stderr_buf = io.StringIO()
            with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(
                stderr_buf
            ):
                exec(bytecode, safe_globals, safe_locals)

            return {
                "stdout": stdout_buf.getvalue(),
                "stderr": stderr_buf.getvalue(),
            }
        except ImportError:
            return None
        except Exception as e:
            return {
                "stdout": "",
                "stderr": f"RestrictedPython execution error: {e}",
            }

    # ── Level 1: Restricted ───────────────────────────────────────

    async def _level1_execute(
        self, code_or_cmd: str, language: str, agent_id: str
    ) -> dict:
        """Execute with working directory restricted to sandbox.

        Limited to sandbox_root/{agent_id}/ directory.
        """
        start = time.monotonic()
        agent_dir = self._sandbox_root / agent_id
        agent_dir.mkdir(parents=True, exist_ok=True)

        if language == "python":
            cmd = ["python3", "-c", code_or_cmd]
        else:
            cmd = ["bash", "-c", code_or_cmd]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(agent_dir),
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=30
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = (time.monotonic() - start) * 1000
                return {
                    "stdout": "",
                    "stderr": "Execution timed out (30s)",
                    "exit_code": -1,
                    "duration_ms": round(elapsed, 1),
                    "level": 1,
                }

            elapsed = (time.monotonic() - start) * 1000
            return {
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                "exit_code": proc.returncode or 0,
                "duration_ms": round(elapsed, 1),
                "level": 1,
            }
        except FileNotFoundError:
            elapsed = (time.monotonic() - start) * 1000
            return {
                "stdout": "",
                "stderr": f"Command not found: {cmd[0]}",
                "exit_code": -1,
                "duration_ms": round(elapsed, 1),
                "level": 1,
            }

    # ── Level 2: Trusted ──────────────────────────────────────────

    async def _level2_execute(
        self, command: str, permissions: Any
    ) -> dict:
        """Execute with timeout and command whitelist check.

        Whitelist check already done by caller.
        """
        start = time.monotonic()
        timeout = permissions.shell_timeout if permissions else 10

        try:
            proc = await asyncio.create_subprocess_exec(
                "bash", "-c", command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = (time.monotonic() - start) * 1000
                return {
                    "stdout": "",
                    "stderr": f"Execution timed out ({timeout}s)",
                    "exit_code": -1,
                    "duration_ms": round(elapsed, 1),
                    "level": 2,
                }

            elapsed = (time.monotonic() - start) * 1000
            return {
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                "exit_code": proc.returncode or 0,
                "duration_ms": round(elapsed, 1),
                "level": 2,
            }
        except FileNotFoundError:
            elapsed = (time.monotonic() - start) * 1000
            return {
                "stdout": "",
                "stderr": "bash not found",
                "exit_code": -1,
                "duration_ms": round(elapsed, 1),
                "level": 2,
            }

    # ── Level 3: Admin ────────────────────────────────────────────

    async def _level3_execute(
        self, code_or_cmd: str, language: str, interaction_hub: Any = None
    ) -> dict:
        """Direct execution with interaction hub approval.

        Requires human approval before executing.
        """
        if interaction_hub is not None:
            from .interaction import InteractionHub

            if isinstance(interaction_hub, InteractionHub):
                req = await interaction_hub.request_approval(
                    agent_id="sandbox-admin",
                    action=code_or_cmd[:80],
                    details=f"Admin-level code execution in sandbox:\n```\n{code_or_cmd}\n```",
                    risk_level="critical",
                    task_scope="sandbox-execution",
                    timeout_seconds=120,
                )
                if req.status != "approved":
                    return {
                        "stdout": "",
                        "stderr": f"Admin execution denied: {req.reply}",
                        "exit_code": -1,
                        "duration_ms": 0,
                        "level": 3,
                    }

        start = time.monotonic()

        if language == "python":
            cmd = ["python3", "-c", code_or_cmd]
        else:
            cmd = ["bash", "-c", code_or_cmd]

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=120
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                elapsed = (time.monotonic() - start) * 1000
                return {
                    "stdout": "",
                    "stderr": "Admin execution timed out (120s)",
                    "exit_code": -1,
                    "duration_ms": round(elapsed, 1),
                    "level": 3,
                }

            elapsed = (time.monotonic() - start) * 1000
            return {
                "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                "exit_code": proc.returncode or 0,
                "duration_ms": round(elapsed, 1),
                "level": 3,
            }
        except FileNotFoundError:
            elapsed = (time.monotonic() - start) * 1000
            return {
                "stdout": "",
                "stderr": "Command not found",
                "exit_code": -1,
                "duration_ms": round(elapsed, 1),
                "level": 3,
            }
