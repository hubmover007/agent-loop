"""Web Tool — web search and fetch for agents.

Provides:
  - web_search: search the web via Brave Search API
  - web_fetch: fetch and extract content from URLs
"""

from __future__ import annotations

import logging
from typing import Any

from ..core import ToolResult, ToolResultStatus
from .base import ToolInterface

logger = logging.getLogger(__name__)


class WebTool(ToolInterface):
    """Web search and fetch tool for agents.

    Uses Brave Search API for search and httpx for fetching.
    """

    name = "web"
    description = "Search the web and fetch web page content"

    def __init__(self, brave_api_key: str | None = None):
        self.brave_api_key = brave_api_key
        self._client = None

    def _ensure_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=30.0)

    async def execute(self, **kwargs) -> ToolResult:
        """Execute web operation.

        Args:
            action: "search" or "fetch" - REQUIRED
            query: search query (for action="search")
            url: URL to fetch (for action="fetch")
            count: number of results (default 5, max 10)
            max_chars: max chars to extract (default 5000)
        """
        action = kwargs.get("action", "search")

        if action == "search":
            return await self._search(
                query=kwargs.get("query", ""),
                count=kwargs.get("count", 5),
            )
        elif action == "fetch":
            return await self._fetch(
                url=kwargs.get("url", ""),
                max_chars=kwargs.get("max_chars", 5000),
            )
        else:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=f"Unknown action: {action}. Use 'search' or 'fetch'."
            )

    async def _search(self, query: str, count: int = 5) -> ToolResult:
        """Search the web via Brave Search API."""
        if not query:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="query is required for search"
            )

        if not self.brave_api_key:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="Brave API key not configured"
            )

        self._ensure_client()

        try:
            resp = await self._client.get(
                "https://api.search.brave.com/res/v1/web/search",
                headers={
                    "X-Subscription-Token": self.brave_api_key,
                    "Accept": "application/json",
                },
                params={
                    "q": query,
                    "count": min(count, 10),
                },
            )
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("web", {}).get("results", [])[:count]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                })

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                data={"results": results, "count": len(results)},
            )

        except Exception as e:
            logger.error("Web search failed: %s", e)
            return ToolResult(
                status=ToolResultStatus.TRANSIENT_ERROR,
                error=f"Search failed: {e}"
            )

    async def _fetch(self, url: str, max_chars: int = 5000) -> ToolResult:
        """Fetch and extract text content from a URL."""
        if not url:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="url is required for fetch"
            )

        self._ensure_client()

        try:
            resp = await self._client.get(
                url,
                follow_redirects=True,
                headers={"User-Agent": "Agent-Loop/1.0"},
            )
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")

            if "text/html" in content_type:
                # Simple HTML to text extraction
                text = self._html_to_text(resp.text)
            elif "application/json" in content_type:
                text = resp.text[:max_chars]
            else:
                text = resp.text[:max_chars]

            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                data={
                    "url": url,
                    "content": text[:max_chars],
                    "content_type": content_type,
                    "status_code": resp.status_code,
                },
            )

        except Exception as e:
            logger.error("Web fetch failed: %s", e)
            return ToolResult(
                status=ToolResultStatus.TRANSIENT_ERROR,
                error=f"Fetch failed: {e}"
            )

    @staticmethod
    def _html_to_text(html: str) -> str:
        """Simple HTML to text conversion."""
        import re

        # Remove scripts and styles
        html = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r'<style[^>]*>.*?</style>', '', html, flags=re.DOTALL | re.IGNORECASE)

        # Remove tags
        text = re.sub(r'<[^>]+>', ' ', html)

        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text


class CodeTool(ToolInterface):
    """Code execution tool for agents.

    Supports:
    - Python code execution (sandboxed subprocess)
    - Shell command execution
    - File read/write
    """

    name = "code"
    description = "Execute Python code, shell commands, or file operations"

    def __init__(self, workspace_dir: str = "/tmp/agent_loop/workspace",
                 timeout: int = 30):
        import os
        self.workspace = os.path.expanduser(workspace_dir)
        self.timeout = timeout
        os.makedirs(self.workspace, exist_ok=True)

    async def execute(self, **kwargs) -> ToolResult:
        """Execute code operation.

        Args:
            action: "python" | "shell" | "read" | "write" - REQUIRED
            code: Python code (for action="python")
            command: shell command (for action="shell")
            path: file path (for action="read" or "write")
            content: file content (for action="write")
        """
        action = kwargs.get("action")

        if action == "python":
            return await self._run_python(kwargs.get("code", ""))
        elif action == "shell":
            return await self._run_shell(kwargs.get("command", ""))
        elif action == "read":
            return await self._read_file(kwargs.get("path", ""))
        elif action == "write":
            return await self._write_file(kwargs.get("path", ""), kwargs.get("content", ""))
        else:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=f"Unknown action: {action}. Use python/shell/read/write."
            )

    async def _run_python(self, code: str) -> ToolResult:
        """Execute Python code in a subprocess."""
        if not code:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="code is required for python action"
            )

        import asyncio
        import tempfile
        import os

        # Write code to temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=self.workspace
        ) as f:
            f.write(code)
            script_path = f.name

        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", script_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            result = {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": proc.returncode,
            }

            return ToolResult(
                status=ToolResultStatus.SUCCESS if proc.returncode == 0
                        else ToolResultStatus.TRANSIENT_ERROR,
                data=result,
                error=result["stderr"] if proc.returncode != 0 else None,
            )

        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolResultStatus.TRANSIENT_ERROR,
                error=f"Python execution timed out after {self.timeout}s"
            )
        finally:
            os.unlink(script_path)

    async def _run_shell(self, command: str) -> ToolResult:
        """Execute a shell command."""
        if not command:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="command is required for shell action"
            )

        import asyncio

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=self.workspace,
            )

            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=self.timeout
            )

            result = {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "exit_code": proc.returncode,
            }

            return ToolResult(
                status=ToolResultStatus.SUCCESS if proc.returncode == 0
                        else ToolResultStatus.TRANSIENT_ERROR,
                data=result,
                error=result["stderr"] if proc.returncode != 0 else None,
            )

        except asyncio.TimeoutError:
            return ToolResult(
                status=ToolResultStatus.TRANSIENT_ERROR,
                error=f"Shell command timed out after {self.timeout}s"
            )

    async def _read_file(self, path: str) -> ToolResult:
        """Read a file."""
        import os

        if not path:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="path is required for read action"
            )

        # Resolve relative to workspace
        full_path = os.path.join(self.workspace, path) if not os.path.isabs(path) else path

        try:
            with open(full_path, "r") as f:
                content = f.read()
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                data={"path": full_path, "content": content},
            )
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=f"Read failed: {e}"
            )

    async def _write_file(self, path: str, content: str) -> ToolResult:
        """Write a file."""
        import os

        if not path:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="path is required for write action"
            )

        full_path = os.path.join(self.workspace, path) if not os.path.isabs(path) else path
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        try:
            with open(full_path, "w") as f:
                f.write(content)
            return ToolResult(
                status=ToolResultStatus.SUCCESS,
                data={"path": full_path, "bytes_written": len(content)},
            )
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=f"Write failed: {e}"
            )
