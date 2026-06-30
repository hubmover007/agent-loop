"""MCP (Model Context Protocol) Tool Registry.

A model-controlled tool registry inspired by the MCP standard.
Supports dynamic registration, JSON Schema validation, namespace isolation,
and automatic LLM function-calling format generation.

Key concepts:
  - Tools are model-controlled — LLM discovers and calls them autonomously
  - Namespace isolation prevents name collisions (fs.read vs web.read)
  - JSON Schema validation runs before every tool call
  - Risk-level gating requires human approval for dangerous operations
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import jsonschema

logger = logging.getLogger(__name__)

# ============================================================
# ToolSpec
# ============================================================


@dataclass
class ToolSpec:
    """Tool specification (MCP-compatible).

    Attributes:
        name: Unique identifier, e.g. "fs.read_file"
        namespace: Logical namespace — "fs" / "web" / "shell" / "custom"
        description: Human-readable description for LLM tool selection
        input_schema: JSON Schema dict for parameter validation
        handler: Async or sync callable that executes the tool
        risk_level: low / medium / high / critical
        tags: Arbitrary tags for filtering
        enabled: Whether this tool can be called (runtime toggle)
        registered_at: ISO 8601 timestamp of registration
    """
    name: str
    namespace: str
    description: str
    input_schema: dict
    handler: Callable
    risk_level: str = "low"
    tags: list[str] = field(default_factory=list)
    enabled: bool = True
    registered_at: str = ""

    def __post_init__(self):
        if not self.registered_at:
            self.registered_at = datetime.now(timezone.utc).isoformat()

    def to_openai_function(self) -> dict:
        """Convert to an OpenAI function-calling definition.

        Returns a dict suitable for the ``tools`` array in a chat completion request:

            {
                "type": "function",
                "function": {
                    "name": "fs.read_file",
                    "description": "...",
                    "parameters": { ... JSON Schema ... }
                }
            }
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._sanitise_schema_for_openai(self.input_schema),
            },
        }

    @staticmethod
    def _sanitise_schema_for_openai(schema: dict) -> dict:
        """Strip JSON Schema keywords that OpenAI rejects (e.g. $schema, definitions).

        Returns a shallow copy with only the keys OpenAI accepts."""
        allowed = {"type", "properties", "required", "description",
                   "enum", "items", "additionalProperties", "default",
                   "anyOf", "oneOf", "allOf", "title"}
        return {k: v for k, v in schema.items() if k in allowed}


# ============================================================
# ToolRegistry
# ============================================================


class ToolRegistry:
    """MCP-style tool registry.

    Features:
      1. Namespace isolation (``fs.read_file`` vs ``web.fetch`` never clash)
      2. JSON Schema parameter validation before every invocation
      3. Dynamic register / unregister at runtime
      4. Filter by namespace, tags, or max risk level
      5. Automatic LLM tool-description generation (OpenAI function-calling format)
    """

    def __init__(self):
        self._tools: dict[str, ToolSpec] = {}

    # ── Registration ──────────────────────────────────────────────

    def register(self, spec: ToolSpec) -> None:
        """Register a tool specification.

        Raises ValueError if a tool with the same name already exists.
        """
        if spec.name in self._tools:
            raise ValueError(f"Tool '{spec.name}' is already registered. Unregister first.")
        self._tools[spec.name] = spec
        logger.debug("ToolRegistry: registered '%s' (ns=%s risk=%s)",
                      spec.name, spec.namespace, spec.risk_level)

    def register_many(self, specs: list[ToolSpec]) -> None:
        """Register a list of tool specs."""
        for spec in specs:
            self.register(spec)

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry. No-op if not registered."""
        removed = self._tools.pop(name, None)
        if removed:
            logger.debug("ToolRegistry: unregistered '%s'", name)

    def get(self, name: str) -> ToolSpec | None:
        """Retrieve a tool specification by name."""
        return self._tools.get(name)

    # ── Listing / Filtering ───────────────────────────────────────

    def list_tools(self, *,
                   namespace: str | None = None,
                   tags: list[str] | None = None,
                   max_risk: str | None = None) -> list[ToolSpec]:
        """List tools with optional filters.

        Args:
            namespace: Only return tools in this namespace.
            tags: Only return tools that have ALL of these tags.
            max_risk: Only return tools at or below this risk level
                      (low < medium < high < critical).
        """
        from .interaction import RISK_LEVEL_ORDER

        results: list[ToolSpec] = []
        for tool in self._tools.values():
            if not tool.enabled:
                continue
            if namespace is not None and tool.namespace != namespace:
                continue
            if tags is not None and not all(t in tool.tags for t in tags):
                continue
            if max_risk is not None:
                tr = RISK_LEVEL_ORDER.get(tool.risk_level, 0)
                mr = RISK_LEVEL_ORDER.get(max_risk, 0)
                if tr > mr:
                    continue
            results.append(tool)
        return results

    def list_namespaces(self) -> list[str]:
        """Return all unique namespaces currently registered."""
        return sorted({t.namespace for t in self._tools.values()})

    # ── OpenAI function-calling bridge ────────────────────────────

    def to_openai_functions(self, namespace: str | None = None) -> list[dict]:
        """Generate the tools array for an OpenAI chat completion request.

        Args:
            namespace: If provided, only include tools from this namespace.
        """
        tools = self.list_tools(namespace=namespace)
        return [t.to_openai_function() for t in tools]

    def to_openai_tools(self, namespace: str | None = None) -> list[dict]:
        """Convert to OpenAI tool calling format (alias for to_openai_functions).

        Returns:
            [{"type": "function", "function": {"name": "...", "description": "...", "parameters": {...}}}]

        Args:
            namespace: If provided, only include tools from this namespace.
        """
        return self.to_openai_functions(namespace=namespace)

    # ── Invocation ────────────────────────────────────────────────

    async def invoke(self, name: str, args: dict,
                     interaction_hub=None,
                     max_retries: int = 3,
                     retry_delay: float = 1.0) -> Any:
        """Invoke a registered tool with automatic retry on failure.

        Flow:
          1. Look up tool — raise ValueError if missing or disabled
          2. Validate args against JSON Schema — raise ValidationError on mismatch
          3. If risk_level >= threshold and interaction_hub exists → request approval
          4. Call handler with exponential backoff retry on failure
          5. Return result

        Args:
            name: Tool name, e.g. "fs.read_file"
            args: Keyword arguments for the handler
            interaction_hub: Optional InteractionHub for human approval gating
            max_retries: Maximum number of retries on failure (default 3)
            retry_delay: Initial delay between retries in seconds (default 1.0),
                doubles on each subsequent retry (exponential backoff)

        Returns:
            Whatever the tool handler returns.
        """
        import inspect

        spec = self._tools.get(name)
        if spec is None:
            raise ValueError(f"Tool not found: {name}")
        if not spec.enabled:
            raise ValueError(f"Tool '{name}' is disabled")

        # 2. Validate args
        try:
            jsonschema.validate(instance=args, schema=spec.input_schema)
        except jsonschema.ValidationError as e:
            raise ValueError(f"Invalid args for '{name}': {e.message}") from e

        # 3. Risk gating
        if interaction_hub is not None:
            from .interaction import RISK_LEVEL_ORDER, detect_risk_level

            risk = detect_risk_level(f"{name} {json.dumps(args)}")
            hub_threshold = getattr(interaction_hub, "_risk_threshold", "medium")

            if risk and RISK_LEVEL_ORDER.get(risk, 0) >= RISK_LEVEL_ORDER.get(hub_threshold, 1):
                try:
                    approval = await interaction_hub.request_approval(
                        agent_id=name,
                        action=f"{name}: {args}",
                        details=json.dumps(args, default=str),
                        risk_level=risk,
                        task_scope=f"tool:{name}",
                    )
                    if approval.status != "approved":
                        raise PermissionError(
                            f"Tool '{name}' requires approval: {approval.status} — {approval.reply}"
                        )
                except Exception as e:
                    if isinstance(e, PermissionError):
                        raise
                    logger.warning("ToolRegistry: approval check failed for '%s': %s", name, e)

        # 4. Call handler with exponential backoff retry
        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                if inspect.iscoroutinefunction(spec.handler):
                    return await spec.handler(**args)
                else:
                    return spec.handler(**args)
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    delay = retry_delay * (2 ** attempt)
                    logger.warning(
                        "ToolRegistry: '%s' attempt %d/%d failed (%s), retrying in %.1fs",
                        name, attempt + 1, max_retries + 1, type(e).__name__, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "ToolRegistry: '%s' all %d attempts failed: %s",
                        name, max_retries + 1, type(e).__name__,
                    )

        raise last_error  # type: ignore[misc]

    # ── Bulk helpers ──────────────────────────────────────────────

    def register_all(self, tools: list[ToolSpec]) -> None:
        """Register multiple tools at once."""
        for t in tools:
            self.register(t)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ============================================================
# BuiltinTools — pre-built tool collections
# ============================================================


class BuiltinTools:
    """Pre-built tool collections that can be registered in one call."""

    @staticmethod
    def filesystem(root_dir: str = ".") -> list[ToolSpec]:
        """File-system tool pack: read_file, write_file, list_dir, search_files.

        Args:
            root_dir: All paths are resolved relative to this directory.
        """
        import os as _os

        def _read_file(path: str, encoding: str = "utf-8") -> str:
            full = _os.path.join(root_dir, path)
            if not _os.path.abspath(full).startswith(_os.path.abspath(root_dir)):
                raise ValueError(f"Path traversal denied: {path}")
            with open(full, encoding=encoding) as f:
                return f.read()

        def _write_file(path: str, content: str, encoding: str = "utf-8") -> dict:
            full = _os.path.join(root_dir, path)
            if not _os.path.abspath(full).startswith(_os.path.abspath(root_dir)):
                raise ValueError(f"Path traversal denied: {path}")
            _os.makedirs(_os.path.dirname(full), exist_ok=True)
            with open(full, "w", encoding=encoding) as f:
                f.write(content)
            size = _os.path.getsize(full)
            return {"path": full, "bytes": size}

        def _list_dir(path: str = ".") -> list[str]:
            full = _os.path.join(root_dir, path)
            if not _os.path.abspath(full).startswith(_os.path.abspath(root_dir)):
                raise ValueError(f"Path traversal denied: {path}")
            return _os.listdir(full)

        def _search_files(path: str = ".", pattern: str = "*") -> list[str]:
            import fnmatch as _fnmatch
            full = _os.path.join(root_dir, path)
            if not _os.path.abspath(full).startswith(_os.path.abspath(root_dir)):
                raise ValueError(f"Path traversal denied: {path}")
            matches = []
            for dirpath, _, filenames in _os.walk(full):
                for fn in filenames:
                    if _fnmatch.fnmatch(fn, pattern):
                        matches.append(_os.path.relpath(_os.path.join(dirpath, fn), root_dir))
            return matches

        return [
            ToolSpec(
                name="fs.read_file",
                namespace="fs",
                description="Read the contents of a file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to root"},
                        "encoding": {"type": "string", "default": "utf-8"},
                    },
                    "required": ["path"],
                },
                handler=_read_file,
                risk_level="low",
                tags=["filesystem", "read"],
            ),
            ToolSpec(
                name="fs.write_file",
                namespace="fs",
                description="Write content to a file (creates or overwrites)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "File path relative to root"},
                        "content": {"type": "string", "description": "Content to write"},
                        "encoding": {"type": "string", "default": "utf-8"},
                    },
                    "required": ["path", "content"],
                },
                handler=_write_file,
                risk_level="medium",
                tags=["filesystem", "write"],
            ),
            ToolSpec(
                name="fs.list_dir",
                namespace="fs",
                description="List contents of a directory",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                    },
                },
                handler=_list_dir,
                risk_level="low",
                tags=["filesystem", "read"],
            ),
            ToolSpec(
                name="fs.search_files",
                namespace="fs",
                description="Search for files matching a glob pattern",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "default": "."},
                        "pattern": {"type": "string", "default": "*"},
                    },
                    "required": ["pattern"],
                },
                handler=_search_files,
                risk_level="low",
                tags=["filesystem", "search"],
            ),
        ]

    @staticmethod
    def shell(allowed_commands: list[str] | None = None) -> list[ToolSpec]:
        """Shell tool pack: run_command (with optional command whitelist).

        Args:
            allowed_commands: If set, only these command prefixes are permitted.
        """
        import subprocess as _sp

        async def _run_command(command: str, timeout: int = 30) -> dict:
            if allowed_commands is not None:
                cmd_name = command.split()[0] if command.strip() else ""
                if cmd_name not in allowed_commands:
                    raise ValueError(
                        f"Command '{cmd_name}' is not in the allowed list: {allowed_commands}"
                    )
            proc = await __import__("asyncio").create_subprocess_shell(
                command,
                stdout=_sp.PIPE,
                stderr=_sp.PIPE,
            )
            try:
                stdout, stderr = await __import__("asyncio").wait_for(
                    proc.communicate(), timeout=timeout
                )
            except __import__("asyncio").TimeoutError:
                proc.kill()
                await proc.wait()
                raise TimeoutError(f"Command timed out after {timeout}s: {command}")
            return {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "returncode": proc.returncode,
            }

        return [
            ToolSpec(
                name="shell.run_command",
                namespace="shell",
                description="Run a shell command and return stdout, stderr, and return code",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "Shell command to run"},
                        "timeout": {"type": "integer", "default": 30, "minimum": 1, "maximum": 300},
                    },
                    "required": ["command"],
                },
                handler=_run_command,
                risk_level="high",
                tags=["shell", "exec"],
            ),
        ]

    @staticmethod
    def web() -> list[ToolSpec]:
        """Web tool pack: fetch_url, search_web.

        Handlers are mock-friendly stubs — replace with real HTTP / search backends.
        """
        async def _fetch_url(url: str, method: str = "GET") -> dict:
            import urllib.request as _ur
            # Stub implementation — replace with real HTTP client in production
            return {"url": url, "method": method, "status": 200, "body": f"Response from {url}"}

        async def _search_web(query: str, max_results: int = 5) -> list[dict]:
            # Stub implementation — replace with real search backend in production
            return [{"title": f"Result {i} for {query}", "url": f"https://example.com/{i}"}
                    for i in range(min(max_results, 5))]

        return [
            ToolSpec(
                name="web.fetch_url",
                namespace="web",
                description="Fetch content from a URL",
                input_schema={
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "format": "uri"},
                        "method": {"type": "string", "enum": ["GET", "POST"], "default": "GET"},
                    },
                    "required": ["url"],
                },
                handler=_fetch_url,
                risk_level="low",
                tags=["web", "http"],
            ),
            ToolSpec(
                name="web.search_web",
                namespace="web",
                description="Search the web for information",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 5, "minimum": 1, "maximum": 50},
                    },
                    "required": ["query"],
                },
                handler=_search_web,
                risk_level="low",
                tags=["web", "search"],
            ),
        ]
