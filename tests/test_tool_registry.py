"""Tests for MCP ToolRegistry."""

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.tool_registry import ToolSpec, ToolRegistry, BuiltinTools


class TestToolSpec:
    """Tests for ToolSpec dataclass."""

    def test_create_and_to_openai_function(self):
        """ToolSpec should convert to OpenAI function-calling format."""
        spec = ToolSpec(
            name="test.echo",
            namespace="test",
            description="Echo back the input",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            handler=lambda message: message,
        )
        func = spec.to_openai_function()

        assert func["type"] == "function"
        assert func["function"]["name"] == "test.echo"
        assert func["function"]["description"] == "Echo back the input"
        assert "message" in func["function"]["parameters"]["properties"]

    def test_to_openai_function_strips_invalid_keys(self):
        """Sanitise removes JSON Schema keys OpenAI rejects."""
        spec = ToolSpec(
            name="test.x",
            namespace="test",
            description="X",
            input_schema={
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "$schema": "http://json-schema.org/draft-07/schema#",
                "definitions": {"foo": {}},
            },
            handler=lambda a: a,
        )
        func = spec.to_openai_function()
        params = func["function"]["parameters"]
        assert "$schema" not in params
        assert "definitions" not in params
        assert "properties" in params


class TestToolRegistry:
    """Tests for ToolRegistry class."""

    @pytest.fixture
    def registry(self):
        return ToolRegistry()

    @pytest.fixture
    def echo_spec(self):
        return ToolSpec(
            name="test.echo",
            namespace="test",
            description="Echo",
            input_schema={
                "type": "object",
                "properties": {"message": {"type": "string"}},
                "required": ["message"],
            },
            handler=lambda message: message,
            risk_level="low",
            tags=["test", "read"],
        )

    # ── test_register_and_get ───────────────────────────────────

    def test_register_and_get(self, registry, echo_spec):
        """Register a tool then retrieve it by name."""
        registry.register(echo_spec)
        got = registry.get("test.echo")
        assert got is not None
        assert got.name == "test.echo"
        assert got.namespace == "test"
        assert got.risk_level == "low"

    def test_register_duplicate_raises(self, registry, echo_spec):
        """Registering the same name twice should raise ValueError."""
        registry.register(echo_spec)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(echo_spec)

    # ── test_unregister ─────────────────────────────────────────

    def test_unregister(self, registry, echo_spec):
        """Unregister removes tool; get returns None."""
        registry.register(echo_spec)
        assert registry.get("test.echo") is not None
        registry.unregister("test.echo")
        assert registry.get("test.echo") is None

    def test_unregister_missing_is_noop(self, registry):
        """Unregister a non-existent tool doesn't raise."""
        registry.unregister("nonexistent")  # Should not raise

    # ── test_list_filter ────────────────────────────────────────

    @pytest.fixture
    def populated_registry(self):
        reg = ToolRegistry()
        specs = [
            ToolSpec("fs.read", "fs", "Read file",
                     {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                     lambda path: path, risk_level="low", tags=["filesystem", "read"]),
            ToolSpec("fs.write", "fs", "Write file",
                     {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
                     lambda path, content: None, risk_level="medium", tags=["filesystem", "write"]),
            ToolSpec("web.fetch", "web", "Fetch URL",
                     {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
                     lambda url: url, risk_level="low", tags=["web", "read"]),
            ToolSpec("shell.exec", "shell", "Execute command",
                     {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
                     lambda cmd: cmd, risk_level="high", tags=["shell", "exec"]),
            # Disabled tool — should not appear in lists
            ToolSpec("fs.delete", "fs", "Delete file",
                     {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
                     lambda path: None, risk_level="critical", tags=["filesystem", "write"],
                     enabled=False),
        ]
        for s in specs:
            reg.register(s)
        return reg

    def test_list_filter_namespace(self, populated_registry):
        """Filter tools by namespace."""
        fs_tools = populated_registry.list_tools(namespace="fs")
        assert len(fs_tools) == 2  # fs.read and fs.write (fs.delete is disabled)
        assert all(t.namespace == "fs" for t in fs_tools)

    def test_list_filter_tags(self, populated_registry):
        """Filter tools by tags (AND semantics)."""
        write_tools = populated_registry.list_tools(tags=["write"])
        assert len(write_tools) == 1
        assert write_tools[0].name == "fs.write"

    def test_list_filter_max_risk(self, populated_registry):
        """Filter tools by max risk level."""
        low_tools = populated_registry.list_tools(max_risk="low")
        assert all(t.risk_level == "low" for t in low_tools)
        assert len(low_tools) == 2  # fs.read and web.fetch

    def test_list_filter_combined(self, populated_registry):
        """Combined namespace + tag filter."""
        result = populated_registry.list_tools(namespace="fs", tags=["read"])
        assert len(result) == 1
        assert result[0].name == "fs.read"

    def test_disabled_not_listed(self, populated_registry):
        """Disabled tools should not appear in any listing."""
        all_tools = populated_registry.list_tools()
        names = {t.name for t in all_tools}
        assert "fs.delete" not in names

    def test_list_namespaces(self, populated_registry):
        """List unique namespaces."""
        nss = populated_registry.list_namespaces()
        assert "fs" in nss
        assert "web" in nss
        assert "shell" in nss

    # ── test_to_openai_functions ─────────────────────────────────

    def test_to_openai_functions(self, populated_registry):
        """Convert all tools to OpenAI function format."""
        funcs = populated_registry.to_openai_functions()
        assert len(funcs) == 4  # 4 enabled tools
        for func in funcs:
            assert func["type"] == "function"
            assert "name" in func["function"]
            assert "description" in func["function"]
            assert "parameters" in func["function"]

    def test_to_openai_functions_namespace_filter(self, populated_registry):
        """Filter OpenAI functions by namespace."""
        funcs = populated_registry.to_openai_functions(namespace="web")
        assert len(funcs) == 1
        assert funcs[0]["function"]["name"] == "web.fetch"

    # ── test_invoke_with_schema_validation ───────────────────────

    @pytest.mark.asyncio
    async def test_invoke_with_schema_validation(self, registry, echo_spec):
        """Invoke validates args against schema before calling handler."""
        registry.register(echo_spec)

        # Valid args
        result = await registry.invoke("test.echo", {"message": "hello"})
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_invoke_missing_required_param(self, registry, echo_spec):
        """Missing required param should raise ValueError."""
        registry.register(echo_spec)
        with pytest.raises(ValueError, match="Invalid args"):
            await registry.invoke("test.echo", {})

    @pytest.mark.asyncio
    async def test_invoke_wrong_type_param(self, registry):
        """Wrong type parameter should raise ValueError."""
        spec = ToolSpec(
            name="math.add",
            namespace="math",
            description="Add numbers",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "number"}, "y": {"type": "number"}},
                "required": ["x", "y"],
            },
            handler=lambda x, y: x + y,
        )
        registry.register(spec)
        with pytest.raises(ValueError, match="Invalid args"):
            await registry.invoke("math.add", {"x": "not_a_number", "y": 2})

    @pytest.mark.asyncio
    async def test_invoke_disabled_tool(self, registry):
        """Invoking a disabled tool should raise ValueError."""
        spec = ToolSpec(
            name="disabled.tool", namespace="test", description="Disabled",
            input_schema={"type": "object", "properties": {}},
            handler=lambda: None, enabled=False,
        )
        registry.register(spec)
        with pytest.raises(ValueError, match="disabled"):
            await registry.invoke("disabled.tool", {})

    @pytest.mark.asyncio
    async def test_invoke_nonexistent_tool(self, registry):
        """Invoking a non-existent tool should raise ValueError."""
        with pytest.raises(ValueError, match="Tool not found"):
            await registry.invoke("nonexistent", {})

    # ── test_invoke_high_risk_needs_approval ─────────────────────

    @pytest.mark.asyncio
    async def test_invoke_high_risk_needs_approval(self, registry):
        """High-risk tool needs human approval via interaction_hub."""
        called = False

        def handler(cmd):
            nonlocal called
            called = True
            return f"ran: {cmd}"

        spec = ToolSpec(
            name="shell.rm",
            namespace="shell",
            description="Remove file",
            input_schema={
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
                "required": ["cmd"],
            },
            handler=handler,
            risk_level="critical",
            tags=["shell", "delete"],
        )
        registry.register(spec)

        # Mock InteractionHub that approves
        hub = MagicMock()
        hub._risk_threshold = "medium"
        approval = MagicMock()
        approval.status = "approved"
        approval.reply = "ok"
        hub.request_approval = AsyncMock(return_value=approval)

        result = await registry.invoke("shell.rm", {"cmd": "rm -rf /tmp/test"}, interaction_hub=hub)
        assert result == "ran: rm -rf /tmp/test"
        assert called
        hub.request_approval.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invoke_high_risk_denied(self, registry):
        """High-risk tool denied by user should raise PermissionError."""
        spec = ToolSpec(
            name="shell.kill",
            namespace="shell",
            description="Kill process",
            input_schema={
                "type": "object",
                "properties": {"pid": {"type": "integer"}},
                "required": ["pid"],
            },
            handler=lambda pid: f"killed {pid}",
            risk_level="high",
            tags=["shell", "dangerous"],
        )
        registry.register(spec)

        # Mock InteractionHub that denies
        hub = MagicMock()
        hub._risk_threshold = "medium"
        approval = MagicMock()
        approval.status = "denied"
        approval.reply = "not allowed"
        hub.request_approval = AsyncMock(return_value=approval)

        with pytest.raises(PermissionError, match="requires approval"):
            await registry.invoke("shell.kill", {"pid": 1234}, interaction_hub=hub)

    @pytest.mark.asyncio
    async def test_invoke_low_risk_no_approval_needed(self, registry, echo_spec):
        """Low-risk tool should not trigger approval request."""
        registry.register(echo_spec)

        hub = MagicMock()
        hub._risk_threshold = "medium"
        hub.request_approval = AsyncMock()

        result = await registry.invoke("test.echo", {"message": "hi"}, interaction_hub=hub)
        assert result == "hi"
        hub.request_approval.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_invoke_async_handler(self, registry):
        """ToolRegistry should support async handler functions."""
        async def async_handler(x: int) -> int:
            return x * 2

        spec = ToolSpec(
            name="async.double",
            namespace="test",
            description="Double a number async",
            input_schema={
                "type": "object",
                "properties": {"x": {"type": "integer"}},
                "required": ["x"],
            },
            handler=async_handler,
        )
        registry.register(spec)
        result = await registry.invoke("async.double", {"x": 21})
        assert result == 42

    # ── register_many / register_all ────────────────────────────

    def test_register_many(self, registry):
        """Register multiple tools at once."""
        specs = [
            ToolSpec("a.one", "a", "One",
                     {"type": "object", "properties": {}}, lambda: 1),
            ToolSpec("a.two", "a", "Two",
                     {"type": "object", "properties": {}}, lambda: 2),
        ]
        registry.register_many(specs)
        assert len(registry) == 2
        assert registry.get("a.one") is not None

    # ── dunder helpers ───────────────────────────────────────────

    def test_len(self, populated_registry):
        assert len(populated_registry) == 5  # 4 enabled + 1 disabled

    def test_contains(self, populated_registry):
        assert "fs.read" in populated_registry
        assert "nonexistent" not in populated_registry


class TestBuiltinTools:
    """Tests for BuiltinTools pre-built tool packs."""

    def test_filesystem_creates_specs(self, tmp_path):
        """filesystem() returns a list of ToolSpecs with valid handlers."""
        specs = BuiltinTools.filesystem(root_dir=str(tmp_path))
        assert len(specs) >= 4
        names = {s.name for s in specs}
        assert "fs.read_file" in names
        assert "fs.write_file" in names
        assert "fs.list_dir" in names
        assert "fs.search_files" in names

        # Test that handlers work
        reg = ToolRegistry()
        for s in specs:
            reg.register(s)

        import asyncio
        # Write & read round-trip
        result = asyncio.run(reg.invoke("fs.write_file", {"path": "test.txt", "content": "hello"}))
        assert result["bytes"] > 0

        content = asyncio.run(reg.invoke("fs.read_file", {"path": "test.txt"}))
        assert content == "hello"

    def test_shell_creates_specs(self):
        """shell() returns a ToolSpec list."""
        specs = BuiltinTools.shell()
        assert len(specs) == 1
        assert specs[0].name == "shell.run_command"
        assert specs[0].risk_level == "high"
        assert specs[0].namespace == "shell"

    def test_web_creates_specs(self):
        """web() returns a ToolSpec list."""
        specs = BuiltinTools.web()
        assert len(specs) >= 2
        names = {s.name for s in specs}
        assert "web.fetch_url" in names
        assert "web.search_web" in names

    @pytest.mark.asyncio
    async def test_shell_with_whitelist(self):
        """Shell tool with whitelist blocks unlisted commands."""
        specs = BuiltinTools.shell(allowed_commands=["echo", "ls"])
        reg = ToolRegistry()
        for s in specs:
            reg.register(s)

        # Allowed command
        result = await reg.invoke("shell.run_command", {"command": "echo hello", "timeout": 5})
        assert "hello" in result["stdout"]
        assert result["returncode"] == 0

        # Blocked command
        with pytest.raises(ValueError, match="not in the allowed list"):
            await reg.invoke("shell.run_command", {"command": "rm -rf /", "timeout": 5})

    @pytest.mark.asyncio
    async def test_filesystem_path_traversal_blocked(self, tmp_path):
        """Filesystem tools should reject path traversal attempts."""
        specs = BuiltinTools.filesystem(root_dir=str(tmp_path))
        reg = ToolRegistry()
        for s in specs:
            reg.register(s)

        with pytest.raises(ValueError, match="Path traversal"):
            await reg.invoke("fs.read_file", {"path": "../../etc/passwd"})
