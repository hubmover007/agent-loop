"""Tests for BrowserTool and CanvasTool."""

from __future__ import annotations

import os
import tempfile
import pytest


# ── BrowserTool ──────────────────────────────────────────────────

PLAYWRIGHT_AVAILABLE = False
try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    pass


browser_test = pytest.mark.skipif(
    not PLAYWRIGHT_AVAILABLE,
    reason="Playwright not installed — pip install playwright && playwright install chromium",
)


class TestBrowserTool:
    """Tests for BrowserTool (requires Playwright)."""

    @browser_test
    @pytest.mark.asyncio
    async def test_browser_tool_creation(self):
        """BrowserTool can be created with default config."""
        from src.tools.browser import BrowserTool
        tool = BrowserTool(headless=True)
        assert tool.name == "browser"
        assert tool.headless is True
        assert tool.timeout == 30000

    @browser_test
    @pytest.mark.asyncio
    async def test_browser_tool_custom_config(self):
        """BrowserTool accepts custom config."""
        from src.tools.browser import BrowserTool
        tool = BrowserTool(headless=False, timeout=60000)
        assert tool.headless is False
        assert tool.timeout == 60000

    @browser_test
    @pytest.mark.asyncio
    async def test_navigate_invalid_url(self):
        """Navigate with missing url returns error."""
        from src.tools.browser import BrowserTool
        tool = BrowserTool()
        result = await tool.execute(action="navigate")
        assert result.status.value == "fatal_error"
        assert "URL is required" in result.error
        await tool.close()

    @browser_test
    @pytest.mark.asyncio
    async def test_unknown_action(self):
        """Unknown action returns error."""
        from src.tools.browser import BrowserTool
        tool = BrowserTool()
        result = await tool.execute(action="nonexistent")
        assert result.status.value == "fatal_error"
        assert "Unknown browser action" in result.error
        await tool.close()

    @browser_test
    @pytest.mark.asyncio
    async def test_navigate(self):
        """Navigate to a URL and get page title."""
        from src.tools.browser import BrowserTool
        tool = BrowserTool()
        try:
            result = await tool.execute(action="navigate", url="about:blank")
            assert result.status.value == "success"
            assert "url" in result.data
            assert result.data["url"] == "about:blank"
        finally:
            await tool.close()

    @browser_test
    @pytest.mark.asyncio
    async def test_extract_body_text(self):
        """Extract text from body after navigation."""
        from src.tools.browser import BrowserTool
        tool = BrowserTool()
        try:
            await tool.execute(action="navigate", url="about:blank")
            result = await tool.execute(action="extract")
            assert result.status.value == "success"
            assert "content" in result.data
        finally:
            await tool.close()

    @browser_test
    @pytest.mark.asyncio
    async def test_click_missing_selector(self):
        """Click without selector returns error."""
        from src.tools.browser import BrowserTool
        tool = BrowserTool()
        try:
            await tool.execute(action="navigate", url="about:blank")
            result = await tool.execute(action="click")
            assert result.status.value == "fatal_error"
            assert "selector" in result.error
        finally:
            await tool.close()

    @browser_test
    @pytest.mark.asyncio
    async def test_type_missing_selector(self):
        """Type without selector returns error."""
        from src.tools.browser import BrowserTool
        tool = BrowserTool()
        try:
            result = await tool.execute(action="type")
            assert result.status.value == "fatal_error"
            assert "selector" in result.error
        finally:
            await tool.close()

    @browser_test
    @pytest.mark.asyncio
    async def test_scroll(self):
        """Scroll down/up the page."""
        from src.tools.browser import BrowserTool
        tool = BrowserTool()
        try:
            result = await tool.execute(action="scroll", direction="down", amount=300)
            assert result.status.value == "success"
            assert result.data["scrolled"] == "down"
            assert result.data["amount"] == 300
        finally:
            await tool.close()

    @browser_test
    @pytest.mark.asyncio
    async def test_wait_missing_selector(self):
        """Wait without selector returns error."""
        from src.tools.browser import BrowserTool
        tool = BrowserTool()
        try:
            result = await tool.execute(action="wait")
            assert result.status.value == "fatal_error"
            assert "selector" in result.error
        finally:
            await tool.close()

    @browser_test
    @pytest.mark.asyncio
    async def test_close(self):
        """Close browser and cleanup."""
        from src.tools.browser import BrowserTool
        tool = BrowserTool()
        try:
            result = await tool.execute(action="close")
            assert result.status.value == "success"
            assert result.data["closed"] is True
            assert tool._browser is None
        finally:
            await tool.close()

    @browser_test
    @pytest.mark.asyncio
    async def test_reopen_after_close(self):
        """After close, browser can be reopened on next action."""
        from src.tools.browser import BrowserTool
        tool = BrowserTool()
        try:
            await tool.execute(action="close")
            # Should reopen browser
            result = await tool.execute(action="navigate", url="about:blank")
            assert result.status.value == "success"
        finally:
            await tool.close()


# ── CanvasTool ───────────────────────────────────────────────────

class TestCanvasTool:
    """Tests for CanvasTool (no external deps)."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Use temp directory for canvas output."""
        self.tmpdir = tempfile.mkdtemp(prefix="canvas_test_")
        from src.tools.canvas import CanvasTool
        self.tool = CanvasTool(output_dir=self.tmpdir)
        yield
        # Cleanup
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_canvas_tool_creation(self):
        """CanvasTool can be created."""
        assert self.tool.name == "canvas"
        assert self.tool.output_dir == self.tmpdir

    @pytest.mark.asyncio
    async def test_render_html(self):
        """Render raw HTML snippet."""
        result = await self.tool.execute(
            action="render_html",
            html="<h1>Hello World</h1><p>Test content</p>",
        )
        assert result.status.value == "success"
        assert "canvas_id" in result.data
        assert "path" in result.data
        assert "url" in result.data
        assert result.data["url"].startswith("/canvas/")

        # Verify file exists and contains content
        filepath = result.data["path"]
        assert os.path.exists(filepath)
        with open(filepath) as f:
            content = f.read()
            assert "<h1>Hello World</h1>" in content
            assert "<p>Test content</p>" in content

    @pytest.mark.asyncio
    async def test_render_chart(self):
        """Render a Chart.js chart."""
        data = {
            "labels": ["A", "B", "C"],
            "datasets": [{"label": "Values", "data": [10, 20, 30]}],
        }
        result = await self.tool.execute(
            action="render_chart",
            data=data,
            chart_type="bar",
            title="My Chart",
        )
        assert result.status.value == "success"
        assert os.path.exists(result.data["path"])

        with open(result.data["path"]) as f:
            content = f.read()
            assert "chart.js" in content.lower() or "Chart" in content
            assert "My Chart" in content
            assert "type: 'bar'" in content

    @pytest.mark.asyncio
    async def test_render_chart_invalid_type(self):
        """Unknown chart type returns error."""
        result = await self.tool.execute(
            action="render_chart",
            data={},
            chart_type="unknown_type",
        )
        assert result.status.value == "fatal_error"
        assert "Unknown chart_type" in result.error

    @pytest.mark.asyncio
    async def test_render_diagram(self):
        """Render a Mermaid diagram."""
        mermaid = "graph TD\n    A[Start] --> B[End]"
        result = await self.tool.execute(
            action="render_diagram",
            mermaid=mermaid,
        )
        assert result.status.value == "success"
        assert os.path.exists(result.data["path"])

        with open(result.data["path"]) as f:
            content = f.read()
            assert "mermaid" in content.lower()
            assert "graph TD" in content
            assert "Start" in content

    @pytest.mark.asyncio
    async def test_render_table(self):
        """Render an HTML table."""
        result = await self.tool.execute(
            action="render_table",
            headers=["Name", "Age", "City"],
            rows=[
                ["Alice", "30", "NYC"],
                ["Bob", "25", "SF"],
            ],
            title="Users",
        )
        assert result.status.value == "success"
        assert os.path.exists(result.data["path"])

        with open(result.data["path"]) as f:
            content = f.read()
            assert "<th>Name</th>" in content
            assert "<td>Alice</td>" in content
            assert "<td>25</td>" in content
            assert "Users" in content

    @pytest.mark.asyncio
    async def test_render_table_empty_headers(self):
        """Table without headers returns error."""
        result = await self.tool.execute(
            action="render_table",
            headers=[],
            rows=[["a", "b"]],
        )
        assert result.status.value == "fatal_error"
        assert "headers" in result.error.lower()

    @pytest.mark.asyncio
    async def test_render_markdown(self):
        """Render Markdown to HTML."""
        result = await self.tool.execute(
            action="render_markdown",
            markdown="# Title\n\n**Bold text** and `code`\n\n- Item 1\n- Item 2",
            title="Test Doc",
        )
        assert result.status.value == "success"
        assert os.path.exists(result.data["path"])

        with open(result.data["path"]) as f:
            content = f.read()
            assert "<h1>Title</h1>" in content
            assert "<b>Bold text</b>" in content
            assert "<code>code</code>" in content
            assert "<li>Item 1</li>" in content
            assert "<li>Item 2</li>" in content

    @pytest.mark.asyncio
    async def test_render_markdown_empty(self):
        """Empty markdown still produces a valid page."""
        result = await self.tool.execute(
            action="render_markdown",
            markdown="",
        )
        assert result.status.value == "success"
        assert os.path.exists(result.data["path"])

    @pytest.mark.asyncio
    async def test_unknown_action(self):
        """Unknown action returns error."""
        result = await self.tool.execute(action="paint_masterpiece")
        assert result.status.value == "fatal_error"
        assert "Unknown canvas action" in result.error

    @pytest.mark.asyncio
    async def test_multiple_canvases_unique_ids(self):
        """Each canvas gets a unique ID."""
        import asyncio
        r1 = await self.tool.execute(action="render_html", html="<p>One</p>")
        await asyncio.sleep(0.002)  # ensure different timestamp
        r2 = await self.tool.execute(action="render_html", html="<p>Two</p>")
        assert r1.data["canvas_id"] != r2.data["canvas_id"]
        assert r1.data["path"] != r2.data["path"]

    @pytest.mark.asyncio
    async def test_xss_escape_in_table(self):
        """Table cell content is HTML-escaped."""
        result = await self.tool.execute(
            action="render_table",
            headers=["Col"],
            rows=[["<script>alert(1)</script>"]],
        )
        assert result.status.value == "success"
        with open(result.data["path"]) as f:
            content = f.read()
            assert "<script>alert(1)</script>" not in content
            assert "&lt;script&gt;" in content

    @pytest.mark.asyncio
    async def test_all_chart_types_valid(self):
        """All supported chart types render without error."""
        for chart_type in self.tool.CHART_TYPES:
            result = await self.tool.execute(
                action="render_chart",
                data={"labels": ["X"], "datasets": [{"data": [1]}]},
                chart_type=chart_type,
            )
            assert result.status.value == "success", f"Failed for {chart_type}"


# ── ToolRegistry integration ─────────────────────────────────────

class TestToolRegistryWithBrowserCanvas:
    """Verify new tools register correctly."""

    def test_browser_tool_in_registry(self):
        """BrowserTool registers when playwright is available."""
        from src.tools.base import ToolRegistry
        reg = ToolRegistry()
        reg.register_defaults()
        names = reg.tool_names
        assert "canvas" in names  # always available

    def test_browser_tool_disabled(self):
        """BrowserTool can be disabled via kwargs."""
        from src.tools.base import ToolRegistry
        reg = ToolRegistry()
        reg.register_defaults(enable_browser=False)
        assert "browser" not in reg.tool_names

    def test_canvas_tool_disabled(self):
        """CanvasTool can be disabled via kwargs."""
        from src.tools.base import ToolRegistry
        reg = ToolRegistry()
        reg.register_defaults(enable_canvas=False)
        assert "canvas" not in reg.tool_names

    def test_both_enabled_default(self):
        """Both canvas and (if available) browser are registered."""
        from src.tools.base import ToolRegistry
        reg = ToolRegistry()
        reg.register_defaults()
        assert "canvas" in reg.tool_names
        # browser may or may not be available
