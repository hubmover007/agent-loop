"""Browser Tool — headless browser automation for agents.

Uses Playwright (async) for headless browser operations.
Supports: navigate, click, type, screenshot, extract, scroll.

Installation:
  pip install playwright
  playwright install chromium
  playwright install-deps chromium

On headless Linux servers, Playwright works out of the box with
headless mode (default). No X server or display required.

Design decision: Playwright over browser-use or Selenium:
- Playwright: lightweight, first-class async Python API, fast
- browser-use: LLM-driven, requires an LLM call per action (expensive/slow),
  Rust core dependency, v0.13+ requires Python >=3.11
- Selenium: heavier, slower, more boilerplate
"""

from __future__ import annotations
import logging
from typing import Any

from ..core import ToolResult, ToolResultStatus
from .base import ToolInterface

logger = logging.getLogger(__name__)


class BrowserTool(ToolInterface):
    """Headless browser automation tool.

    Actions:
    - navigate(url): Open a URL and return page title
    - click(selector): Click an element by CSS selector
    - type(selector, text): Type text into an input field
    - screenshot(): Take a full-page screenshot (saved to state/screenshots/)
    - extract(selector?): Extract text content from page or element
    - scroll(direction, amount): Scroll the page up/down
    - wait(selector, timeout): Wait for an element to appear
    - close(): Close the browser

    All operations run in headless Chromium.
    """

    name = "browser"
    description = "Headless browser automation: navigate, click, type, screenshot, extract"

    def __init__(self, headless: bool = True, timeout: int = 30000):
        self.headless = headless
        self.timeout = timeout
        self._browser = None
        self._page = None
        self._pw = None

    async def _ensure_browser(self):
        """Lazy init browser and page."""
        if self._browser is None:
            try:
                from playwright.async_api import async_playwright
                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(
                    headless=self.headless,
                    args=["--no-sandbox", "--disable-setuid-sandbox"],
                )
                self._page = await self._browser.new_page()
                self._page.set_default_timeout(self.timeout)
                logger.info("BrowserTool: Chromium launched (headless=%s)", self.headless)
            except ImportError:
                raise ImportError(
                    "playwright not installed. Install with:\n"
                    "  pip install playwright\n"
                    "  playwright install chromium\n"
                    "  playwright install-deps chromium"
                )
            except Exception as e:
                raise RuntimeError(f"Failed to launch browser: {e}. "
                    "On Linux, try: playwright install-deps chromium") from e

    async def execute(self, **kwargs) -> ToolResult:
        """Execute browser action.

        Args:
            action: navigate|click|type|screenshot|extract|scroll|wait|close
            url: URL to open (for navigate)
            selector: CSS selector (for click/type/wait/extract)
            text: text to type (for type)
            direction: up|down (for scroll)
            amount: pixels to scroll (for scroll, default 500)
            timeout: wait timeout in ms (for wait, default 5000)
        """
        action = kwargs.get("action", "navigate")

        try:
            if action == "close":
                return await self._close()

            await self._ensure_browser()

            if action == "navigate":
                return await self._navigate(kwargs.get("url", ""))
            elif action == "click":
                return await self._click(kwargs.get("selector", ""))
            elif action == "type":
                return await self._type(kwargs.get("selector", ""), kwargs.get("text", ""))
            elif action == "screenshot":
                return await self._screenshot()
            elif action == "extract":
                return await self._extract(kwargs.get("selector"))
            elif action == "scroll":
                return await self._scroll(
                    kwargs.get("direction", "down"),
                    kwargs.get("amount", 500),
                )
            elif action == "wait":
                return await self._wait(
                    kwargs.get("selector", ""),
                    kwargs.get("timeout", 5000),
                )
            else:
                return ToolResult(
                    status=ToolResultStatus.FATAL_ERROR,
                    error=f"Unknown browser action: {action}",
                )
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=f"BrowserTool.{action}: {e}",
            )

    async def _navigate(self, url: str) -> ToolResult:
        if not url:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="URL is required for navigate",
            )
        await self._page.goto(url, wait_until="domcontentloaded")
        title = await self._page.title()
        current_url = self._page.url
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            data={"url": current_url, "title": title},
        )

    async def _click(self, selector: str) -> ToolResult:
        if not selector:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="selector is required for click",
            )
        await self._page.click(selector)
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            data={"clicked": selector},
        )

    async def _type(self, selector: str, text: str) -> ToolResult:
        if not selector:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="selector is required for type",
            )
        await self._page.fill(selector, text)
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            data={"typed": text, "selector": selector},
        )

    async def _screenshot(self) -> ToolResult:
        import os
        import time
        screenshot_bytes = await self._page.screenshot(full_page=True)
        os.makedirs("state/screenshots", exist_ok=True)
        path = f"state/screenshots/shot_{int(time.time())}.png"
        with open(path, "wb") as f:
            f.write(screenshot_bytes)
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            data={"path": path, "size_bytes": len(screenshot_bytes)},
        )

    async def _extract(self, selector: str | None = None) -> ToolResult:
        if selector:
            content = await self._page.inner_text(selector)
        else:
            # Get visible text only (more useful for LLM than full HTML)
            content = await self._page.inner_text("body")
        # Truncate for LLM context window
        if len(content) > 8000:
            content = content[:8000] + "\n...(truncated)"
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            data={"content": content, "length": len(content)},
        )

    async def _scroll(self, direction: str, amount: int) -> ToolResult:
        delta = amount if direction == "down" else -amount
        await self._page.mouse.wheel(0, delta)
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            data={"scrolled": direction, "amount": amount},
        )

    async def _wait(self, selector: str, timeout: int) -> ToolResult:
        if not selector:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="selector is required for wait",
            )
        await self._page.wait_for_selector(selector, timeout=timeout)
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            data={"found": selector},
        )

    async def _close(self) -> ToolResult:
        """Close the browser and cleanup."""
        if self._browser:
            await self._browser.close()
        if self._pw:
            await self._pw.stop()
        self._browser = None
        self._page = None
        self._pw = None
        logger.info("BrowserTool: browser closed")
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            data={"closed": True},
        )

    async def close(self):
        """Public close method (alias for _close without ToolResult)."""
        await self._close()
