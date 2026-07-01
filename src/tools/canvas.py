"""Canvas Tool — visual rendering for agents.

Agents can render:
- HTML pages (custom visualizations)
- Charts (via Chart.js CDN)
- Diagrams (via Mermaid.js CDN)
- Tables (HTML tables)
- Markdown (simple conversion to HTML)

Design: self-contained HTML files saved to state/canvas/.
Each file is served via a /canvas/{id} web route.
No external Python dependencies — all rendering uses CDN JavaScript.

Alternative approaches considered:
- OpenClaw Canvas: requires OpenClaw Bridge dependency (tight coupling)
- Mermaid CLI: requires Node.js + mermaid-cli (additional system deps)
- Server-side rendering: heavier, more complex

Chose self-contained HTML files for simplicity and zero deps.
"""

from __future__ import annotations
import os
import re
import time
import logging
import json as _json
from typing import Any

from ..core import ToolResult, ToolResultStatus
from .base import ToolInterface

logger = logging.getLogger(__name__)


class CanvasTool(ToolInterface):
    """Visual canvas for agent output.

    Actions:
    - render_html(html): Render raw HTML snippet
    - render_chart(data, chart_type): Render Chart.js chart (bar/line/pie/doughnut/radar)
    - render_diagram(mermaid): Render Mermaid.js diagram
    - render_table(headers, rows): Render HTML table from structured data
    - render_markdown(markdown): Render Markdown text as styled HTML

    Output files: state/canvas/{id}.html
    """

    name = "canvas"
    description = (
        "Render visual content: HTML, charts (Chart.js), diagrams (Mermaid), "
        "tables, markdown"
    )

    CHART_TYPES = {"bar", "line", "pie", "doughnut", "radar", "scatter", "bubble"}

    def __init__(self, output_dir: str = "state/canvas"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    async def execute(self, **kwargs) -> ToolResult:
        action = kwargs.get("action", "render_html")

        try:
            if action == "render_html":
                return self._render_html(kwargs.get("html", ""))
            elif action == "render_chart":
                chart_type = kwargs.get("chart_type", "bar")
                if chart_type not in self.CHART_TYPES:
                    return ToolResult(
                        status=ToolResultStatus.FATAL_ERROR,
                        error=f"Unknown chart_type: {chart_type}. "
                              f"Supported: {', '.join(sorted(self.CHART_TYPES))}",
                    )
                return self._render_chart(
                    kwargs.get("data", {}),
                    chart_type,
                    kwargs.get("title", "Chart"),
                )
            elif action == "render_diagram":
                return self._render_diagram(kwargs.get("mermaid", ""))
            elif action == "render_table":
                headers = kwargs.get("headers", [])
                rows = kwargs.get("rows", [])
                title = kwargs.get("title", "Table")
                return self._render_table(headers, rows, title)
            elif action == "render_markdown":
                return self._render_markdown(
                    kwargs.get("markdown", ""),
                    kwargs.get("title", "Document"),
                )
            else:
                return ToolResult(
                    status=ToolResultStatus.FATAL_ERROR,
                    error=f"Unknown canvas action: {action}",
                )
        except Exception as e:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error=f"CanvasTool.{action}: {e}",
            )

    # ── helpers ──────────────────────────────────────────────

    def _canvas_id(self) -> str:
        return f"canvas_{int(time.time() * 1000)}"

    def _save(self, filename: str, html: str) -> ToolResult:
        path = os.path.join(self.output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        canvas_id = filename.replace(".html", "")
        return ToolResult(
            status=ToolResultStatus.SUCCESS,
            data={
                "canvas_id": canvas_id,
                "path": path,
                "url": f"/canvas/{canvas_id}",
            },
        )

    # ── renderers ────────────────────────────────────────────

    def _render_html(self, html: str) -> ToolResult:
        full = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Canvas</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           max-width: 960px; margin: 40px auto; padding: 0 20px; line-height: 1.6;
           color: #1a1a2e; background: #fafafa; }}
    pre {{ background: #f0f0f5; padding: 12px; border-radius: 6px; overflow-x: auto; }}
    code {{ font-family: "Fira Code", "Cascadia Code", monospace; font-size: 0.9em; }}
  </style>
</head>
<body>{html}</body>
</html>"""
        return self._save(f"{self._canvas_id()}.html", full)

    def _render_chart(self, data: dict, chart_type: str, title: str) -> ToolResult:
        """Render a Chart.js chart.

        Args:
            data: Chart.js data object {labels: [...], datasets: [{...}]}
            chart_type: bar|line|pie|doughnut|radar|scatter|bubble
            title: chart title
        """
        # Optional: accept a simpler data format and convert
        data_json = _json.dumps(data)
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{self._escape(title)}</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #fafafa; display: flex; justify-content: center;
           align-items: center; min-height: 100vh; margin: 0; }}
    .chart-container {{ background: white; padding: 30px; border-radius: 12px;
                        box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: 90%; max-width: 700px; }}
    h2 {{ text-align: center; color: #333; margin-top: 0; }}
  </style>
</head>
<body>
  <div class="chart-container">
    <h2>{self._escape(title)}</h2>
    <canvas id="chart"></canvas>
  </div>
  <script>
    new Chart(document.getElementById('chart'), {{
      type: '{chart_type}',
      data: {data_json},
      options: {{
        responsive: true,
        plugins: {{ legend: {{ position: 'bottom' }} }},
      }},
    }});
  </script>
</body>
</html>"""
        return self._save(f"{self._canvas_id()}.html", html)

    def _render_diagram(self, mermaid: str) -> ToolResult:
        """Render a Mermaid.js diagram.

        Mermaid syntax:
          flowchart TD
            A[Start] --> B{Decision}
            B -->|Yes| C[Do this]
            B -->|No| D[Do that]

        Also supports: sequenceDiagram, classDiagram, stateDiagram, gantt,
                       pie, gitGraph, mindmap, timeline
        """
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Diagram</title>
  <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #fafafa; display: flex; justify-content: center;
           padding: 30px; margin: 0; }}
    .diagram-container {{ background: white; padding: 30px; border-radius: 12px;
                          box-shadow: 0 2px 12px rgba(0,0,0,0.08); width: 90%; max-width: 900px; }}
    .mermaid {{ text-align: center; }}
  </style>
</head>
<body>
  <div class="diagram-container">
    <div class="mermaid">{mermaid}</div>
  </div>
  <script>mermaid.initialize({{ startOnLoad: true, theme: 'default' }});</script>
</body>
</html>"""
        return self._save(f"{self._canvas_id()}.html", html)

    def _render_table(self, headers: list, rows: list, title: str) -> ToolResult:
        if not headers:
            return ToolResult(
                status=ToolResultStatus.FATAL_ERROR,
                error="headers is required for table rendering",
            )
        header_html = "".join(f"<th>{self._escape(str(h))}</th>" for h in headers)
        rows_html = ""
        for row in rows:
            cells = "".join(f"<td>{self._escape(str(c))}</td>" for c in row)
            rows_html += f"<tr>{cells}</tr>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{self._escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           background: #fafafa; padding: 30px; margin: 0; }}
    .table-container {{ background: white; padding: 30px; border-radius: 12px;
                        box-shadow: 0 2px 12px rgba(0,0,0,0.08); max-width: 1000px; margin: 0 auto; }}
    h2 {{ color: #333; margin-top: 0; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ background: #4a6cf7; color: white; padding: 12px 16px; text-align: left;
          font-weight: 600; }}
    td {{ padding: 10px 16px; border-bottom: 1px solid #eee; }}
    tr:hover td {{ background: #f8f9ff; }}
  </style>
</head>
<body>
  <div class="table-container">
    <h2>{self._escape(title)}</h2>
    <table>
      <thead><tr>{header_html}</tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
  </div>
</body>
</html>"""
        return self._save(f"{self._canvas_id()}.html", html)

    def _render_markdown(self, md: str, title: str = "Document") -> ToolResult:
        """Convert simple Markdown to styled HTML.

        Supports: h1-h3, bold, italic, code, inline code, unordered lists,
                  links, blockquotes, horizontal rules, paragraphs.
        For full CommonMark support, consider markdown-it-py or mistune.
        """
        html = self._escape(md)

        # Headers (before other transforms)
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)

        # Horizontal rules
        html = re.sub(r'^---+\s*$', r'<hr>', html, flags=re.MULTILINE)

        # Bold and italic
        html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', html)
        html = re.sub(r'\*(.+?)\*', r'<i>\1</i>', html)

        # Inline code
        html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

        # Links [text](url)
        html = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', html)

        # Blockquotes
        html = re.sub(r'^&gt; (.+)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)

        # Unordered lists
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        # Wrap consecutive <li> in <ul>
        html = re.sub(r'(<li>.*?</li>\n?)+', self._wrap_ul, html)

        # Paragraphs: blank-line separated blocks not already in HTML tags
        html = re.sub(r'\n\n+', r'</p><p>', html)
        html = f"<p>{html}</p>"
        # Clean up empty <p></p>
        html = re.sub(r'<p>\s*</p>', '', html)

        # H1 block shouldn't be inside <p>
        html = re.sub(r'<p><h([123])>(.+?)</h\1></p>', r'<h\1>\2</h\1>', html)

        full = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
           max-width: 800px; margin: 40px auto; padding: 0 20px; line-height: 1.7;
           color: #1a1a2e; background: #fafafa; }}
    h1 {{ font-size: 2em; border-bottom: 2px solid #4a6cf7; padding-bottom: 0.3em; }}
    h2 {{ font-size: 1.5em; margin-top: 1.5em; }}
    h3 {{ font-size: 1.2em; }}
    pre {{ background: #1e1e2e; color: #cdd6f4; padding: 16px; border-radius: 8px;
          overflow-x: auto; }}
    code {{ font-family: "Fira Code", "Cascadia Code", monospace; font-size: 0.9em;
           background: #e8e8f0; padding: 2px 6px; border-radius: 4px; }}
    pre code {{ background: none; padding: 0; }}
    blockquote {{ border-left: 4px solid #4a6cf7; margin: 1em 0; padding: 0.5em 1em;
                 background: #f0f3ff; border-radius: 0 8px 8px 0; }}
    hr {{ border: none; border-top: 1px solid #ddd; margin: 2em 0; }}
    a {{ color: #4a6cf7; }}
    li {{ margin: 0.3em 0; }}
  </style>
</head>
<body>{html}</body>
</html>"""
        return self._save(f"{self._canvas_id()}.html", full)

    @staticmethod
    def _wrap_ul(match: re.Match) -> str:
        return f"<ul>\n{match.group(0)}</ul>\n"

    @staticmethod
    def _escape(text: str) -> str:
        """Escape HTML special characters."""
        return (
            text.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
        )
