"""Tests for web UI endpoints (static files, traces, metrics)."""

import pytest
from unittest.mock import MagicMock, AsyncMock


@pytest.fixture
def web_client():
    """Create httpx test client for the full web app."""
    import httpx
    from src.web import create_app
    from src.loop_engine import LoopConfig
    from src.streaming import Tracer

    mem = MagicMock()
    mem.store = AsyncMock(return_value="memory:ep:test")
    llm = MagicMock()
    llm.provider_name = "mock-llm"
    llm.chat = AsyncMock(return_value=MagicMock(content="{}"))

    tracer = Tracer()
    config = LoopConfig()
    app = create_app(memory=mem, llm=llm, config=config)
    # Inject tracer into main_loop
    app.state.main_loop.tracer = tracer

    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    return client


@pytest.mark.asyncio
async def test_index_html_returns_200(web_client):
    """Static file serving returns index.html at /."""
    async with web_client as client:
        resp = await client.get("/")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_index_html_is_html(web_client):
    """Index page returns text/html content-type."""
    async with web_client as client:
        resp = await client.get("/")
        content_type = resp.headers.get("content-type", "")
        assert "text/html" in content_type


@pytest.mark.asyncio
async def test_app_js_served(web_client):
    """JavaScript file is served from /app.js."""
    async with web_client as client:
        resp = await client.get("/app.js")
        assert resp.status_code == 200
        text = resp.text
        assert "Agent-Loop Console" in text or "apiGet" in text


@pytest.mark.asyncio
async def test_style_css_served(web_client):
    """CSS file is served from /style.css."""
    async with web_client as client:
        resp = await client.get("/style.css")
        assert resp.status_code == 200
        text = resp.text
        assert "sidebar" in text or "dark" in text.lower()


@pytest.mark.asyncio
async def test_api_traces_endpoint_returns_200(web_client):
    """GET /api/traces returns 200 OK."""
    async with web_client as client:
        resp = await client.get("/api/traces")
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_api_traces_endpoint_with_data(web_client):
    """GET /api/traces returns trace data when tracer has spans."""
    async with web_client as client:
        # First, add a trace via the main_loop's tracer
        tracer = web_client._transport.app.state.main_loop.tracer
        span = tracer.start_span("test.op", task="test")
        tracer.end_span(span, "ok")

        resp = await client.get("/api/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert "traces" in data
        assert len(data["traces"]) > 0
        assert data["traces"][0]["name"] == "test.op"


@pytest.mark.asyncio
async def test_api_traces_endpoint_empty(web_client):
    """GET /api/traces returns empty list when no traces."""
    # Clear any existing traces
    tracer = web_client._transport.app.state.main_loop.tracer
    tracer.clear()

    async with web_client as client:
        resp = await client.get("/api/traces")
        assert resp.status_code == 200
        data = resp.json()
        assert data["traces"] == []


@pytest.mark.asyncio
async def test_api_info_endpoint(web_client):
    """GET /api returns API info."""
    async with web_client as client:
        resp = await client.get("/api")
        assert resp.status_code == 200
        data = resp.json()
        assert "endpoints" in data
        assert "name" in data


@pytest.mark.asyncio
async def test_metrics_endpoint_in_web_ui(web_client):
    """GET /metrics returns prometheus metrics."""
    async with web_client as client:
        resp = await client.get("/metrics")
        assert resp.status_code == 200
        content_type = resp.headers.get("content-type", "")
        assert "text/plain" in content_type
