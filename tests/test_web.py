"""Tests for the Web API (src/web/__init__.py).

Uses httpx.ASGITransport + AsyncClient for transport-level testing,
avoiding starlette/httpx version mismatches.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_memory():
    """Create a mock MemoryPool."""
    mem = MagicMock()
    mem.store = AsyncMock(return_value="memory:ep:test")
    return mem


@pytest.fixture
def mock_llm():
    """Create a mock LLMProvider."""
    llm = MagicMock()
    llm.provider_name = "mock-llm"
    llm.chat = AsyncMock(return_value=MagicMock(content="{}"))
    return llm


@pytest.fixture
def create_app_client(mock_memory, mock_llm):
    """Return a factory for httpx.AsyncClient bound to the test app."""
    import httpx
    from src.web import create_app
    from src.loop_engine import LoopConfig

    config = LoopConfig()
    app = create_app(memory=mock_memory, llm=mock_llm, config=config)
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_status_endpoint(create_app_client):
    """GET /api/status returns system status."""
    async with create_app_client as client:
        resp = await client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "loop_phase" in data
        assert "active_loops" in data
        assert "model" in data


@pytest.mark.asyncio
async def test_list_tasks(create_app_client):
    """GET /api/tasks returns task list."""
    async with create_app_client as client:
        resp = await client.get("/api/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert "tasks" in data
        assert isinstance(data["tasks"], list)


@pytest.mark.asyncio
async def test_get_task_no_registry(create_app_client):
    """GET /api/tasks/{id} returns 503 when no task registry exists."""
    async with create_app_client as client:
        resp = await client.get("/api/tasks/nonexistent-task-id")
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data


@pytest.mark.asyncio
async def test_list_agents(create_app_client):
    """GET /api/agents returns agent list."""
    async with create_app_client as client:
        resp = await client.get("/api/agents")
        assert resp.status_code == 200
        data = resp.json()
        assert "agents" in data
        assert isinstance(data["agents"], list)


@pytest.mark.asyncio
async def test_cancel_task_no_manager(create_app_client):
    """POST /api/tasks/{id}/cancel without a manager returns 503."""
    async with create_app_client as client:
        resp = await client.post("/api/tasks/test-task/cancel")
        assert resp.status_code == 503
        data = resp.json()
        assert "error" in data


@pytest.mark.asyncio
async def test_chat_submit(create_app_client):
    """POST /api/chat submits user input and returns response."""
    with patch("src.web.MainLoop.run", new_callable=AsyncMock) as mock_run:
        from src.loop_engine.main_loop import LoopContext
        ctx = LoopContext(
            session_id="test-session",
            user_input="Hello",
        )
        ctx.final_output = "Hello, world!"
        ctx.task_ids = ["task:1", "task:2"]
        ctx.agent_results = [MagicMock(), MagicMock()]
        ctx.discarded_results = []
        ctx.errors = []
        mock_run.return_value = ctx

        async with create_app_client as client:
            resp = await client.post("/api/chat", json={"message": "Hello"})
            assert resp.status_code == 200
            data = resp.json()
            assert "session_id" in data
            assert "output" in data
            assert data["tasks_created"] == 2
            assert data["tasks_done"] == 2


@pytest.mark.asyncio
async def test_not_found(create_app_client):
    """GET /api/nonexistent returns 404."""
    async with create_app_client as client:
        resp = await client.get("/api/nonexistent")
        assert resp.status_code == 404


@pytest.mark.asyncio
async def test_root_endpoint(create_app_client):
    """GET / returns API info."""
    async with create_app_client as client:
        resp = await client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "Agent-Loop API"
        assert "endpoints" in data
