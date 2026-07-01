"""Real SurrealDB integration tests — requires running SurrealDB instance.

Run with:  python3 -m pytest tests/test_surrealdb_real.py -v --tb=short
"""

import pytest
from src.memory import MemoryPool


SURREALDB_URL = "ws://localhost:8001/rpc"
SURREALDB_NS = "agent_loop"
SURREALDB_DB = "agent_loop"

# All SurrealDB tables used by MemoryPool (must stay in sync with schema.surql)
_SURREALDB_TABLES = [
    "fact", "facet", "episode", "project", "edge",
    "agent_registry", "task_registry", "discard_pool",
    "belongs_to", "contains", "part_of", "references_episode",
]


async def _do_clean_surrealdb():
    """Delete all records from SurrealDB tables (internal helper)."""
    from surrealdb import AsyncSurreal
    db = AsyncSurreal(SURREALDB_URL)
    await db.connect()
    await db.signin({"user": "root", "pass": "root"})
    await db.use(SURREALDB_NS, SURREALDB_DB)

    for table in _SURREALDB_TABLES:
        try:
            await db.query(f"DELETE FROM {table}")
        except Exception:
            pass  # Table may not exist yet (first run)

    await db.close()


@pytest.fixture(autouse=True)
def _clean_surrealdb():
    """Clean all SurrealDB tables before each test for test isolation.

    Runs before every test in this module to prevent state leakage
    between tests (e.g. unique index conflicts on idx_fact_name).

    Uses asyncio.run() because pytest-asyncio strict mode does not
    allow autouse async fixtures with sync tests.
    """
    import asyncio
    asyncio.run(_do_clean_surrealdb())
    yield


async def _make_pool():
    mp = MemoryPool(
        url=SURREALDB_URL,
        namespace=SURREALDB_NS,
        database=SURREALDB_DB,
        user="root",
        password="root",
    )
    await mp.connect()
    return mp


class TestSurrealDBRealConnect:
    """Verify MemoryPool can connect to real SurrealDB."""

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self):
        """MemoryPool should connect and disconnect cleanly."""
        mp = await _make_pool()
        assert mp._db is not None
        await mp.disconnect()
        assert mp._db is None

    @pytest.mark.asyncio
    async def test_schema_initialization(self):
        """Schema initialization should succeed without errors."""
        mp = await _make_pool()
        try:
            await mp.initialize_schema()
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise
        finally:
            await mp.disconnect()

    @pytest.mark.asyncio
    async def test_register_agent(self):
        """Register an agent in SurrealDB."""
        mp = await _make_pool()
        try:
            await mp.register_agent("test-agent-real-1", {"specialty": "testing"})
            await mp.update_agent_status("test-agent-real-1", "active")
        finally:
            await mp.disconnect()


class TestSurrealDBRealWrite:
    """Verify data can be written to and read from SurrealDB."""

    @pytest.mark.asyncio
    async def test_store_episode(self):
        """Write an episode and read it back."""
        mp = await _make_pool()
        try:
            ep_id = await mp.write_episode(
                title="Test Episode",
                summary="A real SurrealDB integration test episode",
                content="Detailed test content",
                tags=["test", "integration"],
                user_input="Run integration test",
                output="All passed",
                task_count=1,
                session_id="session-real-001",
            )
            assert ep_id is not None
            assert "episode" in ep_id

            episode = await mp.get_episode(ep_id)
            assert episode is not None
            assert episode["title"] == "Test Episode"
        finally:
            await mp.disconnect()

    @pytest.mark.asyncio
    async def test_write_fact_and_retrieve(self):
        """Write a fact and retrieve it by name."""
        mp = await _make_pool()
        try:
            fact_id = await mp.write_fact(
                fact_type="entity",
                name="test_real_fact",
                value={"key": "value"},
            )
            assert fact_id is not None

            result = await mp.get_fact("test_real_fact")
            assert result is not None
            assert result["name"] == "test_real_fact"
        finally:
            await mp.disconnect()

    @pytest.mark.asyncio
    async def test_register_and_update_task(self):
        """Register a task and update its status."""
        mp = await _make_pool()
        try:
            await mp.register_task(
                task_id="task-real-001",
                parent_id=None,
                scope="integration_test",
                priority=1,
            )
            await mp.update_task_status("task-real-001", "completed", {"ok": True})
        finally:
            await mp.disconnect()

    @pytest.mark.asyncio
    async def test_discard_pool(self):
        """Write discarded results and query them."""
        mp = await _make_pool()
        try:
            await mp.discard_result(
                agent_id="agent-x",
                task_id="task-x",
                reason="Test discard",
                result={"status": "failed"},
                agent_log=["line1", "line2"],
            )

            results = await mp.get_discarded(agent_id="agent-x")
            assert len(results) > 0
            assert results[0]["agent_id"] == "agent-x"
        finally:
            await mp.disconnect()


class TestSurrealDBRealEdgeWrite:
    """Verify graph edges can be written to SurrealDB."""

    @pytest.mark.asyncio
    async def test_write_edge_between_nodes(self):
        """Create two nodes and an edge between them."""
        mp = await _make_pool()
        try:
            fact_a = await mp.write_fact("entity", "node_a", {"val": 1})
            fact_b = await mp.write_fact("entity", "node_b", {"val": 2})
            assert fact_a is not None
            assert fact_b is not None

            edge_id = await mp.write_edge(
                source=fact_a,
                target=fact_b,
                relation="depends_on",
            )
            assert edge_id is not None
            assert "edge" in edge_id
        finally:
            await mp.disconnect()

    @pytest.mark.asyncio
    async def test_get_neighbors(self):
        """Create nodes + edges, then traverse neighbors."""
        mp = await _make_pool()
        try:
            a = await mp.write_fact("entity", "hub_a", {})
            b = await mp.write_fact("entity", "hub_b", {})
            await mp.write_edge(a, b, "linked_to")

            neighbors = await mp.get_neighbors(a)
            assert len(neighbors) > 0
        finally:
            await mp.disconnect()


class TestSurrealDBRealProjectMemory:
    """Verify MemoryPool project docs work (in-memory/SQLite fallback)."""

    @pytest.mark.asyncio
    async def test_save_load_project_docs_inmem(self):
        """Project docs use in-memory store even when SurrealDB is connected."""
        mp = MemoryPool(db_path=":memory:")
        try:
            await mp.save_project_doc("agent-real-1", "prompt", "Build a CLI tool")
            await mp.save_project_doc("agent-real-1", "plan", "1. Parse args\n2. Execute")

            docs = await mp.load_project_docs("agent-real-1")
            assert docs["prompt"] == "Build a CLI tool"
            assert docs["plan"] == "1. Parse args\n2. Execute"
        finally:
            mp.clear()
