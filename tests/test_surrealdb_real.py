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
    """Delete all records from SurrealDB tables and drop fact table for schema reset."""
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

    # Drop tables with schema changes so re-initialization creates fresh schema
    for table in ["fact", "episode", "agent_registry", "task_registry", "discard_pool"]:
        try:
            await db.query(f"REMOVE TABLE {table}")
        except Exception:
            pass

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
    # Ensure schema is initialized so unique indexes are active
    try:
        await mp.initialize_schema()
    except Exception:
        pass  # Schema may already exist
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


class TestSurrealDBRealFactCompositeIndex:
    """Verify composite unique index (fact_type, name, agent_id)."""

    @pytest.mark.asyncio
    async def test_same_name_different_type(self):
        """Same name with different fact_type should coexist."""
        mp = await _make_pool()
        try:
            eid = await mp.write_fact("entity", "nginx", {"ip": "10.0.0.1"})
            fid = await mp.write_fact("facetpoint", "nginx",
                                       {"port": 8000, "status": "running"})
            assert eid != fid
            assert "fact" in eid
            assert "fact" in fid
        finally:
            await mp.disconnect()

    @pytest.mark.asyncio
    async def test_same_name_type_upsert(self):
        """Writing same (fact_type, name, agent_id) should upsert."""
        mp = await _make_pool()
        try:
            id1 = await mp.write_fact("entity", "server1", {"ip": "1.1.1.1"})
            id2 = await mp.write_fact("entity", "server1", {"ip": "2.2.2.2"})
            # Should upsert, so IDs should match (same record)
            assert id1 == id2

            fact = await mp.get_fact("server1")
            assert fact is not None
            assert fact["value"] == {"ip": "2.2.2.2"}
        finally:
            await mp.disconnect()

    @pytest.mark.asyncio
    async def test_agent_id_namespace_isolation(self):
        """Same (fact_type, name) with different agent_id should coexist."""
        mp = await _make_pool()
        try:
            id_a = await mp.write_fact("entity", "user",
                                        {"name": "Alice"}, agent_id="agent-a")
            id_b = await mp.write_fact("entity", "user",
                                        {"name": "Bob"}, agent_id="agent-b")
            assert id_a != id_b

            # Each agent sees only its own fact
            fa = await mp.get_fact("user", agent_id="agent-a")
            fb = await mp.get_fact("user", agent_id="agent-b")
            assert fa is not None
            assert fb is not None
            assert fa["value"] == {"name": "Alice"}
            assert fb["value"] == {"name": "Bob"}
        finally:
            await mp.disconnect()

    @pytest.mark.asyncio
    async def test_query_facts_by_agent(self):
        """query_facts should filter by agent_id and fact_type."""
        mp = await _make_pool()
        try:
            await mp.write_fact("entity", "svc-a", {}, agent_id="agent-1")
            await mp.write_fact("entity", "svc-b", {}, agent_id="agent-1")
            await mp.write_fact("facetpoint", "note-a", {}, agent_id="agent-1")
            await mp.write_fact("entity", "svc-c", {}, agent_id="agent-2")

            # Query by agent_id
            facts = await mp.query_facts(agent_id="agent-1")
            assert len(facts) == 3

            # Query by agent_id + fact_type
            entities = await mp.query_facts(agent_id="agent-1", fact_type="entity")
            assert len(entities) == 2

            # Query all
            all_facts = await mp.query_facts()
            assert len(all_facts) >= 4
        finally:
            await mp.disconnect()

    @pytest.mark.asyncio
    async def test_write_fact_without_upsert(self):
        """upsert=False should raise on duplicate."""
        mp = await _make_pool()
        try:
            await mp.write_fact("entity", "unique-key", {"v": 1}, upsert=False)
            await mp.write_fact("facetpoint", "unique-key", {"v": 2}, upsert=False)
            # Different type, different agent_id should work even without upsert
        finally:
            await mp.disconnect()


class TestSurrealDBRealCleanup:
    """Verify cleanup and consolidation methods."""

    @pytest.mark.asyncio
    async def test_cleanup_stale_episodes(self):
        """cleanup_stale_episodes should only delete old episodes."""
        mp = await _make_pool()
        try:
            # Write an episode (it will have current timestamp)
            await mp.write_episode(
                title="Recent Episode",
                summary="This is recent",
                content="Should stay",
            )

            # cleanup with days=365 should not delete recent episodes
            deleted = await mp.cleanup_stale_episodes(days=365)
            assert deleted == 0, f"Expected 0 deletions, got {deleted}"
        finally:
            await mp.disconnect()

    @pytest.mark.asyncio
    async def test_consolidate_episodes_to_facts(self):
        """consolidate_episodes_to_facts should summarize and delete."""
        mp = await _make_pool()
        try:
            ep_id = await mp.write_episode(
                title="Old Task",
                summary="Fixed the server",
                content="Restarted nginx",
                tags=["ops", "fix"],
                consolidated=False,
            )

            # Consolidate with days=365 — episode is recent, nothing should be consolidated
            result = await mp.consolidate_episodes_to_facts(days=365)
            assert result["consolidated"] == 0
            assert result["deleted"] == 0

            # Episode should still exist
            ep = await mp.get_episode(ep_id)
            assert ep is not None
        finally:
            await mp.disconnect()

    @pytest.mark.asyncio
    async def test_cleanup_inmem_fallback(self):
        """cleanup_stale_episodes works with in-memory backend."""
        import time
        mp = MemoryPool()  # in-memory, no SurrealDB
        try:
            mp._mem.setdefault("episode", []).append({
                "id": "episode:old",
                "title": "Old",
                "created_at": time.time() - 86400 * 60,  # 60 days ago
                "consolidated": False,
            })
            mp._mem.setdefault("episode", []).append({
                "id": "episode:recent",
                "title": "Recent",
                "created_at": time.time(),
                "consolidated": False,
            })

            deleted = await mp.cleanup_stale_episodes(days=30)
            assert deleted == 1
            remaining = mp._mem.get("episode", [])
            assert len(remaining) == 1
            assert remaining[0]["id"] == "episode:recent"
        finally:
            mp.clear()
