"""Tests for Anchor Layer — P6-A: stable key-fact storage."""

import os
import pytest
import tempfile
from pathlib import Path

from src.memory.anchor import (
    AnchorEntry,
    AnchorFile,
    AnchorManager,
    ANCHOR_DIR,
    ANCHOR_AGENT_ID,
)

# ================================================================
# SurrealDB test helpers
# ================================================================

SURREALDB_URL = "ws://localhost:8001/rpc"
SURREALDB_NS = "agent_loop"
SURREALDB_DB = "agent_loop"


async def _make_anchor_db_pool():
    """Create a connected MemoryPool for anchor DB tests."""
    from src.memory import MemoryPool
    mp = MemoryPool(
        url=SURREALDB_URL,
        namespace=SURREALDB_NS,
        database=SURREALDB_DB,
        user="root",
        password="root",
    )
    await mp.connect()
    try:
        await mp.initialize_schema()
    except Exception:
        pass
    return mp


# ================================================================
# AnchorEntry
# ================================================================

def test_anchor_entry_creation():
    e = AnchorEntry(key="github", value="https://github.com/hubmover007/agent-loop", category="repository")
    assert e.key == "github"
    assert e.value == "https://github.com/hubmover007/agent-loop"
    assert e.category == "repository"


def test_anchor_entry_empty_category():
    e = AnchorEntry(key="name", value="张云飞")
    assert e.category == ""


def test_anchor_entry_general_category_markdown():
    """Entries with empty category render under ## General."""
    af = AnchorFile(
        name="test",
        title="Test",
        entries=[
            AnchorEntry(key="k1", value="v1", category="CatA"),
            AnchorEntry(key="k2", value="v2"),  # no category
        ],
    )
    md = af.to_markdown()
    assert "## CatA" in md
    assert "## General" in md
    assert "- **k2**: v2" in md


# ================================================================
# AnchorFile to_markdown / from_markdown
# ================================================================

def test_anchor_file_to_markdown():
    af = AnchorFile(
        name="system",
        title="System Anchor",
        entries=[
            AnchorEntry(key="name", value="张云飞", category="Owner"),
            AnchorEntry(key="timezone", value="Asia/Shanghai", category="System"),
        ],
    )
    md = af.to_markdown()
    assert "# System Anchor" in md
    assert "## Owner" in md
    assert "## System" in md
    assert "- **name**: 张云飞" in md
    assert "- **timezone**: Asia/Shanghai" in md


def test_anchor_file_to_markdown_no_category():
    """Entries without category still render (under ## General)."""
    af = AnchorFile(
        name="simple",
        title="Simple",
        entries=[
            AnchorEntry(key="key1", value="val1"),
            AnchorEntry(key="key2", value="val2"),
        ],
    )
    md = af.to_markdown()
    assert "- **key1**: val1" in md
    assert "- **key2**: val2" in md
    assert "## General" in md


def test_anchor_file_from_markdown():
    content = """# System Anchor

## Owner
- **name**: 张云飞
- **feishu_id**: ou_69a3ae3daba191c4b20326fbc501ef36

## System
- **host**: iZt4n3a5m5fwvch523fldgZ
- **os**: Linux 6.8.0-90-generic
"""
    af = AnchorFile.from_markdown(content, "system")
    assert af.name == "system"
    assert af.title == "System Anchor"
    assert len(af.entries) == 4
    assert af.entries[0].key == "name"
    assert af.entries[0].value == "张云飞"
    assert af.entries[0].category == "Owner"
    assert af.entries[2].key == "host"
    assert af.entries[2].category == "System"


def test_anchor_file_roundtrip():
    """from_markdown(to_markdown(af)) should be equivalent.

    Empty categories are rendered as ## General for roundtrip fidelity.
    """
    af = AnchorFile(
        name="test",
        title="Test Anchor",
        entries=[
            AnchorEntry(key="k1", value="v1", category="CatA"),
            AnchorEntry(key="k2", value="v2", category="CatB"),
            AnchorEntry(key="k3_no_cat", value="v3"),
        ],
    )
    md = af.to_markdown()
    af2 = AnchorFile.from_markdown(md, "test")
    assert af2.name == af.name
    assert af2.title == af.title
    assert len(af2.entries) == 3
    assert {(e.key, e.value, e.category) for e in af2.entries} == {
        ("k1", "v1", "CatA"),
        ("k2", "v2", "CatB"),
        ("k3_no_cat", "v3", "General"),
    }


def test_anchor_file_empty():
    af = AnchorFile(name="empty", title="Empty")
    md = af.to_markdown()
    assert "# Empty" in md
    assert "- **" not in md


def test_anchor_file_from_markdown_edge_cases():
    """Test parsing edge cases."""
    # Empty content
    af = AnchorFile.from_markdown("", "empty")
    assert af.title == ""
    assert af.entries == []

    # Only title
    af = AnchorFile.from_markdown("# Just Title", "title-only")
    assert af.title == "Just Title"
    assert af.entries == []

    # Malformed entry lines should be skipped
    content = """# Test
## Cat
- not a valid entry
- **valid**: works
- *star item*: ignored
"""
    af = AnchorFile.from_markdown(content, "test")
    assert len(af.entries) == 1
    assert af.entries[0].key == "valid"


# ================================================================
# AnchorManager — file operations
# ================================================================

def test_anchor_manager_write_and_read():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AnchorManager(base_dir=tmpdir)
        entries = [
            AnchorEntry(key="k1", value="v1", category="A"),
            AnchorEntry(key="k2", value="v2", category="B"),
        ]
        path = mgr.write_anchor("test", "Test Title", entries)
        assert os.path.exists(path)

        af = mgr.read_anchor("test")
        assert af is not None
        assert af.title == "Test Title"
        assert len(af.entries) == 2
        assert af.entries[0].key == "k1"


def test_anchor_manager_overwrite():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AnchorManager(base_dir=tmpdir)
        mgr.write_anchor("test", "First", [AnchorEntry(key="a", value="1")])
        mgr.write_anchor("test", "Second", [AnchorEntry(key="b", value="2")])

        af = mgr.read_anchor("test")
        assert af.title == "Second"
        assert len(af.entries) == 1
        assert af.entries[0].key == "b"


def test_anchor_manager_read_nonexistent():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AnchorManager(base_dir=tmpdir)
        assert mgr.read_anchor("nonexistent") is None


def test_anchor_manager_list_anchors():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AnchorManager(base_dir=tmpdir)
        mgr.write_anchor("a", "A", [AnchorEntry(key="x", value="1")])
        mgr.write_anchor("b", "B", [AnchorEntry(key="y", value="2")])

        names = mgr.list_anchors()
        assert sorted(names) == ["a", "b"]


def test_anchor_manager_list_empty():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AnchorManager(base_dir=tmpdir)
        assert mgr.list_anchors() == []


def test_anchor_manager_lookup():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AnchorManager(base_dir=tmpdir)
        mgr.write_anchor("system", "System", [
            AnchorEntry(key="name", value="张云飞", category="Owner"),
            AnchorEntry(key="host", value="myhost", category="System"),
        ])

        assert mgr.lookup("system", "name") == "张云飞"
        assert mgr.lookup("system", "host") == "myhost"
        assert mgr.lookup("system", "nonexistent") is None
        assert mgr.lookup("nonexistent", "any") is None


def test_anchor_manager_get_all_entries():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AnchorManager(base_dir=tmpdir)
        mgr.write_anchor("a", "A", [AnchorEntry(key="k1", value="v1")])
        mgr.write_anchor("b", "B", [AnchorEntry(key="k2", value="v2")])

        all_entries = mgr.get_all_entries()
        assert len(all_entries) == 2
        anchor_names = {name for name, _ in all_entries}
        assert anchor_names == {"a", "b"}


# ================================================================
# AnchorManager — DB sync (real SurrealDB)
# ================================================================

@pytest.mark.asyncio
async def test_anchor_sync_to_db():
    """Sync anchor entries to SurrealDB fact table."""
    memory = await _make_anchor_db_pool()

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AnchorManager(base_dir=tmpdir, memory_pool=memory)
        mgr.write_anchor("sync_test", "Sync Test", [
            AnchorEntry(key="k1", value="v1"),
            AnchorEntry(key="k2", value="v2"),
        ])

        count = await mgr.sync_to_db(name="sync_test")
        assert count == 2

        # Verify facts were written
        facts = await memory.query_facts(agent_id=ANCHOR_AGENT_ID, limit=100)
        synced_keys = {f["name"] for f in facts if f["name"].startswith("sync_test.")}
        assert "sync_test.k1" in synced_keys
        assert "sync_test.k2" in synced_keys

    await memory.disconnect()


@pytest.mark.asyncio
async def test_anchor_load_from_db():
    """Load anchor from SurrealDB as fallback."""
    memory = await _make_anchor_db_pool()

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AnchorManager(base_dir=tmpdir, memory_pool=memory)
        mgr.write_anchor("db_load_test", "DB Load", [
            AnchorEntry(key="foo", value="bar"),
        ])
        await mgr.sync_to_db(name="db_load_test")

        # Read back from DB
        af = await mgr.load_from_db("db_load_test")
        assert af is not None
        assert af.name == "db_load_test"
        assert len(af.entries) >= 1
        found = any(e.key == "foo" and e.value == "bar" for e in af.entries)
        assert found

    await memory.disconnect()


@pytest.mark.asyncio
async def test_anchor_sync_no_db():
    """Sync should return 0 when no DB connection."""
    mgr = AnchorManager(base_dir="state/anchors", memory_pool=None)
    count = await mgr.sync_to_db()
    assert count == 0


@pytest.mark.asyncio
async def test_anchor_load_from_db_no_db():
    """load_from_db should return None when no DB connection."""
    mgr = AnchorManager(base_dir="state/anchors", memory_pool=None)
    af = await mgr.load_from_db("test")
    assert af is None


@pytest.mark.asyncio
async def test_anchor_delete_from_db():
    """Delete anchor facts from DB."""
    memory = await _make_anchor_db_pool()

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AnchorManager(base_dir=tmpdir, memory_pool=memory)
        mgr.write_anchor("del_test", "Delete Test", [
            AnchorEntry(key="x", value="1"),
        ])
        await mgr.sync_to_db(name="del_test")

        # Delete and verify
        count = await mgr.delete_anchor_from_db("del_test")
        assert count >= 1

        # Verify gone
        af = await mgr.load_from_db("del_test")
        assert af is None

    await memory.disconnect()


@pytest.mark.asyncio
async def test_anchor_sync_all():
    """Sync all anchors at once."""
    memory = await _make_anchor_db_pool()

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AnchorManager(base_dir=tmpdir, memory_pool=memory)
        mgr.write_anchor("a", "A", [AnchorEntry(key="a1", value="va1")])
        mgr.write_anchor("b", "B", [AnchorEntry(key="b1", value="vb1")])

        count = await mgr.sync_to_db()  # sync all
        assert count == 2

    await memory.disconnect()


# ================================================================
# MainLoop Anchor Integration
# ================================================================

def test_main_loop_check_anchors_hit():
    """MainLoop._check_anchors should hit when query matches anchor name."""
    from unittest.mock import MagicMock
    from src.loop_engine.main_loop import MainLoop
    from src.memory.anchor import AnchorManager

    memory = MagicMock()
    llm = MagicMock()

    with tempfile.TemporaryDirectory() as tmpdir:
        mgr = AnchorManager(base_dir=tmpdir)
        mgr.write_anchor("system", "System", [
            AnchorEntry(key="name", value="张云飞"),
            AnchorEntry(key="host", value="iZxxx"),
        ])
        mgr.write_anchor("project", "Project", [
            AnchorEntry(key="github", value="https://github.com/x/y"),
        ])

        loop = MainLoop(memory=memory, llm=llm, anchor_manager=mgr)

        # Query mentioning anchor name
        hits = loop._check_anchors("what is the system config?")
        assert hits is not None
        assert "system" in hits

        # Query mentioning a key
        hits2 = loop._check_anchors("what is the github repo?")
        assert hits2 is not None
        assert "project" in hits2

        # Query with no match
        hits3 = loop._check_anchors("random question about weather")
        assert hits3 is None


def test_main_loop_check_anchors_no_manager():
    """_check_anchors should return None when anchor_manager is None."""
    from unittest.mock import MagicMock
    from src.loop_engine.main_loop import MainLoop

    memory = MagicMock()
    llm = MagicMock()
    loop = MainLoop(memory=memory, llm=llm, anchor_manager=None)

    assert loop._check_anchors("system config") is None


def test_main_loop_constructor_with_anchor_manager():
    """MainLoop should accept and store anchor_manager."""
    from unittest.mock import MagicMock
    from src.loop_engine.main_loop import MainLoop
    from src.memory.anchor import AnchorManager

    memory = MagicMock()
    llm = MagicMock()
    mgr = AnchorManager(base_dir="state/anchors")
    loop = MainLoop(memory=memory, llm=llm, anchor_manager=mgr)

    assert loop.anchor_manager is mgr


def test_main_loop_constructor_no_anchor_manager():
    """MainLoop should work without anchor_manager (optional)."""
    from unittest.mock import MagicMock
    from src.loop_engine.main_loop import MainLoop

    memory = MagicMock()
    llm = MagicMock()
    loop = MainLoop(memory=memory, llm=llm)

    assert loop.anchor_manager is None


# ================================================================
# Default Anchor Files
# ================================================================

def test_default_system_anchor_exists():
    """Default system.md anchor file should exist and be valid."""
    mgr = AnchorManager(base_dir=ANCHOR_DIR)
    af = mgr.read_anchor("system")
    assert af is not None, "state/anchors/system.md should exist"
    assert af.title == "System Anchor"
    assert len(af.entries) >= 4

    keys = {e.key for e in af.entries}
    assert "name" in keys
    assert "host" in keys
    assert "os" in keys


def test_default_agent_loop_anchor_exists():
    """Default agent-loop.md anchor file should exist and be valid."""
    mgr = AnchorManager(base_dir=ANCHOR_DIR)
    af = mgr.read_anchor("agent-loop")
    assert af is not None, "state/anchors/agent-loop.md should exist"
    assert af.title == "Agent-Loop Project"

    keys = {e.key for e in af.entries}
    assert "github" in keys
    assert "local_path" in keys
    assert "language" in keys


def test_default_anchors_parse_correctly():
    """Both default anchor files should roundtrip through markdown."""
    mgr = AnchorManager(base_dir=ANCHOR_DIR)
    for name in mgr.list_anchors():
        af = mgr.read_anchor(name)
        assert af is not None
        md = af.to_markdown()
        af2 = AnchorFile.from_markdown(md, name)
        assert af2.title == af.title
        assert len(af2.entries) == len(af.entries)


# ================================================================
# CLI integration smoke test
# ================================================================

def test_cli_anchor_list():
    """Smoke test: agent-loop anchor list should not crash."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "src/cli.py", "anchor", "list"],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "system.md" in result.stdout or "system" in result.stdout


def test_cli_anchor_show():
    """Smoke test: agent-loop anchor show system."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "src/cli.py", "anchor", "show", "system"],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "System Anchor" in result.stdout


def test_cli_anchor_lookup():
    """Smoke test: agent-loop anchor lookup system name."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "src/cli.py", "anchor", "lookup", "system", "name"],
        cwd=str(Path(__file__).parent.parent),
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "system.name" in result.stdout
    assert "张云飞" in result.stdout
