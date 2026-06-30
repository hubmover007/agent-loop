"""Memory Pool - shared memory for all agents.

Four-layer inverted cone topology (M-FLOW inspired):
  Layer 0: Facts (Entity + FacetPoint) - 锥尖
  Layer 1: Facets - 语义维度
  Layer 2: Episodes - 事件上下文
  Layer 3: Projects - 项目全景

All layers connected by semantic edges with vectorized descriptions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from surrealdb import Surreal

from ..core import MemoryLayer

logger = logging.getLogger(__name__)

# Default embedding dimension (configurable)
DEFAULT_EMBEDDING_DIM = 1024


class MemoryPool:
    """Unified memory pool backed by SurrealDB.

    All agents read/write through this single pool.
    No copies, no sync - shared access through SurrealDB.

    Falls back to SQLite or in-memory dict when SurrealDB is not connected.
    """

    _DB_SCHEMA = (
        "CREATE TABLE IF NOT EXISTS memories ("
        "  id TEXT PRIMARY KEY,"
        "  agent_id TEXT,"
        "  content TEXT,"
        "  metadata TEXT,"
        "  created_at REAL"
        ")"
    )

    def __init__(self, url: str = "ws://127.0.0.1:8000", namespace: str = "agent_loop",
                 database: str = "memory", user: str = "root", password: str = "root",
                 db_path: str | None = None):
        self.url = url
        self.namespace = namespace
        self.database = database
        self._auth = {"user": user, "pass": password}
        self._db: Surreal | None = None
        self._embedding_fn: Any = None  # Set via configure_embedding()
        self._mem: dict[str, list[dict]] = {}  # in-memory fallback
        self._db_path: str | None = db_path
        self._sqlite: Any = None  # sqlite3.Connection when db_path is set
        if db_path:
            self._init_sqlite(db_path)

    async def connect(self) -> None:
        """Connect to SurrealDB and initialize schema."""
        self._db = Surreal(self.url)
        await self._db.signin(self._auth)
        await self._db.use(self.namespace, self.database)
        logger.info("MemoryPool connected to %s", self.url)

    async def disconnect(self) -> None:
        """Disconnect from SurrealDB."""
        if self._db:
            await self._db.close()
            self._db = None
        if self._sqlite:
            self._sqlite.close()
            self._sqlite = None

    def _init_sqlite(self, db_path: str) -> None:
        """Initialize SQLite backend and create schema."""
        import sqlite3
        self._sqlite = sqlite3.connect(db_path, check_same_thread=False)
        self._sqlite.execute(self._DB_SCHEMA)
        self._sqlite.commit()
        logger.info("MemoryPool: SQLite backend initialized at %s", db_path)

    def _save_to_db(self, key: str, entry: dict) -> None:
        """Persist a single memory entry to SQLite."""
        if not self._sqlite:
            return
        import json as _json
        import time
        self._sqlite.execute(
            "INSERT OR REPLACE INTO memories (id, agent_id, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                key,
                entry.get("agent_id", ""),
                entry.get("content", _json.dumps(entry, default=str)),
                _json.dumps(entry.get("metadata", {}), default=str),
                entry.get("created_at", time.time()),
            ),
        )
        self._sqlite.commit()

    def _load_from_db(self) -> dict[str, list[dict]]:
        """Load all memories from SQLite."""
        if not self._sqlite:
            return {}
        import json as _json
        rows = self._sqlite.execute("SELECT id, agent_id, content, metadata, created_at FROM memories").fetchall()
        result: dict[str, list[dict]] = {}
        for row in rows:
            entry_id, agent_id, content, metadata, created_at = row
            try:
                entry = _json.loads(content)
            except (json.JSONDecodeError, TypeError):
                entry = {"content": content}
            entry["id"] = entry_id
            entry["agent_id"] = agent_id
            entry["created_at"] = created_at
            try:
                entry["metadata"] = _json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                entry["metadata"] = {}
            # Group by collection (extracted from id pattern 'collection:N')
            collection = entry_id.split(":")[0] if ":" in entry_id else "default"
            result.setdefault(collection, []).append(entry)
        return result

    async def initialize_schema(self) -> None:
        """Load and execute the schema definition."""
        import os
        schema_path = os.path.join(os.path.dirname(__file__), "schema.surql")
        with open(schema_path) as f:
            schema_sql = f.read()

        # Execute each statement separately
        for statement in schema_sql.split(";\n"):
            statement = statement.strip()
            if statement and not statement.startswith("--"):
                try:
                    await self._db.query(statement)
                except Exception as e:
                    # Ignore "already exists" errors
                    if "already exists" not in str(e).lower():
                        logger.warning("Schema statement failed: %s", e)

        logger.info("MemoryPool schema initialized")

    def configure_embedding(self, fn: Any) -> None:
        """Set the embedding function used for vectorization.

        Args:
            fn: Callable that takes text and returns list[float].
                e.g. lambda text: openai.Embedding.create(input=text, model="text-embedding-3-small")
        """
        self._embedding_fn = fn

    async def embed(self, text: str) -> list[float]:
        """Vectorize text using configured embedding function."""
        if self._embedding_fn is None:
            raise RuntimeError("Embedding function not configured. Call configure_embedding() first.")
        result = self._embedding_fn(text)
        if hasattr(result, "data") and hasattr(result.data, "__getitem__"):
            # OpenAI-style response
            return result.data[0].embedding
        return list(result)

    # ============================================================
    # Memory Compression
    # ============================================================

    async def maybe_compress(self, agent_id: str,
                            threshold: int = 100,
                            keep_recent: int = 20) -> str | None:
        """If an agent's memory exceeds the threshold, compress old entries into a summary.

        Steps:
          1. Retrieve all memory entries for the agent
          2. If count <= threshold, return None (no compression needed)
          3. Take the oldest (total - keep_recent) entries
          4. Generate a summary from their content
          5. Delete old entries, save the summary
          6. Return the summary text

        Args:
            agent_id: The agent whose memories to compress
            threshold: Maximum number of memories before triggering compression
            keep_recent: Number of most recent memories to keep uncompressed

        Returns:
            The compression summary string, or None if no compression was needed.
        """
        collection_key = f"agent_{agent_id}"
        memories = self._mem.get(collection_key, [])

        if len(memories) <= threshold:
            return None

        # Separate old and recent
        split_idx = len(memories) - keep_recent
        old_memories = memories[:split_idx]
        recent_memories = memories[split_idx:]

        # Generate summary from old memories
        summary_lines = ["## 压缩记忆"]
        for m in old_memories:
            content = str(m.get("content", ""))[:100]
            if content:
                summary_lines.append(f"- {content}")
        summary = "\n".join(summary_lines)

        # Save summary as a compressed entry
        import time
        summary_entry = {
            "content": summary,
            "metadata": {"type": "compression_summary", "compressed_count": len(old_memories)},
            "agent_id": agent_id,
            "created_at": time.time(),
        }

        # Persist to SQLite if available
        if self._sqlite:
            import json as _json
            for m in old_memories:
                mid = m.get("id", "")
                if mid:
                    self._sqlite.execute("DELETE FROM memories WHERE id = ?", (mid,))
            self._sqlite.commit()
            # Save the summary
            self._sqlite.execute(
                "INSERT OR REPLACE INTO memories (id, agent_id, content, metadata, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    f"{collection_key}:compressed:{int(time.time())}",
                    agent_id,
                    _json.dumps(summary_entry, default=str),
                    _json.dumps(summary_entry.get("metadata", {}), default=str),
                    time.time(),
                ),
            )
            self._sqlite.commit()

        # Update in-memory cache
        self._mem[collection_key] = recent_memories + [summary_entry]

        logger.info(
            "MemoryPool: compressed %d memories for %s (threshold=%d, kept=%d)",
            len(old_memories), agent_id, threshold, keep_recent,
        )
        return summary

    # ============================================================
    # Generic memory save/load (for project docs, etc.)
    # ============================================================

    async def save(self, agent_id: str, content: str,
                   metadata: dict | None = None) -> str:
        """Save a generic memory entry for an agent.

        Used by project docs and other general-purpose memory.
        Returns the entry key.
        """
        import time as _time
        key = f"agent_{agent_id}:mem:{int(_time.time() * 1000)}"
        entry = {
            "id": key,
            "agent_id": agent_id,
            "content": content,
            "metadata": metadata or {},
            "created_at": _time.time(),
        }
        collection = f"agent_{agent_id}"
        self._mem.setdefault(collection, []).append(entry)
        self._save_to_db(key, entry)
        return key

    def _get_by_agent(self, agent_id: str) -> list[dict]:
        """Get all memory entries for a given agent.

        Returns entries from either in-memory store or SQLite.
        """
        collection = f"agent_{agent_id}"
        if collection in self._mem:
            return list(self._mem[collection])
        # Try to load from SQLite
        if self._sqlite:
            loaded = self._load_from_db()
            if collection in loaded:
                self._mem[collection] = loaded[collection]
                return list(self._mem[collection])
        return []

    async def save_project_doc(self, agent_id: str, doc_type: str,
                               content: str) -> None:
        """Save a project-level document.

        doc_type: prompt / plan / implement / documentation
        """
        await self.save(agent_id, content, {
            "type": "project_doc",
            "doc_type": doc_type,
        })
        logger.debug("MemoryPool: saved project doc '%s' for %s", doc_type, agent_id)

    async def load_project_docs(self, agent_id: str) -> dict[str, str]:
        """Load all project documents for an agent.

        Returns dict of doc_type → content.
        """
        memories = self._get_by_agent(agent_id)
        docs: dict[str, str] = {}
        for m in memories:
            meta = m.get("metadata", {})
            if meta.get("type") == "project_doc":
                docs[meta["doc_type"]] = m.get("content", "")
        return docs

    async def write_fact(self, fact_type: str, name: str, value: Any = None,
                         embedding_text: str | None = None) -> str:
        """Write a Fact (entity or facetpoint) to Layer 0."""
        embedding = await self.embed(embedding_text or name) if self._embedding_fn else None
        record = {
            "fact_type": fact_type,
            "name": name,
            "value": value,
            "embedding": embedding or [],
        }
        if self._db:
            result = await self._db.create("fact", record)
            return result["id"]
        else:
            self._mem.setdefault("fact", []).append(record)
            return f"fact:{len(self._mem['fact'])}"

    async def write_facet(self, name: str, description: str) -> str:
        """Write a Facet to Layer 1."""
        embedding = await self.embed(description) if self._embedding_fn else None
        result = await self._db.create("facet", {
            "name": name,
            "description": description,
            "embedding": embedding or [],
        })
        return result["id"]

    async def write_episode(self, title: str, summary: str, content: str = "",
                            tags: list[str] | None = None,
                            user_input: str = "", output: str = "",
                            task_count: int = 0, session_id: str = "",
                            consolidated: bool = False) -> str:
        """Write an Episode to Layer 2."""
        embedding = await self.embed(summary) if self._embedding_fn else None
        record = {
            "title": title,
            "summary": summary,
            "content": content,
            "embedding": embedding or [],
            "tags": tags or [],
            "type": "episode",
            "user_input": user_input,
            "output": output,
            "task_count": task_count,
            "session_id": session_id,
            "consolidated": consolidated,
        }
        if self._db:
            result = await self._db.create("episode", record)
            return result["id"]
        else:
            self._mem.setdefault("episode", []).append(record)
            ep_id = f"episode:{len(self._mem['episode'])}"
            record["id"] = ep_id
            return ep_id

    async def write_project(self, name: str, description: str) -> str:
        """Write a Project to Layer 3."""
        embedding = await self.embed(description) if self._embedding_fn else None
        result = await self._db.create("project", {
            "name": name,
            "description": description,
            "embedding": embedding or [],
        })
        return result["id"]

    async def write_edge(self, source: str, target: str, relation: str) -> str:
        """Write a semantic edge between two nodes."""
        embedding = await self.embed(relation) if self._embedding_fn else None
        result = await self._db.create("edge", {
            "source_id": source,
            "target_id": target,
            "relation": relation,
            "embedding": embedding or [],
        })
        return result["id"]

    # ============================================================
    # Read Operations
    # ============================================================

    async def get_fact(self, name: str) -> dict | None:
        """Get a Fact by name."""
        if self._db:
            result = await self._db.query(
                "SELECT * FROM fact WHERE name = $name LIMIT 1",
                {"name": name}
            )
            rows = result if isinstance(result, list) else result.get("result", [])
            return rows[0] if rows else None
        else:
            for r in self._mem.get("fact", []):
                if r.get("name") == name:
                    return r
            return None

    async def get_facet(self, name: str) -> dict | None:
        """Get a Facet by name."""
        result = await self._db.query(
            "SELECT * FROM facet WHERE name = $name LIMIT 1",
            {"name": name}
        )
        rows = result if isinstance(result, list) else result.get("result", [])
        return rows[0] if rows else None

    async def get_episode(self, episode_id: str) -> dict | None:
        """Get an Episode by ID."""
        if self._db:
            return await self._db.select(episode_id)
        else:
            for r in self._mem.get("episode", []):
                if r.get("id") == episode_id:
                    return r
            return None

    async def store(self, data: dict) -> str:
        """Generic store: accepts a dict with 'type' key, routes to appropriate writer.

        Supported types: 'episode', 'fact'.

        Note: creates a shallow copy of data to avoid mutating the caller's dict.
        """
        data_copy = dict(data)
        record_type = data_copy.pop("type", "episode")
        if record_type == "episode":
            return await self.write_episode(
                title=data_copy.pop("title", data_copy.get("user_input", "")[:80]),
                summary=data_copy.pop("summary", data_copy.get("output", "")),
                content=data_copy.pop("content", ""),
                tags=data_copy.pop("tags", None),
                user_input=data_copy.pop("user_input", ""),
                output=data_copy.pop("output", ""),
                task_count=data_copy.pop("task_count", 0),
                session_id=data_copy.pop("session_id", ""),
                consolidated=data_copy.pop("consolidated", False),
            )
        elif record_type == "fact":
            return await self.write_fact(
                fact_type=data_copy.pop("fact_type", "entity"),
                name=data_copy.pop("name", ""),
                value=data_copy.pop("value", None),
                embedding_text=data_copy.pop("embedding_text", None),
            )
        else:
            raise ValueError(f"Unknown record type: {record_type}")

    async def get_unconsolidated_episodes(self, limit: int = 50) -> list[dict]:
        """Get episodes that haven't been consolidated yet."""
        if self._db:
            try:
                result = await self._db.query(
                    "SELECT * FROM episode WHERE consolidated = false OR consolidated IS NONE LIMIT $limit",
                    {"limit": limit}
                )
                return result if isinstance(result, list) else result.get("result", [])
            except Exception as e:
                logger.warning("Query unconsolidated episodes failed: %s", e)
                return []
        else:
            return [r for r in self._mem.get("episode", [])
                    if not r.get("consolidated", False)][:limit]

    async def mark_episode_consolidated(self, episode_id: str) -> None:
        """Mark an episode as consolidated."""
        if self._db:
            try:
                await self._db.query(
                    "UPDATE episode SET consolidated = true WHERE id = $id",
                    {"id": episode_id}
                )
            except Exception as e:
                logger.warning("Mark episode consolidated failed: %s", e)
        else:
            for r in self._mem.get("episode", []):
                if r.get("id") == episode_id:
                    r["consolidated"] = True
                    break

    # ============================================================
    # Graph Traversal
    # ============================================================

    async def get_edges_from(self, node_id: str, max_hops: int = 1) -> list[dict]:
        """Get all edges and connected nodes from a given node."""
        result = await self._db.query(f"""
            SELECT *, ->edge->{MemoryLayer.EPISODE.value} AS episodes
            FROM {node_id}
            LIMIT {max_hops * 10}
        """)
        return result if isinstance(result, list) else result.get("result", [])

    async def get_neighbors(self, node_id: str) -> list[dict]:
        """Get 1-hop neighbors of a node via edges."""
        result = await self._db.query(f"""
            SELECT out.* as target, relation, embedding
            FROM edge
            WHERE source_id = {node_id}
        """)
        return result if isinstance(result, list) else result.get("result", [])

    # ============================================================
    # Agent/Task registry operations
    # ============================================================

    async def register_agent(self, agent_id: str, expert_profile: dict | None = None) -> None:
        """Register an agent in the system."""
        entry = {"agent_id": agent_id, "status": "idle", "expert_profile": expert_profile}
        if self._db:
            await self._db.create("agent_registry", entry)
        else:
            registry = self._mem.setdefault("agent_registry", [])
            registry.append(entry)

    async def update_agent_status(self, agent_id: str, status: str) -> None:
        """Update agent status."""
        if self._db:
            await self._db.query(
                "UPDATE agent_registry SET status = $status WHERE agent_id = $agent_id",
                {"agent_id": agent_id, "status": status}
            )
        else:
            for entry in self._mem.get("agent_registry", []):
                if entry.get("agent_id") == agent_id:
                    entry["status"] = status

    async def register_task(self, task_id: str, parent_id: str | None, scope: str,
                            priority: int = 3) -> None:
        """Register a task in the system."""
        entry = {"task_id": task_id, "parent_id": parent_id, "scope": scope,
                 "status": "pending", "priority": priority}
        if self._db:
            await self._db.create("task_registry", entry)
        else:
            registry = self._mem.setdefault("task_registry", [])
            registry.append(entry)

    async def update_task_status(self, task_id: str, status: str,
                                 result: dict | None = None) -> None:
        """Update task status and optionally set result."""
        if self._db:
            updates = {"status": status, "updated_at": datetime.now(timezone.utc)}
            if result:
                updates["result"] = result
            await self._db.query(
                "UPDATE task_registry MERGE $updates WHERE task_id = $task_id",
                {"task_id": task_id, "updates": updates}
            )
        else:
            for entry in self._mem.get("task_registry", []):
                if entry.get("task_id") == task_id:
                    entry["status"] = status
                    if result:
                        entry["result"] = result

    DISCARD_POOL_MAX = 1000  # cap in-memory discard pool to prevent unbounded growth

    async def discard_result(self, agent_id: str, task_id: str, reason: str,
                             result: dict | None = None, agent_log: list | None = None) -> None:
        """Write a discarded result to the audit pool."""
        entry = {
            "agent_id": agent_id,
            "task_id": task_id,
            "reason": reason,
            "result": result,
            "agent_log": agent_log or [],
            "discarded_at": datetime.now(timezone.utc).isoformat(),
        }
        if self._db:
            await self._db.create("discard_pool", entry)
        else:
            pool = self._mem.setdefault("discard_pool", [])
            pool.append(entry)
            # Trim oldest entries when over cap
            if len(pool) > self.DISCARD_POOL_MAX:
                self._mem["discard_pool"] = pool[-self.DISCARD_POOL_MAX:]

    async def get_discarded(self, agent_id: str | None = None,
                            task_id: str | None = None) -> list[dict]:
        """Query discarded results for audit."""
        if not self._db:
            pool = self._mem.get("discard_pool", [])
            results = pool
            if agent_id:
                results = [e for e in results if e.get("agent_id") == agent_id]
            if task_id:
                results = [e for e in results if e.get("task_id") == task_id]
            return results

        conditions = []
        params = {}
        if agent_id:
            conditions.append("agent_id = $agent_id")
            params["agent_id"] = agent_id
        if task_id:
            conditions.append("task_id = $task_id")
            params["task_id"] = task_id

        where = " AND ".join(conditions) if conditions else "true"
        result = await self._db.query(
            f"SELECT * FROM discard_pool WHERE {where} ORDER BY discarded_at DESC",
            params
        )
        return result if isinstance(result, list) else result.get("result", [])
