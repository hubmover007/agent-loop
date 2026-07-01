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
from datetime import datetime, timedelta, timezone
from typing import Any, TYPE_CHECKING

from ..core import MemoryLayer

_Surreal: Any = None  # lazily resolved in connect()

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
        self._db: Any = None  # surrealdb.Surreal instance
        self._embedding_fn: Any = None  # Set via configure_embedding()
        self._mem: dict[str, list[dict]] = {}  # in-memory fallback
        self._db_path: str | None = db_path
        self._sqlite: Any = None  # sqlite3.Connection when db_path is set
        if db_path:
            self._init_sqlite(db_path)

    @staticmethod
    async def _get_surreal() -> Any:
        """Lazily import AsyncSurreal; raises ImportError if not installed."""
        global _Surreal
        if _Surreal is None:
            from surrealdb import AsyncSurreal as _S
            _Surreal = _S
        return _Surreal

    async def connect(self) -> None:
        """Connect to SurrealDB and initialize schema."""
        Surreal_cls = await self._get_surreal()
        self._db = Surreal_cls(self.url)
        await self._db.connect()
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

        # Execute each statement separately, stripping comment lines
        for statement in schema_sql.split(";\n"):
            # Filter out comment lines and blank lines
            lines = [line for line in statement.split("\n")
                     if line.strip() and not line.strip().startswith("--")]
            statement = "\n".join(lines).strip()
            if statement:
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

    def clear(self) -> None:
        """Clear all in-memory data (for testing)."""
        self._mem.clear()
        if self._sqlite:
            self._sqlite.execute("DELETE FROM memories")
            self._sqlite.commit()

    async def write_fact(self, fact_type: str, name: str, value: Any = None,
                         embedding_text: str | None = None,
                         upsert: bool = True,
                         agent_id: str = "shared") -> str:
        """Write a Fact (entity or facetpoint) to Layer 0.

        By default uses upsert semantics: if a fact with the same (fact_type,
        name, agent_id) already exists, it updates the existing record instead
        of creating a duplicate. Set upsert=False to enforce strict
        create-only behavior.

        Args:
            agent_id: Namespace isolation key (defaults to "shared" for
                      backward compatibility).
        """
        embedding = await self.embed(embedding_text or name) if self._embedding_fn else None
        record = {
            "fact_type": fact_type,
            "name": name,
            "value": value,
            "agent_id": agent_id,
            "embedding": embedding or [],
            "updated_at": datetime.now(timezone.utc),
        }
        if self._db:
            try:
                result = await self._db.create("fact", record)
            except Exception as e:
                if upsert and ("already exists" in str(e).lower()
                               or "already contains" in str(e).lower()):
                    result = await self._db.query(
                        "UPDATE fact MERGE $record"
                        " WHERE fact_type = $fact_type AND name = $name"
                        " AND agent_id = $agent_id RETURN AFTER",
                        {"fact_type": fact_type, "name": name,
                         "agent_id": agent_id, "record": record}
                    )
                    rows = result if isinstance(result, list) else result.get("result", [])
                    if rows:
                        row = rows[0] if isinstance(rows, list) else rows
                        rid = row.get("id", row) if isinstance(row, dict) else row
                        return str(rid) if rid is not None else str(row)
                    raise RuntimeError(f"Failed to upsert fact: {name}")
                else:
                    raise
            # surrealdb v2 returns RecordID; normalize to string
            rid = result.get("id", result) if isinstance(result, dict) else result
            return str(rid) if rid is not None else str(result)
        else:
            self._mem.setdefault("fact", []).append(record)
            return f"fact:{len(self._mem['fact'])}"

    async def write_facet(self, name: str, description: str) -> str:
        """Write a Facet to Layer 1 (upsert: update if name already exists)."""
        embedding = await self.embed(description) if self._embedding_fn else None
        record = {
            "name": name,
            "description": description,
            "embedding": embedding or [],
        }
        try:
            result = await self._db.create("facet", record)
        except Exception as e:
            if "already exists" in str(e).lower():
                result = await self._db.query(
                    "UPDATE facet MERGE $record WHERE name = $name RETURN AFTER",
                    {"name": name, "record": record}
                )
                rows = result if isinstance(result, list) else result.get("result", [])
                if rows:
                    row = rows[0] if isinstance(rows, list) else rows
                    rid = row.get("id", row) if isinstance(row, dict) else row
                    return str(rid) if rid is not None else str(row)
                raise RuntimeError(f"Failed to upsert facet: {name}")
            else:
                raise
        rid = result.get("id", result) if isinstance(result, dict) else result
        return str(rid) if rid is not None else str(result)

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
            # surrealdb v2 returns RecordID; normalize to string
            rid = result.get("id", result) if isinstance(result, dict) else result
            return str(rid) if rid is not None else str(result)
        else:
            self._mem.setdefault("episode", []).append(record)
            ep_id = f"episode:{len(self._mem['episode'])}"
            record["id"] = ep_id
            return ep_id

    async def write_project(self, name: str, description: str) -> str:
        """Write a Project to Layer 3 (upsert: update if name already exists)."""
        embedding = await self.embed(description) if self._embedding_fn else None
        record = {
            "name": name,
            "description": description,
            "embedding": embedding or [],
        }
        try:
            result = await self._db.create("project", record)
        except Exception as e:
            if "already exists" in str(e).lower():
                result = await self._db.query(
                    "UPDATE project MERGE $record WHERE name = $name RETURN AFTER",
                    {"name": name, "record": record}
                )
                rows = result if isinstance(result, list) else result.get("result", [])
                if rows:
                    row = rows[0] if isinstance(rows, list) else rows
                    rid = row.get("id", row) if isinstance(row, dict) else row
                    return str(rid) if rid is not None else str(row)
                raise RuntimeError(f"Failed to upsert project: {name}")
            else:
                raise
        rid = result.get("id", result) if isinstance(result, dict) else result
        return str(rid) if rid is not None else str(result)

    async def write_edge(self, source: str, target: str, relation: str) -> str:
        """Write a semantic edge between two nodes."""
        embedding = await self.embed(relation) if self._embedding_fn else None
        # Parse source/target table names from RecordID strings like "fact:xxx"
        src_type = source.split(":")[0] if ":" in source else "unknown"
        tgt_type = target.split(":")[0] if ":" in target else "unknown"
        # Convert string IDs to RecordID objects (required by surrealdb v2.0)
        from surrealdb import RecordID
        src_rid = RecordID.parse(source) if isinstance(source, str) else source
        tgt_rid = RecordID.parse(target) if isinstance(target, str) else target
        result = await self._db.create("edge", {
            "source_type": src_type,
            "source_id": src_rid,
            "target_type": tgt_type,
            "target_id": tgt_rid,
            "relation": relation,
            "embedding": embedding or [],
        })
        # surrealdb v2 returns RecordID; normalize to string
        rid = result.get("id", result) if isinstance(result, dict) else result
        return str(rid) if rid is not None else str(result)

    # ============================================================
    # Read Operations
    # ============================================================

    async def get_fact(self, name: str,
                       agent_id: str | None = None) -> dict | None:
        """Get a Fact by name (optionally scoped to an agent)."""
        if self._db:
            if agent_id:
                result = await self._db.query(
                    "SELECT * FROM fact WHERE name = $name AND agent_id = $agent_id LIMIT 1",
                    {"name": name, "agent_id": agent_id}
                )
            else:
                result = await self._db.query(
                    "SELECT * FROM fact WHERE name = $name LIMIT 1",
                    {"name": name}
                )
            rows = result if isinstance(result, list) else result.get("result", [])
            return rows[0] if rows else None
        else:
            for r in self._mem.get("fact", []):
                if r.get("name") == name:
                    if agent_id is None or r.get("agent_id") == agent_id:
                        return r
            return None

    async def query_facts(self,
                          agent_id: str | None = None,
                          fact_type: str | None = None,
                          limit: int = 100) -> list[dict]:
        """Query facts with optional agent_id and fact_type filters.

        Args:
            agent_id: Filter by agent namespace (None = all agents).
            fact_type: Filter by fact type ("entity" / "facetpoint").
            limit: Maximum number of results (default 100).
        """
        if self._db:
            conditions = []
            params = {"limit": limit}
            if agent_id:
                conditions.append("agent_id = $agent_id")
                params["agent_id"] = agent_id
            if fact_type:
                conditions.append("fact_type = $fact_type")
                params["fact_type"] = fact_type
            where = " AND ".join(conditions) if conditions else "true"
            result = await self._db.query(
                f"SELECT * FROM fact WHERE {where} ORDER BY updated_at DESC LIMIT $limit",
                params
            )
            return result if isinstance(result, list) else result.get("result", [])
        else:
            results = self._mem.get("fact", [])
            if agent_id:
                results = [r for r in results if r.get("agent_id") == agent_id]
            if fact_type:
                results = [r for r in results if r.get("fact_type") == fact_type]
            return results[:limit]

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
            result = await self._db.select(episode_id)
            # surrealdb v2.0 returns list; unwrap first element
            if isinstance(result, list):
                return result[0] if result else None
            return result
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
                agent_id=data_copy.pop("agent_id", "shared"),
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
    # Data Cleanup & Retention
    # ============================================================

    async def cleanup_stale_episodes(self, days: int = 30) -> int:
        """Delete episodes older than `days` days.

        Returns the number of episodes deleted.
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        if self._db:
            # Select episodes older than cutoff, then delete them one by one
            result = await self._db.query(
                "SELECT id, created_at FROM episode"
                " WHERE created_at < <datetime>($cutoff)"
                " ORDER BY created_at ASC LIMIT 1000",
                {"cutoff": cutoff.isoformat()}
            )
            rows = result if isinstance(result, list) else result.get("result", [])
            deleted = 0
            for row in rows:
                eid = row.get("id") if isinstance(row, dict) else row
                try:
                    await self._db.query(
                        "DELETE FROM episode WHERE id = $id", {"id": eid}
                    )
                    deleted += 1
                except Exception as e:
                    logger.warning("Delete stale episode %s failed: %s", eid, e)
            if deleted:
                logger.info(
                    "MemoryPool: cleaned up %d stale episodes (older than %d days)",
                    deleted, days
                )
            return deleted
        else:
            episodes = self._mem.get("episode", [])
            before_count = len(episodes)
            cutoff_ts = cutoff.timestamp()
            kept = [e for e in episodes
                     if float(e.get("created_at", 0)) > cutoff_ts]
            self._mem["episode"] = kept
            deleted = before_count - len(kept)
            if deleted:
                logger.info(
                    "MemoryPool: cleaned up %d stale episodes (older than %d days)",
                    deleted, days
                )
            return deleted

    async def consolidate_episodes_to_facts(self, days: int = 30) -> dict:
        """Convert old unconsolidated episodes into summary facts, then delete them.

        This is a lightweight consolidation: old episodes are deleted after
        recording a summary fact. No LLM summarization is performed.

        Args:
            days: Age threshold in days.

        Returns:
            dict with 'consolidated' (count of episodes processed) and
            'deleted' (count of episodes deleted).
        """
        if not self._db:
            return {"consolidated": 0, "deleted": 0}

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        result = await self._db.query(
            "SELECT id, title, summary, tags, created_at FROM episode"
            " WHERE (consolidated = false OR consolidated IS NONE)"
            " AND created_at < <datetime>($cutoff)"
            " ORDER BY created_at ASC LIMIT 100",
            {"cutoff": cutoff.isoformat()}
        )
        rows = result if isinstance(result, list) else result.get("result", [])

        consolidated_count = 0
        deleted_count = 0

        for row in rows:
            try:
                # Write a summary fact before deleting
                await self.write_fact(
                    fact_type="facetpoint",
                    name=f"consolidated_episode:{row.get('id', 'unknown')}",
                    value={
                        "title": row.get("title", ""),
                        "summary": row.get("summary", ""),
                        "tags": row.get("tags", []),
                    },
                    upsert=False,
                    agent_id="system",
                )

                # Delete the episode
                del_result = await self._db.query(
                    "DELETE FROM episode WHERE id = $id",
                    {"id": row["id"]}
                )
                deleted_count += 1
                consolidated_count += 1
            except Exception as e:
                logger.warning(
                    "Consolidate episode %s failed: %s",
                    row.get("id", "?"), e
                )

        if consolidated_count:
            logger.info(
                "MemoryPool: consolidated %d episodes into facts, deleted %d",
                consolidated_count, deleted_count
            )
        return {"consolidated": consolidated_count, "deleted": deleted_count}

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
        """Register an agent in the system (idempotent — upserts if already registered)."""
        entry = {"agent_id": agent_id, "status": "idle", "expert_profile": expert_profile}
        if self._db:
            try:
                await self._db.create("agent_registry", entry)
            except Exception as e:
                if "already exists" in str(e).lower():
                    await self._db.query(
                        "UPDATE agent_registry MERGE $entry WHERE agent_id = $agent_id",
                        {"agent_id": agent_id, "entry": entry}
                    )
                else:
                    raise
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
        """Register a task in the system (idempotent — upserts if already registered)."""
        entry = {"task_id": task_id, "parent_id": parent_id, "scope": scope,
                 "status": "pending", "priority": priority}
        if self._db:
            try:
                await self._db.create("task_registry", entry)
            except Exception as e:
                if "already exists" in str(e).lower():
                    await self._db.query(
                        "UPDATE task_registry MERGE $entry WHERE task_id = $task_id",
                        {"task_id": task_id, "entry": entry}
                    )
                else:
                    raise
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
