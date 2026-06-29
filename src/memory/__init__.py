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
from datetime import datetime
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
    """

    def __init__(self, url: str = "ws://127.0.0.1:8000", namespace: str = "agent_loop",
                 database: str = "memory", user: str = "root", password: str = "root"):
        self.url = url
        self.namespace = namespace
        self.database = database
        self._auth = {"user": user, "pass": password}
        self._db: Surreal | None = None
        self._embedding_fn: Any = None  # Set via configure_embedding()

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
    # Write Operations
    # ============================================================

    async def write_fact(self, fact_type: str, name: str, value: Any = None,
                         embedding_text: str | None = None) -> str:
        """Write a Fact (entity or facetpoint) to Layer 0."""
        embedding = await self.embed(embedding_text or name) if self._embedding_fn else None
        result = await self._db.create("fact", {
            "fact_type": fact_type,
            "name": name,
            "value": value,
            "embedding": embedding or [],
        })
        return result["id"]

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
                            tags: list[str] | None = None) -> str:
        """Write an Episode to Layer 2."""
        embedding = await self.embed(summary) if self._embedding_fn else None
        result = await self._db.create("episode", {
            "title": title,
            "summary": summary,
            "content": content,
            "embedding": embedding or [],
            "tags": tags or [],
        })
        return result["id"]

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
        result = await self._db.query(
            "SELECT * FROM fact WHERE name = $name LIMIT 1",
            {"name": name}
        )
        rows = result if isinstance(result, list) else result.get("result", [])
        return rows[0] if rows else None

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
        return await self._db.select(episode_id)

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
        await self._db.create("agent_registry", {
            "agent_id": agent_id,
            "status": "idle",
            "expert_profile": expert_profile,
        })

    async def update_agent_status(self, agent_id: str, status: str) -> None:
        """Update agent status."""
        await self._db.query(
            "UPDATE agent_registry SET status = $status WHERE agent_id = $agent_id",
            {"agent_id": agent_id, "status": status}
        )

    async def register_task(self, task_id: str, parent_id: str | None, scope: str,
                            priority: int = 3) -> None:
        """Register a task in the system."""
        await self._db.create("task_registry", {
            "task_id": task_id,
            "parent_id": parent_id,
            "scope": scope,
            "status": "pending",
            "priority": priority,
        })

    async def update_task_status(self, task_id: str, status: str,
                                 result: dict | None = None) -> None:
        """Update task status and optionally set result."""
        updates = {"status": status, "updated_at": datetime.now()}
        if result:
            updates["result"] = result
        await self._db.query(
            "UPDATE task_registry MERGE $updates WHERE task_id = $task_id",
            {"task_id": task_id, "updates": updates}
        )

    async def discard_result(self, agent_id: str, task_id: str, reason: str,
                             result: dict | None = None, agent_log: list | None = None) -> None:
        """Write a discarded result to the audit pool."""
        await self._db.create("discard_pool", {
            "agent_id": agent_id,
            "task_id": task_id,
            "reason": reason,
            "result": result,
            "agent_log": agent_log or [],
        })

    async def get_discarded(self, agent_id: str | None = None,
                            task_id: str | None = None) -> list[dict]:
        """Query discarded results for audit."""
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
