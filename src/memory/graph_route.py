"""Graph-routed retrieval - M-FLOW inspired.

Core algorithm:
  1. Anchor search at cone tip (Facts) - vector similarity
  2. Project anchors into the graph - extract subgraph + 1-hop neighbors
  3. Propagate cost from cone tip to cone base - find optimal Episode paths

Key innovations:
  - Semantic edge filtering: edges have vectorized descriptions, filter noise
  - Minimum path cost: one strong evidence chain proves relevance
  - Direct-hit penalty: prefer precise anchor paths over wide Episode matches
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field

from . import MemoryPool
from ..core import MemoryLayer

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A candidate node from vector search."""
    node_id: str
    node_type: str  # "fact" | "facet" | "episode" | "project"
    distance: float
    data: dict = field(default_factory=dict)


@dataclass
class GraphPath:
    """A path through the graph from anchor to episode."""
    anchor: SearchResult
    edges: list[dict] = field(default_factory=list)
    nodes: list[str] = field(default_factory=list)

    @property
    def hops(self) -> int:
        return len(self.edges)


@dataclass
class RetrieveResult:
    """Final retrieval result with episode and score."""
    episode_id: str
    episode_data: dict
    score: float
    path: GraphPath


class GraphRouter:
    """M-FLOW style graph-routed retrieval engine."""

    # Cost weights
    START_COST_WEIGHT = 1.0       # Weight for anchor vector distance
    EDGE_COST_WEIGHT = 1.0        # Weight for edge semantic distance
    HOP_PENALTY = 0.1             # Penalty per hop
    DIRECT_HIT_PENALTY = 0.3      # Penalty for direct Episode match
    MAX_HOPS = 3                  # Maximum path length

    def __init__(self, pool: MemoryPool):
        self.pool = pool

    async def retrieve(self, query: str, top_k: int = 5) -> list[RetrieveResult]:
        """Execute graph-routed retrieval for a query.

        Args:
            query: Natural language query
            top_k: Number of top episodes to return

        Returns:
            List of RetrieveResult sorted by score (lower = better)
        """
        query_embedding = await self.pool.embed(query)

        # Phase 1: Broad anchor search across all layers
        anchors = await self._search_all_layers(query, query_embedding, top_n=100)

        if not anchors:
            logger.info("No anchors found for query: %s", query[:50])
            return []

        # Phase 2: Extract subgraph around anchors
        subgraph = await self._extract_subgraph(anchors)

        # Phase 3: Propagate cost from anchors to episodes
        episode_scores: dict[str, tuple[float, GraphPath]] = {}
        for anchor in anchors:
            for episode_id, episode_data in subgraph.items():
                paths = await self._find_paths(anchor, episode_id)
                for path in paths:
                    cost = self._compute_path_cost(path, query_embedding)
                    if episode_id not in episode_scores or cost < episode_scores[episode_id][0]:
                        episode_scores[episode_id] = (cost, path)

        # Sort by cost (lower is better) and return top-k
        sorted_episodes = sorted(episode_scores.items(), key=lambda x: x[1][0])
        results = []
        for episode_id, (score, path) in sorted_episodes[:top_k]:
            episode_data = subgraph[episode_id]
            results.append(RetrieveResult(
                episode_id=episode_id,
                episode_data=episode_data,
                score=score,
                path=path,
            ))

        logger.info("Graph route retrieved %d episodes for query: %s", len(results), query[:50])
        return results

    # ============================================================
    # Phase 1: Anchor Search
    # ============================================================

    async def _search_all_layers(self, query: str, query_embedding: list[float],
                                  top_n: int = 100) -> list[SearchResult]:
        """Search across all memory layers simultaneously."""
        anchors = []

        # Search Facts (Layer 0 - cone tip, highest precision)
        fact_anchors = await self._vector_search("fact", query_embedding, top_n)
        anchors.extend(fact_anchors)

        # Search Facets (Layer 1)
        facet_anchors = await self._vector_search("facet", query_embedding, top_n // 2)
        anchors.extend(facet_anchors)

        # Search Episodes (Layer 2 - cone base, broad match)
        episode_anchors = await self._vector_search("episode", query_embedding, top_n // 2)
        # Mark episode anchors for direct-hit penalty
        for a in episode_anchors:
            a.node_type = "episode_direct"
        anchors.extend(episode_anchors)

        # Search Projects (Layer 3)
        project_anchors = await self._vector_search("project", query_embedding, top_n // 4)
        anchors.extend(project_anchors)

        # Deduplicate by node_id and sort by distance
        seen = set()
        unique = []
        for a in sorted(anchors, key=lambda x: x.distance):
            if a.node_id not in seen:
                seen.add(a.node_id)
                unique.append(a)

        return unique[:top_n]

    async def _vector_search(self, table: str, query_embedding: list[float],
                             top_n: int) -> list[SearchResult]:
        """Vector similarity search on a table."""
        try:
            result = await self.pool._db.query(f"""
                SELECT *, vector::distance::cosine(embedding, $query_embedding) AS _dist
                FROM {table}
                WHERE embedding IS NOT NONE
                ORDER BY _dist ASC
                LIMIT $top_n
            """, {
                "query_embedding": query_embedding,
                "top_n": top_n,
            })
            rows = result if isinstance(result, list) else result.get("result", [])
            return [
                SearchResult(
                    node_id=row["id"],
                    node_type=table,
                    distance=self._cosine_distance(row.get("embedding", []), query_embedding),
                    data=row,
                )
                for row in rows
            ]
        except Exception as e:
            logger.warning("Vector search on %s failed: %s", table, e)
            return []

    # ============================================================
    # Phase 2: Subgraph Extraction
    # ============================================================

    async def _extract_subgraph(self, anchors: list[SearchResult]) -> dict[str, dict]:
        """Extract subgraph: all episodes connected to anchors via semantic edges."""
        episodes: dict[str, dict] = {}

        for anchor in anchors:
            # If anchor is already an episode, add directly
            if anchor.node_type in ("episode", "episode_direct"):
                episodes[anchor.node_id] = anchor.data
                continue

            # Follow edges from anchor to find connected episodes
            neighbors = await self.pool.get_neighbors(anchor.node_id)

            # Also check graph relations (belongs_to, contains, part_of)
            relations = await self._get_graph_relations(anchor.node_id)

            for rel in relations:
                target = rel.get("in") or rel.get("out")
                if target and isinstance(target, str):
                    # Try to get the target node
                    target_data = await self._get_node(target)
                    if target_data:
                        target_type = self._infer_type(target)
                        if target_type == "episode":
                            episodes[target] = target_data

        return episodes

    async def _get_graph_relations(self, node_id: str) -> list[dict]:
        """Get graph relation edges for a node."""
        try:
            result = await self.pool._db.query(f"""
                SELECT *
                FROM {node_id}->belongs_to->contains->part_of
            """)
            return result if isinstance(result, list) else result.get("result", [])
        except Exception:
            return []

    async def _get_node(self, node_id: str) -> dict | None:
        """Get a node by its record ID."""
        try:
            return await self.pool._db.select(node_id)
        except Exception:
            return None

    def _infer_type(self, node_id: str) -> str:
        """Infer node type from its record ID."""
        for t in ["fact", "facet", "episode", "project", "edge"]:
            if node_id.startswith(t):
                return t
        return "unknown"

    # ============================================================
    # Phase 3: Cost Propagation
    # ============================================================

    async def _find_paths(self, anchor: SearchResult, episode_id: str) -> list[GraphPath]:
        """Find all paths from anchor to episode (up to MAX_HOPS)."""
        paths = []

        # Direct path: anchor IS the episode
        if anchor.node_id == episode_id:
            paths.append(GraphPath(anchor=anchor))
            return paths

        # 1-hop: anchor → episode
        edges_1hop = await self._get_edges_between(anchor.node_id, episode_id)
        for edge in edges_1hop:
            paths.append(GraphPath(
                anchor=anchor,
                edges=[edge],
                nodes=[anchor.node_id, episode_id],
            ))

        # 2-hop: anchor → intermediate → episode
        if self.MAX_HOPS >= 2:
            neighbors = await self.pool.get_neighbors(anchor.node_id)
            for neighbor in neighbors:
                target_id = neighbor.get("target_id") or neighbor.get("target")
                if not target_id:
                    continue
                target_id = str(target_id)
                edges_to_ep = await self._get_edges_between(target_id, episode_id)
                for edge2 in edges_to_ep:
                    paths.append(GraphPath(
                        anchor=anchor,
                        edges=[neighbor, edge2],
                        nodes=[anchor.node_id, target_id, episode_id],
                    ))

        # 3-hop (limited, to avoid explosion)
        if self.MAX_HOPS >= 3 and not paths:  # Only if no shorter path found
            # Simplified: try via facet intermediary
            for facet_name in ["facet:ops", "facet:feishu", "facet:aws"]:
                inter_edges = await self._get_edges_between(anchor.node_id, f"facet:{facet_name}")
                if inter_edges:
                    for ie in inter_edges[:1]:  # One edge is enough
                        ep_edges = await self._get_edges_between(f"facet:{facet_name}", episode_id)
                        for ee in ep_edges[:1]:
                            paths.append(GraphPath(
                                anchor=anchor,
                                edges=[ie, ee],
                                nodes=[anchor.node_id, f"facet:{facet_name}", episode_id],
                            ))
                            break
                    if paths:
                        break

        return paths

    async def _get_edges_between(self, source: str, target: str) -> list[dict]:
        """Get semantic edges between two nodes."""
        try:
            result = await self.pool._db.query(f"""
                SELECT *, embedding
                FROM edge
                WHERE (source_id = {source} AND target_id = {target})
                   OR (source_id = {target} AND target_id = {source})
            """)
            return result if isinstance(result, list) else result.get("result", [])
        except Exception:
            return []

    def _compute_path_cost(self, path: GraphPath, query_embedding: list[float]) -> float:
        """Compute total cost of a path (lower = better relevance).

        Cost = start_cost + Σ(edge_cost) + hop_penalty * hops + direct_hit_penalty
        """
        # Start cost: vector distance of the anchor
        start_cost = path.anchor.distance * self.START_COST_WEIGHT

        # Edge cost: sum of semantic distances of edges
        edge_cost = 0.0
        for edge in path.edges:
            edge_embedding = edge.get("embedding", [])
            if edge_embedding:
                edge_cost += self._cosine_distance(edge_embedding, query_embedding) * self.EDGE_COST_WEIGHT
            else:
                # Edge without embedding gets a default moderate cost
                edge_cost += 0.3

        # Hop penalty
        hop_penalty = path.hops * self.HOP_PENALTY

        # Direct hit penalty: if anchor is an episode (not a precise fact anchor)
        direct_hit = self.DIRECT_HIT_PENALTY if path.anchor.node_type == "episode_direct" else 0.0

        total = start_cost + edge_cost + hop_penalty + direct_hit
        logger.debug(
            "Path cost: %.4f (start=%.4f, edge=%.4f, hop=%.4f, direct=%.4f, hops=%d)",
            total, start_cost, edge_cost, hop_penalty, direct_hit, path.hops
        )
        return total

    # ============================================================
    # Utility
    # ============================================================

    @staticmethod
    def _cosine_distance(a: list[float], b: list[float]) -> float:
        """Compute cosine distance between two vectors."""
        if not a or not b or len(a) != len(b):
            return 1.0  # Maximum distance for invalid inputs

        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

        if norm_a == 0 or norm_b == 0:
            return 1.0

        # Cosine distance = 1 - cosine_similarity
        return 1.0 - (dot / (norm_a * norm_b))
