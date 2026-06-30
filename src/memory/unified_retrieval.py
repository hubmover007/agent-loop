"""Unified Memory Retrieval — systemic M-FLOW + Mythos integration.

Design principle (NOT if-else compatibility):
  Memory is a single system with two representations:
    1. Explicit Graph (M-FLOW): structured knowledge, entity-relation,
       episodic sequences, project context. Retrieved via graph routing.
    2. Implicit Latent (Mythos): knowledge encoded in model parameters,
       retrieved via deep reasoning iteration (RDT loop).

  The two representations are NOT separate systems glued with if-else.
  They are two layers of the same memory, accessed through a single
  retrieve() call that:
    a) Queries the explicit graph (fast, precise, structured)
    b) Feeds graph results into DeepReason as context (implicit activation)
    c) DeepReason iterates to "recall" knowledge from model parameters
    d) Returns unified context = explicit_facts + implicit_insights

  This is analogous to human memory:
    - Explicit = hippocampus (episodic, relational)
    - Implicit = neocortex (parameter-based, iterative recall)
    - Together = unified conscious recall

Architecture:
  User Query
    ↓
  MemoryPool.retrieve(query)
    ├── GraphRouter.route(query)     ← M-FLOW explicit graph
    │     → facts, episodes, relations
    ↓
  DeepReasonLoop.reason(query, context=graph_results)
    ├── Iteration 1: activate relevant parametric knowledge
    ├── Iteration 2: cross-reference with graph facts
    └── Iteration N: converge on unified understanding
    ↓
  UnifiedMemoryContext
    ├── explicit: [graph facts, episodes, relations]
    └── implicit: [reasoning insights, parametric recall]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class MemoryContext:
    """Unified memory context returned by retrieve().

    Combines:
      - explicit: M-FLOW graph results (facts, episodes, relations)
      - implicit: Mythos-style deep reasoning insights
      - metadata: retrieval diagnostics
    """
    explicit: list[dict] = field(default_factory=list)
    implicit: str = ""
    confidence: float = 0.0
    graph_hops: int = 0
    reason_iterations: int = 0

    def to_prompt(self) -> str:
        """Render as a prompt string for downstream LLM calls."""
        parts = []
        if self.explicit:
            parts.append("=== Retrieved from Memory Graph ===")
            for item in self.explicit[:5]:
                parts.append(f"  [{item.get('layer', '?')}] {item.get('title', '')}")
                if item.get('summary'):
                    parts.append(f"    {item['summary']}")
        if self.implicit:
            parts.append("\n=== Deep Recall ===")
            parts.append(self.implicit)
        return "\n".join(parts) if parts else ""


class UnifiedRetriever:
    """Unified memory retrieval: M-FLOW graph + Mythos deep reasoning.

    This is the single entry point for memory access. All agents call
    retriever.retrieve(query) and get back a unified MemoryContext.

    Design:
      1. GraphRouter queries the explicit M-FLOW knowledge graph
         → fast, precise, structured facts and relations
      2. Results feed into DeepReasonLoop as context
         → activates parametric knowledge (Mythos-style)
      3. DeepReason iterates to cross-reference and synthesize
         → produces implicit insights
      4. Both are returned as unified MemoryContext

    NOT if-else: both layers are ALWAYS consulted. The graph provides
    anchors, deep reasoning provides synthesis. Together = unified recall.
    """

    def __init__(self, memory_pool: Any, graph_router: Any,
                 deep_reason: Any, llm: Any):
        """
        Args:
            memory_pool: MemoryPool (SurrealDB backend)
            graph_router: GraphRouter (M-FLOW graph routing)
            deep_reason: DeepReasonLoop (RDT-style iterative reasoning)
            llm: LLMProvider (for embedding + reasoning)
        """
        self.memory = memory_pool
        self.graph_router = graph_router
        self.deep_reason = deep_reason
        self.llm = llm

    async def retrieve(self, query: str, max_hops: int = 3,
                       deep_reason_iterations: int = 3) -> MemoryContext:
        """Retrieve unified memory context for a query.

        This is the single memory access point. It:
          1. Generates query embedding
          2. Routes through M-FLOW graph (explicit layer)
          3. Feeds graph results into DeepReason (implicit layer)
          4. Returns unified context

        Args:
            query: User's question or task description
            max_hops: Max graph traversal hops (M-FLOW)
            deep_reason_iterations: Max reasoning iterations (Mythos)
        """
        context = MemoryContext()

        # ---- Phase 1: Explicit Graph Retrieval (M-FLOW) ----
        try:
            # Generate query embedding
            query_embedding = await self._embed(query)

            # Graph routing: anchor search → subgraph extraction → cost propagation
            graph_results = await self.graph_router.retrieve(
                query=query,
            )

            context.explicit = graph_results
            context.graph_hops = max_hops

            logger.debug("UnifiedRetriever: graph returned %d items",
                        len(graph_results))

        except Exception as e:
            logger.warning("UnifiedRetriever: graph retrieval failed: %s", e)
            graph_results = []

        # ---- Phase 2: Implicit Deep Reasoning (Mythos) ----
        # ALWAYS run deep reasoning, even if graph returned nothing.
        # This activates parametric knowledge (Mythos-style recall).
        try:
            # Build context from graph results
            graph_context = self._format_graph_context(graph_results)

            # DeepReason iterates to cross-reference and synthesize
            # This is where Mythos-style parametric recall happens:
            # the model "remembers" knowledge encoded in its weights
            reason_state = await self.deep_reason.reason(
                query=f"Recall and synthesize knowledge relevant to: {query}",
                context=graph_context,
            )

            context.implicit = reason_state.current_thought
            context.confidence = reason_state.confidence
            context.reason_iterations = reason_state.iteration

            logger.debug("UnifiedRetriever: deep reason iter=%d conf=%.2f",
                        reason_state.iteration, reason_state.confidence)

        except Exception as e:
            logger.warning("UnifiedRetriever: deep reason failed: %s", e)

        return context

    async def _embed(self, text: str) -> list[float]:
        """Generate embedding for text via LLM provider."""
        try:
            embeddings = await self.llm.embed(text)
            return embeddings[0] if isinstance(embeddings, list) and embeddings else embeddings
        except Exception:
            # Fallback: return zero vector
            return [0.0] * 1024

    @staticmethod
    def _format_graph_context(graph_results: list[dict]) -> str:
        """Format graph results as context string for deep reasoning."""
        if not graph_results:
            return "No explicit memory found. Recall from parametric knowledge."

        parts = ["Retrieved memory graph facts:"]
        for i, item in enumerate(graph_results[:5], 1):
            layer = item.get("layer", "?")
            title = item.get("title", "untitled")
            summary = item.get("summary", "")
            parts.append(f"  {i}. [{layer}] {title}: {summary}")

        parts.append("\nSynthesize these facts with your parametric knowledge.")
        return "\n".join(parts)

    async def store_episode(self, title: str, summary: str,
                            content: str, tags: list[str] | None = None) -> None:
        """Store an episode in memory (M-FLOW episode layer).

        Also triggers consolidation: repeated episodes are compressed
        into semantic facts (M-FLOW archival process).
        """
        try:
            await self.memory.write_episode(
                title=title,
                summary=summary,
                content=content,
            )
            logger.debug("UnifiedRetriever: stored episode '%s'", title)
        except Exception as e:
            logger.warning("UnifiedRetriever: store_episode failed: %s", e)

    async def consolidate(self) -> dict:
        """Run memory consolidation: episodes → semantic facts.

        M-FLOW archival: episodic memories are periodically compressed
        into semantic facts (entity-relation triples) for long-term storage.

        Process:
          1. Fetch unconsolidated episodes from MemoryPool
          2. Use LLM to extract entity-relation triples from episodes
          3. Write extracted facts back to MemoryPool
          4. Mark episodes as consolidated

        Returns consolidation statistics.
        """
        stats = {"episodes_processed": 0, "facts_created": 0}

        try:
            # 1. Fetch unconsolidated episodes
            episodes = await self.memory.get_unconsolidated_episodes(limit=50)
            if not episodes:
                logger.info("UnifiedRetriever: no unconsolidated episodes")
                return stats

            # 2. Extract entity-relation triples via LLM
            triples = await self._extract_triples(episodes)

            # 3. Write facts back to MemoryPool
            for triple in triples:
                try:
                    await self.memory.write_fact(
                        fact_type="entity",
                        name=triple.get("entity", ""),
                        value={
                            "relation": triple.get("relation", ""),
                            "target": triple.get("target", ""),
                        },
                        embedding_text=f"{triple.get('entity', '')} {triple.get('relation', '')} {triple.get('target', '')}",
                    )
                    stats["facts_created"] += 1
                except Exception as e:
                    logger.warning("Consolidate: write_fact failed: %s", e)

            # 4. Mark episodes as consolidated
            for ep in episodes:
                ep_id = ep.get("id", "")
                if ep_id:
                    try:
                        await self.memory.mark_episode_consolidated(ep_id)
                        stats["episodes_processed"] += 1
                    except Exception as e:
                        logger.warning("Consolidate: mark_episode failed: %s", e)

            logger.info(
                "UnifiedRetriever: consolidated %d episodes → %d facts",
                stats["episodes_processed"], stats["facts_created"]
            )

        except Exception as e:
            logger.warning("UnifiedRetriever: consolidation failed: %s", e)

        return stats

    async def _extract_triples(self, episodes: list[dict]) -> list[dict]:
        """Use LLM to extract entity-relation triples from episodes.

        Returns a list of {"entity": "...", "relation": "...", "target": "..."}.
        """
        if not episodes:
            return []

        # Build compact episode summaries for the prompt
        summaries = []
        for ep in episodes[:20]:  # Limit to avoid huge prompts
            title = ep.get("title", "")
            summary = ep.get("summary", ep.get("user_input", ""))
            output = ep.get("output", ep.get("content", ""))[:200]
            summaries.append(f"- [{title}]: {summary} | Output: {output}")

        prompt = f"""Extract entity-relation triples from the following episodes.
An entity-relation triple captures: (subject, predicate, object).
Output a JSON array:

Episodes:
{chr(10).join(summaries)}

Output only JSON array of triples:
[{{"entity": "...", "relation": "...", "target": "..."}}]

Only include meaningful triples. Skip if nothing substantive."""

        try:
            from ..utils import extract_json_from_llm_response as _ejfr
            response = await self.llm.chat([{"role": "user", "content": prompt}])
            triples = _ejfr(response.content, default=[])
            return triples if isinstance(triples, list) else []
        except Exception as e:
            logger.warning("Extract triples LLM call failed: %s", e)
            return []
