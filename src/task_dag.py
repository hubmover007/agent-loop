"""Task DAG — dependency graph for task orchestration.

Supports:
  - Linear chains: A → B → C
  - Parallel branches: A → [B, C]
  - Failure branches: if A fails → run D
  - Topological sort
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class DAGTaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class DAGNode:
    """A single node in a task dependency graph."""

    id: str
    name: str
    handler: Callable[..., Awaitable[Any]]
    dependencies: list[str] = field(default_factory=list)  # Predecessor node IDs
    on_failure: str | None = None  # Node ID to jump to on failure
    status: DAGTaskStatus = DAGTaskStatus.PENDING
    result: Any = None
    error: str | None = None


class TaskDAG:
    """Task dependency graph.

    Nodes are connected via directed edges (dependencies).
    Execution order is determined by topological sort.
    On failure, nodes can branch to alternate paths via on_failure.
    """

    def __init__(self):
        self.nodes: dict[str, DAGNode] = {}

    def add_node(self, node: DAGNode) -> None:
        """Register a node in the DAG."""
        self.nodes[node.id] = node

    def add_edge(self, from_id: str, to_id: str,
                 on_failure: bool = False) -> None:
        """Add a dependency edge between two nodes.

        Args:
            from_id: Source node ID
            to_id: Target node ID
            on_failure: If True, to is executed when from fails.
                        If False (default), to depends on from's success.
        """
        if from_id not in self.nodes:
            raise KeyError(f"Source node not found: {from_id}")
        if to_id not in self.nodes:
            raise KeyError(f"Target node not found: {to_id}")

        if on_failure:
            self.nodes[from_id].on_failure = to_id
        else:
            if from_id not in self.nodes[to_id].dependencies:
                self.nodes[to_id].dependencies.append(from_id)

    def topological_sort(self) -> list[str]:
        """Produce a topological ordering of nodes.

        Uses Kahn's algorithm (BFS-based).
        Returns node IDs in execution order.

        Raises ValueError if the graph contains a cycle.
        """
        in_degree: dict[str, int] = {nid: 0 for nid in self.nodes}
        adj: dict[str, list[str]] = {nid: [] for nid in self.nodes}

        for node in self.nodes.values():
            for dep in node.dependencies:
                if dep in self.nodes:
                    in_degree[node.id] += 1
                    adj[dep].append(node.id)

        # BFS queue for nodes with zero in-degree
        queue: deque[str] = deque()
        for nid, deg in in_degree.items():
            if deg == 0:
                queue.append(nid)

        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for neighbor in adj.get(current, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(order) != len(self.nodes):
            raise ValueError("TaskDAG contains a cycle; topological sort is impossible")

        return order

    async def execute(self, context: dict | None = None) -> dict[str, Any]:
        """Execute all nodes in dependency order.

        Steps:
          1. Topological sort to determine execution order
          2. For each node:
             a. Check if all dependencies succeeded — skip if not
             b. Execute the node's handler
             c. On success: mark SUCCESS
             d. On failure: mark FAILED and check on_failure branch

        Returns:
            dict mapping node_id → {status, result, error}
        """
        ctx = context or {}
        order = self.topological_sort()
        results: dict[str, Any] = {}

        processed: set[str] = set()
        # Track nodes that should execute because they were re-queued
        # by a failure branch (on_failure)
        extra_nodes: dict[str, str] = {}  # node_id → reason

        i = 0
        while i < len(order):
            node_id = order[i]
            node = self.nodes.get(node_id)
            if not node:
                i += 1
                continue

            # If this node was already processed (e.g., triggered by failure branch),
            # skip the original occurrence in order
            if node_id in processed:
                i += 1
                continue

            # Check if all dependencies succeeded
            failed_deps = [
                d for d in node.dependencies
                if d in self.nodes and self.nodes[d].status != DAGTaskStatus.SUCCESS
            ]
            if failed_deps:
                node.status = DAGTaskStatus.SKIPPED
                logger.debug("TaskDAG: skipping '%s' — unmet dependencies: %s",
                             node_id, failed_deps)
                results[node_id] = {
                    "status": DAGTaskStatus.SKIPPED.value,
                    "result": None,
                    "error": f"Skipped: dependencies {failed_deps} not successful",
                }
                processed.add(node_id)
                i += 1
                continue

            # Execute the node
            node.status = DAGTaskStatus.RUNNING
            logger.debug("TaskDAG: running '%s' (%s)", node_id, node.name)
            try:
                node.result = await node.handler(ctx)
                node.status = DAGTaskStatus.SUCCESS
            except Exception as e:
                node.status = DAGTaskStatus.FAILED
                node.error = str(e)
                logger.warning("TaskDAG: '%s' failed: %s", node_id, e)

                # Check on_failure branch
                if node.on_failure and node.on_failure in self.nodes:
                    failure_id = node.on_failure
                    failure_node = self.nodes[failure_id]
                    # Clear dependencies so it can execute unconditionally
                    failure_node.dependencies = []
                    self.nodes[failure_id] = failure_node
                    logger.info("TaskDAG: '%s' failed → branching to '%s'",
                                node_id, failure_id)

                    # Re-compute topological sort after clearing deps
                    try:
                        order = self.topological_sort()
                        # Reset i since order changed
                        i = 0
                    except ValueError:
                        pass  # If sort fails (shouldn't happen), continue with old order

            results[node_id] = {
                "status": node.status.value,
                "result": node.result,
                "error": node.error,
            }
            processed.add(node_id)
            i += 1

        return results
