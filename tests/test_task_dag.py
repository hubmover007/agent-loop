"""Tests for P2 TaskDAG — dependency graph orchestration."""

import pytest

from src.task_dag import (
    TaskDAG, DAGNode, DAGTaskStatus,
)


async def _handler(val: str):
    """Simple handler that returns a value."""
    return f"done:{val}"


async def _failing_handler(val: str):
    """Handler that raises an exception."""
    raise RuntimeError(f"Failed: {val}")


def _make_node(node_id: str, name: str, succeeds: bool = True) -> DAGNode:
    handler = _handler if succeeds else _failing_handler
    return DAGNode(
        id=node_id,
        name=name,
        handler=lambda ctx, n=name: handler(n),
    )


class TestTaskDAGTopologicalSort:
    """Tests for topological sort."""

    def test_linear_sort(self):
        """A→B→C should yield ['A', 'B', 'C']."""
        dag = TaskDAG()
        dag.add_node(_make_node("a", "A"))
        dag.add_node(_make_node("b", "B"))
        dag.add_node(_make_node("c", "C"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")

        order = dag.topological_sort()
        assert order == ["a", "b", "c"]

    def test_parallel_sort(self):
        """A→[B,C] where B and C have no inter-dependency."""
        dag = TaskDAG()
        dag.add_node(_make_node("a", "A"))
        dag.add_node(_make_node("b", "B"))
        dag.add_node(_make_node("c", "C"))
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")

        order = dag.topological_sort()
        assert order[0] == "a"  # A must come first
        assert set(order[1:]) == {"b", "c"}  # B and C after A, order doesn't matter

    def test_cycle_detection(self):
        """Cycle should raise ValueError."""
        dag = TaskDAG()
        dag.add_node(_make_node("a", "A"))
        dag.add_node(_make_node("b", "B"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")  # Creates cycle

        with pytest.raises(ValueError, match="cycle"):
            dag.topological_sort()

    def test_no_dependencies(self):
        """Nodes with no dependencies sort in insertion order."""
        dag = TaskDAG()
        dag.add_node(_make_node("x", "X"))
        dag.add_node(_make_node("y", "Y"))
        dag.add_node(_make_node("z", "Z"))

        order = dag.topological_sort()
        assert set(order) == {"x", "y", "z"}
        assert len(order) == 3


class TestTaskDAGExecute:
    """Tests for DAG execution."""

    @pytest.mark.asyncio
    async def test_linear_execution(self):
        """A→B→C: all succeed, execute in order."""
        dag = TaskDAG()
        dag.add_node(_make_node("a", "A"))
        dag.add_node(_make_node("b", "B"))
        dag.add_node(_make_node("c", "C"))
        dag.add_edge("a", "b")
        dag.add_edge("b", "c")

        results = await dag.execute({"ctx": "test"})
        assert results["a"]["status"] == "success"
        assert results["a"]["result"] == "done:A"
        assert results["b"]["status"] == "success"
        assert results["b"]["result"] == "done:B"
        assert results["c"]["status"] == "success"
        assert results["c"]["result"] == "done:C"

    @pytest.mark.asyncio
    async def test_skipped_on_dep_failure(self):
        """If A fails, B (which depends on A) is skipped."""
        dag = TaskDAG()
        dag.add_node(DAGNode(
            id="a", name="A", handler=lambda ctx: _failing_handler("A"),
        ))
        dag.add_node(_make_node("b", "B"))
        dag.add_edge("a", "b")

        results = await dag.execute()
        assert results["a"]["status"] == "failed"
        assert results["a"]["error"] == "Failed: A"
        assert results["b"]["status"] == "skipped"

    @pytest.mark.asyncio
    async def test_failure_branch(self):
        """If A fails, on_failure→C should execute."""
        dag = TaskDAG()
        dag.add_node(DAGNode(
            id="a", name="A",
            handler=lambda ctx: _failing_handler("A"),
            on_failure="c",
        ))
        dag.add_node(_make_node("b", "B"))
        dag.add_node(_make_node("c", "C"))
        dag.add_edge("a", "b")  # B depends on A succeeding

        results = await dag.execute()
        assert results["a"]["status"] == "failed"
        assert results["b"]["status"] == "skipped"
        assert results["c"]["status"] == "success"
        assert results["c"]["result"] == "done:C"

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """A→[B,C] where both B and C run after A succeeds."""
        dag = TaskDAG()
        dag.add_node(_make_node("a", "A"))
        dag.add_node(_make_node("b", "B"))
        dag.add_node(_make_node("c", "C"))
        dag.add_edge("a", "b")
        dag.add_edge("a", "c")

        results = await dag.execute()
        assert results["a"]["status"] == "success"
        assert results["b"]["status"] == "success"
        assert results["c"]["status"] == "success"

    @pytest.mark.asyncio
    async def test_add_edge_unknown_node(self):
        """Adding an edge with unknown node raises KeyError."""
        dag = TaskDAG()
        dag.add_node(_make_node("a", "A"))
        with pytest.raises(KeyError):
            dag.add_edge("a", "nonexistent")

    @pytest.mark.asyncio
    async def test_status_values(self):
        """DAGTaskStatus enum has correct values."""
        assert DAGTaskStatus.PENDING.value == "pending"
        assert DAGTaskStatus.RUNNING.value == "running"
        assert DAGTaskStatus.SUCCESS.value == "success"
        assert DAGTaskStatus.FAILED.value == "failed"
        assert DAGTaskStatus.SKIPPED.value == "skipped"

    @pytest.mark.asyncio
    async def test_all_fail(self):
        """All nodes fail gracefully."""
        dag = TaskDAG()
        dag.add_node(DAGNode(
            id="a", name="A", handler=lambda ctx: _failing_handler("A"),
        ))
        dag.add_node(DAGNode(
            id="b", name="B", handler=lambda ctx: _failing_handler("B"),
        ))

        results = await dag.execute()
        assert results["a"]["status"] == "failed"
        assert results["b"]["status"] == "failed"
