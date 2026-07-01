"""Tests for distributed tracing (TraceSpan, Tracer in src/streaming.py)."""

import time
import pytest
from src.streaming import TraceSpan, Tracer


class TestTraceSpan:
    """Tests for TraceSpan lifecycle."""

    def test_span_start_end(self):
        """Span can be started and finished, computing duration."""
        span = TraceSpan("test.operation")
        assert span.name == "test.operation"
        assert span.span_id is not None
        assert len(span.span_id) == 8
        assert span.status == "ok"
        assert span.end_time is None

        span.finish("ok")
        assert span.end_time is not None
        assert span.duration_ms >= 0
        assert span.status == "ok"

    def test_span_error_status(self):
        """Span can be finished with error status."""
        span = TraceSpan("test.error")
        span.finish("error")
        assert span.status == "error"
        assert span.duration_ms >= 0

    def test_span_attributes(self):
        """Span supports setting arbitrary attributes."""
        span = TraceSpan("test.attrs")
        span.set_attribute("component", "agent-loop")
        span.set_attribute("task", "test-task")
        span.set_attribute("count", 42)

        assert span.attributes["component"] == "agent-loop"
        assert span.attributes["task"] == "test-task"
        assert span.attributes["count"] == 42

    def test_span_events(self):
        """Span supports adding timestamped events."""
        span = TraceSpan("test.events")
        span.add_event("tool_call", tool="fs.read_file")
        span.add_event("tool_result", result="ok")

        assert len(span.events) == 2
        assert span.events[0]["name"] == "tool_call"
        assert span.events[0]["tool"] == "fs.read_file"
        assert span.events[1]["name"] == "tool_result"
        assert span.events[1]["result"] == "ok"
        assert "timestamp" in span.events[0]

    def test_span_to_dict(self):
        """Span exports to dict with all fields."""
        span = TraceSpan("test.dict")
        span.set_attribute("x", "y")
        span.add_event("e1")
        span.finish("ok")

        d = span.to_dict()
        assert d["span_id"] == span.span_id
        assert d["name"] == "test.dict"
        assert d["status"] == "ok"
        assert d["attributes"]["x"] == "y"
        assert len(d["events"]) == 1
        assert d["duration_ms"] >= 0
        assert d["parent_id"] is None

    def test_span_parent_child(self):
        """Span records parent_id for trace hierarchy."""
        parent = TraceSpan("parent")
        child = TraceSpan("child", parent_id=parent.span_id)

        assert child.parent_id == parent.span_id
        assert parent.parent_id is None

    def test_span_duration_before_finish(self):
        """Duration is 0 before span is finished."""
        span = TraceSpan("test")
        assert span.duration_ms == 0


class TestTracer:
    """Tests for Tracer span management."""

    def test_tracer_start_end_single_span(self):
        """Tracer can start and end a single span."""
        tracer = Tracer()
        span = tracer.start_span("test.op")
        assert len(tracer._spans) == 1
        assert tracer._current is span

        tracer.end_span(span, "ok")
        assert span.end_time is not None
        assert tracer._current is None

    def test_tracer_span_hierarchy(self):
        """Tracer maintains parent-child span hierarchy."""
        tracer = Tracer()
        parent = tracer.start_span("parent.op")
        child = tracer.start_span("child.op")

        assert child.parent_id == parent.span_id
        assert tracer._current is child

        # End child → current returns to parent
        tracer.end_span(child)
        assert tracer._current is parent

        # End parent → current is None
        tracer.end_span(parent)
        assert tracer._current is None

    def test_tracer_deep_nesting(self):
        """Tracer handles 3-level deep nesting."""
        tracer = Tracer()
        s1 = tracer.start_span("l1")
        s2 = tracer.start_span("l2")
        s3 = tracer.start_span("l3")

        assert tracer._current is s3
        tracer.end_span(s3)
        assert tracer._current is s2
        tracer.end_span(s2)
        assert tracer._current is s1
        tracer.end_span(s1)
        assert tracer._current is None

    def test_tracer_get_traces(self):
        """get_traces returns list of span dicts."""
        tracer = Tracer()
        tracer.start_span("op1")
        tracer.start_span("op2")
        tracer.end_span(tracer._current)

        traces = tracer.get_traces()
        assert len(traces) == 2
        assert traces[0]["name"] == "op1"
        assert traces[1]["name"] == "op2"

    def test_tracer_clear(self):
        """Clear removes all spans and resets current."""
        tracer = Tracer()
        tracer.start_span("op1")
        tracer.start_span("op2")

        tracer.clear()
        assert len(tracer._spans) == 0
        assert tracer._current is None
        assert tracer.get_traces() == []

    def test_tracer_span_with_attributes(self):
        """Tracer.start_span accepts attributes."""
        tracer = Tracer()
        span = tracer.start_span("op", task="test-task", priority=1)
        assert span.attributes["task"] == "test-task"
        assert span.attributes["priority"] == 1
        # component is always set
        assert span.attributes["component"] == "agent-loop"

    def test_tracer_end_span_error_status(self):
        """End span with error status."""
        tracer = Tracer()
        span = tracer.start_span("op")
        tracer.end_span(span, "error")
        assert span.status == "error"

    def test_tracer_end_span_sibling_children(self):
        """Multiple children of same parent: ending one doesn't affect others."""
        tracer = Tracer()
        parent = tracer.start_span("parent")
        child1 = tracer.start_span("child1")
        tracer.end_span(child1)
        assert tracer._current is parent

        child2 = tracer.start_span("child2")
        assert child2.parent_id == parent.span_id
        tracer.end_span(child2)
        assert tracer._current is parent

    def test_tracer_end_nonexistent_parent(self):
        """Ending a span with nonexistent parent sets current=None."""
        tracer = Tracer()
        span = TraceSpan("orphan", parent_id="nonexist")
        tracer._spans.append(span)
        tracer._current = span
        tracer.end_span(span)
        assert tracer._current is None
