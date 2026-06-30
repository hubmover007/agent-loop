"""Tests for ProgressEmitter streaming output."""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from src.streaming import ProgressEvent, ProgressEmitter


class TestProgressEvent:
    """Tests for ProgressEvent data class."""

    def test_roundtrip(self):
        ev = ProgressEvent(
            type="phase_start", agent_id="a1", phase="PLAN",
            message="Starting plan",
            data={"step": 0},
        )
        d = ev.to_dict()
        restored = ProgressEvent.from_dict(d)
        assert restored.type == "phase_start"
        assert restored.agent_id == "a1"
        assert restored.data["step"] == 0

    def test_defaults(self):
        ev = ProgressEvent(
            type="tool_call", agent_id="a1", phase="EXECUTE",
            message="Calling tool",
        )
        assert ev.timestamp != ""
        assert ev.data == {}


@pytest.mark.asyncio
class TestProgressEmitterCore:
    """Core emitter functionality tests."""

    async def test_emit_and_subscribe(self):
        """Emitted events are received by subscriber queue."""
        emitter = ProgressEmitter("agent-1")
        q = await emitter.subscribe()

        emitter.emit("phase_start", "PLAN", "Starting plan", step=0)

        # Should be available immediately
        event = q.get_nowait()
        assert event.type == "phase_start"
        assert event.phase == "PLAN"
        assert event.message == "Starting plan"
        assert event.data["step"] == 0

    async def test_callback(self):
        """Synchronous callbacks are called on emit."""
        emitter = ProgressEmitter("agent-1")
        received = []

        def collector(ev: ProgressEvent):
            received.append(ev)

        emitter.on_event(collector)
        emitter.emit("phase_done", "PLAN", "Done planning")

        assert len(received) == 1
        assert received[0].type == "phase_done"
        assert received[0].message == "Done planning"

    async def test_callback_error_isolation(self):
        """Callback exceptions don't disrupt emitter or other callbacks."""
        emitter = ProgressEmitter("agent-1")
        called = [False, False]

        def bad_callback(ev):
            called[0] = True
            raise RuntimeError("oops")

        def good_callback(ev):
            called[1] = True

        emitter.on_event(bad_callback)
        emitter.on_event(good_callback)
        emitter.emit("phase_start", "TEST", "should not crash")

        assert called[0] is True
        assert called[1] is True  # good callback still fires

    async def test_log_file(self):
        """Events are written to JSONL log file when configured."""
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "progress.jsonl"
            emitter = ProgressEmitter("agent-2", log_path=str(log_path))

            emitter.emit("phase_start", "PLAN", "plan a")
            emitter.emit("phase_done", "PLAN", "plan done")

            assert log_path.exists()
            lines = log_path.read_text().strip().split("\n")
            assert len(lines) == 2

            data0 = json.loads(lines[0])
            assert data0["type"] == "phase_start"
            assert data0["agent_id"] == "agent-2"

    async def test_multiple_subscribers(self):
        """Multiple subscriber queues all receive events."""
        emitter = ProgressEmitter("agent-3")
        q1 = await emitter.subscribe()
        q2 = await emitter.subscribe()
        q3 = await emitter.subscribe()

        emitter.emit("tool_call", "EXECUTE", "calling ssh")

        # All three queues should have the event
        for q in [q1, q2, q3]:
            event = q.get_nowait()
            assert event.type == "tool_call"
            assert event.message == "calling ssh"

    async def test_unsubscribe(self):
        """Unsubscribed queue no longer receives events."""
        emitter = ProgressEmitter("agent-4")
        q1 = await emitter.subscribe()
        q2 = await emitter.subscribe()

        emitter.unsubscribe(q1)
        emitter.emit("phase_start", "TEST", "msg")

        # q1 should be empty now (already removed)
        assert q1.empty()
        # q2 should have the event
        event = q2.get_nowait()
        assert event.type == "phase_start"

    async def test_detach(self):
        """detach clears all subscribers and callbacks."""
        emitter = ProgressEmitter("agent-5")
        q = await emitter.subscribe()
        called = []

        emitter.on_event(lambda ev: called.append(ev))
        emitter.detach()

        emitter.emit("phase_start", "TEST", "msg")

        # Queue is detached but the emit might still push to it.
        # The key test: callback should NOT fire.
        assert len(called) == 0

    async def test_stream_to_sink_sync(self):
        """stream_to_sink forwards events to a sync callable."""
        emitter = ProgressEmitter("agent-6")
        received = []

        def sink(ev: ProgressEvent):
            received.append(ev.message)

        # Start streaming in background
        task = asyncio.create_task(emitter.stream_to_sink(sink))

        # Give the consumer loop time to subscribe and start consuming
        await asyncio.sleep(0.05)

        # Emit some events
        emitter.emit("phase_start", "PLAN", "hello")
        emitter.emit("phase_done", "PLAN", "world")

        # Give it a moment to process
        await asyncio.sleep(0.15)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "hello" in received
        assert "world" in received

    async def test_stream_to_sink_async(self):
        """stream_to_sink forwards events to an async callable."""
        emitter = ProgressEmitter("agent-7")
        received = []

        async def sink(ev: ProgressEvent):
            received.append(ev.message)

        task = asyncio.create_task(emitter.stream_to_sink(sink))

        # Give the consumer loop time to subscribe and start consuming
        await asyncio.sleep(0.05)

        emitter.emit("tool_call", "EXECUTE", "async hello")
        await asyncio.sleep(0.15)

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "async hello" in received


# ============================================================
# P0: Token & Tool Call Streaming
# ============================================================

@pytest.mark.asyncio
class TestStreamingTokens:
    """Tests for P0 streaming events: token, tool_call, tool_result."""

    async def test_emit_token(self):
        """emit_token should emit a 'token' type progress event."""
        emitter = ProgressEmitter("agent-tok")
        q = await emitter.subscribe()

        emitter.emit_token("Hello")
        emitter.emit_token(" world")

        ev1 = q.get_nowait()
        assert ev1.type == "token"
        assert ev1.message == "Hello"

        ev2 = q.get_nowait()
        assert ev2.type == "token"
        assert ev2.message == " world"

    async def test_emit_tool_call(self):
        """emit_tool_call should emit a 'tool_call' type event with tool name and args."""
        emitter = ProgressEmitter("agent-tc")
        q = await emitter.subscribe()

        emitter.emit_tool_call("search", {"query": "weather"})

        ev = q.get_nowait()
        assert ev.type == "tool_call"
        assert ev.data["tool"] == "search"
        assert ev.data["args"] == {"query": "weather"}

    async def test_emit_tool_result(self):
        """emit_tool_result should emit a 'tool_result' type event."""
        emitter = ProgressEmitter("agent-tr")
        q = await emitter.subscribe()

        emitter.emit_tool_result("search", "Found: sunny, 25°C")

        ev = q.get_nowait()
        assert ev.type == "tool_result"
        assert ev.data["tool"] == "search"
        assert ev.data["result"] == "Found: sunny, 25°C"

    async def test_loop_streaming_integration(self):
        """AgentLoop integration: emitter receives events during execution."""
        from src.loop_engine import AgentLoop, LoopConfig, ToolLoop
        from src.tool_registry import ToolRegistry

        # Mock LLM that returns a simple plan
        class StreamTestLLM:
            async def chat(self, messages, **kwargs):
                from src.loop_engine import LLMResponse
                import json
                return LLMResponse(
                    content=json.dumps([
                        {"description": "Say hello", "tool": None,
                         "params": {}, "output_key": None},
                    ]),
                    model="mock",
                )

            async def embed(self, text):
                return [[0.0]]

        emitter = ProgressEmitter("agent-loop-stream")
        q = await emitter.subscribe()

        config = LoopConfig(max_agent_steps=3)
        llm = StreamTestLLM()
        tool_loop = ToolLoop(ToolRegistry(), config)

        agent = AgentLoop(
            tool_loop=tool_loop,
            llm=llm,
            config=config,
            emitter=emitter,
        )

        result = await agent.run(
            agent_id="stream-test",
            task_scope="Simple test",
            context={"task_id": "st-1"},
            allowed_tools=[],
        )

        # Should have received phase_start and phase_done events
        events = []
        while not q.empty():
            events.append(q.get_nowait())

        event_types = {ev.type for ev in events}
        assert "phase_start" in event_types or "phase_done" in event_types
        # At minimum, some events were emitted
        assert len(events) > 0
