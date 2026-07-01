"""Tests for LoopMetrics — per-phase tracking and summary."""

import time

import pytest

from src.loop_engine.metrics import LoopMetrics, PhaseMetric


class TestLoopMetrics:
    """Test LoopMetrics basic functionality."""

    def test_basic_flow(self):
        """Test basic start → phases → finish flow."""
        m = LoopMetrics()
        m.start()

        m.start_phase("input")
        time.sleep(0.01)
        m.end_phase("input")

        m.start_phase("reason")
        m.record_llm_call(prompt_tokens=100, completion_tokens=50, cost=0.001)
        m.end_phase("reason")

        m.finish()

        assert m.total_duration > 0
        assert m.total_llm_calls == 1
        assert m.total_tokens == 150
        assert m.total_prompt_tokens == 100
        assert m.total_completion_tokens == 50
        assert m.total_cost == pytest.approx(0.001)
        assert "input" in m.phases
        assert "reason" in m.phases
        assert m.phases["reason"].llm_calls == 1
        assert m.phases["reason"].tokens_used == 150

    def test_multiple_llm_calls(self):
        """Test multiple LLM calls in same phase."""
        m = LoopMetrics()
        m.start()
        m.start_phase("reason")

        m.record_llm_call(prompt_tokens=100, completion_tokens=50, cost=0.001)
        m.record_llm_call(prompt_tokens=200, completion_tokens=100, cost=0.002)

        m.end_phase("reason")
        m.finish()

        assert m.total_llm_calls == 2
        assert m.total_tokens == 450
        assert m.total_cost == pytest.approx(0.003)
        assert m.phases["reason"].llm_calls == 2

    def test_error_recording(self):
        """Test error counting."""
        m = LoopMetrics()
        m.start()
        m.start_phase("retrieve")
        m.record_error()
        m.record_error()
        m.end_phase("retrieve")
        m.finish()

        assert m.total_errors == 2
        assert m.phases["retrieve"].errors == 2

    def test_phase_duration(self):
        """Test that phase duration is measured."""
        m = LoopMetrics()
        m.start()
        m.start_phase("input")
        time.sleep(0.05)
        m.end_phase("input")
        m.finish()

        assert m.phases["input"].duration >= 0.04

    def test_summary_string(self):
        """Test that summary() returns a readable string."""
        m = LoopMetrics()
        m.start()
        m.start_phase("reason")
        m.record_llm_call(prompt_tokens=100, completion_tokens=50, cost=0.001)
        m.end_phase("reason")
        m.finish()

        s = m.summary()
        assert "Loop completed" in s
        assert "150 tokens" in s
        assert "1 LLM" in s or "1 calls" in s
        assert "reason:" in s

    def test_to_dict(self):
        """Test serialization to dict."""
        m = LoopMetrics()
        m.start()
        m.start_phase("input")
        m.end_phase("input")
        m.finish()

        d = m.to_dict()
        assert "total_duration" in d
        assert "total_tokens" in d
        assert "phases" in d
        assert "input" in d["phases"]

    def test_no_phases(self):
        """Test metrics with no phases."""
        m = LoopMetrics()
        m.start()
        m.finish()

        assert m.total_duration >= 0
        assert m.total_llm_calls == 0
        assert m.total_tokens == 0
        assert len(m.phases) == 0

    def test_multiple_phases(self):
        """Test metrics with all 8 phases."""
        m = LoopMetrics()
        m.start()

        for phase in ["input", "retrieve", "reason", "decompose",
                       "dispatch", "execute", "collect", "output"]:
            m.start_phase(phase)
            m.record_llm_call(prompt_tokens=10, completion_tokens=5, cost=0.0001)
            m.end_phase(phase)

        m.finish()

        assert len(m.phases) == 8
        assert m.total_llm_calls == 8
        assert m.total_tokens == 120  # 15 * 8

    def test_phase_metric_dataclass(self):
        """Test PhaseMetric dataclass."""
        p = PhaseMetric(name="test", start=100.0)
        p.duration = 1.5
        p.llm_calls = 2
        p.tokens_used = 200
        assert p.name == "test"
        assert p.duration == 1.5
        assert p.llm_calls == 2
        assert p.tokens_used == 200
