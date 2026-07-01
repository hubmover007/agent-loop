"""Tests for Prometheus metrics collector (src/metrics.py)."""

import pytest
from src.metrics import MetricsCollector, get_collector


class TestMetricsCollector:
    """Tests for MetricsCollector counter/histogram/gauge operations."""

    def test_counter_increment(self):
        """Counter increments with and without labels."""
        coll = MetricsCollector()
        coll.inc("test_requests_total")
        assert coll._counters["test_requests_total"][""] == 1
        coll.inc("test_requests_total")
        assert coll._counters["test_requests_total"][""] == 2

    def test_counter_with_labels(self):
        """Counter with labels creates separate label keys."""
        coll = MetricsCollector()
        coll.inc("agent_loop_tasks_total", {"status": "success"})
        coll.inc("agent_loop_tasks_total", {"status": "failed"})
        coll.inc("agent_loop_tasks_total", {"status": "success"})

        assert coll._counters["agent_loop_tasks_total"]['status="success"'] == 2
        assert coll._counters["agent_loop_tasks_total"]['status="failed"'] == 1

    def test_gauge_set(self):
        """Gauge sets and overwrites values."""
        coll = MetricsCollector()
        coll.set("agent_loop_active_agents", 5)
        assert coll._gauges["agent_loop_active_agents"] == 5
        coll.set("agent_loop_active_agents", 3)
        assert coll._gauges["agent_loop_active_agents"] == 3

    def test_histogram_observe(self):
        """Histogram records multiple observations."""
        coll = MetricsCollector()
        coll.observe("agent_loop_llm_latency_seconds", 0.5)
        coll.observe("agent_loop_llm_latency_seconds", 1.0)
        coll.observe("agent_loop_llm_latency_seconds", 1.5)

        vals = coll._histograms["agent_loop_llm_latency_seconds"][""]
        assert len(vals) == 3
        assert vals == [0.5, 1.0, 1.5]

    def test_histogram_with_labels(self):
        """Histogram with labels separates observations."""
        coll = MetricsCollector()
        coll.observe("agent_loop_llm_latency_seconds", 0.3, {"provider": "deepseek"})
        coll.observe("agent_loop_llm_latency_seconds", 0.8, {"provider": "openai"})
        coll.observe("agent_loop_llm_latency_seconds", 0.5, {"provider": "deepseek"})

        deepseek_vals = coll._histograms["agent_loop_llm_latency_seconds"]['provider="deepseek"']
        openai_vals = coll._histograms["agent_loop_llm_latency_seconds"]['provider="openai"']
        assert deepseek_vals == [0.3, 0.5]
        assert openai_vals == [0.8]


class TestPrometheusFormat:
    """Tests for Prometheus text format output."""

    def test_prometheus_format_counter(self):
        """Counter outputs Prometheus format correctly."""
        coll = MetricsCollector()
        coll.inc("test_total")
        text = coll.format_prometheus()
        assert "test_total 1" in text

    def test_prometheus_format_counter_with_labels(self):
        """Labeled counter outputs Prometheus format with labels."""
        coll = MetricsCollector()
        coll.inc("test_total", {"status": "ok"})
        text = coll.format_prometheus()
        assert 'test_total{status="ok"} 1' in text

    def test_prometheus_format_gauge(self):
        """Gauge outputs as bare value."""
        coll = MetricsCollector()
        coll.set("test_gauge", 42.0)
        text = coll.format_prometheus()
        assert "test_gauge 42.0" in text

    def test_prometheus_format_histogram(self):
        """Histogram outputs count, sum, avg."""
        coll = MetricsCollector()
        coll.observe("test_latency", 1.0)
        coll.observe("test_latency", 2.0)
        text = coll.format_prometheus()
        assert "test_latency_count 2" in text
        assert "test_latency_sum 3.0" in text

    def test_prometheus_format_empty(self):
        """Empty collector produces only newline."""
        coll = MetricsCollector()
        text = coll.format_prometheus()
        assert text == "\n"

    def test_combined_format(self):
        """Multiple metric types all appear in output."""
        coll = MetricsCollector()
        coll.inc("counter_a")
        coll.set("gauge_a", 1)
        coll.observe("hist_a", 0.5)
        text = coll.format_prometheus()
        assert "counter_a 1" in text
        assert "gauge_a 1" in text
        assert "hist_a_count 1" in text


class TestMetricsReset:
    """Tests for reset functionality."""

    def test_reset_clears_all(self):
        """Reset clears counters, gauges, and histograms."""
        coll = MetricsCollector()
        coll.inc("test_total")
        coll.set("test_gauge", 42)
        coll.observe("test_hist", 1.0)
        coll.reset()
        assert coll._counters == {}
        assert coll._gauges == {}
        assert coll._histograms == {}

    def test_reset_then_format(self):
        """After reset, format returns only newline."""
        coll = MetricsCollector()
        coll.inc("test_total")
        coll.reset()
        assert coll.format_prometheus() == "\n"


class TestGlobalCollector:
    """Tests for the global singleton."""

    def test_get_collector_returns_singleton(self):
        """get_collector returns the same instance."""
        c1 = get_collector()
        c2 = get_collector()
        assert c1 is c2


class TestMetricsEndpoint:
    """Integration tests for the /metrics endpoint."""

    @pytest.fixture
    def metrics_client(self):
        """Create a test client for the web app."""
        from unittest.mock import MagicMock, AsyncMock
        import httpx
        from src.web import create_app
        from src.loop_engine import LoopConfig

        mem = MagicMock()
        mem.store = AsyncMock(return_value="memory:ep:test")
        llm = MagicMock()
        llm.provider_name = "mock-llm"
        llm.chat = AsyncMock(return_value=MagicMock(content="{}"))

        config = LoopConfig()
        app = create_app(memory=mem, llm=llm, config=config)
        transport = httpx.ASGITransport(app=app)
        client = httpx.AsyncClient(transport=transport, base_url="http://testserver")
        return client

    @pytest.mark.asyncio
    async def test_metrics_endpoint_returns_200(self, metrics_client):
        """GET /metrics returns 200 OK."""
        async with metrics_client as client:
            resp = await client.get("/metrics")
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_endpoint_content_type(self, metrics_client):
        """GET /metrics returns text/plain content type."""
        async with metrics_client as client:
            resp = await client.get("/metrics")
            assert "text/plain" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_metrics_endpoint_has_standard_metric(self, metrics_client):
        """GET /metrics includes standard metrics after some activity."""
        from src.metrics import get_collector as gc
        gc().inc("agent_loop_tasks_total", {"status": "success"})
        gc().set("agent_loop_active_agents", 1)

        async with metrics_client as client:
            resp = await client.get("/metrics")
            text = resp.text
            assert "agent_loop_tasks_total" in text
            assert "agent_loop_active_agents" in text

    @pytest.mark.asyncio
    async def test_api_metrics_endpoint(self, metrics_client):
        """GET /api/metrics also returns metrics."""
        async with metrics_client as client:
            resp = await client.get("/api/metrics")
            assert resp.status_code == 200
            assert "text/plain" in resp.headers.get("content-type", "")
