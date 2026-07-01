"""Prometheus metrics for Agent-Loop.

Metrics exposed at /metrics endpoint:
  - agent_loop_tasks_total{status} — 任务总数（按状态）
  - agent_loop_iterations_total — 总迭代次数
  - agent_loop_llm_calls_total{provider} — LLM 调用次数
  - agent_loop_llm_latency_seconds{provider} — LLM 延迟
  - agent_loop_tool_calls_total{tool} — 工具调用次数
  - agent_loop_tool_errors_total{tool} — 工具错误次数
  - agent_loop_active_agents — 活跃 Agent 数
  - agent_loop_cost_total{provider} — 总花费
"""

import time
from collections import defaultdict
from typing import Any


class MetricsCollector:
    """轻量级 metrics 收集器（不依赖 prometheus_client）"""

    def __init__(self):
        self._counters: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._gauges: dict[str, float] = defaultdict(float)
        self._histograms: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    def inc(self, metric: str, labels: dict[str, str] | None = None, value: float = 1):
        """递增计数器"""
        key = self._label_key(labels)
        self._counters[metric][key] += value

    def set(self, metric: str, value: float):
        """设置 gauge"""
        self._gauges[metric] = value

    def observe(self, metric: str, value: float, labels: dict[str, str] | None = None):
        """记录直方图值"""
        key = self._label_key(labels)
        self._histograms[metric][key].append(value)

    def _label_key(self, labels: dict[str, str] | None) -> str:
        if not labels:
            return ""
        return ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))

    def format_prometheus(self) -> str:
        """输出 Prometheus 格式文本"""
        lines = []
        for metric, label_map in self._counters.items():
            for label_str, value in label_map.items():
                if label_str:
                    lines.append(f'{metric}{{{label_str}}} {value}')
                else:
                    lines.append(f'{metric} {value}')
        for metric, value in self._gauges.items():
            lines.append(f'{metric} {value}')
        for metric, label_map in self._histograms.items():
            for label_str, values in label_map.items():
                if values:
                    if label_str:
                        lines.append(f'{metric}_count{{{label_str}}} {len(values)}')
                        lines.append(f'{metric}_sum{{{label_str}}} {sum(values)}')
                        lines.append(f'{metric}_avg{{{label_str}}} {sum(values)/len(values):.4f}')
                    else:
                        lines.append(f'{metric}_count {len(values)}')
                        lines.append(f'{metric}_sum {sum(values)}')
                        lines.append(f'{metric}_avg {sum(values)/len(values):.4f}')
        return "\n".join(lines) + "\n"

    def reset(self):
        """Reset all metrics."""
        self._counters.clear()
        self._gauges.clear()
        self._histograms.clear()


# 全局单例
_collector = MetricsCollector()


def get_collector() -> MetricsCollector:
    return _collector
