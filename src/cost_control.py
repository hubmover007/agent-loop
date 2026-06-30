"""Budget Control — prevent LLM call costs from exceeding limits.

CostController sits between LLMPool and the actual LLM calls, checking whether
each call stays within daily/monthly/per-task budgets before allowing execution.
After each call, it records the actual cost for tracking.

Usage:
    budget = Budget(daily_limit=5.0, monthly_limit=50.0, per_task_limit=1.0)
    controller = CostController(budget)

    if controller.check(estimated_cost=0.01, task_scope="agent-1"):
        # ... make LLM call ...
        controller.record(actual_cost=0.008, provider_id="gpt-4", tokens_in=100, tokens_out=200)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================
# Budget
# ============================================================


@dataclass
class BudgetLimitError(RuntimeError):
    """Raised when a call would exceed the budget."""
    limit_type: str  # 'daily', 'monthly', 'per_task'
    current: float
    limit: float
    estimated: float


@dataclass
class Budget:
    """Budget configuration for LLM usage.

    Attributes:
        daily_limit: Maximum daily cost ($). Default $5.
        monthly_limit: Maximum monthly cost ($). Default $50.
        per_task_limit: Maximum cost per single task ($). Default $1.
        alert_threshold: Fraction (0-1) that triggers an alert. Default 0.8.
    """
    daily_limit: float = 5.0
    monthly_limit: float = 50.0
    per_task_limit: float = 1.0
    alert_threshold: float = 0.8

    def to_dict(self) -> dict:
        return {
            "daily_limit": self.daily_limit,
            "monthly_limit": self.monthly_limit,
            "per_task_limit": self.per_task_limit,
            "alert_threshold": self.alert_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Budget":
        return cls(
            daily_limit=data.get("daily_limit", 5.0),
            monthly_limit=data.get("monthly_limit", 50.0),
            per_task_limit=data.get("per_task_limit", 1.0),
            alert_threshold=data.get("alert_threshold", 0.8),
        )


# ============================================================
# CostController
# ============================================================


class CostController:
    """Cost controller that tracks and enforces LLM spending limits.

    Attributes:
        budget: Budget configuration.
        _daily_cost: Accumulated cost for the current day.
        _monthly_cost: Accumulated cost for the current month.
        _daily_date: Date string (YYYY-MM-DD) for daily tracking.
        _monthly_key: Date string (YYYY-MM) for monthly tracking.
        _state_path: Path to persist cost state.
    """

    def __init__(self, budget: Budget | None = None,
                 state_path: str | Path = "state/cost.json"):
        """
        Args:
            budget: Budget limits. Creates default Budget($5/$50/$1) if None.
            state_path: Path to the JSON state file for persistence.
        """
        self.budget = budget or Budget()
        self._state_path = Path(state_path)
        self._daily_cost: float = 0.0
        self._monthly_cost: float = 0.0
        self._daily_date: str = ""
        self._monthly_key: str = ""
        self._lock = asyncio.Lock()

        # Per-task tracking: task_scope → accumulated cost
        self._task_costs: dict[str, float] = {}

    # ── Check ────────────────────────────────────────────────────────

    def check(self, estimated_cost: float, task_scope: str = "") -> bool:
        """Check if an LLM call fits within all budget limits.

        Args:
            estimated_cost: Estimated cost of this call ($).
            task_scope: Task identifier for per-task limit checking.

        Returns:
            True if the call is within budget, False otherwise.

        Raises:
            BudgetLimitError: If the call would exceed any limit.
        """
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        month = now.strftime("%Y-%m")

        # Reset accumulators if date changed
        if self._daily_date != today:
            self._daily_cost = 0.0
            self._daily_date = today
        if self._monthly_key != month:
            self._monthly_cost = 0.0
            self._monthly_key = month

        # Check daily limit
        if self._daily_cost + estimated_cost > self.budget.daily_limit:
            self._alert("daily", self._daily_cost, self.budget.daily_limit)
            return False

        # Check monthly limit
        if self._monthly_cost + estimated_cost > self.budget.monthly_limit:
            self._alert("monthly", self._monthly_cost, self.budget.monthly_limit)
            return False

        # Check per-task limit
        if task_scope:
            task_cost = self._task_costs.get(task_scope, 0.0)
            if task_cost + estimated_cost > self.budget.per_task_limit:
                self._alert("per_task", task_cost, self.budget.per_task_limit)
                return False

        return True

    # ── Record ───────────────────────────────────────────────────────

    def record(self, actual_cost: float, provider_id: str = "",
               tokens_in: int = 0, tokens_out: int = 0) -> None:
        """Record the actual cost of an LLM call.

        Args:
            actual_cost: Actual cost of this call ($).
            provider_id: ID of the LLM provider used.
            tokens_in: Input token count.
            tokens_out: Output token count.
        """
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        month = now.strftime("%Y-%m")

        # Reset accumulators if date changed
        if self._daily_date != today:
            self._daily_cost = 0.0
            self._daily_date = today
        if self._monthly_key != month:
            self._monthly_cost = 0.0
            self._monthly_key = month

        self._daily_cost += actual_cost
        self._monthly_cost += actual_cost

        logger.debug(
            "CostController: recorded $%.6f (%s) — daily=$%.4f, monthly=$%.2f",
            actual_cost, provider_id, self._daily_cost, self._monthly_cost,
        )

    def record_task(self, task_scope: str, actual_cost: float) -> None:
        """Accumulate cost for a specific task scope.

        Args:
            task_scope: Task identifier.
            actual_cost: Cost to add to the task.
        """
        current = self._task_costs.get(task_scope, 0.0)
        self._task_costs[task_scope] = current + actual_cost

    def reset_task(self, task_scope: str) -> None:
        """Reset accumulated cost for a task (e.g., task completed)."""
        self._task_costs.pop(task_scope, None)

    # ── Query ────────────────────────────────────────────────────────

    def get_remaining(self) -> dict:
        """Get remaining budget across all dimensions.

        Returns:
            Dict with daily_remaining, monthly_remaining, and alert flags.
        """
        now = datetime.now(timezone.utc)
        today = now.strftime("%Y-%m-%d")
        month = now.strftime("%Y-%m")

        if self._daily_date != today:
            self._daily_cost = 0.0
            self._daily_date = today
        if self._monthly_key != month:
            self._monthly_cost = 0.0
            self._monthly_key = month

        daily_ratio = self._daily_cost / max(self.budget.daily_limit, 0.001)
        monthly_ratio = self._monthly_cost / max(self.budget.monthly_limit, 0.001)

        return {
            "daily_remaining": round(max(0, self.budget.daily_limit - self._daily_cost), 4),
            "daily_spent": round(self._daily_cost, 4),
            "daily_limit": self.budget.daily_limit,
            "daily_alert": daily_ratio >= self.budget.alert_threshold,
            "monthly_remaining": round(max(0, self.budget.monthly_limit - self._monthly_cost), 4),
            "monthly_spent": round(self._monthly_cost, 4),
            "monthly_limit": self.budget.monthly_limit,
            "monthly_alert": monthly_ratio >= self.budget.alert_threshold,
            "per_task_limit": self.budget.per_task_limit,
            "alert_threshold": self.budget.alert_threshold,
        }

    def _alert(self, limit_type: str, current: float, limit: float) -> None:
        """Log an alert when approaching the budget limit."""
        ratio = current / max(limit, 0.001)
        logger.warning(
            "CostController: %s budget limit approaching! $%.4f / $%.2f (%.0f%%)",
            limit_type, current, limit, ratio * 100,
        )

    # ── Persist ──────────────────────────────────────────────────────

    async def persist(self) -> None:
        """Persist current cost state to JSON file."""
        async with self._lock:
            try:
                self._state_path.parent.mkdir(parents=True, exist_ok=True)
                state = {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "daily_cost": self._daily_cost,
                    "monthly_cost": self._monthly_cost,
                    "daily_date": self._daily_date,
                    "monthly_key": self._monthly_key,
                    "task_costs": self._task_costs,
                    "budget": self.budget.to_dict(),
                    "remaining": self.get_remaining(),
                }
                tmp = self._state_path.with_suffix(".tmp")
                tmp.write_text(json.dumps(state, indent=2, ensure_ascii=False))
                os.replace(tmp, self._state_path)
                logger.debug("CostController: persisted to %s", self._state_path)
            except Exception as e:
                logger.warning("CostController: persist failed: %s", e)

    async def load(self) -> None:
        """Load cost state from JSON file if it exists."""
        async with self._lock:
            if not self._state_path.exists():
                return
            try:
                data = json.loads(self._state_path.read_text())
                now = datetime.now(timezone.utc)

                stored_daily_date = data.get("daily_date", "")
                stored_monthly_key = data.get("monthly_key", "")

                today = now.strftime("%Y-%m-%d")
                month = now.strftime("%Y-%m")

                # Only restore if dates match (reset otherwise)
                if stored_daily_date == today:
                    self._daily_cost = data.get("daily_cost", 0.0)
                else:
                    self._daily_cost = 0.0

                if stored_monthly_key == month:
                    self._monthly_cost = data.get("monthly_cost", 0.0)
                else:
                    self._monthly_cost = 0.0

                self._daily_date = today
                self._monthly_key = month
                self._task_costs = data.get("task_costs", {})

                logger.debug("CostController: loaded from %s", self._state_path)
            except Exception as e:
                logger.warning("CostController: load failed: %s", e)
