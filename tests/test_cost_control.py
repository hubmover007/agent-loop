"""Tests for src/cost_control.py — Budget-based LLM cost control."""

import json
import pytest
import tempfile
from pathlib import Path
from src.cost_control import Budget, CostController, BudgetLimitError


class TestBudget:
    """Tests for Budget dataclass."""

    def test_defaults(self):
        """Budget has sensible defaults."""
        b = Budget()
        assert b.daily_limit == 5.0
        assert b.monthly_limit == 50.0
        assert b.per_task_limit == 1.0
        assert b.alert_threshold == 0.8

    def test_custom_limits(self):
        """Custom limits are set correctly."""
        b = Budget(daily_limit=10.0, monthly_limit=100.0, per_task_limit=2.0, alert_threshold=0.5)
        assert b.daily_limit == 10.0
        assert b.alert_threshold == 0.5

    def test_to_dict_from_dict(self):
        """Serialization round-trip works."""
        b = Budget(daily_limit=7.5, monthly_limit=75.0, per_task_limit=1.5, alert_threshold=0.6)
        d = b.to_dict()
        restored = Budget.from_dict(d)
        assert restored.daily_limit == 7.5
        assert restored.monthly_limit == 75.0
        assert restored.per_task_limit == 1.5
        assert restored.alert_threshold == 0.6


class TestWithinBudget:
    """Tests for budget checks that pass."""

    def test_within_budget(self):
        """Small call within budget should pass."""
        controller = CostController(Budget(daily_limit=5.0, monthly_limit=50.0))
        assert controller.check(0.01) is True

    def test_multiple_small_calls(self):
        """Multiple small calls within budget should all pass."""
        controller = CostController(Budget(daily_limit=5.0))
        for _ in range(10):
            assert controller.check(0.1) is True
            controller.record(0.1, "gpt-4", 100, 200)
        # 10 * 0.1 = 1.0, still within $5
        assert controller.check(0.5) is True

    def test_get_remaining_defaults(self):
        """get_remaining() returns correct defaults."""
        controller = CostController()
        remaining = controller.get_remaining()
        assert remaining["daily_remaining"] == 5.0
        assert remaining["monthly_remaining"] == 50.0
        assert not remaining["daily_alert"]
        assert not remaining["monthly_alert"]


class TestExceedLimits:
    """Tests for budget limit violations."""

    def test_exceed_daily(self):
        """Check fails when daily limit exceeded."""
        controller = CostController(Budget(daily_limit=1.0))
        controller._daily_cost = 0.95  # Simulate near-limit
        assert controller._daily_date == ""  # Manually set date to prevent reset
        # We need to simulate: the daily limit check happens before reset
        # So set date manually
        from datetime import datetime, timezone
        controller._daily_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        assert controller.check(0.1) is False  # 0.95 + 0.1 > 1.0

    def test_exceed_monthly(self):
        """Check fails when monthly limit exceeded."""
        controller = CostController(Budget(monthly_limit=10.0))
        from datetime import datetime, timezone
        controller._monthly_key = datetime.now(timezone.utc).strftime("%Y-%m")
        controller._monthly_cost = 9.9
        assert controller.check(0.5) is False  # 9.9 + 0.5 > 10.0

    def test_exceed_per_task(self):
        """Check fails when per-task limit exceeded."""
        controller = CostController(Budget(per_task_limit=0.5))
        controller.record_task("task-1", 0.4)
        assert controller.check(0.2, task_scope="task-1") is False  # 0.4 + 0.2 > 0.5


class TestRecordAndRemaining:
    """Tests for record() and get_remaining()."""

    def test_record_updates_daily(self):
        """Record updates daily cost."""
        controller = CostController(Budget(daily_limit=10.0))
        controller.record(0.5, "gpt-4", 1000, 2000)
        remaining = controller.get_remaining()
        assert remaining["daily_spent"] == 0.5
        assert remaining["daily_remaining"] == 9.5

    def test_record_updates_monthly(self):
        """Record updates monthly cost."""
        controller = CostController(Budget(monthly_limit=100.0))
        controller.record(2.5, "gpt-4", 5000, 10000)
        remaining = controller.get_remaining()
        assert remaining["monthly_spent"] == 2.5
        assert remaining["monthly_remaining"] == 97.5

    def test_multiple_records_accumulate(self):
        """Multiple records accumulate correctly."""
        controller = CostController(Budget(daily_limit=10.0, monthly_limit=100.0))
        controller.record(1.0, "p1", 100, 200)
        controller.record(2.0, "p2", 200, 400)
        controller.record(0.5, "p1", 50, 100)

        remaining = controller.get_remaining()
        assert remaining["daily_spent"] == 3.5
        assert remaining["monthly_spent"] == 3.5
        assert remaining["daily_remaining"] == 6.5
        assert remaining["monthly_remaining"] == 96.5

    def test_record_task_accumulates(self):
        """record_task accumulates per-task costs."""
        controller = CostController(Budget(per_task_limit=10.0))
        controller.record_task("task-a", 0.5)
        controller.record_task("task-a", 0.3)
        controller.record_task("task-b", 0.2)

        assert controller._task_costs["task-a"] == 0.8
        assert controller._task_costs["task-b"] == 0.2

    def test_reset_task_clears_cost(self):
        """reset_task clears per-task cost."""
        controller = CostController()
        controller.record_task("task-a", 0.9)
        controller.reset_task("task-a")
        assert "task-a" not in controller._task_costs
        assert controller.check(0.9, task_scope="task-a") is True


class TestAlertThreshold:
    """Tests for alert threshold behavior."""

    def test_alert_threshold_daily(self):
        """Alert triggers at 80% of daily limit by default."""
        controller = CostController(Budget(daily_limit=10.0))
        controller.record(8.1, "p1", 1000, 2000)  # 81%
        remaining = controller.get_remaining()
        assert remaining["daily_alert"] is True

    def test_no_alert_below_threshold(self):
        """No alert below threshold."""
        controller = CostController(Budget(daily_limit=10.0))
        controller.record(7.0, "p1", 1000, 2000)  # 70%
        remaining = controller.get_remaining()
        assert remaining["daily_alert"] is False

    def test_alert_threshold_monthly(self):
        """Alert triggers at 80% of monthly limit."""
        controller = CostController(Budget(monthly_limit=100.0))
        controller.record(81.0, "p1", 1000, 2000)  # 81%
        remaining = controller.get_remaining()
        assert remaining["monthly_alert"] is True

    def test_custom_alert_threshold(self):
        """Custom alert threshold is respected."""
        controller = CostController(Budget(daily_limit=10.0, alert_threshold=0.5))
        controller.record(5.1, "p1", 1000, 2000)  # 51% > 50%
        remaining = controller.get_remaining()
        assert remaining["daily_alert"] is True


class TestPersistence:
    """Tests for persist() and load()."""

    @pytest.mark.asyncio
    async def test_persist_and_load(self):
        """Persist and load round-trip works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "cost.json"
            controller = CostController(
                Budget(daily_limit=10.0),
                state_path=state_path,
            )
            controller.record(3.5, "gpt-4", 1000, 2000)
            controller.record_task("task-x", 0.5)

            await controller.persist()
            assert state_path.exists()

            # Load into new controller
            loaded = CostController(
                Budget(daily_limit=10.0),
                state_path=state_path,
            )
            await loaded.load()

            assert loaded._daily_cost == 3.5
            assert loaded._task_costs.get("task-x") == 0.5

    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self):
        """Loading non-existent file doesn't crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = CostController(state_path=Path(tmpdir) / "nonexistent.json")
            await controller.load()
            assert controller._daily_cost == 0.0

    @pytest.mark.asyncio
    async def test_load_stale_daily_reset(self):
        """Loading stale daily data resets to 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            state_path = Path(tmpdir) / "cost.json"
            # Write state with yesterday's date
            from datetime import datetime, timezone, timedelta
            yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
            state_path.write_text(json.dumps({
                "daily_cost": 4.0,
                "monthly_cost": 20.0,
                "daily_date": yesterday,
                "monthly_key": datetime.now(timezone.utc).strftime("%Y-%m"),
            }))

            controller = CostController(state_path=state_path)
            await controller.load()
            assert controller._daily_cost == 0.0  # Reset because yesterday
            assert controller._monthly_cost == 20.0  # Same month, kept
