"""Tests for src/interaction.py — Human-in-the-Loop approval system."""

import asyncio
import pytest
from src.interaction import (
    InteractionHub, ApprovalRequest,
    detect_risk_level, is_high_risk,
    HIGH_RISK_PATTERNS, RISK_LEVEL_ORDER,
)


class TestRiskDetection:
    """Tests for high-risk operation detection."""

    def test_detect_rm_command(self):
        assert detect_risk_level("rm -rf /tmp") == "critical"
        assert detect_risk_level("rm /var/log/app.log") == "critical"

    def test_detect_delete_command(self):
        assert detect_risk_level("delete from users") == "high"
        assert detect_risk_level("DELETE * FROM orders") == "high"

    def test_detect_drop_command(self):
        assert detect_risk_level("drop table users") == "critical"

    def test_detect_kill_command(self):
        assert detect_risk_level("kill -9 1234") == "high"

    def test_detect_reboot_command(self):
        assert detect_risk_level("sudo reboot") == "high"

    def test_detect_sudo_command(self):
        assert detect_risk_level("sudo apt update") == "critical"

    def test_detect_chmod_chown(self):
        assert detect_risk_level("chmod 777 /etc/passwd") == "medium"
        assert detect_risk_level("chown root:root /app") == "medium"

    def test_no_risk_for_safe_commands(self):
        assert detect_risk_level("ls -la") is None
        assert detect_risk_level("echo hello") is None
        assert detect_risk_level("cat /etc/hosts") is None
        assert detect_risk_level("") is None

    def test_is_high_risk_with_threshold(self):
        assert is_high_risk("rm file.txt", threshold="medium") is True
        assert is_high_risk("chmod 644 file.txt", threshold="medium") is True
        assert is_high_risk("chmod 644 file.txt", threshold="high") is False
        assert is_high_risk("sudo apt update", threshold="critical") is True
        assert is_high_risk("sudo reboot", threshold="critical") is False
        assert is_high_risk("kill process", threshold="critical") is False

    def test_low_risk_no_interrupt(self):
        """Low-risk operations should not trigger interrupt at medium threshold."""
        assert is_high_risk("ls -la", threshold="medium") is False
        assert is_high_risk("echo test", threshold="medium") is False


class TestApprovalFlow:
    """Tests for approval request flow."""

    @pytest.mark.asyncio
    async def test_approval_flow(self):
        """Request → Approve → Complete flow."""
        hub = InteractionHub(risk_threshold="low")

        async def approve_later():
            await asyncio.sleep(0.05)
            await hub.approve(list(hub._pending.keys())[0], "go ahead")

        asyncio.create_task(approve_later())

        req = await hub.request_approval(
            agent_id="agent-1",
            action="rm -rf /data",
            details="Deleting production data",
            risk_level="critical",
            timeout_seconds=5,
        )

        assert req.status == "approved"
        assert req.reply == "go ahead"
        assert hub.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_deny_flow(self):
        """Request → Deny flow."""
        hub = InteractionHub(risk_threshold="low")

        async def deny_later():
            await asyncio.sleep(0.05)
            await hub.deny(list(hub._pending.keys())[0], "not safe")

        asyncio.create_task(deny_later())

        req = await hub.request_approval(
            agent_id="agent-2",
            action="drop table users",
            details="Dropping user table",
            risk_level="critical",
            timeout_seconds=5,
        )

        assert req.status == "denied"
        assert req.reply == "not safe"
        assert hub.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_timeout(self):
        """Request should expire after timeout."""
        hub = InteractionHub(risk_threshold="low")

        req = await hub.request_approval(
            agent_id="agent-3",
            action="reboot server",
            details="Restarting production",
            risk_level="high",
            timeout_seconds=0.1,
        )

        assert req.status == "expired"
        assert req.reply is not None
        assert "Timeout" in req.reply
        assert hub.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_low_risk_no_interrupt(self):
        """Low-risk actions below threshold should auto-approve."""
        hub = InteractionHub(risk_threshold="high")  # Only high+ need approval

        req = await hub.request_approval(
            agent_id="agent-4",
            action="chmod 644 file.txt",
            details="Setting permissions",
            risk_level="medium",
            timeout_seconds=5,
        )

        # Should auto-approve because medium < high threshold
        assert req.status == "approved"
        assert "below threshold" in (req.reply or "")
        assert hub.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_get_pending(self):
        """get_pending() returns all waiting requests."""
        hub = InteractionHub(risk_threshold="low")

        # Start a request but don't approve it
        async def request_task():
            await hub.request_approval(
                agent_id="agent-5",
                action="delete all",
                details="Test",
                risk_level="high",
                timeout_seconds=5,
            )

        asyncio.create_task(request_task())
        await asyncio.sleep(0.05)

        pending = hub.get_pending()
        assert len(pending) == 1
        assert pending[0].agent_id == "agent-5"
        assert pending[0].status == "pending"

    @pytest.mark.asyncio
    async def test_multiple_requests(self):
        """Multiple agents can request approval concurrently."""
        hub = InteractionHub(risk_threshold="low")

        async def request_and_wait(agent_id: str):
            return await hub.request_approval(
                agent_id=agent_id,
                action="test action",
                details="test",
                risk_level="high",
                timeout_seconds=0.5,
            )

        # Start 3 concurrent requests
        tasks = [
            asyncio.create_task(request_and_wait(f"agent-{i}"))
            for i in range(3)
        ]

        await asyncio.sleep(0.05)

        # All 3 should be pending
        assert hub.get_pending_count() == 3

        # Approve all
        for req in list(hub.get_pending()):
            await hub.approve(req.id, "ok")

        results = await asyncio.gather(*tasks)
        for r in results:
            assert r.status == "approved"

        assert hub.get_pending_count() == 0

    @pytest.mark.asyncio
    async def test_deny_nonexistent_request(self):
        """Denying a non-existent request should raise ValueError."""
        hub = InteractionHub()
        with pytest.raises(ValueError, match="not found"):
            await hub.deny("nonexistent-id", "no")

    @pytest.mark.asyncio
    async def test_approve_nonexistent_request(self):
        """Approving a non-existent request should raise ValueError."""
        hub = InteractionHub()
        with pytest.raises(ValueError, match="not found"):
            await hub.approve("nonexistent-id", "ok")

    def test_risk_threshold_property(self):
        """risk_threshold can be read and written."""
        hub = InteractionHub()
        assert hub.risk_threshold == "medium"

        hub.risk_threshold = "high"
        assert hub.risk_threshold == "high"

        with pytest.raises(ValueError, match="Invalid risk threshold"):
            hub.risk_threshold = "invalid"

    def test_approval_request_to_dict(self):
        """ApprovalRequest.to_dict() serializes correctly."""
        req = ApprovalRequest(
            id="test-id",
            agent_id="agent-1",
            action="test",
            details="details",
            risk_level="high",
        )
        d = req.to_dict()
        assert d["id"] == "test-id"
        assert d["agent_id"] == "agent-1"
        assert d["risk_level"] == "high"
        assert d["status"] == "pending"
