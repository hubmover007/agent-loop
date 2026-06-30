"""Human-in-the-Loop — approval-based interrupt for high-risk operations.

When an Agent attempts a high-risk operation, InteractionHub pauses execution,
sends an approval request to the user, and waits for confirmation before proceeding.

Usage:
    hub = InteractionHub()
    req = await hub.request_approval("agent-1", "rm -rf /data", "Deleting production data", "critical")
    # Hub blocks until user calls approve()/deny() or timeout expires
    if req.status == "approved":
        ...  # proceed with operation
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# ============================================================
# High-risk operation patterns
# ============================================================

HIGH_RISK_PATTERNS: list[tuple[str, str]] = [
    ("rm -rf", "critical"),
    ("rm", "high"),
    ("delete", "high"),
    ("drop", "critical"),
    ("kill", "high"),
    ("reboot", "high"),
    ("sudo", "critical"),
    ("chmod", "medium"),
    ("chown", "medium"),
]

RISK_LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def detect_risk_level(command: str) -> str | None:
    """Detect risk level from a command/action string.

    Returns the risk level string ('low'/'medium'/'high'/'critical') or None
    if no risk pattern is detected.
    """
    if not command:
        return None
    cmd_lower = command.lower()
    for pattern, level in HIGH_RISK_PATTERNS:
        if pattern in cmd_lower:
            return level
    return None


def is_high_risk(command: str, threshold: str = "medium") -> bool:
    """Check if a command exceeds the given risk threshold.

    Args:
        command: The command or action string to check.
        threshold: Minimum risk level that triggers an interrupt.
                   One of: 'low', 'medium', 'high', 'critical'.

    Returns True if the command's risk level >= threshold.
    """
    risk = detect_risk_level(command)
    if risk is None:
        return False
    return RISK_LEVEL_ORDER.get(risk, 0) >= RISK_LEVEL_ORDER.get(threshold, 1)


# ============================================================
# ApprovalRequest
# ============================================================


@dataclass
class ApprovalRequest:
    """An approval request sent by an Agent to the user.

    Attributes:
        id: Unique request ID (uuid).
        agent_id: Agent that initiated the request.
        task_scope: The task context in which the action is being taken.
        action: Human-readable description of the action.
        details: Detailed information about the action.
        risk_level: One of 'low', 'medium', 'high', 'critical'.
        created_at: ISO 8601 timestamp.
        status: Current status: 'pending', 'approved', 'denied', 'expired'.
        timeout_seconds: How long to wait before auto-expiring.
        reply: User-provided reply (only set after approval/denial).
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    task_scope: str = ""
    action: str = ""
    details: str = ""
    risk_level: str = "medium"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "pending"  # pending | approved | denied | expired
    timeout_seconds: int = 300  # 5 minutes
    reply: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "agent_id": self.agent_id,
            "task_scope": self.task_scope,
            "action": self.action,
            "details": self.details,
            "risk_level": self.risk_level,
            "created_at": self.created_at,
            "status": self.status,
            "timeout_seconds": self.timeout_seconds,
            "reply": self.reply,
        }


# ============================================================
# InteractionHub
# ============================================================


class InteractionHub:
    """Human-in-the-loop interaction hub.

    Singleton held by AgentManagerAgent. Agents request approval through it
    before executing high-risk operations. The hub blocks the Agent until
    the user approves, denies, or the request times out.

    Usage:
        # Agent side
        req = await hub.request_approval("agent-1", "rm -rf /data",
                                         "Deleting production data", "critical")
        if req.status != "approved":
            raise RuntimeError("User denied the operation")

        # User side (via UI or CLI)
        await hub.approve(req.id, "ok, go ahead")
    """

    def __init__(self, risk_threshold: str = "medium"):
        """
        Args:
            risk_threshold: Minimum risk level that triggers approval.
                            One of: 'low', 'medium', 'high', 'critical'.
        """
        self._pending: dict[str, ApprovalRequest] = {}
        self._events: dict[str, asyncio.Event] = {}
        self._risk_threshold = risk_threshold

    # ── Agent-facing ─────────────────────────────────────────────────

    async def request_approval(
        self,
        agent_id: str,
        action: str,
        details: str = "",
        risk_level: str = "medium",
        task_scope: str = "",
        timeout_seconds: int = 300,
    ) -> ApprovalRequest:
        """Request user approval for an action. Blocks until response or timeout.

        Args:
            agent_id: ID of the requesting agent.
            action: Human-readable action description.
            details: Detailed context for the user.
            risk_level: 'low' | 'medium' | 'high' | 'critical'.
            task_scope: Task context string.
            timeout_seconds: How long to wait (default 300s).

        Returns:
            ApprovalRequest with status 'approved', 'denied', or 'expired'.
        """
        # Skip if risk below threshold
        if RISK_LEVEL_ORDER.get(risk_level, 1) < RISK_LEVEL_ORDER.get(self._risk_threshold, 1):
            req = ApprovalRequest(
                agent_id=agent_id,
                task_scope=task_scope,
                action=action,
                details=details,
                risk_level=risk_level,
                timeout_seconds=timeout_seconds,
            )
            req.status = "approved"
            req.reply = "auto-approved (below threshold)"
            return req

        req = ApprovalRequest(
            agent_id=agent_id,
            task_scope=task_scope,
            action=action,
            details=details,
            risk_level=risk_level,
            timeout_seconds=timeout_seconds,
        )
        event = asyncio.Event()

        self._pending[req.id] = req
        self._events[req.id] = event

        logger.info(
            "InteractionHub: approval requested by '%s' — %s [%s]",
            agent_id, action, risk_level,
        )

        try:
            # Wait for event or timeout
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                req.status = "expired"
                req.reply = "Timeout expired — no user response"
                logger.warning(
                    "InteractionHub: request %s expired after %ds",
                    req.id, timeout_seconds,
                )
                return req

            # Return the request with user's response
            return self._pending.get(req.id, req)
        finally:
            self._pending.pop(req.id, None)
            self._events.pop(req.id, None)

    # ── User-facing ──────────────────────────────────────────────────

    async def approve(self, request_id: str, reply: str = "") -> ApprovalRequest:
        """Approve a pending request.

        Args:
            request_id: ID of the pending ApprovalRequest.
            reply: Optional reply message from the user.

        Returns:
            Updated ApprovalRequest. Raises ValueError if request not found.
        """
        req = self._pending.get(request_id)
        if req is None:
            raise ValueError(f"ApprovalRequest '{request_id}' not found or already resolved")

        req.status = "approved"
        req.reply = reply or "Approved by user"
        logger.info("InteractionHub: request %s approved by user", request_id)

        event = self._events.get(request_id)
        if event:
            event.set()

        return req

    async def deny(self, request_id: str, reply: str = "") -> ApprovalRequest:
        """Deny a pending request.

        Args:
            request_id: ID of the pending ApprovalRequest.
            reply: Optional reason for denial.

        Returns:
            Updated ApprovalRequest. Raises ValueError if request not found.
        """
        req = self._pending.get(request_id)
        if req is None:
            raise ValueError(f"ApprovalRequest '{request_id}' not found or already resolved")

        req.status = "denied"
        req.reply = reply or "Denied by user"
        logger.info("InteractionHub: request %s denied by user", request_id)

        event = self._events.get(request_id)
        if event:
            event.set()

        return req

    # ── Query ────────────────────────────────────────────────────────

    def get_pending(self) -> list[ApprovalRequest]:
        """Get all currently pending approval requests (for UI polling)."""
        return list(self._pending.values())

    def get_pending_count(self) -> int:
        """Get count of pending requests."""
        return len(self._pending)

    @property
    def risk_threshold(self) -> str:
        return self._risk_threshold

    @risk_threshold.setter
    def risk_threshold(self, value: str) -> None:
        if value not in RISK_LEVEL_ORDER:
            raise ValueError(f"Invalid risk threshold: {value}")
        self._risk_threshold = value
