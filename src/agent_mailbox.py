"""Agent-to-Agent Communication — mailbox-based messaging system.

Agents can send messages to each other for delegation, questions, status updates,
and handoffs. Each Agent has a personal Mailbox; a global MailRouter routes
messages between mailboxes.

Usage:
    router = MailRouter()
    mailbox_a = router.register("agent-a")
    mailbox_b = router.register("agent-b")

    msg = await mailbox_a.send("agent-b", "ask", "Help!", "How do I fix this error?")
    reply = await mailbox_b.receive()
    await mailbox_b.reply(reply, "Try restarting the service.")
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ============================================================
# AgentMessage
# ============================================================


@dataclass
class AgentMessage:
    """A message between two agents.

    Attributes:
        id: Unique message ID (uuid).
        from_agent: Sender agent ID.
        to_agent: Recipient agent ID, or 'any' for broadcast.
        type: Message type: 'delegate', 'ask', 'inform', 'handoff'.
        subject: Short subject line.
        body: Full message body.
        created_at: ISO 8601 timestamp.
        reply_to: ID of the message this is replying to.
        status: 'unread' or 'read'.
    """
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    from_agent: str = ""
    to_agent: str = ""
    type: str = "inform"  # delegate | ask | inform | handoff
    subject: str = ""
    body: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reply_to: str | None = None
    status: str = "unread"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "from_agent": self.from_agent,
            "to_agent": self.to_agent,
            "type": self.type,
            "subject": self.subject,
            "body": self.body,
            "created_at": self.created_at,
            "reply_to": self.reply_to,
            "status": self.status,
        }


# ============================================================
# AgentMailbox
# ============================================================


class AgentMailbox:
    """A single Agent's message mailbox.

    Each Agent gets one Mailbox. Messages arrive via MailRouter.route()
    and are consumed via receive() or peek().

    Usage:
        mailbox = AgentMailbox("agent-1")
        await mailbox.send("agent-2", "ask", "Help", "...")
        msg = await mailbox.receive()  # blocks until a message arrives
    """

    def __init__(self, agent_id: str, router: MailRouter | None = None):
        self.agent_id = agent_id
        self._router = router  # Back-reference for send()
        self._inbox: list[AgentMessage] = []
        self._sent: list[AgentMessage] = []
        self._new_message_event = asyncio.Event()

    # ── Send ─────────────────────────────────────────────────────────

    async def send(
        self,
        to_agent: str,
        type: str,
        subject: str,
        body: str,
        reply_to: str | None = None,
    ) -> AgentMessage:
        """Send a message to another Agent.

        Routes via the global MailRouter if available; otherwise
        just records locally.

        Args:
            to_agent: Recipient agent ID or 'any' for broadcast.
            type: Message type ('delegate', 'ask', 'inform', 'handoff').
            subject: Short subject.
            body: Message body.
            reply_to: ID of the message being replied to (optional).

        Returns:
            The sent AgentMessage.
        """
        msg = AgentMessage(
            from_agent=self.agent_id,
            to_agent=to_agent,
            type=type,
            subject=subject,
            body=body,
            reply_to=reply_to,
        )
        self._sent.append(msg)

        if self._router:
            await self._router.route(msg)

        logger.debug("Mailbox[%s]: sent %s → %s: %s", self.agent_id, type, to_agent, subject)
        return msg

    # ── Receive ──────────────────────────────────────────────────────

    async def receive(self, timeout: float = 5.0) -> AgentMessage | None:
        """Receive one message from the inbox (blocking with timeout).

        Args:
            timeout: Maximum wait time in seconds (default 5.0).

        Returns:
            The next AgentMessage, or None if timeout.
        """
        if self._inbox:
            return self._deliver_one()

        try:
            await asyncio.wait_for(self._new_message_event.wait(), timeout=timeout)
            return self._deliver_one()
        except asyncio.TimeoutError:
            return None

    def _deliver_one(self) -> AgentMessage | None:
        """Pop and mark the oldest unread message as read."""
        if self._inbox:
            msg = self._inbox.pop(0)
            msg.status = "read"
            if not self._inbox:
                self._new_message_event.clear()
            return msg
        return None

    def peek(self) -> list[AgentMessage]:
        """View all unread messages without consuming them."""
        return [m for m in self._inbox if m.status == "unread"]

    # ── Reply ────────────────────────────────────────────────────────

    async def reply(self, msg: AgentMessage, body: str) -> AgentMessage:
        """Reply to a received message.

        Args:
            msg: The message to reply to.
            body: Reply body.

        Returns:
            The reply AgentMessage.
        """
        return await self.send(
            to_agent=msg.from_agent,
            type=msg.type,
            subject=f"Re: {msg.subject}",
            body=body,
            reply_to=msg.id,
        )

    # ── Internal: receive a routed message ───────────────────────────

    def _receive(self, msg: AgentMessage) -> None:
        """Internal: add a message to this mailbox (called by MailRouter)."""
        self._inbox.append(msg)
        self._new_message_event.set()

    # ── Query ────────────────────────────────────────────────────────

    @property
    def unread_count(self) -> int:
        return len([m for m in self._inbox if m.status == "unread"])

    @property
    def sent_count(self) -> int:
        return len(self._sent)


# ============================================================
# MailRouter
# ============================================================


class MailRouter:
    """Global message router — singleton held by AgentManagerAgent.

    Maintains a registry of Mailboxes keyed by agent_id.
    Routes messages from one mailbox to another.

    Usage:
        router = MailRouter()
        mailbox = router.register("agent-1")
        ...
        router.unregister("agent-1")
    """

    def __init__(self):
        self._mailboxes: dict[str, AgentMailbox] = {}

    def register(self, agent_id: str) -> AgentMailbox:
        """Create and register a Mailbox for an Agent.

        If a Mailbox for agent_id already exists, returns the existing one.

        Args:
            agent_id: The agent's ID.

        Returns:
            The AgentMailbox for this agent.
        """
        if agent_id in self._mailboxes:
            return self._mailboxes[agent_id]

        mailbox = AgentMailbox(agent_id=agent_id, router=self)
        self._mailboxes[agent_id] = mailbox
        logger.debug("MailRouter: registered '%s'", agent_id)
        return mailbox

    async def route(self, msg: AgentMessage) -> bool:
        """Route a message to the target Agent's mailbox.

        Args:
            msg: The AgentMessage to route.

        Returns:
            True if delivered, False if target not found.
        """
        target = msg.to_agent

        # Broadcast to 'any' — deliver to first available mailbox
        if target == "any":
            delivered = False
            for aid, mb in self._mailboxes.items():
                if aid != msg.from_agent:
                    mb._receive(msg)
                    delivered = True
                    break
            if not delivered:
                logger.warning("MailRouter: broadcast to 'any' — no other agent registered")
            return delivered

        mailbox = self._mailboxes.get(target)
        if mailbox is None:
            logger.warning("MailRouter: target '%s' not registered", target)
            return False

        mailbox._receive(msg)
        return True

    def unregister(self, agent_id: str) -> None:
        """Remove a Mailbox from the registry (Agent destroyed).

        Args:
            agent_id: The agent's ID to unregister.
        """
        if agent_id in self._mailboxes:
            del self._mailboxes[agent_id]
            logger.debug("MailRouter: unregistered '%s'", agent_id)

    @property
    def registered_agents(self) -> list[str]:
        """Return list of registered agent IDs."""
        return list(self._mailboxes.keys())

    @property
    def agent_count(self) -> int:
        """Return number of registered agents."""
        return len(self._mailboxes)
