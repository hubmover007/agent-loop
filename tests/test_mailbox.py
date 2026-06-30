"""Tests for src/agent_mailbox.py — Agent-to-Agent messaging system."""

import asyncio
import pytest
from src.agent_mailbox import AgentMailbox, MailRouter, AgentMessage


class TestAgentMailbox:
    """Tests for AgentMailbox standalone operations."""

    def test_create_mailbox(self):
        """Mailbox initializes with correct agent_id."""
        mb = AgentMailbox("agent-1")
        assert mb.agent_id == "agent-1"
        assert mb.unread_count == 0
        assert mb.sent_count == 0

    def test_peek_empty(self):
        """Peek on empty inbox returns empty list."""
        mb = AgentMailbox("agent-1")
        assert mb.peek() == []

    @pytest.mark.asyncio
    async def test_receive_timeout(self):
        """Receive with no messages returns None after timeout."""
        mb = AgentMailbox("agent-1")
        msg = await mb.receive(timeout=0.1)
        assert msg is None

    @pytest.mark.asyncio
    async def test_send_without_router(self):
        """Send without router records locally but doesn't route."""
        mb = AgentMailbox("agent-1")
        msg = await mb.send("agent-2", "ask", "Help", "How do I...")
        assert msg.from_agent == "agent-1"
        assert msg.to_agent == "agent-2"
        assert msg.type == "ask"
        assert mb.sent_count == 1

    @pytest.mark.asyncio
    async def test_internal_receive(self):
        """Internal _receive adds message to inbox."""
        mb = AgentMailbox("agent-1")
        msg = AgentMessage(from_agent="agent-2", to_agent="agent-1", type="inform",
                          subject="Hello", body="Hi there!")
        mb._receive(msg)
        assert mb.unread_count == 1
        assert mb.peek()[0].subject == "Hello"

        # Receive consumes it
        received = await mb.receive(timeout=0.1)
        assert received is not None
        assert received.subject == "Hello"
        assert received.status == "read"
        assert mb.unread_count == 0


class TestSendReceive:
    """Tests for full send-receive cycle via MailRouter."""

    @pytest.mark.asyncio
    async def test_send_receive(self):
        """Agent A sends to Agent B, B receives."""
        router = MailRouter()
        mb_a = router.register("agent-a")
        mb_b = router.register("agent-b")

        await mb_a.send("agent-b", "ask", "Question", "What is the answer?")

        msg = await mb_b.receive(timeout=0.5)
        assert msg is not None
        assert msg.from_agent == "agent-a"
        assert msg.to_agent == "agent-b"
        assert msg.type == "ask"
        assert msg.subject == "Question"
        assert msg.body == "What is the answer?"

    @pytest.mark.asyncio
    async def test_route(self):
        """MailRouter correctly routes messages."""
        router = MailRouter()
        mb_a = router.register("agent-a")
        mb_b = router.register("agent-b")

        msg = AgentMessage(
            from_agent="agent-a",
            to_agent="agent-b",
            type="delegate",
            subject="Task",
            body="Please handle this",
        )
        result = await router.route(msg)
        assert result is True
        assert mb_b.unread_count == 1

    @pytest.mark.asyncio
    async def test_route_to_nonexistent(self):
        """Routing to unregistered agent returns False."""
        router = MailRouter()
        mb_a = router.register("agent-a")

        msg = AgentMessage(from_agent="agent-a", to_agent="agent-x",
                          type="inform", subject="X", body="X")
        result = await router.route(msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_reply(self):
        """Reply chain works correctly."""
        router = MailRouter()
        mb_a = router.register("agent-a")
        mb_b = router.register("agent-b")

        # A → B
        msg = await mb_a.send("agent-b", "ask", "Question", "What?")
        received = await mb_b.receive(timeout=0.5)
        assert received is not None

        # B → A (reply)
        reply = await mb_b.reply(received, "Answer: 42")
        assert reply.reply_to == msg.id
        assert reply.subject == "Re: Question"

        received_reply = await mb_a.receive(timeout=0.5)
        assert received_reply is not None
        assert received_reply.body == "Answer: 42"

    @pytest.mark.asyncio
    async def test_unregister(self):
        """Messages to unregistered agents are discarded."""
        router = MailRouter()
        mb_a = router.register("agent-a")
        mb_b = router.register("agent-b")

        router.unregister("agent-b")
        msg = AgentMessage(from_agent="agent-a", to_agent="agent-b",
                          type="inform", subject="X", body="X")
        result = await router.route(msg)
        assert result is False

    @pytest.mark.asyncio
    async def test_broadcast_any(self):
        """Broadcast to 'any' delivers to first available agent."""
        router = MailRouter()
        mb_a = router.register("agent-a")
        mb_b = router.register("agent-b")

        msg = AgentMessage(from_agent="agent-a", to_agent="any",
                          type="inform", subject="Broadcast", body="Hello all")
        result = await router.route(msg)
        assert result is True

        # Should be delivered to agent-b (not sender)
        received = await mb_b.receive(timeout=0.5)
        assert received is not None
        assert received.body == "Hello all"

    @pytest.mark.asyncio
    async def test_send_delegate_type(self):
        """Send with 'delegate' type."""
        router = MailRouter()
        mb_a = router.register("agent-a")
        mb_b = router.register("agent-b")

        await mb_a.send("agent-b", "delegate", "Sub-task", "Run analysis")
        msg = await mb_b.receive(timeout=0.5)
        assert msg is not None
        assert msg.type == "delegate"

    @pytest.mark.asyncio
    async def test_send_handoff_type(self):
        """Send with 'handoff' type."""
        router = MailRouter()
        mb_a = router.register("agent-a")
        mb_b = router.register("agent-b")

        await mb_a.send("agent-b", "handoff", "Transfer", "Taking over task")
        msg = await mb_b.receive(timeout=0.5)
        assert msg is not None
        assert msg.type == "handoff"


class TestMailRouter:
    """Tests for MailRouter."""

    def test_register_duplicate(self):
        """Registering same agent twice returns same mailbox."""
        router = MailRouter()
        mb1 = router.register("agent-1")
        mb2 = router.register("agent-1")
        assert mb1 is mb2
        assert router.agent_count == 1

    def test_agent_count(self):
        """agent_count tracks correctly."""
        router = MailRouter()
        assert router.agent_count == 0
        router.register("agent-1")
        router.register("agent-2")
        assert router.agent_count == 2
        router.register("agent-3")
        assert router.agent_count == 3

    def test_registered_agents(self):
        """registered_agents returns correct list."""
        router = MailRouter()
        router.register("agent-1")
        router.register("agent-2")
        agents = router.registered_agents
        assert "agent-1" in agents
        assert "agent-2" in agents
        assert len(agents) == 2


class TestAgentMessage:
    """Tests for AgentMessage dataclass."""

    def test_defaults(self):
        msg = AgentMessage()
        assert msg.status == "unread"
        assert msg.reply_to is None

    def test_to_dict(self):
        msg = AgentMessage(
            id="msg-1",
            from_agent="a",
            to_agent="b",
            type="ask",
            subject="S",
            body="B",
            reply_to="msg-0",
        )
        d = msg.to_dict()
        assert d["id"] == "msg-1"
        assert d["type"] == "ask"
        assert d["reply_to"] == "msg-0"
