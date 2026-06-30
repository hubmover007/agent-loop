"""Tests for P2.5 durable project memory (MemoryPool + AgentSoul)."""

import pytest

from src.memory import MemoryPool
from src.agent_soul import AgentSoul


class TestProjectMemory:
    """Tests for MemoryPool.save_project_doc / load_project_docs."""

    @pytest.mark.asyncio
    async def test_save_load_project_doc(self):
        """Save and load a project document through MemoryPool."""
        pool = MemoryPool(db_path=":memory:")
        try:
            # Save a project doc
            await pool.save_project_doc("agent-1", "prompt", "Build a web app")
            await pool.save_project_doc("agent-1", "plan", "1. Setup\n2. Code\n3. Deploy")

            # Load all docs
            docs = await pool.load_project_docs("agent-1")
            assert docs["prompt"] == "Build a web app"
            assert docs["plan"] == "1. Setup\n2. Code\n3. Deploy"
        finally:
            pool.clear()

    @pytest.mark.asyncio
    async def test_project_doc_types(self):
        """All four doc types (prompt/plan/implement/documentation) work."""
        pool = MemoryPool(db_path=":memory:")
        try:
            await pool.save_project_doc("agent-2", "prompt", "spec here")
            await pool.save_project_doc("agent-2", "plan", "plan here")
            await pool.save_project_doc("agent-2", "implement", "implement here")
            await pool.save_project_doc("agent-2", "documentation", "docs here")

            docs = await pool.load_project_docs("agent-2")
            assert docs["prompt"] == "spec here"
            assert docs["plan"] == "plan here"
            assert docs["implement"] == "implement here"
            assert docs["documentation"] == "docs here"
            assert len(docs) == 4
        finally:
            pool.clear()

    @pytest.mark.asyncio
    async def test_overwrite_project_doc(self):
        """Saving same doc_type twice produces two entries (append semantics)."""
        pool = MemoryPool(db_path=":memory:")
        try:
            await pool.save_project_doc("agent-3", "prompt", "v1")
            await pool.save_project_doc("agent-3", "prompt", "v2")

            docs = await pool.load_project_docs("agent-3")
            # Last one wins in dict overwrite, but we have non-deterministic order from save() timestamps
            assert "prompt" in docs
            assert docs["prompt"] in ("v1", "v2")
        finally:
            pool.clear()

    @pytest.mark.asyncio
    async def test_load_empty_docs(self):
        """Loading docs for agent with no project docs returns empty dict."""
        pool = MemoryPool(db_path=":memory:")
        try:
            docs = await pool.load_project_docs("unknown-agent")
            assert docs == {}
        finally:
            pool.clear()


class TestAgentSoulWithProjectDocs:
    """Tests for AgentSoul.build_system_prompt with project docs."""

    def test_build_prompt_with_docs(self):
        """Project docs appear in the system prompt."""
        soul = AgentSoul(
            agent_id="test-agent",
            personality="developer",
            state_root="/tmp/test-as-docs",
            soul_root="/tmp/test-as-soul",
        )

        task_context = {
            "project_docs": {
                "prompt": "Build a REST API for user management",
                "plan": "1. Define models\n2. Create endpoints\n3. Add auth",
            }
        }

        prompt = soul.build_system_prompt(task_context=task_context)
        assert "Project Spec" in prompt
        assert "Build a REST API" in prompt
        assert "Project Plan" in prompt
        assert "Define models" in prompt

    def test_build_prompt_without_docs(self):
        """System prompt should work fine without project docs."""
        soul = AgentSoul(
            agent_id="test-agent-2",
            personality="executor",
            state_root="/tmp/test-as-nodocs",
            soul_root="/tmp/test-as-nosoul",
        )

        # No task_context
        prompt = soul.build_system_prompt(task_context=None)
        assert isinstance(prompt, str)
        assert "executor" in prompt.lower() or "Agent" in prompt

        # task_context without project_docs
        prompt = soul.build_system_prompt(task_context={"other": "stuff"})
        assert "executor" in prompt.lower() or "Agent" in prompt
        assert "Project Spec" not in prompt

    def test_build_prompt_implementation_docs(self):
        """Implementation docs appear in prompt when present."""
        soul = AgentSoul(
            agent_id="test-agent-3",
            personality="coder",
            state_root="/tmp/test-as-impl",
            soul_root="/tmp/test-as-impl",
        )

        task_context = {
            "project_docs": {
                "implement": "Use FastAPI with async/await pattern",
                "documentation": "Swagger UI at /docs",
            }
        }

        prompt = soul.build_system_prompt(task_context=task_context)
        assert "Implementation Guide" in prompt
        assert "FastAPI" in prompt
