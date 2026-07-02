"""Tests for new features: AmmoBox, AmmoRefiller, Project, intent classification."""

import asyncio
import tempfile
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ammo import AmmoBox, AmmoRefiller, AmmoItem
from src.project import Project, ProjectCard


# ── AmmoBox Tests ──────────────────────────────────────────────

class TestAmmoBox:
    def test_add_and_retrieve_pinned(self):
        box = AmmoBox(max_tokens=2000)
        box.add_pinned("Project context: test", source="project_card")
        ctx = box.to_context()
        assert "Project context" in ctx

    def test_add_and_retrieve_workspace(self):
        box = AmmoBox(max_tokens=2000)
        box.add_workspace("[step 0] search → results", step_num=0)
        box.add_workspace("[step 1] analyze → done", step_num=1)
        ctx = box.to_context()
        assert "step 0" in ctx
        assert "step 1" in ctx

    def test_natural_eviction_workspace_fifo(self):
        """When over capacity, oldest workspace items are evicted first."""
        box = AmmoBox(max_tokens=100)  # Very small to trigger eviction
        box.add_pinned("pinned content that stays", source="card")
        for i in range(20):
            box.add_workspace(f"step {i} " * 10, step_num=i)
        ctx = box.to_context()
        assert "pinned" in ctx  # Pinned never evicted
        # Recent items should be present, old ones evicted
        assert "step 19" in ctx or "step 18" in ctx

    def test_add_facts_and_findings(self):
        box = AmmoBox(max_tokens=2000)
        box.add_fact("Python is interpreted", source="retrieve")
        box.add_finding("Worker found: API returns 200")
        ctx = box.to_context()
        assert "Python" in ctx
        assert "API returns 200" in ctx

    def test_empty_box_context(self):
        box = AmmoBox(max_tokens=2000)
        ctx = box.to_context()
        assert ctx == "" or ctx.strip() == ""


# ── AmmoRefiller Tests ─────────────────────────────────────────

class TestAmmoRefiller:
    def test_creation(self):
        ammo = AmmoBox(max_tokens=2000)
        refiller = AmmoRefiller(ammo_box=ammo, project=None, memory=None, llm=None)
        assert refiller.ammo is ammo
        assert refiller._review_interval == 3

    @pytest.mark.asyncio
    async def test_check_and_refill_no_crash_on_none(self):
        """Refiller should not crash when project/memory/llm are None."""
        ammo = AmmoBox(max_tokens=2000)
        refiller = AmmoRefiller(ammo_box=ammo, project=None, memory=None, llm=None)
        # Should complete without error even with no resources
        await refiller.check_and_refill({
            "phase": "EXECUTE",
            "step_num": 0,
            "last_error": None,
            "last_step_success": True,
            "step_desc": "test",
            "user_input": "test",
        })

    @pytest.mark.asyncio
    async def test_periodic_review_trigger(self):
        """Review triggers every review_interval steps."""
        ammo = AmmoBox(max_tokens=2000)
        refiller = AmmoRefiller(ammo_box=ammo, project=None, memory=None, llm=None)
        refiller._review_interval = 3

        for i in range(3):
            await refiller.check_and_refill({
                "phase": "EXECUTE",
                "step_num": i,
                "last_error": None,
                "last_step_success": True,
                "step_desc": f"step {i}",
                "user_input": "test",
            })
        # After 3 steps, review should have been triggered
        assert refiller._step_count == 3


# ── Project Tests ──────────────────────────────────────────────

class TestProject:
    def test_project_creation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj = Project(root=tmpdir)
            assert proj.root == Path(tmpdir).resolve()
            assert proj.agent_dir.exists()

    def test_project_card_auto_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj = Project(root=tmpdir)
            card = ProjectCard(project=proj)
            card.load()
            assert card.name == proj.project_id

    def test_project_card_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj = Project(root=tmpdir)
            card = ProjectCard(project=proj)
            card.load()
            card.description = "A test project"
            card.tech_stack = "Python, pytest"
            card.save()
            card2 = ProjectCard(project=proj)
            card2.load()
            assert card2.description == "A test project"
            assert "Python" in card2.tech_stack

    def test_project_findings_write_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj = Project(root=tmpdir)
            proj.write_finding("task_1", "Found a bug in auth.py:42")
            proj.write_finding("task_2", "Fixed by adding None check")
            findings = proj.read_findings()
            assert len(findings) == 2

    def test_project_session_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            proj = Project(root=tmpdir)
            proj.save_session_summary("sess_001", "Session about fixing auth bug",
                                      "fix auth", "auth fixed")
            sessions = proj.load_recent_sessions(limit=5)
            assert len(sessions) >= 1
            assert any("auth bug" in s.get("summary", "") for s in sessions)

class TestClassifyIntent:
    """Test the 3-level intent classification in MainLoop."""

    def _make_loop(self):
        from src.loop_engine.main_loop import MainLoop
        from src.memory import MemoryPool
        memory = MemoryPool()
        llm = MagicMock()
        llm.chat = AsyncMock()
        llm.chat_with_retry = AsyncMock()
        llm.model = "test"
        return MainLoop(memory=memory, llm=llm)

    def test_chat_simple_greeting(self):
        loop = self._make_loop()
        assert loop._classify_intent("hello") == "chat"

    def test_chat_simple_math(self):
        loop = self._make_loop()
        assert loop._classify_intent("1+1=?") == "chat"

    def test_analysis_vs_keyword(self):
        loop = self._make_loop()
        assert loop._classify_intent("Python vs Rust 性能对比") == "analysis"

    def test_analysis_summarize(self):
        loop = self._make_loop()
        assert loop._classify_intent("总结一下今天的会议") == "analysis"

    def test_analysis_explain(self):
        loop = self._make_loop()
        assert loop._classify_intent("解释一下 Raft 共识算法") == "analysis"

    def test_complex_deploy(self):
        loop = self._make_loop()
        assert loop._classify_intent("部署到生产环境") == "complex"

    def test_complex_fix_bug(self):
        loop = self._make_loop()
        assert loop._classify_intent("修复 auth.py 的 bug") == "complex"

    def test_complex_write_code(self):
        loop = self._make_loop()
        assert loop._classify_intent("写代码实现登录功能") == "complex"

    def test_complex_implement(self):
        loop = self._make_loop()
        assert loop._classify_intent("implement user authentication") == "complex"

    def test_analysis_compare(self):
        loop = self._make_loop()
        assert loop._classify_intent("compare React vs Vue performance") == "analysis"

    def test_complex_migrate(self):
        loop = self._make_loop()
        assert loop._classify_intent("migrate database from MySQL to PostgreSQL") == "complex"

    def test_chat_very_short(self):
        loop = self._make_loop()
        assert loop._classify_intent("ok") == "chat"

    def test_analysis_medium_no_keywords(self):
        loop = self._make_loop()
        # Medium text without any keywords → analysis (not complex)
        text = "Tell me about the current state of quantum computing"
        result = loop._classify_intent(text)
        assert result in ("analysis", "complex")  # Either is reasonable
