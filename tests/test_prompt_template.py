"""Tests for AgentSoul build_system_prompt with template variable interpolation."""

import pytest
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from src.agent_soul import AgentSoul


@pytest.fixture
def soul_dirs():
    """Create a temporary soul directory with test files."""
    with TemporaryDirectory() as tmp:
        soul_root = Path(tmp) / "soul"
        state_root = Path(tmp) / "state"

        # Shared constitution files
        shared = soul_root / "shared"
        shared.mkdir(parents=True)
        (shared / "SAFETY.md").write_text("# SAFETY\nAlways be safe.")
        (shared / "CONSTRAINTS.md").write_text("# CONSTRAINTS\nRespect boundaries.")
        (shared / "PRINCIPLES.md").write_text("# PRINCIPLES\nBe helpful.")

        # Card templates
        for sub in ["cards/personalities", "cards/roles"]:
            (soul_root / sub).mkdir(parents=True)
        (soul_root / "cards/personalities/executor.md").write_text("# Executor\nEfficient and focused.")
        (soul_root / "cards/roles/coder.md").write_text("# Coder\nWrite {{language}} code.")

        yield soul_root, state_root


@pytest.fixture
def soul_with_template(soul_dirs):
    """AgentSoul with template variables in IDENTITY.md."""
    soul_root, state_root = soul_dirs
    # Create agent with template variables in IDENTITY
    agent_id = "agent-template-test"
    private = state_root / agent_id
    private.mkdir(parents=True)
    (private / "IDENTITY.md").write_text(
        "# Identity\nI am a {{task_type}} agent, I write {{language}} code."
    )
    (private / "ROLE.md").write_text(f"# Role\nCurrent role: coder")
    (private / "JOURNAL.md").write_text("# Journal\nStarted today.")
    (private / "profile.json").write_text(json.dumps({
        "curiosity": 0.5, "cautiousness": 0.5,
        "assertiveness": 0.5, "creativity": 0.5, "efficiency": 0.5,
    }))
    (private / "meta.json").write_text(json.dumps({
        "agent_id": agent_id, "created_at": "2024-01-01T00:00:00",
        "total_tasks": 0, "success_rate": 0.0,
        "personality": "executor", "role": "coder",
    }))

    return AgentSoul(
        agent_id=agent_id,
        personality="executor",
        role="coder",
        soul_root=soul_root,
        state_root=state_root,
    )


class TestPromptTemplate:
    """Tests for AgentSoul template variable interpolation."""

    def test_build_prompt_with_context(self, soul_with_template):
        """build_system_prompt should replace {{var}} placeholders with context values."""
        context = {"task_type": "coding", "language": "python"}
        prompt = soul_with_template.build_system_prompt(task_context=context)

        assert "I am a coding agent" in prompt
        assert "I write python code" in prompt
        assert "{{task_type}}" not in prompt
        assert "{{language}}" not in prompt

    def test_build_prompt_no_context(self, soul_with_template):
        """build_system_prompt without context should work exactly as before (backward compat)."""
        prompt = soul_with_template.build_system_prompt()

        # Template placeholders should remain unmodified
        assert "{{task_type}}" in prompt
        assert "{{language}}" in prompt
        # SAFETY should still appear
        assert "SAFETY" in prompt
        assert "Always be safe" in prompt

    def test_build_prompt_missing_var(self, soul_with_template):
        """Missing context variables should leave placeholders as-is."""
        context = {"task_type": "coding"}  # language is missing
        prompt = soul_with_template.build_system_prompt(task_context=context)

        assert "I am a coding agent" in prompt
        assert "{{language}}" in prompt  # remains unrendered

    def test_build_prompt_empty_context(self, soul_with_template):
        """Empty context dict should behave same as no context."""
        prompt = soul_with_template.build_system_prompt(task_context={})

        assert "{{task_type}}" in prompt  # unchanged
        assert "SAFETY" in prompt

    def test_build_prompt_none_context(self, soul_with_template):
        """None context should behave same as no context (explicit None arg)."""
        prompt = soul_with_template.build_system_prompt(task_context=None)

        assert "{{task_type}}" in prompt
        assert "SAFETY" in prompt

    def test_build_prompt_shared_layers_not_affected(self, soul_with_template):
        """Template interpolation should still include all shared layers."""
        context = {"task_type": "debugging", "language": "rust"}
        prompt = soul_with_template.build_system_prompt(task_context=context)

        assert "SAFETY" in prompt
        assert "CONSTRAINTS" in prompt
        assert "PRINCIPLES" in prompt
        assert "Efficient and focused" in prompt  # from personality card
        assert "I am a debugging agent" in prompt
