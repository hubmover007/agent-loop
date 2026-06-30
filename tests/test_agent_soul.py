"""Tests for AgentSoul — personality/role management with md file layering."""

from __future__ import annotations

import os
import pytest
import tempfile
from pathlib import Path

# ── helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def soul_dirs(tmp_path):
    """Create temp directories for shared, cards, state."""
    shared = tmp_path / "config/agents/shared"
    cards_pers = tmp_path / "config/agents/cards/personalities"
    cards_roles = tmp_path / "config/agents/cards/roles"
    state = tmp_path / "state/agents"

    for d in [shared, cards_pers, cards_roles, state]:
        d.mkdir(parents=True, exist_ok=True)

    # Write shared files
    (shared / "SAFETY.md").write_text("# 安全铁律\n绝对红线，不可违反。")
    (shared / "CONSTRAINTS.md").write_text("# 约束\n最大步骤 10 步。")
    (shared / "PRINCIPLES.md").write_text("# 准则\n诚实 > 讨好。")

    # Write personality cards
    (cards_pers / "executor.md").write_text("# 执行者\n话少，直接行动。")
    (cards_pers / "analyst.md").write_text("# 分析师\n深度思考，谨慎下结论。")
    (cards_pers / "creative.md").write_text("# 创意者\n发散联想，敢提方案。")
    (cards_pers / "guardian.md").write_text("# 守护者\n安全第一，凡事确认。")

    # Write role cards
    (cards_roles / "coder.md").write_text("# 编码者\n写之前先读项目结构。")
    (cards_roles / "researcher.md").write_text("# 研究者\n交叉验证信息来源。")
    (cards_roles / "ops.md").write_text("# 运维者\n操作前确认影响范围。")

    yield {
        "shared": shared,
        "cards_pers": cards_pers,
        "cards_roles": cards_roles,
        "state": state,
        "root": tmp_path / "config/agents",
    }


# ── Tests ────────────────────────────────────────────────────────────────────


def test_soul_builder_creates_soul(soul_dirs):
    """SoulBuilder creates an AgentSoul with specified personality and role."""
    from src.agent_soul import SoulBuilder, AgentSoul

    soul = (SoulBuilder(agent_id="test-agent-1",
                        soul_root=soul_dirs["root"],
                        state_root=soul_dirs["state"])
            .with_personality("executor")
            .with_role("coder")
            .build())

    assert isinstance(soul, AgentSoul)
    assert soul.agent_id == "test-agent-1"
    assert soul.personality == "executor"
    assert soul.role == "coder"


def test_soul_loads_shared_md(soul_dirs):
    """Shared md files (SAFETY, CONSTRAINTS, PRINCIPLES) are loaded."""
    from src.agent_soul import SoulBuilder

    soul = (SoulBuilder(agent_id="test-agent-2",
                        soul_root=soul_dirs["root"],
                        state_root=soul_dirs["state"])
            .with_personality("executor")
            .with_role("coder")
            .build())

    prompt = soul.build_system_prompt()
    assert "安全铁律" in prompt
    assert "绝对红线" in prompt
    assert "最大步骤 10 步" in prompt
    assert "诚实 > 讨好" in prompt


def test_soul_builds_system_prompt(soul_dirs):
    """build_system_prompt includes all layers."""
    from src.agent_soul import SoulBuilder

    soul = (SoulBuilder(agent_id="test-agent-3",
                        soul_root=soul_dirs["root"],
                        state_root=soul_dirs["state"])
            .with_personality("executor")
            .with_role("coder")
            .build())

    prompt = soul.build_system_prompt()

    # Shared layer
    assert "安全铁律" in prompt
    assert "约束" in prompt
    assert "准则" in prompt

    # Cards layer
    assert "执行者" in prompt
    assert "编码者" in prompt

    # Private layer
    assert "test-agent-3" in prompt
    assert "executor" in prompt


def test_soul_safety_has_highest_priority(soul_dirs):
    """SAFETY.md must appear first in the system prompt."""
    from src.agent_soul import SoulBuilder

    soul = (SoulBuilder(agent_id="test-agent-4",
                        soul_root=soul_dirs["root"],
                        state_root=soul_dirs["state"])
            .with_personality("executor")
            .with_role("coder")
            .build())

    prompt = soul.build_system_prompt()

    # SAFETY.md content should be the first section
    sections = prompt.split("\n\n---\n\n")
    first_section = sections[0].strip()
    assert first_section.startswith("# 安全铁律") or "安全铁律" in first_section

    # Verify safety content appears before personality content
    safety_pos = prompt.index("安全铁律")
    executor_pos = prompt.index("执行者")
    assert safety_pos < executor_pos, "SAFETY.md must appear before personality"


@pytest.mark.asyncio
async def test_soul_evolve_appends_journal(soul_dirs):
    """evolve appends to JOURNAL.md."""
    from src.agent_soul import SoulBuilder

    soul = (SoulBuilder(agent_id="test-agent-5",
                        soul_root=soul_dirs["root"],
                        state_root=soul_dirs["state"])
            .with_personality("executor")
            .with_role("coder")
            .build())

    # Read initial journal
    journal_path = soul_dirs["state"] / "test-agent-5" / "JOURNAL.md"
    initial_content = journal_path.read_text()
    initial_lines = len(initial_content.splitlines())

    # Evolve
    await soul.evolve("Completed: fix bug #42. Score: 0.85")

    # Check journal was appended
    updated_content = journal_path.read_text()
    updated_lines = len(updated_content.splitlines())
    assert updated_lines > initial_lines
    assert "fix bug #42" in updated_content
    assert "0.85" in updated_content


@pytest.mark.asyncio
async def test_soul_role_upgrade(soul_dirs):
    """update_role changes ROLE.md and updates meta.json."""
    from src.agent_soul import SoulBuilder

    soul = (SoulBuilder(agent_id="test-agent-6",
                        soul_root=soul_dirs["root"],
                        state_root=soul_dirs["state"])
            .with_personality("executor")
            .with_role("coder")
            .build())

    assert soul.role == "coder"

    # Upgrade role
    await soul.update_role("architect")
    assert soul.role == "architect"

    # Check ROLE.md was updated
    role_path = soul_dirs["state"] / "test-agent-6" / "ROLE.md"
    content = role_path.read_text()
    assert "architect" in content

    # Check meta.json was updated
    import json
    meta_path = soul_dirs["state"] / "test-agent-6" / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["role"] == "architect"
    assert "role_upgraded_at" in meta


@pytest.mark.asyncio
async def test_soul_update_identity_trait(soul_dirs):
    """update_identity_trait adjusts profile.json values within [0, 1]."""
    from src.agent_soul import SoulBuilder

    soul = (SoulBuilder(agent_id="test-agent-7",
                        soul_root=soul_dirs["root"],
                        state_root=soul_dirs["state"])
            .with_personality("executor")
            .with_role("coder")
            .build())

    await soul.update_identity_trait("curiosity", +0.3)
    await soul.update_identity_trait("cautiousness", -0.2)

    import json
    profile_path = soul_dirs["state"] / "test-agent-7" / "profile.json"
    profile = json.loads(profile_path.read_text())

    assert profile["curiosity"] == 0.8  # 0.5 + 0.3
    assert profile["cautiousness"] == 0.3  # 0.5 - 0.2

    # Test clamping
    await soul.update_identity_trait("curiosity", +1.0)
    profile2 = json.loads(profile_path.read_text())
    assert profile2["curiosity"] == 1.0

    await soul.update_identity_trait("cautiousness", -1.0)
    profile3 = json.loads(profile_path.read_text())
    assert profile3["cautiousness"] == 0.0


def test_soul_different_personalities(soul_dirs):
    """Different personalities load different cards."""
    from src.agent_soul import SoulBuilder

    executor_soul = (SoulBuilder(agent_id="agent-exec",
                                 soul_root=soul_dirs["root"],
                                 state_root=soul_dirs["state"])
                     .with_personality("executor")
                     .with_role("coder")
                     .build())

    analyst_soul = (SoulBuilder(agent_id="agent-analyst",
                               soul_root=soul_dirs["root"],
                               state_root=soul_dirs["state"])
                   .with_personality("analyst")
                   .with_role("coder")
                   .build())

    exec_prompt = executor_soul.build_system_prompt()
    analyst_prompt = analyst_soul.build_system_prompt()

    assert "执行者" in exec_prompt
    assert "分析师" in analyst_prompt
    assert "深度思考" in analyst_prompt


@pytest.mark.asyncio
async def test_soul_record_task_updates_meta(soul_dirs):
    """record_task updates total_tasks and success_rate."""
    from src.agent_soul import SoulBuilder
    import json

    soul = (SoulBuilder(agent_id="test-agent-8",
                        soul_root=soul_dirs["root"],
                        state_root=soul_dirs["state"])
            .with_personality("executor")
            .with_role("coder")
            .build())

    # 3 successes, 1 failure
    for _ in range(3):
        await soul.record_task(success=True)
    await soul.record_task(success=False)

    meta_path = soul_dirs["state"] / "test-agent-8" / "meta.json"
    meta = json.loads(meta_path.read_text())

    assert meta["total_tasks"] == 4
    assert meta["success_count"] == 3
    assert meta["success_rate"] == 0.75
