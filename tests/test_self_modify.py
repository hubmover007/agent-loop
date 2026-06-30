"""Tests for AgentSoul self-modification — autonomous file editing with audit trail."""

from __future__ import annotations

import json
import pytest
import tempfile
from pathlib import Path


# ── helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def soul_dirs(tmp_path):
    """Create temp directories for AgentSoul tests."""
    shared = tmp_path / "config/agents/shared"
    cards_pers = tmp_path / "config/agents/cards/personalities"
    cards_roles = tmp_path / "config/agents/cards/roles"
    state = tmp_path / "state/agents"

    for d in [shared, cards_pers, cards_roles, state]:
        d.mkdir(parents=True, exist_ok=True)

    (shared / "SAFETY.md").write_text("# Safety\nDo not delete.")

    yield {
        "root": tmp_path / "config/agents",
        "state": state,
    }


@pytest.fixture
def permissions_config():
    """Create a temporary permissions.json."""
    config = {
        "templates": {
            "coder": {
                "trust_level": "restricted",
                "self_modification": {
                    "can_modify_identity": True,
                    "can_modify_role": True,
                    "can_append_journal": True,
                    "can_modify_knowledge": True,
                    "can_modify_profile": False
                }
            }
        }
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    yield config_path
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def coder_perms(permissions_config):
    """AgentPermissions for coder."""
    from src.permissions import AgentPermissions
    return AgentPermissions("coder", "agent-self-01", permissions_config)


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_self_modify_identity(soul_dirs, coder_perms):
    """Agent can modify its own IDENTITY.md."""
    from src.agent_soul import SoulBuilder

    soul = SoulBuilder(
        agent_id="agent-self-01",
        soul_root=soul_dirs["root"],
        state_root=soul_dirs["state"],
    ).with_personality("executor").with_role("coder").build()

    new_content = "# New Identity\nI am an evolved agent."

    result = await soul.self_modify("identity", new_content, permissions=coder_perms)
    assert result is True

    identity_path = soul_dirs["state"] / "agent-self-01" / "IDENTITY.md"
    assert identity_path.read_text() == new_content

    # Check audit log in JOURNAL
    journal_path = soul_dirs["state"] / "agent-self-01" / "JOURNAL.md"
    journal = journal_path.read_text()
    assert "Self-Modify" in journal
    assert "identity" in journal


@pytest.mark.asyncio
async def test_self_modify_blocked_profile(soul_dirs, coder_perms):
    """Agent cannot modify profile.json (permissions deny)."""
    from src.agent_soul import SoulBuilder

    soul = SoulBuilder(
        agent_id="agent-self-02",
        soul_root=soul_dirs["root"],
        state_root=soul_dirs["state"],
    ).with_personality("executor").with_role("coder").build()

    result = await soul.self_modify("profile", '{"curiosity": 1.0}', permissions=coder_perms)
    assert result is False

    # Profile should be unchanged
    profile_path = soul_dirs["state"] / "agent-self-02" / "profile.json"
    profile = json.loads(profile_path.read_text())
    assert profile["curiosity"] == 0.5  # default, not 1.0


@pytest.mark.asyncio
async def test_self_modify_without_permissions(soul_dirs):
    """Without permissions parameter, self_modify bypasses checks."""
    from src.agent_soul import SoulBuilder

    soul = SoulBuilder(
        agent_id="agent-self-03",
        soul_root=soul_dirs["root"],
        state_root=soul_dirs["state"],
    ).with_personality("executor").with_role("coder").build()

    new_content = "# New Role\nrole: architect"
    result = await soul.self_modify("role", new_content, permissions=None)
    assert result is True  # No permissions → allowed

    role_path = soul_dirs["state"] / "agent-self-03" / "ROLE.md"
    assert role_path.read_text() == new_content


@pytest.mark.asyncio
async def test_self_modify_role(soul_dirs, coder_perms):
    """Agent can modify its own ROLE.md."""
    from src.agent_soul import SoulBuilder

    soul = SoulBuilder(
        agent_id="agent-self-04",
        soul_root=soul_dirs["root"],
        state_root=soul_dirs["state"],
    ).with_personality("executor").with_role("coder").build()

    new_content = "# Role: Senior Developer"
    result = await soul.self_modify("role", new_content, permissions=coder_perms)
    assert result is True

    role_path = soul_dirs["state"] / "agent-self-04" / "ROLE.md"
    assert "Senior Developer" in role_path.read_text()


@pytest.mark.asyncio
async def test_self_modify_unknown_file_type(soul_dirs, coder_perms):
    """Unknown file_type returns False."""
    from src.agent_soul import SoulBuilder

    soul = SoulBuilder(
        agent_id="agent-self-05",
        soul_root=soul_dirs["root"],
        state_root=soul_dirs["state"],
    ).with_personality("executor").with_role("coder").build()

    result = await soul.self_modify("nonexistent", "content", permissions=coder_perms)
    assert result is False


@pytest.mark.asyncio
async def test_request_safety_change(soul_dirs):
    """Agent can suggest SAFETY.md changes (written to JOURNAL)."""
    from src.agent_soul import SoulBuilder

    soul = SoulBuilder(
        agent_id="agent-self-06",
        soul_root=soul_dirs["root"],
        state_root=soul_dirs["state"],
    ).with_personality("executor").with_role("coder").build()

    suggestion = "Consider adding a rule: must verify before deleting files."

    # Read journal before
    journal_path = soul_dirs["state"] / "agent-self-06" / "JOURNAL.md"
    journal_before = journal_path.read_text()

    await soul.request_safety_change(suggestion)

    # Read journal after
    journal_after = journal_path.read_text()
    assert len(journal_after) > len(journal_before)
    assert "SAFETY" in journal_after
    assert "verify before deleting" in journal_after

    # SAFETY.md itself should NOT be modified
    safety_path = soul_dirs["root"] / "shared/SAFETY.md"
    assert safety_path.read_text() == "# Safety\nDo not delete."


@pytest.mark.asyncio
async def test_audit_log_written(soul_dirs, coder_perms):
    """Self-modification writes audit log to JOURNAL.md."""
    from src.agent_soul import SoulBuilder

    soul = SoulBuilder(
        agent_id="agent-self-07",
        soul_root=soul_dirs["root"],
        state_root=soul_dirs["state"],
    ).with_personality("executor").with_role("coder").build()

    # Clear journal to start fresh
    journal_path = soul_dirs["state"] / "agent-self-07" / "JOURNAL.md"
    journal_before = journal_path.read_text()

    await soul.self_modify("identity", "# ID\nI changed myself.", permissions=coder_perms)

    journal_after = journal_path.read_text()
    assert len(journal_after) > len(journal_before)
    assert "Self-Modify" in journal_after
    assert "identity" in journal_after


@pytest.mark.asyncio
async def test_self_modify_journal(soul_dirs, coder_perms):
    """Agent can append/modify its own JOURNAL.md."""
    from src.agent_soul import SoulBuilder

    soul = SoulBuilder(
        agent_id="agent-self-08",
        soul_root=soul_dirs["root"],
        state_root=soul_dirs["state"],
    ).with_personality("executor").with_role("coder").build()

    new_content = "# My Journal\nI did stuff."
    result = await soul.self_modify("journal", new_content, permissions=coder_perms)
    assert result is True

    journal_path = soul_dirs["state"] / "agent-self-08" / "JOURNAL.md"
    content = journal_path.read_text()
    assert "I did stuff" in content
