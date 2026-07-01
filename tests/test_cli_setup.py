"""Tests for CLI setup, doctor, and helper functions."""
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# Ensure src/cli.py is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli import _generate_llm_pool, _create_default_anchors


@contextmanager
def _tmp_cwd():
    """Context manager that creates a temp dir and changes into it, restoring on exit."""
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            yield tmpdir
        finally:
            os.chdir(old_cwd)


class TestGenerateLLMPool:
    """Tests for _generate_llm_pool function."""

    def test_easyrouter_pool_structure(self):
        """EasyRouter pool should have 8 providers with correct structure."""
        config = {"provider": "easyrouter", "models": ["deepseek-v4-pro"]}
        with _tmp_cwd():
            _generate_llm_pool(config, "easyrouter")

            pool_path = Path("config/llm_pool.json")
            assert pool_path.exists(), "llm_pool.json should be created"

            pool = json.loads(pool_path.read_text())
            assert "providers" in pool
            assert "selection" in pool
            assert len(pool["providers"]) == 8

            p0 = pool["providers"][0]
            assert p0["type"] == "openai"
            assert p0["endpoint"] == "https://easyrouter.io/v1"
            assert p0["api_key_source"] == "env:EASYROUTER_API_KEY"
            assert "capabilities" in p0
            assert "modality" in p0
            assert "tags" in p0

            assert pool["selection"]["default_strategy"] == "balanced"
            assert "task_mapping" in pool["selection"]
            assert "strategies" in pool["selection"]

    def test_openai_pool_structure(self):
        """OpenAI pool should have 2 providers."""
        config = {"provider": "openai", "models": ["gpt-4o"]}
        with _tmp_cwd():
            _generate_llm_pool(config, "openai")

            pool = json.loads(Path("config/llm_pool.json").read_text())
            assert len(pool["providers"]) == 2
            assert pool["providers"][0]["api_key_source"] == "env:OPENAI_API_KEY"
            assert pool["providers"][0]["model"] == "gpt-4o"

    def test_deepseek_pool_structure(self):
        """DeepSeek pool should have 2 providers."""
        config = {"provider": "deepseek", "models": ["deepseek-chat"]}
        with _tmp_cwd():
            _generate_llm_pool(config, "deepseek")

            pool = json.loads(Path("config/llm_pool.json").read_text())
            assert len(pool["providers"]) == 2
            assert pool["providers"][0]["api_key_source"] == "env:DEEPSEEK_API_KEY"

    def test_local_pool_structure(self):
        """Local pool should have 1 provider with no API key."""
        config = {"provider": "local", "models": ["llama3"]}
        with _tmp_cwd():
            _generate_llm_pool(config, "local")

            pool = json.loads(Path("config/llm_pool.json").read_text())
            assert len(pool["providers"]) == 1
            assert pool["providers"][0]["api_key_source"] == "none"
            assert pool["providers"][0]["endpoint"] == "http://localhost:11434/v1"

    def test_easyrouter_provider_ids(self):
        """EasyRouter provider IDs should follow naming convention."""
        config = {"provider": "easyrouter", "models": []}
        with _tmp_cwd():
            _generate_llm_pool(config, "easyrouter")

            pool = json.loads(Path("config/llm_pool.json").read_text())
            ids = [p["id"] for p in pool["providers"]]
            assert all(i.startswith("easyrouter-") for i in ids)
            assert "easyrouter-deepseek-v4-pro" in ids
            assert "easyrouter-gpt-5.5" in ids


class TestCreateDefaultAnchors:
    """Tests for _create_default_anchors function."""

    def test_creates_system_anchor(self):
        """Should create system.md anchor file."""
        with _tmp_cwd():
            _create_default_anchors("UTC", "en")

            system_path = Path("state/anchors/system.md")
            assert system_path.exists()

            content = system_path.read_text()
            assert "System Anchor" in content
            assert "timezone" in content
            assert "UTC" in content
            assert "language" in content
            assert "en" in content
            assert "created_at" in content

    def test_creates_with_custom_timezone_language(self):
        """Should use provided timezone and language."""
        with _tmp_cwd():
            _create_default_anchors("Asia/Shanghai", "zh")

            content = Path("state/anchors/system.md").read_text()
            assert "Asia/Shanghai" in content
            assert "zh" in content

    def test_idempotent(self):
        """Should be safe to call multiple times (overwrites, not errors)."""
        with _tmp_cwd():
            _create_default_anchors("UTC", "en")
            _create_default_anchors("Asia/Shanghai", "zh")

            content = Path("state/anchors/system.md").read_text()
            assert "Asia/Shanghai" in content  # overwritten


class TestCmdDoctor:
    """Tests for cmd_doctor function."""

    def test_doctor_imports_and_runs(self):
        """Doctor command should be importable and runnable."""
        from src.cli import cmd_doctor

        args = MagicMock()
        result = cmd_doctor(args)
        assert result in (0, 1)


class TestCmdTest:
    """Tests for cmd_test function."""

    def test_cmd_test_imports(self):
        """cmd_test function should be importable."""
        from src.cli import cmd_test
        assert callable(cmd_test)


class TestCLIEntrypoints:
    """Tests for CLI argument parsing."""

    def test_help_output(self):
        """agent-loop --help should include new commands."""
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "src.cli", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
        assert "setup" in result.stdout
        assert "test" in result.stdout
        assert "doctor" in result.stdout

    def test_setup_help(self):
        """agent-loop setup --help should work."""
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "src.cli", "setup", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0

    def test_doctor_help(self):
        """agent-loop doctor --help should work."""
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "src.cli", "doctor", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0

    def test_test_help(self):
        """agent-loop test --help should work."""
        import subprocess
        result = subprocess.run(
            ["python3", "-m", "src.cli", "test", "--help"],
            capture_output=True, text=True,
            cwd=Path(__file__).parent.parent,
        )
        assert result.returncode == 0
