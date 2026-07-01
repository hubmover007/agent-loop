"""Tests for CLI helper functions: _load_env, _resolve_api_key, _create_llm_from_config,
and the easyrouter provider type in create_provider."""
import json
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cli import _load_env, _resolve_api_key


@contextmanager
def _tmp_cwd():
    old_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        try:
            yield tmpdir
        finally:
            os.chdir(old_cwd)


# ────────────────────────────────────────────────────────────
# _load_env
# ────────────────────────────────────────────────────────────

class TestLoadEnv:
    def test_loads_env_file(self):
        with _tmp_cwd():
            Path(".env").write_text("MY_KEY=hello123\nOTHER_KEY=world\n")
            _load_env()
            assert os.environ.get("MY_KEY") == "hello123"
            assert os.environ.get("OTHER_KEY") == "world"

    def test_skips_comments_and_blanks(self):
        with _tmp_cwd():
            Path(".env").write_text("# comment\n\nKEY1=value1\n# another comment\n\nKEY2=value2\n")
            _load_env()
            assert os.environ.get("KEY1") == "value1"
            assert os.environ.get("KEY2") == "value2"

    def test_setdefault_does_not_overwrite(self):
        with _tmp_cwd():
            os.environ["EXISTING"] = "original"
            Path(".env").write_text("EXISTING=newvalue\n")
            _load_env()
            assert os.environ["EXISTING"] == "original"  # not overwritten

    def test_loads_cwd_env_file(self):
        """Should load .env from current working directory."""
        with _tmp_cwd():
            Path(".env").write_text("CWD_KEY=cwd_value\n")
            _load_env()
            assert os.environ.get("CWD_KEY") == "cwd_value"

    def test_no_env_file_no_error(self):
        """Should not raise when .env doesn't exist."""
        with _tmp_cwd():
            _load_env()  # no .env file, should not error

    def test_quoted_values_stripped(self):
        """Values in quotes should have quotes stripped."""
        with _tmp_cwd():
            Path(".env").write_text('QKEY1="stripped_val"\nQKEY2=\'single_val\'\n')
            _load_env()
            assert os.environ.get("QKEY1") == "stripped_val"
            assert os.environ.get("QKEY2") == "single_val"


# ────────────────────────────────────────────────────────────
# _resolve_api_key
# ────────────────────────────────────────────────────────────

class TestResolveApiKey:
    def test_env_prefix(self):
        os.environ["TEST_EASY_KEY"] = "sk-test-123"
        assert _resolve_api_key("env:TEST_EASY_KEY") == "sk-test-123"

    def test_env_not_set(self):
        result = _resolve_api_key("env:NONEXISTENT_VAR_XYZZY")
        assert result == ""

    def test_none_source(self):
        assert _resolve_api_key("none") == "not-needed"

    def test_empty_source(self):
        assert _resolve_api_key("") == ""

    def test_literal_key(self):
        assert _resolve_api_key("sk-literal-key-abc") == "sk-literal-key-abc"

    def test_env_easyrouter_api_key(self):
        os.environ["EASYROUTER_API_KEY"] = "er-key-456"
        assert _resolve_api_key("env:EASYROUTER_API_KEY") == "er-key-456"


# ────────────────────────────────────────────────────────────
# _create_llm_from_config
# ────────────────────────────────────────────────────────────

class TestCreateLLMFromConfig:
    def test_fallback_to_easyrouter_when_no_pool(self):
        """When no config/llm_pool.json exists, fall back to easyrouter."""
        with _tmp_cwd():
            os.environ["EASYROUTER_API_KEY"] = "test-er-key"
            from src.cli import _create_llm_from_config

            args = MagicMock()
            args.provider = None
            args.api_key = None

            llm, pool = _create_llm_from_config(args)
            from src.llm import OpenAICompatibleProvider
            assert isinstance(llm, OpenAICompatibleProvider)
            assert llm.base_url == "https://easyrouter.io"
            assert llm.api_key == "test-er-key"
            assert llm.default_model == "gemini-2.5-flash"

    def test_fallback_to_easyrouter_via_create_provider(self):
        """When no pool, use create_provider('easyrouter')."""
        with _tmp_cwd():
            os.environ["EASYROUTER_API_KEY"] = "test-er-key-2"
            from src.cli import _create_llm_from_config

            args = MagicMock()
            args.provider = None
            args.api_key = None

            llm, pool = _create_llm_from_config(args)
            assert llm.base_url == "https://easyrouter.io"
            assert llm.default_model == "gemini-2.5-flash"

    def test_uses_arg_provider_when_set(self):
        """When --provider is explicitly set, use create_provider."""
        with _tmp_cwd():
            os.environ["EASYROUTER_API_KEY"] = "test-key"
            from src.cli import _create_llm_from_config

            args = MagicMock()
            args.provider = "easyrouter"
            args.api_key = None

            llm, pool = _create_llm_from_config(args)
            from src.llm import OpenAICompatibleProvider
            assert isinstance(llm, OpenAICompatibleProvider)

    def test_llm_pool_priority(self):
        """When llm_pool.json exists, use first enabled provider."""
        with _tmp_cwd():
            os.makedirs("config", exist_ok=True)
            os.environ["EASYROUTER_API_KEY"] = "pool-key"

            pool_config = {
                "providers": [
                    {
                        "id": "test-provider-1",
                        "type": "openai",
                        "endpoint": "https://api.test.com/v1",
                        "model": "test-model",
                        "api_key_source": "env:EASYROUTER_API_KEY",
                        "capabilities": ["general"],
                        "modality": ["text"],
                        "enabled": True,
                        "verified": False,
                        "tags": [],
                        "cost_per_1m_input": 0.0,
                        "cost_per_1m_output": 0.0,
                        "max_concurrent": 5,
                    },
                ],
                "selection": {
                    "default_strategy": "balanced",
                    "task_mapping": {},
                    "strategies": {},
                },
            }
            Path("config/llm_pool.json").write_text(json.dumps(pool_config))

            from src.cli import _create_llm_from_config

            args = MagicMock()
            args.provider = None
            args.api_key = None

            llm, pool = _create_llm_from_config(args)
            from src.llm import OpenAICompatibleProvider
            assert isinstance(llm, OpenAICompatibleProvider)
            assert llm.base_url == "https://api.test.com/v1"
            assert llm.default_model == "test-model"
            assert llm.api_key == "pool-key"

    def test_llm_pool_disabled_skipped(self):
        """Disabled providers should be skipped."""
        with _tmp_cwd():
            os.makedirs("config", exist_ok=True)
            os.environ["EASYROUTER_API_KEY"] = "pool-key-2"

            pool_config = {
                "providers": [
                    {
                        "id": "disabled-provider",
                        "type": "openai",
                        "endpoint": "https://disabled.example.com/v1",
                        "model": "disabled-model",
                        "api_key_source": "env:EASYROUTER_API_KEY",
                        "capabilities": ["general"],
                        "modality": ["text"],
                        "enabled": False,  # DISABLED
                        "verified": False,
                        "tags": [],
                        "cost_per_1m_input": 0.0,
                        "cost_per_1m_output": 0.0,
                        "max_concurrent": 5,
                    },
                    {
                        "id": "enabled-provider",
                        "type": "openai",
                        "endpoint": "https://enabled.example.com/v1",
                        "model": "enabled-model",
                        "api_key_source": "env:EASYROUTER_API_KEY",
                        "capabilities": ["general"],
                        "modality": ["text"],
                        "enabled": True,  # ENABLED
                        "verified": False,
                        "tags": [],
                        "cost_per_1m_input": 0.0,
                        "cost_per_1m_output": 0.0,
                        "max_concurrent": 5,
                    },
                ],
                "selection": {
                    "default_strategy": "balanced",
                    "task_mapping": {},
                    "strategies": {},
                },
            }
            Path("config/llm_pool.json").write_text(json.dumps(pool_config))

            from src.cli import _create_llm_from_config

            args = MagicMock()
            args.provider = None
            args.api_key = None

            llm, pool = _create_llm_from_config(args)
            assert llm.base_url == "https://enabled.example.com/v1"
            assert llm.default_model == "enabled-model"

    def test_no_key_without_env(self):
        """When no env var is set and no pool, still return provider (empty key)."""
        with _tmp_cwd():
            # Clear env
            old = os.environ.pop("EASYROUTER_API_KEY", None)
            try:
                from src.cli import _create_llm_from_config

                args = MagicMock()
                args.provider = None
                args.api_key = None

                llm, pool = _create_llm_from_config(args)
                from src.llm import OpenAICompatibleProvider
                assert isinstance(llm, OpenAICompatibleProvider)
                # Key will be empty, which is acceptable (provider handles error on use)
            finally:
                if old:
                    os.environ["EASYROUTER_API_KEY"] = old


# ────────────────────────────────────────────────────────────
# create_provider("easyrouter")
# ────────────────────────────────────────────────────────────

class TestCreateProviderEasyRouter:
    def test_easyrouter_returns_openai_compatible(self):
        """create_provider('easyrouter') should return OpenAICompatibleProvider."""
        from src.llm import create_provider, OpenAICompatibleProvider

        provider = create_provider("easyrouter", api_key="test-key-123")
        assert isinstance(provider, OpenAICompatibleProvider)
        assert provider.base_url == "https://easyrouter.io"
        assert provider.api_key == "test-key-123"
        assert provider.default_model == "gemini-2.5-flash"

    def test_easyrouter_uses_env_key(self):
        """EasyRouter should use EASYROUTER_API_KEY from env when no explicit key."""
        os.environ["EASYROUTER_API_KEY"] = "env-key-789"
        from src.llm import create_provider

        provider = create_provider("easyrouter")
        assert provider.api_key == "env-key-789"

    def test_easyrouter_explicit_key_overrides_env(self):
        """Explicit api_key should override env var."""
        os.environ["EASYROUTER_API_KEY"] = "env-key"
        from src.llm import create_provider

        provider = create_provider("easyrouter", api_key="explicit-key")
        assert provider.api_key == "explicit-key"

    def test_easyrouter_default_model(self):
        """EasyRouter should default to deepseek-v4-pro."""
        from src.llm import create_provider

        provider = create_provider("easyrouter", api_key="test")
        assert provider.default_model == "gemini-2.5-flash"

    def test_easyrouter_custom_model(self):
        """EasyRouter should accept custom default_model."""
        from src.llm import create_provider

        provider = create_provider("easyrouter", api_key="test", default_model="gpt-4o")
        assert provider.default_model == "gpt-4o"


# ────────────────────────────────────────────────────────────
# Existing providers still work
# ────────────────────────────────────────────────────────────

class TestCreateProviderExisting:
    def test_deepseek_still_works(self):
        from src.llm import create_provider, DeepSeekProvider
        provider = create_provider("deepseek", api_key="sk-test")
        assert isinstance(provider, DeepSeekProvider)

    def test_openai_compatible_still_works(self):
        from src.llm import create_provider, OpenAICompatibleProvider
        provider = create_provider("openai-compatible", base_url="http://localhost:8080/v1", api_key="test")
        assert isinstance(provider, OpenAICompatibleProvider)

    def test_unknown_provider_raises(self):
        from src.llm import create_provider
        with pytest.raises(ValueError, match="Unknown provider type"):
            create_provider("nonexistent-provider")
