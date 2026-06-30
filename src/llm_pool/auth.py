"""AuthResolver — resolves credentials without storing plaintext secrets.

Security model:
  - Config YAML stores only metadata (env var names, AWS profile, etc.)
  - Actual secrets are loaded at runtime from:
      1. Environment variables (highest priority)
      2. ~/.openclaw/.env file
      3. AWS SDK credential chain (for aws_sdk method)
  - Resolved credentials are NEVER logged, written to disk, or passed to sub-agents
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Path to the .env file (not committed to git)
_DOTENV_PATH = Path.home() / ".openclaw" / ".env"
_dotenv_cache: dict[str, str] | None = None


def _load_dotenv() -> dict[str, str]:
    """Load ~/.openclaw/.env once, cache in memory for session."""
    global _dotenv_cache
    if _dotenv_cache is not None:
        return _dotenv_cache

    env: dict[str, str] = {}
    if _DOTENV_PATH.exists():
        for line in _DOTENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            # Strip surrounding quotes
            val = val.strip().strip('"').strip("'")
            env[key.strip()] = val
    _dotenv_cache = env
    return env


class AuthResolver:
    """Resolves auth credentials from config without storing secrets."""

    def resolve(self, auth_cfg) -> dict[str, Any]:
        """Resolve credentials from auth config.

        Returns a dict of kwargs to pass to the protocol adapter.
        Never returns plaintext keys in logs.
        """
        method = auth_cfg.method

        if method == "env_var":
            return self._resolve_env_var(auth_cfg)
        elif method == "aws_sdk":
            return self._resolve_aws_sdk(auth_cfg)
        elif method == "gcp_adc":
            return self._resolve_gcp_adc()
        elif method == "none":
            return {}
        else:
            logger.warning("Unknown auth method: %s, treating as 'none'", method)
            return {}

    def _resolve_env_var(self, auth_cfg) -> dict[str, Any]:
        key_env = auth_cfg.key_env
        if not key_env:
            return {}

        # Priority 1: environment variable
        api_key = os.environ.get(key_env)

        # Priority 2: ~/.openclaw/.env
        if not api_key:
            api_key = _load_dotenv().get(key_env)

        if not api_key:
            logger.warning(
                "API key env var '%s' not found in environment or .env — "
                "provider may fail. Set it with: export %s=<your-key>",
                key_env, key_env
            )
            return {"api_key": None}

        # Log only that we found it, never the value
        logger.debug("Resolved API key from %s (length=%d)", key_env, len(api_key))
        return {"api_key": api_key}

    def _resolve_aws_sdk(self, auth_cfg) -> dict[str, Any]:
        """Return AWS profile/region info; boto3 handles the actual credential chain."""
        return {
            "aws_profile": auth_cfg.profile,
            "aws_region": auth_cfg.region or "us-east-1",
        }

    def _resolve_gcp_adc(self) -> dict[str, Any]:
        """Signal to use Google Application Default Credentials."""
        return {"use_adc": True}

    def validate_required(self, auth_cfg) -> tuple[bool, str]:
        """Check if required credentials are available without resolving them."""
        method = auth_cfg.method

        if method == "env_var":
            key_env = auth_cfg.key_env
            if not key_env:
                return False, "key_env not specified"
            in_env = key_env in os.environ
            in_dotenv = key_env in _load_dotenv()
            if not (in_env or in_dotenv):
                return False, f"${key_env} not set"
            return True, "ok"

        elif method == "aws_sdk":
            try:
                import boto3  # type: ignore
                return True, "boto3 available"
            except ImportError:
                return False, "boto3 not installed"

        elif method == "gcp_adc":
            gac = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
            if gac and Path(gac).exists():
                return True, "GOOGLE_APPLICATION_CREDENTIALS set"
            return False, "GOOGLE_APPLICATION_CREDENTIALS not set"

        elif method == "none":
            return True, "no auth required"

        return False, f"unknown method: {method}"
