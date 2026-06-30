"""Shared utility functions for Agent-Loop."""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_json_from_llm_response(content: str, default: Any = None) -> Any:
    """Extract JSON from an LLM response that may be wrapped in markdown code blocks.

    Handles these patterns:
    - Raw JSON: {"key": "value"}
    - JSON in code block: ```json ... ```
    - JSON in generic code block: ``` ... ```

    Args:
        content: Raw LLM response text
        default: Value to return if JSON parsing fails

    Returns:
        Parsed JSON object, or default on failure.
    """
    text = content.strip()
    try:
        if "```json" in text:
            text = text.split("```json", 1)[1].split("```", 1)[0].strip()
        elif "```" in text:
            text = text.split("```", 1)[1].split("```", 1)[0].strip()
        return json.loads(text)
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Failed to extract JSON from LLM response: %s", e)
        return default
