"""Coerce native or stringified LLM tool arguments."""

from __future__ import annotations

import json
from typing import Any


def coerce_index(index: Any) -> int:
    """Accept a 1-based int or a digit string (``1`` / ``\"1\"``)."""
    if isinstance(index, bool) or index is None:
        raise ValueError(f"invalid index: {index!r}")
    if isinstance(index, int):
        return index
    if isinstance(index, str):
        text = index.strip()
        if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
            return int(text)
        raise ValueError(f"invalid index: {index!r}")
    raise ValueError(f"invalid index: {index!r}")


def coerce_checklist(value: Any) -> list[str] | None:
    """Accept a list or a JSON array string."""
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid checklist: {value!r}") from exc
        if not isinstance(parsed, list):
            raise ValueError(f"invalid checklist: {value!r}")
        return [str(item) for item in parsed]
    raise ValueError(f"invalid checklist: {value!r}")


def coerce_agent_props(value: Any) -> dict[str, Any] | None:
    """Accept a dict or a JSON object string."""
    if value is None:
        return None
    if isinstance(value, dict):
        return dict(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid agent_props: {value!r}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"invalid agent_props: {value!r}")
        return parsed
    raise ValueError(f"invalid agent_props: {value!r}")
