"""OpenAI tool schemas + callables for coordinator orchestration."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any


def openai_tool_schema(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> dict[str, Any]:
    """Build a chat.completions tools[] entry from a callable's signature/doc."""
    tool_name = name or fn.__name__
    doc = (description or inspect.getdoc(fn) or "").strip()
    sig = inspect.signature(fn)
    properties: dict[str, Any] = {}
    required: list[str] = []

    try:
        hints = inspect.get_annotations(fn, eval_str=True)
    except Exception:
        hints = dict(getattr(fn, "__annotations__", {}) or {})

    for pname, param in sig.parameters.items():
        if pname in {"self", "cls"}:
            continue
        prop: dict[str, Any] = {"type": _json_type(hints.get(pname, str))}
        properties[pname] = prop
        if param.default is inspect.Parameter.empty:
            required.append(pname)

    return {
        "type": "function",
        "function": {
            "name": tool_name,
            "description": doc.split("\n\n")[0] if doc else tool_name,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def _json_type(annotation: Any) -> str:
    origin = getattr(annotation, "__origin__", None)
    if origin is list:
        return "array"
    if origin is dict:
        return "object"
    name = getattr(annotation, "__name__", "") or str(annotation)
    mapping = {
        "str": "string",
        "int": "integer",
        "float": "number",
        "bool": "boolean",
        "list": "array",
        "dict": "object",
    }
    # Optional[X] / X | None
    if "None" in str(annotation) or "Optional" in str(annotation):
        args = getattr(annotation, "__args__", ())
        for a in args:
            if a is not type(None):
                return _json_type(a)
    return mapping.get(name, "string")


def attach_schema(fn: Callable[..., Any], schema: dict[str, Any]) -> Callable[..., Any]:
    """Attach OpenAI tool schema on the callable for host runners."""
    setattr(fn, "__oa_tool_schema__", schema)
    return fn


def tool_schemas(tools: list[Callable[..., Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fn in tools:
        schema = getattr(fn, "__oa_tool_schema__", None)
        if schema is None:
            schema = openai_tool_schema(fn)
        out.append(schema)
    return out
