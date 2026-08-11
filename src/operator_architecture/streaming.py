"""Streaming / observability types for Operator Architecture."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, TypedDict


class StreamEvent(TypedDict, total=False):
    """Structured event forwarded from host agents or emitted by the SM."""

    agent: str
    phase: str  # start | token | tool_start | tool_result | tool_error | commissioned | done | fail
    detail: str
    model: str
    objective: str
    commission_id: str
    status: str
    duration_ms: float
    index: int


StreamingCallback = (
    Callable[[StreamEvent], None]
    | Callable[[StreamEvent], Awaitable[None]]
    | None
)


async def emit_stream(
    callback: StreamingCallback,
    event: StreamEvent | dict[str, Any],
) -> None:
    if callback is None:
        return
    result = callback(event)  # type: ignore[arg-type]
    if hasattr(result, "__await__"):
        await result  # type: ignore[misc]
