"""Operator Architecture — framework-agnostic multi-agent orchestration SDK."""

from __future__ import annotations

from operator_architecture.agent import (
    AgentHandle,
    AgentRequest,
    AgentResult,
    AgentRunner,
    AgentSpec,
    ObjectiveSlot,
    callable_agent,
)
from operator_architecture.coordinator import DEFAULT_COORDINATOR_SKILL, Coordinator
from operator_architecture.machine import StateMachine
from operator_architecture.messages import Message, Messages
from operator_architecture.orchestration import openai_tool_schema, tool_schemas
from operator_architecture.streaming import StreamEvent, StreamingCallback, emit_stream

__all__ = [
    "AgentHandle",
    "AgentRequest",
    "AgentResult",
    "AgentRunner",
    "AgentSpec",
    "Coordinator",
    "DEFAULT_COORDINATOR_SKILL",
    "Message",
    "Messages",
    "ObjectiveSlot",
    "StateMachine",
    "StreamEvent",
    "StreamingCallback",
    "callable_agent",
    "emit_stream",
    "openai_tool_schema",
    "tool_schemas",
]

__version__ = "0.2.0"
