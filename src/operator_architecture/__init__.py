"""Operator Architecture — framework-agnostic multi-agent orchestration SDK."""

from __future__ import annotations

from operator_architecture.agent import (
    AgentHandle,
    AgentRequest,
    AgentResult,
    AgentRunner,
    AgentSpec,
    ObjectiveSlot,
    SlotStatus,
    callable_agent,
)
from operator_architecture.coordinator import DEFAULT_COORDINATOR_SKILL, Coordinator
from operator_architecture.machine import StateMachine
from operator_architecture.messages import Message, Messages
from operator_architecture.orchestration import openai_tool_schema, tool_schemas
from operator_architecture.serialize import (
    AgentHandleModel,
    AgentSpecModel,
    ChatMessage,
    CoordinatorModel,
    StateMachineModel,
    agent_handle_from_model,
    agent_handle_to_model,
    agent_spec_from_model,
    agent_spec_to_model,
    coordinator_from_model,
    coordinator_to_model,
    messages_from_model,
    messages_to_model,
    state_machine_from_json,
    state_machine_from_model,
    state_machine_to_json,
    state_machine_to_model,
)
from operator_architecture.streaming import StreamEvent, StreamingCallback, emit_stream

__all__ = [
    "AgentHandle",
    "AgentHandleModel",
    "AgentRequest",
    "AgentResult",
    "AgentRunner",
    "AgentSpec",
    "AgentSpecModel",
    "ChatMessage",
    "Coordinator",
    "CoordinatorModel",
    "DEFAULT_COORDINATOR_SKILL",
    "Message",
    "Messages",
    "ObjectiveSlot",
    "SlotStatus",
    "StateMachine",
    "StateMachineModel",
    "StreamEvent",
    "StreamingCallback",
    "agent_handle_from_model",
    "agent_handle_to_model",
    "agent_spec_from_model",
    "agent_spec_to_model",
    "callable_agent",
    "coordinator_from_model",
    "coordinator_to_model",
    "emit_stream",
    "messages_from_model",
    "messages_to_model",
    "openai_tool_schema",
    "state_machine_from_json",
    "state_machine_from_model",
    "state_machine_to_json",
    "state_machine_to_model",
    "tool_schemas",
]

__version__ = "0.4.0"
