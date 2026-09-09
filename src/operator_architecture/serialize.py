"""Pydantic snapshot models and dump/load helpers for Operator Architecture.

Live ``AgentRunner`` callables, streaming callbacks, and orchestration-tool
closures are process-local. Snapshots store ``runner_id`` keys; hosts rebound
runners from a mapping on load.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from operator_architecture.agent import AgentHandle, AgentRunner, AgentSpec, ObjectiveSlot
from operator_architecture.coordinator import Coordinator
from operator_architecture.machine import StateMachine
from operator_architecture.messages import Messages
from operator_architecture.streaming import StreamingCallback


class ChatMessage(BaseModel):
    """OpenAI-shaped chat message. Extra keys (tool_calls, tool_call_id, …) are kept."""

    model_config = ConfigDict(extra="allow")

    role: str
    content: Any = None


class AgentSpecModel(BaseModel):
    """Serializable agent spec — ``runner`` is rebound via ``runner_id``."""

    name: str
    description: str
    skill: str
    runner_id: str
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class CoordinatorModel(BaseModel):
    """Serializable coordinator persona — ``runner`` is rebound via ``runner_id``."""

    skill: str
    runner_id: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentHandleModel(BaseModel):
    """Serializable handle: spec snapshot plus objective slots."""

    spec: AgentSpecModel
    slots: list[ObjectiveSlot] = Field(default_factory=list)


class StateMachineModel(BaseModel):
    """Durable snapshot of a ``StateMachine`` (no live runners or callbacks)."""

    schema_version: Literal[1] = 1
    coordinator: CoordinatorModel
    coordinator_messages: list[ChatMessage] = Field(default_factory=list)
    active_agent: str | None = None
    agents: list[AgentHandleModel] = Field(default_factory=list)


def _require_runner(runners: Mapping[str, AgentRunner], runner_id: str) -> AgentRunner:
    if runner_id not in runners:
        raise KeyError(f"No runner registered for {runner_id!r}")
    return runners[runner_id]


def agent_spec_to_model(spec: AgentSpec) -> AgentSpecModel:
    """Serialize an ``AgentSpec``, dropping the live runner."""
    return AgentSpecModel(
        name=spec.name,
        description=spec.description,
        skill=spec.skill,
        runner_id=spec.runner_id or spec.name,
        model=spec.model,
        metadata=dict(spec.metadata),
    )


def agent_spec_from_model(
    model: AgentSpecModel,
    runners: Mapping[str, AgentRunner],
) -> AgentSpec:
    """Rehydrate an ``AgentSpec`` by resolving ``runner_id`` in ``runners``."""
    return AgentSpec(
        name=model.name,
        description=model.description,
        skill=model.skill,
        runner=_require_runner(runners, model.runner_id),
        runner_id=model.runner_id,
        model=model.model,
        metadata=dict(model.metadata),
    )


def coordinator_to_model(coordinator: Coordinator) -> CoordinatorModel:
    """Serialize a ``Coordinator``, dropping the live runner."""
    return CoordinatorModel(
        skill=coordinator.skill,
        runner_id=coordinator.runner_id,
        model=coordinator.model,
        metadata=dict(coordinator.metadata),
    )


def coordinator_from_model(
    model: CoordinatorModel,
    *,
    runners: Mapping[str, AgentRunner],
    coordinator_runner: AgentRunner | None = None,
) -> Coordinator:
    """Rehydrate a ``Coordinator``.

    Uses ``coordinator_runner`` when given; otherwise ``runners[runner_id]``
    or ``runners["coordinator"]``. A snapshot with no ``runner_id`` loads
    without a runner (manual commission/accept mode).
    """
    runner: AgentRunner | None
    runner_id = model.runner_id
    if coordinator_runner is not None:
        runner = coordinator_runner
        if not runner_id:
            runner_id = "coordinator"
    elif model.runner_id is None:
        runner = None
    else:
        runner = runners.get(model.runner_id)
        if runner is None:
            runner = runners.get("coordinator")
        if runner is None:
            raise KeyError(f"No runner registered for {model.runner_id!r}")
    return Coordinator(
        skill=model.skill,
        runner=runner,
        runner_id=runner_id,
        model=model.model,
        metadata=dict(model.metadata),
    )


def agent_handle_to_model(handle: AgentHandle) -> AgentHandleModel:
    """Serialize an ``AgentHandle`` including its objective slots."""
    return AgentHandleModel(
        spec=agent_spec_to_model(handle.spec),
        slots=[slot.model_copy(deep=True) for slot in handle.slots],
    )


def agent_handle_from_model(
    model: AgentHandleModel,
    runners: Mapping[str, AgentRunner],
) -> AgentHandle:
    """Rehydrate an ``AgentHandle`` by resolving the spec's ``runner_id``."""
    return AgentHandle(
        spec=agent_spec_from_model(model.spec, runners),
        slots=[slot.model_copy(deep=True) for slot in model.slots],
    )


def messages_to_model(messages: Messages) -> list[ChatMessage]:
    """Validate OpenAI-shaped message dicts as ``ChatMessage`` models."""
    return [ChatMessage.model_validate(m) for m in messages.to_list()]


def messages_from_model(data: list[ChatMessage]) -> Messages:
    """Rebuild a ``Messages`` helper from snapshot chat messages."""
    return Messages([m.model_dump() for m in data])


def state_machine_to_model(sm: StateMachine) -> StateMachineModel:
    """Snapshot a live ``StateMachine`` (runners and callbacks omitted)."""
    return StateMachineModel(
        coordinator=coordinator_to_model(sm.coordinator),
        coordinator_messages=messages_to_model(sm.coordinator_messages),
        active_agent=sm.active_agent,
        agents=[agent_handle_to_model(handle) for handle in sm.agents()],
    )


def state_machine_from_model(
    model: StateMachineModel,
    *,
    runners: Mapping[str, AgentRunner],
    coordinator_runner: AgentRunner | None = None,
    streaming_callback: StreamingCallback = None,
) -> StateMachine:
    """Rehydrate a ``StateMachine`` from a snapshot, rebinding runners.

    Orchestration-tool closures are not restored; they are rebuilt on the
    next ``run()``. ``status=="running"`` slots are restored as-is.
    """
    coordinator = coordinator_from_model(
        model.coordinator,
        runners=runners,
        coordinator_runner=coordinator_runner,
    )
    sm = StateMachine(coordinator=coordinator)
    sm.coordinator_messages = messages_from_model(model.coordinator_messages)
    sm.active_agent = model.active_agent
    for handle_model in model.agents:
        handle = agent_handle_from_model(handle_model, runners)
        if not handle.name or handle.name == "coordinator":
            raise ValueError("agent name required and must not be 'coordinator'")
        if handle.name in sm._agents:
            raise ValueError(f"agent already registered: {handle.name}")
        sm._agents[handle.name] = handle
    if streaming_callback is not None:
        sm.set_streaming_callback(streaming_callback)
    return sm


def state_machine_to_json(sm: StateMachine) -> str:
    """JSON-encode a ``StateMachine`` snapshot."""
    return state_machine_to_model(sm).model_dump_json()


def state_machine_from_json(
    data: str,
    *,
    runners: Mapping[str, AgentRunner],
    coordinator_runner: AgentRunner | None = None,
    streaming_callback: StreamingCallback = None,
) -> StateMachine:
    """JSON-decode a snapshot and rehydrate a ``StateMachine``."""
    return state_machine_from_model(
        StateMachineModel.model_validate_json(data),
        runners=runners,
        coordinator_runner=coordinator_runner,
        streaming_callback=streaming_callback,
    )
