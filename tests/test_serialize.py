"""Serialize / rehydrate StateMachine snapshots."""

from __future__ import annotations

import asyncio
import json

import pytest

from operator_architecture import (
    AgentRequest,
    AgentResult,
    AgentSpec,
    Coordinator,
    StateMachine,
    StateMachineModel,
    callable_agent,
    state_machine_from_json,
    state_machine_from_model,
    state_machine_to_json,
    state_machine_to_model,
)


LONG_REPORT = ("Finding line. " * 80).strip()


async def _research(request: AgentRequest) -> AgentResult:
    return AgentResult(
        content=LONG_REPORT + f"\nObjective: {request.objective}",
        raw={"n": 1},
    )


class _ToyCoordinator:
    async def run(self, request: AgentRequest, *, streaming_callback=None) -> AgentResult:
        _ = streaming_callback
        tools = {fn.__name__: fn for fn in request.metadata["tools"]}
        agents = await tools["list_agents"]()
        name = agents[0]["name"]
        staged = await tools["commission"](name, request.objective)
        accepted = await tools["accept_agent_result"](name, staged["index"])
        return AgentResult(content=accepted["summary"])


def _research_runner():
    return callable_agent(_research)


def _sm(*, runner=None) -> StateMachine:
    return StateMachine(
        coordinator=Coordinator(runner=runner),
        agents=[
            AgentSpec(
                name="researcher",
                description="Read-only exploration",
                skill="Be concrete.",
                runner=_research_runner(),
            )
        ],
    )


def _runners(*, coordinator=None, researcher=None) -> dict:
    out = {"researcher": researcher or _research_runner()}
    if coordinator is not None:
        out["coordinator"] = coordinator
    return out


def test_agent_spec_constructor_and_runner_excluded() -> None:
    runner = _research_runner()
    spec = AgentSpec(
        name="researcher",
        description="Read-only exploration",
        skill="Be concrete.",
        runner=runner,
    )
    assert spec.runner is runner
    assert spec.runner_id == "researcher"
    dumped = spec.model_dump()
    assert "runner" not in dumped
    assert dumped["runner_id"] == "researcher"
    assert dumped["name"] == "researcher"


def test_agent_result_raw_excluded_from_dump() -> None:
    result = AgentResult(content="hi", raw={"secret": 1})
    assert result.raw == {"secret": 1}
    assert "raw" not in result.model_dump()
    assert "raw" not in json.loads(result.model_dump_json())


def test_round_trip_after_commission_and_accept() -> None:
    coord = _ToyCoordinator()
    research = _research_runner()
    sm = StateMachine(
        coordinator=Coordinator(runner=coord),
        agents=[
            AgentSpec(
                name="researcher",
                description="Read-only exploration",
                skill="Be concrete.",
                runner=research,
            )
        ],
    )
    reply = asyncio.run(sm.run("Map vLLM config"))
    slot = sm.agent("researcher")[1]
    assert slot.status == "accepted"

    model = sm.to_model()
    assert model.schema_version == 1
    assert model.coordinator.runner_id == "coordinator"
    assert model.agents[0].spec.runner_id == "researcher"
    assert model.agents[0].slots[0].index == 1
    assert model.agents[0].slots[0].commission_id == "researcher-1"
    assert model.agents[0].slots[0].status == "accepted"
    assert model.agents[0].slots[0].result is not None
    assert LONG_REPORT in (model.agents[0].slots[0].agent_message or "")
    assert any(m.role == "system" for m in model.coordinator_messages)
    assert "runner" not in model.agents[0].spec.model_dump()

    sm2 = StateMachine.from_model(
        model,
        runners={"researcher": research, "coordinator": coord},
    )
    slot2 = sm2.agent("researcher")[1]
    assert slot2.status == "accepted"
    assert slot2.index == 1
    assert slot2.commission_id == "researcher-1"
    assert slot2.messages == slot.messages
    assert slot2.agent_message == slot.agent_message
    assert slot2.result == slot.result
    assert sm2.coordinator_messages.to_list() == sm.coordinator_messages.to_list()
    assert sm2.active_agent == sm.active_agent
    assert sm2.coordinator.skill == sm.coordinator.skill
    assert sm2.coordinator.model == sm.coordinator.model
    assert reply == slot2.agent_message


def test_round_trip_while_staged() -> None:
    research = _research_runner()
    sm = StateMachine(
        agents=[
            AgentSpec(
                name="researcher",
                description="Read-only exploration",
                skill="Be concrete.",
                runner=research,
            )
        ],
    )
    staged = asyncio.run(sm.commission("researcher", "Map vLLM config"))
    assert staged["status"] == "staged"
    model = state_machine_to_model(sm)
    sm2 = state_machine_from_model(model, runners={"researcher": research})
    slot = sm2.agent("researcher")[1]
    assert slot.status == "staged"
    assert slot.index == 1
    assert slot.commission_id == "researcher-1"
    assert slot.agent_message == sm.agent("researcher")[1].agent_message
    assert sm2.coordinator.runner is None


def test_running_status_restored_as_is() -> None:
    research = _research_runner()
    sm = _sm()
    asyncio.run(sm.commission("researcher", "Map vLLM config"))
    sm.agent("researcher")[1].status = "running"
    sm.active_agent = "researcher"
    sm2 = StateMachine.from_model(sm.to_model(), runners=_runners(researcher=research))
    assert sm2.agent("researcher")[1].status == "running"
    assert sm2.active_agent == "researcher"


def test_tool_calls_survive_json_round_trip() -> None:
    research = _research_runner()
    sm = StateMachine(
        agents=[
            AgentSpec(
                name="researcher",
                description="Read-only exploration",
                skill="Be concrete.",
                runner=research,
            )
        ],
    )
    sm.coordinator_messages.assistant(
        "",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "list_agents", "arguments": "{}"},
            }
        ],
    )
    blob = state_machine_to_json(sm)
    sm2 = state_machine_from_json(blob, runners={"researcher": research})
    dumped = sm2.coordinator_messages.to_list()
    tool_msg = next(m for m in dumped if m.get("tool_calls"))
    assert tool_msg["tool_calls"][0]["id"] == "call_1"
    assert tool_msg["tool_calls"][0]["function"]["name"] == "list_agents"


def test_missing_runner_id_raises_key_error() -> None:
    sm = _sm()
    asyncio.run(sm.commission("researcher", "Map vLLM config"))
    model = sm.to_model()
    with pytest.raises(KeyError, match="researcher"):
        StateMachine.from_model(model, runners={})


def test_missing_coordinator_runner_id_raises_key_error() -> None:
    sm = _sm(runner=_ToyCoordinator())
    model = sm.to_model()
    assert model.coordinator.runner_id == "coordinator"
    with pytest.raises(KeyError, match="coordinator"):
        StateMachine.from_model(model, runners={"researcher": _research_runner()})


def test_json_dumps_loads_round_trip() -> None:
    research = _research_runner()
    sm = StateMachine(
        agents=[
            AgentSpec(
                name="researcher",
                description="Read-only exploration",
                skill="Be concrete.",
                runner=research,
            )
        ],
    )
    asyncio.run(sm.commission("researcher", "Map vLLM config"))
    blob = state_machine_to_json(sm)
    parsed = json.loads(blob)
    assert parsed["schema_version"] == 1
    sm2 = state_machine_from_json(blob, runners={"researcher": research})
    assert sm2.agent("researcher")[1].status == "staged"
    assert sm2.agent("researcher")[1].objective == "Map vLLM config"


def test_rehydrated_machine_can_commission_and_run() -> None:
    coord = _ToyCoordinator()
    research = _research_runner()
    sm = StateMachine(
        coordinator=Coordinator(runner=coord),
        agents=[
            AgentSpec(
                name="researcher",
                description="Read-only exploration",
                skill="Be concrete.",
                runner=research,
            )
        ],
    )
    sm2 = StateMachine.from_model(
        sm.to_model(),
        runners={"researcher": research, "coordinator": coord},
    )
    staged = asyncio.run(sm2.commission("researcher", "Find Dockerfiles"))
    assert staged["status"] == "staged"
    assert sm2.agent("researcher")[1].index == 1

    reply = asyncio.run(sm2.run("Map vLLM config"))
    assert LONG_REPORT in reply
    assert sm2.agent("researcher")[2].status == "accepted"


def test_from_model_accepts_state_machine_model_directly() -> None:
    research = _research_runner()
    sm = _sm()
    asyncio.run(sm.commission("researcher", "Map vLLM config"))
    model = StateMachineModel.model_validate(sm.to_model().model_dump())
    sm2 = state_machine_from_model(model, runners={"researcher": research})
    assert sm2.agent("researcher")[1].status == "staged"


def test_coordinator_runner_kwarg_overrides_mapping() -> None:
    coord = _ToyCoordinator()
    research = _research_runner()
    sm = _sm(runner=_ToyCoordinator())
    sm2 = StateMachine.from_model(
        sm.to_model(),
        runners={"researcher": research},
        coordinator_runner=coord,
    )
    reply = asyncio.run(sm2.run("Map vLLM config"))
    assert LONG_REPORT in reply
