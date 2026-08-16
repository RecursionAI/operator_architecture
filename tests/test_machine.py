"""StateMachine: flexible tool args and untruncated summaries."""

from __future__ import annotations

import asyncio

from operator_architecture import (
    AgentRequest,
    AgentResult,
    AgentSpec,
    Coordinator,
    StateMachine,
    callable_agent,
    tool_schemas,
)


LONG_REPORT = ("Finding line. " * 80).strip()  # well over 400 chars


async def _research(request: AgentRequest) -> AgentResult:
    return AgentResult(content=LONG_REPORT + f"\nObjective: {request.objective}")


def _sm() -> StateMachine:
    return StateMachine(
        coordinator=Coordinator(),
        agents=[
            AgentSpec(
                name="researcher",
                description="Read-only exploration",
                skill="Be concrete.",
                runner=callable_agent(_research),
            )
        ],
    )


def test_commission_summary_is_full_agent_message() -> None:
    sm = _sm()
    staged = asyncio.run(sm.commission("researcher", "Map vLLM config"))
    full = sm.agent("researcher")[1].agent_message
    assert full is not None
    assert len(full) > 400
    assert staged["summary"] == full
    assert staged["preview"] == full[:400]
    assert sm.agent("researcher")[1].result is not None
    assert sm.agent("researcher")[1].result["report"] == full
    assert sm.agent("researcher")[1].result["summary"] == full


def test_accept_and_peek_accept_int_or_str_index() -> None:
    sm = _sm()
    asyncio.run(sm.commission("researcher", "Map vLLM config"))
    peeked_int = sm.get_agent_message("researcher", 1)
    peeked_str = sm.get_agent_message("researcher", "1")
    assert peeked_str["agent_message"] == peeked_int["agent_message"]
    assert peeked_str["index"] == 1

    accepted_str = sm.accept("researcher", "1")
    assert accepted_str["status"] == "accepted"
    assert accepted_str["summary"] == peeked_int["agent_message"]
    assert accepted_str["result"]["report"] == peeked_int["agent_message"]
    assert accepted_str["result"]["summary"] == peeked_int["agent_message"]


def test_accept_int_index_still_works() -> None:
    sm = _sm()
    asyncio.run(sm.commission("researcher", "Map vLLM config"))
    accepted = sm.accept_agent_result("researcher", 1)
    assert accepted["status"] == "accepted"
    assert accepted["index"] == 1


def test_bad_string_index_is_structured_error() -> None:
    sm = _sm()
    asyncio.run(sm.commission("researcher", "Map vLLM config"))
    out = sm.get_agent_message("researcher", "nope")
    assert out["error"] == "bad_index"
    out = sm.accept("researcher", "nope")
    assert out["error"] == "bad_index"


def test_commission_accepts_stringified_checklist_and_props() -> None:
    sm = _sm()
    staged = asyncio.run(
        sm.commission(
            "researcher",
            "Map vLLM config",
            checklist='["find config", "note versions"]',
            agent_props='{"repo": "operator-py"}',
        )
    )
    assert staged["status"] == "staged"
    slot = sm.agent("researcher")[1]
    assert slot.checklist == ["find config", "note versions"]
    assert slot.agent_props == {"repo": "operator-py"}


def test_instruct_string_index() -> None:
    sm = _sm()
    asyncio.run(sm.commission("researcher", "Map vLLM config"))
    again = asyncio.run(sm.instruct("researcher", "1", "Also check Dockerfiles"))
    assert again["status"] == "staged"
    assert again["summary"] == sm.agent("researcher")[1].agent_message


def test_orchestration_schemas_keep_native_types() -> None:
    sm = _sm()
    schemas = {s["function"]["name"]: s for s in tool_schemas(sm.orchestration_tools())}
    assert (
        schemas["accept_agent_result"]["function"]["parameters"]["properties"]["index"][
            "type"
        ]
        == "integer"
    )
    assert (
        schemas["commission"]["function"]["parameters"]["properties"]["checklist"]["type"]
        == "array"
    )
    assert (
        schemas["commission"]["function"]["parameters"]["properties"]["agent_props"][
            "type"
        ]
        == "object"
    )
