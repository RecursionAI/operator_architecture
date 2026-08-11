"""Minimal OA example — host-provided callable agents, no LLM libraries."""

from __future__ import annotations

import asyncio

from operator_architecture import (
    AgentRequest,
    AgentResult,
    AgentSpec,
    Coordinator,
    StateMachine,
    callable_agent,
)


async def research(request: AgentRequest) -> AgentResult:
    lines = [
        f"Skill: {request.skill[:60]}…",
        f"Objective: {request.objective}",
    ]
    if request.checklist:
        lines.append("Checklist: " + ", ".join(request.checklist))
    return AgentResult(content="\n".join(lines))


async def main() -> None:
    sm = StateMachine(
        coordinator=Coordinator(),
        agents=[
            AgentSpec(
                name="researcher",
                description="Read-only exploration",
                skill="You are a researcher. Be concrete.",
                runner=callable_agent(research),
            ),
        ],
    )

    events: list[str] = []

    async def on_stream(event: dict) -> None:
        events.append(f"{event.get('agent')}:{event.get('phase')}")

    staged = await sm.commission(
        "researcher",
        "Map where vLLM is configured",
        checklist=["find config", "note versions"],
        streaming_callback=on_stream,
    )
    print("staged:", staged["status"], staged["index"])
    print("message:\n", sm.agent("researcher")[1].agent_message)
    print("accept:", sm.accept("researcher", 1)["status"])
    print("stream phases:", events)
    print("agents:", sm.list_agents())
    print("objectives:", sm.list_objectives())


if __name__ == "__main__":
    asyncio.run(main())
