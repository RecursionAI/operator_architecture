"""Minimal OA example — sm.run is the whole operator turn (no LLM libraries)."""

from __future__ import annotations

import asyncio

from operator_architecture import (
    AgentRequest,
    AgentResult,
    AgentSpec,
    Coordinator,
    StateMachine,
)


class CoordinatorRunner:
    async def run(self, request: AgentRequest, *, streaming_callback=None) -> AgentResult:
        # Handle the agentic loop here.
        # request.messages — coordinator thread so far
        # request.metadata["tools"] / ["tool_schemas"] — commission, instruct, accept, ...
        # Call your model, invoke those tools, repeat until you can reply to the user.
        _ = streaming_callback
        return AgentResult(content=f"Acknowledged: {request.objective}")


class ResearchRunner:
    async def run(self, request: AgentRequest, *, streaming_callback=None) -> AgentResult:
        # Junior work: Relay, LangChain, HTTP, ...
        _ = streaming_callback
        return AgentResult(content=f"Findings for: {request.objective}")


async def main() -> None:
    sm = StateMachine(
        coordinator=Coordinator(runner=CoordinatorRunner()),
        agents=[
            AgentSpec(
                name="researcher",
                description="Read-only exploration",
                skill="You are a researcher. Be concrete.",
                runner=ResearchRunner(),
            ),
        ],
    )

    events: list[str] = []

    async def on_stream(event: dict) -> None:
        events.append(f"{event.get('agent')}:{event.get('phase')}")

    reply = await sm.run(
        "Map where vLLM is configured",
        streaming_callback=on_stream,
    )
    print("reply:\n", reply)
    print("stream phases:", events)
    print("agents:", sm.list_agents())


if __name__ == "__main__":
    asyncio.run(main())
