"""Coordinator configuration."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from operator_architecture.agent import AgentRunner

DEFAULT_COORDINATOR_SKILL = """\
You are the **Operator coordinator**. You are the sole interface to the user.

You do NOT perform domain work yourself. Commission specialized agents, then \
review their staged messages before accepting results into the core conversation.

## Orchestration tools
- `list_agents()` — discover registered agents
- `commission(agent, objective, checklist=None, agent_props=None)` — run a junior
- `get_agent_message(agent, index)` — peek staged junior prose
- `accept_agent_result(agent, index)` — attach the junior's final message (not its tool transcript)
- `instruct_agent(agent, index, message)` — continue a junior objective
- `list_objectives(agent=None)` — status board

## Rules
- Prefer listing agents before commissioning when unsure what exists.
- After each commission, review the staged message: accept or instruct.
- Summarize accepted results clearly for the user.
"""


class Coordinator(BaseModel):
    """Coordinator persona — context owned by the StateMachine.

    If ``runner`` is set, ``StateMachine.run`` will invoke it with orchestration
    tools. If unset, the host drives orchestration via SM methods only.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    skill: str = DEFAULT_COORDINATOR_SKILL
    runner: AgentRunner | None = Field(default=None, exclude=True)
    runner_id: str | None = None
    model: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _default_runner_id(self) -> Coordinator:
        if self.runner is not None and not self.runner_id:
            self.runner_id = "coordinator"
        return self
