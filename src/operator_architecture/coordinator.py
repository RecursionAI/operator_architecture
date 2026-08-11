"""Coordinator configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from operator_architecture.agent import AgentRunner

DEFAULT_COORDINATOR_SKILL = """\
You are the **Operator coordinator**. You are the sole interface to the user.

You do NOT perform domain work yourself. Commission specialized agents, then \
review their staged messages before accepting results into the core conversation.

## Orchestration tools
- `list_agents()` — discover registered agents
- `commission(agent, objective, checklist=None, agent_props=None)` — run a junior
- `get_agent_message(agent, index)` — peek staged junior prose
- `accept_agent_result(agent, index)` — attach compact result to the core thread
- `instruct_agent(agent, index, message)` — continue a junior objective
- `list_objectives(agent=None)` — status board

## Rules
- Prefer listing agents before commissioning when unsure what exists.
- After each commission, review the staged message: accept or instruct.
- Summarize accepted results clearly for the user.
"""


@dataclass
class Coordinator:
    """Coordinator persona — context owned by the StateMachine.

    If ``runner`` is set, ``StateMachine.run`` will invoke it with orchestration
    tools. If unset, the host drives orchestration via SM methods only.
    """

    skill: str = DEFAULT_COORDINATOR_SKILL
    runner: AgentRunner | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
