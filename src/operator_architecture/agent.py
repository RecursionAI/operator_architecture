"""Agent registration and runner protocol."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from operator_architecture.streaming import StreamingCallback


@dataclass
class AgentRequest:
    """Input handed to a host ``AgentRunner`` when an objective is run."""

    agent: str
    objective: str
    skill: str
    messages: list[dict[str, Any]]
    checklist: list[str] | None = None
    agent_props: dict[str, Any] | None = None
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    """Output from a host agent — staged onto the objective slot."""

    content: str
    messages: list[dict[str, Any]] | None = None
    usage: dict[str, Any] | None = None
    raw: Any = None


@runtime_checkable
class AgentRunner(Protocol):
    """Host-provided agent runtime (Relay, LangChain, HTTP, callable, …)."""

    async def run(
        self,
        request: AgentRequest,
        *,
        streaming_callback: StreamingCallback = None,
    ) -> AgentResult: ...


@dataclass
class AgentSpec:
    """Register a named sub-agent with the state machine."""

    name: str
    description: str
    skill: str
    runner: AgentRunner
    model: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ObjectiveSlot:
    """One commissioned objective for a sub-agent."""

    index: int
    agent: str
    objective: str
    checklist: list[str] = field(default_factory=list)
    agent_props: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, Any]] = field(default_factory=list)
    agent_message: str | None = None
    result: dict[str, Any] | None = None
    status: str = "pending"  # pending | running | staged | accepted | failed
    model: str | None = None
    duration_ms: float | None = None
    commission_id: str = ""

    def __post_init__(self) -> None:
        if not self.commission_id:
            self.commission_id = f"{self.agent}-{self.index}"


@dataclass
class AgentHandle:
    """Runtime handle for one registered agent + its objective slots."""

    spec: AgentSpec
    slots: list[ObjectiveSlot] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def runner(self) -> AgentRunner:
        return self.spec.runner

    def __len__(self) -> int:
        return len(self.slots)

    def __getitem__(self, index: int) -> ObjectiveSlot:
        if index < 1 or index > len(self.slots):
            raise IndexError(
                f"{self.name} has {len(self.slots)} objectives; "
                f"requested index {index} (1-based)"
            )
        return self.slots[index - 1]

    def next_index(self) -> int:
        return len(self.slots) + 1

    def create_objective(
        self,
        objective: str,
        *,
        checklist: list[str] | None = None,
        agent_props: dict[str, Any] | None = None,
    ) -> ObjectiveSlot:
        idx = self.next_index()
        slot = ObjectiveSlot(
            index=idx,
            agent=self.name,
            objective=objective,
            checklist=list(checklist or []),
            agent_props=dict(agent_props or {}),
            model=self.spec.model,
            messages=[
                {"role": "system", "content": self.spec.skill},
            ],
        )
        self.slots.append(slot)
        return slot

    def objectives(self) -> list[ObjectiveSlot]:
        return list(self.slots)


def callable_agent(
    fn: Callable[[AgentRequest], Awaitable[AgentResult] | AgentResult],
) -> AgentRunner:
    """Wrap an async/sync callable as an ``AgentRunner``."""

    class _CallableRunner:
        async def run(
            self,
            request: AgentRequest,
            *,
            streaming_callback: StreamingCallback = None,
        ) -> AgentResult:
            _ = streaming_callback
            result = fn(request)
            if hasattr(result, "__await__"):
                result = await result  # type: ignore[misc]
            if isinstance(result, AgentResult):
                return result
            return AgentResult(content=str(result))

    return _CallableRunner()
