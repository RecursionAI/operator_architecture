# Operator Architecture

**Framework-agnostic multi-agent orchestration SDK.**

Operator Architecture (OA) manages **state**, **context**, **sub-agents**, and **orchestration**. The product API is one call: `await sm.run(user_text)`. That turn is the full agentic loop — the coordinator receives the user message, commissions sub-agents, instructs or accepts their work, and replies.

Compatible with any agent runtime — Relay, LangChain, OpenAI Agents, HTTP services, or a plain async function. Runtime dependencies: **none** (stdlib only).

## Install

```bash
uv add operator-architecture
# or
pip install operator-architecture
```

```python
from operator_architecture import (
    StateMachine,
    Coordinator,
    AgentSpec,
    AgentRequest,
    AgentResult,
    callable_agent,
)
```

## The loop: `sm.run`

Every agent is an `AgentRunner`: `AgentRequest` in, `AgentResult` out. Pass one to the coordinator and one to each junior, then call `sm.run`. Hosts should not drive `commission` / `accept` / `instruct` themselves — that is what the coordinator does inside the turn.

```python
class CoordinatorRunner:
    async def run(self, request: AgentRequest, *, streaming_callback=None) -> AgentResult:
        # Handle the agentic loop here.
        # request.messages — coordinator thread so far
        # request.metadata["tools"] / ["tool_schemas"] — commission, instruct, accept, ...
        # Call your model, invoke those tools, repeat until you can reply to the user.
        return AgentResult(content="...")


class ResearchRunner:
    async def run(self, request: AgentRequest, *, streaming_callback=None) -> AgentResult:
        # Junior work: Relay, LangChain, HTTP, ...
        return AgentResult(content=f"Findings for: {request.objective}")


sm = StateMachine(
    coordinator=Coordinator(runner=CoordinatorRunner()),
    agents=[
        AgentSpec(
            name="researcher",
            description="Read-only exploration",
            skill="You are a careful researcher. Answer with concrete findings.",
            runner=ResearchRunner(),
        ),
    ],
)

reply = await sm.run("Map where vLLM is configured")
```

OA does not ship an LLM. `sm.run` is the **turn API**; `Coordinator.runner` is where the model + tool iteration lives. If that runner ignores `request.metadata["tools"]`, juniors never run.

Full runnable copy: [`examples/minimal_callable.py`](examples/minimal_callable.py).

## What `sm.run` owns

```python
await sm.run(
    user_text,
    *,
    streaming_callback=None,
    orchestration_tools=True,
) -> str
```

One call is one operator turn:

1. Requires `Coordinator.runner`. Unset → `RuntimeError`. Empty / whitespace input → `""`.
2. Appends the user message to `sm.coordinator_messages`.
3. Builds an `AgentRequest` (`agent="coordinator"`, `objective=user_text`, coordinator skill, current messages).
4. Injects orchestration tools when `orchestration_tools=True` (the default):
   - `request.metadata["tools"]` — callables
   - `request.metadata["tool_schemas"]` — OpenAI `tools[]` schemas
5. Invokes `Coordinator.runner.run(...)` once. That runner must loop on tool calls until it has a user-facing reply.
6. Persists the result onto the coordinator thread: `result.messages` replaces the thread; otherwise `result.content` is appended as an assistant message.
7. Returns the reply string (or `"(no response)"`).

```text
host: await sm.run(user_text)
  → append user message
  → Coordinator.runner
       → model round
            → commission / instruct_agent / accept_agent_result
            → junior AgentRunners
            → …until a final assistant message
  → persist reply on coordinator thread
  → return str
```

## Coordinator runner

You supply inference. OA supplies the orchestration tools the coordinator must call:

| Tool | Role |
|------|------|
| `list_agents()` | Discover registered juniors |
| `commission(agent, objective, checklist=None, agent_props=None)` | Run a junior; result is **staged** |
| `get_agent_message(agent, index)` | Peek staged prose without accepting |
| `instruct_agent(agent, index, message)` | Continue the same objective |
| `accept_agent_result(agent, index)` | Accept into the core thread |
| `list_objectives(agent=None)` | Status board |

`commission` / `instruct` return `summary` (full junior final message, no cap) plus `preview` (first 400 chars). OpenAI schemas keep native types (`index` integer, `checklist` array, `agent_props` object); at runtime OA also accepts stringified LLM values (`"1"`, `"{\"k\": \"v\"}"`). Invalid args return `{ "error": ... }` instead of raising.

Default coordinator skill already describes this loop. Override `Coordinator.skill` only if you need a different persona.

Return `AgentResult.messages` when you own the full coordinator transcript (typical for a real tool loop). Return only `content` when you do not.

## Juniors: `AgentSpec` + `AgentRunner`

Each sub-agent is a named spec whose runner OA calls on `commission` / `instruct`:

```python
class AgentRunner(Protocol):
    async def run(
        self,
        request: AgentRequest,
        *,
        streaming_callback: StreamingCallback = None,
    ) -> AgentResult: ...
```

`AgentRequest` carries OpenAI-shaped `messages`, `objective`, `skill`, optional `checklist` / `agent_props`.  
`AgentResult.content` is staged as `agent_message`. The junior’s full session stays on the slot (`sm.agent(name)[index].messages`). Core context only sees what the coordinator’s tool loop stores (the orchestration tool result, including `summary`).

`callable_agent(fn)` wraps a plain async/sync function. For Relay, LangChain, or HTTP, keep the adapter in your host — OA never imports those libraries.

## Manual APIs available if required

`sm.commission`, `sm.accept` / `sm.accept_agent_result`, `sm.instruct`, `sm.list_agents`, and `sm.list_objectives` exist for tests and debugging. Do not use them as a second way to run OA. The coordinator inside `sm.run` is supposed to call those tools.

## `streaming_callback`

Optional observability hook (UI, logs, websockets). Sync or async:

```python
async def on_stream(event: dict) -> None:
    print(event["phase"], event.get("detail", "")[:80])

await sm.run("…", streaming_callback=on_stream)
```

Phases: `start`, `token`, `tool_start`, `tool_result`, `tool_error`, `commissioned`, `done`, `fail`.  
Runners may emit events; OA forwards them and also emits lifecycle events around commission.

## OpenAI-compatible context

Coordinator and junior threads are lists of chat.completions-style dicts:

```python
{"role": "system"|"user"|"assistant"|"tool", "content": "...", ...}
```

Helpers: `Messages` (`.system()`, `.user()`, `.assistant()`, `.to_list()`).

## Appendix: SDK spec

Public surface from `operator_architecture` (`__all__`). Field lists are constructor / dataclass args unless noted.

### `Coordinator`

Dataclass. Persona handed to `StateMachine`.

- `skill: str = DEFAULT_COORDINATOR_SKILL` — coordinator system prompt
- `runner: AgentRunner | None = None` — required for `sm.run`
- `model: str | None = None` — metadata only
- `metadata: dict[str, Any] = {}` — merged into the coordinator `AgentRequest`

### `AgentSpec`

Dataclass. Registers one junior.

- `name: str` — unique; must not be `"coordinator"`
- `description: str` — shown by `list_agents`
- `skill: str` — junior system prompt
- `runner: AgentRunner` — called on commission / instruct
- `model: str | None = None` — metadata only
- `metadata: dict[str, Any] = {}` — copied onto junior `AgentRequest`

### `AgentRunner`

Protocol. Anything with this method counts (class, `callable_agent(fn)`, …).

```python
async def run(
    self,
    request: AgentRequest,
    *,
    streaming_callback: StreamingCallback = None,
) -> AgentResult: ...
```

### `AgentRequest`

Dataclass. Input to every runner.

- `agent: str`
- `objective: str`
- `skill: str`
- `messages: list[dict[str, Any]]` — OpenAI-shaped thread
- `checklist: list[str] | None = None`
- `agent_props: dict[str, Any] | None = None`
- `model: str | None = None`
- `metadata: dict[str, Any] = {}` — coordinator requests also get `tools` and `tool_schemas`

### `AgentResult`

Dataclass. Output from every runner.

- `content: str` — user-facing reply (coordinator) or staged `agent_message` (junior)
- `messages: list[dict[str, Any]] | None = None` — if set, replaces the thread
- `usage: dict[str, Any] | None = None`
- `raw: Any = None`

### `StateMachine`

```python
StateMachine(
    coordinator: Coordinator | None = None,
    agents: Sequence[AgentSpec] | None = None,
)
```

Defaults to `Coordinator()` and no juniors. No process-global singleton.

**Instance attrs**

- `coordinator: Coordinator`
- `coordinator_messages: Messages`
- `active_agent: str | None`

**Turn / registry**

- `async run(user_text, *, streaming_callback=None, orchestration_tools=True) -> str`
- `add_agent(spec: AgentSpec) -> AgentHandle`
- `agent(name: str) -> AgentHandle` — 1-based slots: `sm.agent(name)[index]`
- `agents() -> list[AgentHandle]`
- `set_streaming_callback(callback: StreamingCallback) -> None`
- `orchestration_tools() -> list` — callables + OpenAI schemas for the coordinator runner

**Last-resort (tests / debugging)**

- `async commission(agent, objective, checklist=None, agent_props=None, *, streaming_callback=None) -> dict`
- `async instruct(agent, index, message, *, streaming_callback=None) -> dict`
- `async instruct_agent(...)` — alias of `instruct`
- `accept(agent, index) -> dict` / `accept_agent_result(agent, index) -> dict`
- `get_agent_message(agent, index) -> dict`
- `list_agents() -> list[dict]`
- `list_objectives(agent=None) -> list[dict]`

### `AgentHandle`

Runtime handle for one registered junior.

- `spec: AgentSpec`
- `slots: list[ObjectiveSlot] = []`
- `name` — `spec.name`
- `runner` — `spec.runner`
- `handle[index] -> ObjectiveSlot` — **1-based** (`index` int or `"1"`)
- `objectives() -> list[ObjectiveSlot]`
- `len(handle)` — slot count

### `ObjectiveSlot`

Dataclass. One commissioned objective.

- `index: int` — 1-based
- `agent: str`
- `objective: str`
- `checklist: list[str] = []`
- `agent_props: dict[str, Any] = {}`
- `messages: list[dict[str, Any]] = []` — junior thread
- `agent_message: str | None = None` — last junior prose
- `result: dict[str, Any] | None = None`
- `status: str = "pending"` — `pending | running | staged | accepted | failed`
- `model: str | None = None`
- `duration_ms: float | None = None`
- `commission_id: str = ""` — defaults to `"{agent}-{index}"`

### `Messages` / `Message`

`Message` is `dict[str, Any]` (chat.completions shape).

```python
Messages(messages: list[Message] | None = None)
```

- `system(content) -> Messages`
- `user(content) -> Messages`
- `assistant(content, *, tool_calls=None) -> Messages`
- `tool(content, *, tool_call_id) -> Messages`
- `append(message) -> Messages` / `extend(messages) -> Messages`
- `replace(messages) -> Messages`
- `to_list() -> list[Message]` — deep copy
- `clear() -> None`

### `StreamEvent` / `StreamingCallback`

`StreamEvent` is a `TypedDict` (all keys optional):

- `agent: str`
- `phase: str` — `start | token | tool_start | tool_result | tool_error | commissioned | done | fail`
- `detail: str`
- `model: str`
- `objective: str`
- `commission_id: str`
- `status: str`
- `duration_ms: float`
- `index: int`

`StreamingCallback` — `Callable[[StreamEvent], None]` or async equivalent, or `None`.

### Helpers

- `callable_agent(fn) -> AgentRunner` — wrap `fn(request) -> AgentResult | str` (sync or async)
- `emit_stream(callback, event) -> None` — no-op if `callback` is `None`
- `openai_tool_schema(fn, *, name=None, description=None) -> dict` — chat.completions `tools[]` entry
- `tool_schemas(tools) -> list[dict]` — schemas for a list of callables
- `DEFAULT_COORDINATOR_SKILL: str` — default `Coordinator.skill`

## License / status

Early SDK (`0.3.0`). API may evolve; `sm.run` as the operator turn is the stable idea.
