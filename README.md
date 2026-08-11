# Operator Architecture

**Framework-agnostic multi-agent orchestration SDK.**

Operator Architecture (OA) manages **state**, **context**, **sub-agents**, and **orchestration**. It does **not** call LLMs, run tool loops, or ship coding tools. You plug in any agent runtime — Relay, LangChain, OpenAI Agents, HTTP services, or a plain async function.

For the earlier coding-CLI / five-pillars vision, see [ORIGINAL_CONCEPT.md](ORIGINAL_CONCEPT.md).

## Install

```bash
pip install -e .
# or
uv pip install -e .
```

Runtime dependencies: **none** (stdlib only).

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

## What OA owns vs what you own

| Operator Architecture | Your host |
|-----------------------|-----------|
| `StateMachine`, `Coordinator`, `AgentSpec` | Agent implementations (`AgentRunner`) |
| OpenAI-compatible message threads | Relay / LangChain / custom loops |
| `commission` → stage → `accept` / `instruct` | Models, API keys, tools, MCP, FS |
| Optional `streaming_callback` fan-in | Emitting stream events from runners |

## Core objects

### `AgentSpec` + `AgentRunner`

Register any number of sub-agents. Each needs a **runner** OA will call:

```python
async def research(request: AgentRequest) -> AgentResult:
    # call Relay, LangChain, HTTP, … — OA does not care
    return AgentResult(content=f"Findings for: {request.objective}")

researcher = AgentSpec(
    name="researcher",
    description="Read-only exploration",
    skill="You are a careful researcher. Answer with concrete findings.",
    runner=callable_agent(research),
    model="my-model",  # metadata only
)
```

Protocol:

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
`AgentResult.content` is staged as `agent_message`.

### `Coordinator`

Owns the user-facing skill string and optional runner for `sm.run()`:

```python
coordinator = Coordinator(
    skill="You operate the state machine…",  # default skill provided
    runner=my_coordinator_runner,            # optional
    model="coord-model",
)
```

### `StateMachine`

```python
sm = StateMachine(coordinator=coordinator, agents=[researcher])
```

No process-global singleton — hold the instance yourself (`sm = StateMachine(...)`).

## Lifecycle

```text
user → (optional) sm.run / host
     → commission(agent, objective)
         → AgentRunner.run(AgentRequest)
         → stage agent_message  (status=staged)
     → accept_agent_result(agent, index)   # compact result for core thread
        or instruct_agent(agent, index, message)  # continue junior
```

### Direct API (always available)

```python
staged = await sm.commission("researcher", "Find all uses of vLLM")
msg = sm.get_agent_message("researcher", 1)
accepted = sm.accept("researcher", 1)          # or accept_agent_result
# or:
await sm.instruct("researcher", 1, "Also check Dockerfiles")

sm.list_agents()
sm.list_objectives()
```

Indexed access: `sm.agent("researcher")[1].agent_message`.

### `sm.run` (optional)

If `Coordinator.runner` is set, `await sm.run(user_text, streaming_callback=...)` appends the user message and invokes that runner with:

- `metadata["tools"]` — orchestration callables
- `metadata["tool_schemas"]` — OpenAI `tools[]` schemas

**OA does not execute a tool loop.** Your runner (Relay, LangChain, …) must invoke those callables when the model requests them.

```python
tools = sm.orchestration_tools()
# list_agents, commission, get_agent_message,
# accept_agent_result, instruct_agent, list_objectives
```

## OpenAI-compatible context

Coordinator and junior threads are lists of chat.completions-style dicts:

```python
{"role": "system"|"user"|"assistant"|"tool", "content": "...", ...}
```

Helpers: `Messages` (`.system()`, `.user()`, `.assistant()`, `.to_list()`).

## `streaming_callback`

Optional observability hook (UI, logs, websockets). Sync or async:

```python
async def on_stream(event: dict) -> None:
    print(event["phase"], event.get("detail", "")[:80])

await sm.commission("researcher", "…", streaming_callback=on_stream)
# or
await sm.run("…", streaming_callback=on_stream)
```

Phases: `start`, `token`, `tool_start`, `tool_result`, `tool_error`, `commissioned`, `done`, `fail`.  
Runners may emit events; OA forwards them and also emits lifecycle events around commission.

## Writing adapters

| Adapter idea | Wraps |
|--------------|--------|
| `callable_agent(fn)` | Plain async/sync function (shipped) |
| Relay agent | `encode.relay_async` / `courier_os.relay` inside `run()` |
| LangChain agent | AgentExecutor / LangGraph; map messages ↔ LC messages |
| HTTP agent | POST OpenAI-compatible or custom JSON API |

OA never imports those libraries. Keep adapters in your host.

### Minimal Relay sketch (host code)

```python
class RelayRunner:
    def __init__(self, model, api_key, base_url, tools):
        self.model, self.api_key, self.base_url, self.tools = model, api_key, base_url, tools

    async def run(self, request, *, streaming_callback=None):
        import encode
        messages = encode.Messages()
        for m in request.messages:
            # map dicts into encode.Messages as needed
            ...
        out = await encode.relay_async(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            messages=messages,
            tools=self.tools or request.metadata.get("tools"),
        )
        return AgentResult(content=out.content or "", raw=out)
```

## Example

See [`examples/minimal_callable.py`](examples/minimal_callable.py).

## Design boundaries

- **Not** a coding CLI or competitor to Cursor/Claude Code
- **Not** an inference or tool-loop SDK (use Courier OS, encode, agentloop, LangChain, …)
- **Not** coupled to AXE or Courier OS — hosts may wrap them as runners

## License / status

Early SDK (`0.2.0`). API may evolve; the orchestration contract (commission / stage / accept) is the stable idea.
