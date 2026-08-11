# Operator Architecture memory

## What this is
- **Pure orchestration SDK** (`operator-architecture` → `import operator_architecture`)
- Owns state, OpenAI-shaped context, sub-agent registry, commission → stage → accept/instruct
- Does **not** call LLMs, run tool loops, or ship coding tools / TUI / CLI

## Public API
- `StateMachine`, `Coordinator`, `AgentSpec`, `AgentRequest`, `AgentResult`, `AgentRunner`
- `Messages` (OpenAI chat dicts), `callable_agent`, `orchestration_tools` / `tool_schemas`
- `streaming_callback` for host observability (`StreamEvent` phases)

## Host contract
- Each `AgentSpec.runner` implements `async run(request, *, streaming_callback=None) -> AgentResult`
- Adapters (Relay, LangChain, HTTP) live in the host — zero coupling to encode / axe / courier-os

## Layout
- `src/operator_architecture/` — SDK only
- `examples/minimal_callable.py` — no-LLM smoke example
- `ORIGINAL_CONCEPT.md` — archived coding-operator vision
- `README.md` — current SDK docs

## Non-goals (here)
- AXE / agentloop, Courier OS inference, Scout FS tools, SQLite/ACP (future optional modules)
