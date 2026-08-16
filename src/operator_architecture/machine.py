"""StateMachine — orchestration engine for the Operator Architecture."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

from operator_architecture.agent import (
    AgentHandle,
    AgentRequest,
    AgentResult,
    AgentSpec,
    ObjectiveSlot,
)
from operator_architecture.coerce import coerce_agent_props, coerce_checklist, coerce_index
from operator_architecture.coordinator import Coordinator
from operator_architecture.messages import Messages
from operator_architecture.orchestration import attach_schema, openai_tool_schema, tool_schemas
from operator_architecture.streaming import StreamingCallback, emit_stream


class StateMachine:
    """Framework-agnostic multi-agent orchestration engine.

    Owns coordinator context, sub-agent registry, and the commission → stage →
    accept/instruct lifecycle. Does **not** call LLMs — hosts supply
    ``AgentRunner`` implementations.
    """

    def __init__(
        self,
        coordinator: Coordinator | None = None,
        agents: Sequence[AgentSpec] | None = None,
    ) -> None:
        self.coordinator = coordinator or Coordinator()
        self.coordinator_messages = Messages().system(self.coordinator.skill)
        self.active_agent: str | None = None
        self._streaming_callback: StreamingCallback = None
        self._agents: dict[str, AgentHandle] = {}
        for spec in agents or []:
            self.add_agent(spec)

    # ── registry ──────────────────────────────────────────────────────

    def add_agent(self, spec: AgentSpec) -> AgentHandle:
        if not spec.name or spec.name == "coordinator":
            raise ValueError("agent name required and must not be 'coordinator'")
        if spec.name in self._agents:
            raise ValueError(f"agent already registered: {spec.name}")
        handle = AgentHandle(spec=spec)
        self._agents[spec.name] = handle
        return handle

    def agent(self, name: str) -> AgentHandle:
        if name not in self._agents:
            raise KeyError(f"Unknown agent: {name}")
        return self._agents[name]

    def agents(self) -> list[AgentHandle]:
        return list(self._agents.values())

    def set_streaming_callback(self, callback: StreamingCallback) -> None:
        self._streaming_callback = callback

    # ── discovery / status ────────────────────────────────────────────

    def list_agents(self) -> list[dict[str, Any]]:
        """Return registered agents (name, description, model, metadata keys)."""
        out: list[dict[str, Any]] = []
        for handle in self._agents.values():
            spec = handle.spec
            entry: dict[str, Any] = {
                "name": spec.name,
                "description": spec.description,
                "model": spec.model,
            }
            meta_tools = spec.metadata.get("tools")
            if meta_tools is not None:
                entry["tools"] = meta_tools
            out.append(entry)
        return out

    def list_objectives(self, agent: str | None = None) -> list[dict[str, Any]]:
        """Return objective status board for one agent or all."""
        handles = [self.agent(agent)] if agent else self.agents()
        rows: list[dict[str, Any]] = []
        for handle in handles:
            for slot in handle.objectives():
                rows.append(
                    {
                        "agent": slot.agent,
                        "index": slot.index,
                        "commission_id": slot.commission_id,
                        "objective": slot.objective,
                        "status": slot.status,
                        "model": slot.model,
                        "duration_ms": slot.duration_ms,
                    }
                )
        return rows

    def get_agent_message(self, agent: str, index: Any) -> dict[str, Any]:
        """Peek staged junior prose without accepting it."""
        if agent not in self._agents:
            return {"error": "unknown_agent", "agent": agent}
        try:
            idx = coerce_index(index)
            slot = self.agent(agent)[idx]
        except (ValueError, IndexError) as exc:
            return {"error": "bad_index", "detail": str(exc)}
        return {
            "agent": agent,
            "index": idx,
            "status": slot.status,
            "agent_message": slot.agent_message,
            "objective": slot.objective,
        }

    # ── commission / instruct / accept ────────────────────────────────

    async def commission(
        self,
        agent: str,
        objective: str,
        checklist: list[str] | str | None = None,
        agent_props: dict[str, Any] | str | None = None,
        *,
        streaming_callback: StreamingCallback = None,
    ) -> dict[str, Any]:
        """Create an objective slot, run the host agent, stage the result."""
        goal = (objective or "").strip()
        if not goal:
            return {"error": "missing_objective", "detail": "objective is required."}
        try:
            checklist = coerce_checklist(checklist)
            agent_props = coerce_agent_props(agent_props)
        except ValueError as exc:
            return {"error": "bad_args", "detail": str(exc)}

        handle = self.agent(agent)
        slot = handle.create_objective(
            goal, checklist=checklist, agent_props=agent_props
        )
        slot.status = "running"
        self.active_agent = agent
        cb = streaming_callback or self._streaming_callback

        await emit_stream(
            cb,
            {
                "agent": agent,
                "phase": "commissioned",
                "detail": goal,
                "model": slot.model or "",
                "objective": goal,
                "commission_id": slot.commission_id,
                "index": slot.index,
            },
        )

        brief = self._build_brief(goal, checklist, agent_props)
        slot.messages.append({"role": "user", "content": brief})

        return await self._run_slot(handle, slot, streaming_callback=cb)

    async def instruct(
        self,
        agent: str,
        index: Any,
        message: str,
        *,
        streaming_callback: StreamingCallback = None,
    ) -> dict[str, Any]:
        """Append instruction to a slot and re-run the host agent."""
        text = (message or "").strip()
        if not text:
            return {"error": "empty_message"}
        if agent not in self._agents:
            return {"error": "unknown_agent", "agent": agent}
        handle = self.agent(agent)
        try:
            idx = coerce_index(index)
            slot = handle[idx]
        except (ValueError, IndexError) as exc:
            return {"error": "bad_index", "detail": str(exc)}

        slot.messages.append({"role": "user", "content": text})
        slot.status = "running"
        self.active_agent = agent
        cb = streaming_callback or self._streaming_callback
        return await self._run_slot(handle, slot, streaming_callback=cb)

    def accept(self, agent: str, index: Any) -> dict[str, Any]:
        """Accept staged result into the coordinator core thread (compact)."""
        return self.accept_agent_result(agent, index)

    def accept_agent_result(self, agent: str, index: Any) -> dict[str, Any]:
        """Attach slot.result onto coordinator messages context."""
        if agent not in self._agents:
            return {"error": "unknown_agent", "agent": agent}
        try:
            idx = coerce_index(index)
            slot = self.agent(agent)[idx]
        except (ValueError, IndexError) as exc:
            return {"error": "bad_index", "detail": str(exc)}

        if slot.agent_message is None and slot.result is None:
            return {"error": "nothing_staged", "agent": agent, "index": idx}

        summary = slot.agent_message or ""
        compact = slot.result or {
            "status": "accepted",
            "agent": agent,
            "index": idx,
            "report": summary,
            "summary": summary,
            "objective": slot.objective,
            "model": slot.model,
            "duration_ms": slot.duration_ms,
        }
        compact = {
            **compact,
            "status": "accepted",
            "report": compact.get("report") or summary,
            "summary": compact.get("summary") or summary,
        }
        slot.result = compact
        slot.status = "accepted"

        # Hosts attach this return value to the coordinator thread (tool result
        # or manual append). OA does not invent a second copy here.
        return {
            "status": "accepted",
            "agent": agent,
            "index": idx,
            "summary": summary,
            "result": compact,
        }

    # aliases matching tool names
    async def instruct_agent(
        self,
        agent: str,
        index: Any,
        message: str,
        *,
        streaming_callback: StreamingCallback = None,
    ) -> dict[str, Any]:
        return await self.instruct(
            agent, index, message, streaming_callback=streaming_callback
        )

    # ── coordinator turn ──────────────────────────────────────────────

    async def run(
        self,
        user_text: str,
        *,
        streaming_callback: StreamingCallback = None,
        orchestration_tools: bool = True,
    ) -> str:
        """Append user text and invoke the coordinator runner (if configured).

        OA does not execute a tool loop. The host ``Coordinator.runner`` owns
        inference/tool-calling and may call back into orchestration tools.
        """
        text = (user_text or "").strip()
        if not text:
            return ""
        if self.coordinator.runner is None:
            raise RuntimeError(
                "Coordinator.runner is not set. "
                "Provide a runner for sm.run(), or call sm.commission/accept directly."
            )

        self.coordinator_messages.user(text)
        self.active_agent = "coordinator"
        cb = streaming_callback or self._streaming_callback
        self._streaming_callback = cb

        await emit_stream(
            cb,
            {
                "agent": "coordinator",
                "phase": "start",
                "detail": "starting",
                "model": self.coordinator.model or "",
            },
        )

        tools = self.orchestration_tools() if orchestration_tools else []
        request = AgentRequest(
            agent="coordinator",
            objective=text,
            skill=self.coordinator.skill,
            messages=self.coordinator_messages.to_list(),
            model=self.coordinator.model,
            metadata={
                "tools": tools,
                "tool_schemas": tool_schemas(tools),
                **self.coordinator.metadata,
            },
        )

        try:
            result = await self.coordinator.runner.run(
                request, streaming_callback=cb
            )
            if result.messages is not None:
                self.coordinator_messages.replace(result.messages)
            elif result.content:
                self.coordinator_messages.assistant(result.content)
            reply = (result.content or "").strip() or "(no response)"
            await emit_stream(
                cb,
                {
                    "agent": "coordinator",
                    "phase": "done",
                    "detail": reply[:500],
                    "model": self.coordinator.model or "",
                    "status": "done",
                },
            )
            return reply
        except Exception as exc:  # noqa: BLE001
            err = f"Coordinator error: {type(exc).__name__}: {exc}"
            await emit_stream(
                cb,
                {
                    "agent": "coordinator",
                    "phase": "fail",
                    "detail": err,
                    "status": "failed",
                },
            )
            raise
        finally:
            self.active_agent = None
            if streaming_callback is not None:
                # only clear if we temporarily owned the callback for this call
                pass

    def orchestration_tools(self) -> list:
        """Return plain callables + OpenAI schemas for host coordinator runners."""
        sm = self

        async def list_agents() -> list[dict[str, Any]]:
            """List registered sub-agents with descriptions."""
            return sm.list_agents()

        async def commission(
            agent: str,
            objective: str,
            checklist: list[str] | None = None,
            agent_props: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            """Commission a sub-agent. Result is staged until accept_agent_result."""
            return await sm.commission(
                agent, objective, checklist=checklist, agent_props=agent_props
            )

        async def get_agent_message(agent: str, index: int) -> dict[str, Any]:
            """Peek a staged sub-agent message without accepting it."""
            return sm.get_agent_message(agent, index)

        async def accept_agent_result(agent: str, index: int) -> dict[str, Any]:
            """Accept a staged result into the coordinator core thread."""
            return sm.accept_agent_result(agent, index)

        async def instruct_agent(
            agent: str, index: int, message: str
        ) -> dict[str, Any]:
            """Send further instruction to a sub-agent objective and re-run it."""
            return await sm.instruct(agent, index, message)

        async def list_objectives(agent: str | None = None) -> list[dict[str, Any]]:
            """List objective slots and their statuses."""
            return sm.list_objectives(agent)

        tools = [
            list_agents,
            commission,
            get_agent_message,
            accept_agent_result,
            instruct_agent,
            list_objectives,
        ]
        for fn in tools:
            attach_schema(fn, openai_tool_schema(fn))
        return tools

    # ── internals ─────────────────────────────────────────────────────

    def _build_brief(
        self,
        objective: str,
        checklist: list[str] | None,
        agent_props: dict[str, Any] | None,
    ) -> str:
        parts = [f"Objective:\n{objective.strip()}"]
        if checklist:
            parts.append(
                "Checklist:\n"
                + "\n".join(f"- {item}" for item in checklist if str(item).strip())
            )
        if agent_props:
            props_lines = [f"- {k}: {v}" for k, v in agent_props.items()]
            parts.append("Agent props:\n" + "\n".join(props_lines))
        return "\n\n".join(parts)

    async def _run_slot(
        self,
        handle: AgentHandle,
        slot: ObjectiveSlot,
        *,
        streaming_callback: StreamingCallback = None,
    ) -> dict[str, Any]:
        await emit_stream(
            streaming_callback,
            {
                "agent": handle.name,
                "phase": "start",
                "detail": "starting",
                "model": slot.model or "",
                "commission_id": slot.commission_id,
                "index": slot.index,
                "objective": slot.objective,
            },
        )

        started = time.perf_counter()
        request = AgentRequest(
            agent=handle.name,
            objective=slot.objective,
            skill=handle.spec.skill,
            messages=list(slot.messages),
            checklist=list(slot.checklist) or None,
            agent_props=dict(slot.agent_props) or None,
            model=slot.model,
            metadata=dict(handle.spec.metadata),
        )

        try:
            result: AgentResult = await handle.runner.run(
                request, streaming_callback=streaming_callback
            )
            duration_ms = (time.perf_counter() - started) * 1000
            report = (result.content or "").strip() or "(empty report)"
            if result.messages is not None:
                slot.messages = list(result.messages)
            else:
                slot.messages.append({"role": "assistant", "content": report})

            slot.agent_message = report
            slot.duration_ms = duration_ms
            slot.status = "staged"
            slot.result = {
                "status": "staged",
                "agent": handle.name,
                "index": slot.index,
                "report": report,
                "summary": report,
                "duration_ms": duration_ms,
                "model": slot.model,
                "objective": slot.objective,
            }
            await emit_stream(
                streaming_callback,
                {
                    "agent": handle.name,
                    "phase": "done",
                    "detail": report[:500],
                    "model": slot.model or "",
                    "commission_id": slot.commission_id,
                    "status": "staged",
                    "duration_ms": duration_ms,
                    "objective": slot.objective,
                    "index": slot.index,
                },
            )
            self.active_agent = None
            return {
                "status": "staged",
                "agent": handle.name,
                "index": slot.index,
                "commission_id": slot.commission_id,
                "summary": report,
                "hint": (
                    f"Junior reply staged at sm.agent('{handle.name}')[{slot.index}].agent_message. "
                    f"Call accept_agent_result('{handle.name}', {slot.index}) to attach "
                    f"the compact result to the core thread, or instruct_agent(...) to continue."
                ),
                "preview": report[:400],
                "duration_ms": duration_ms,
                "model": slot.model,
            }
        except Exception as exc:  # noqa: BLE001
            duration_ms = (time.perf_counter() - started) * 1000
            slot.status = "failed"
            slot.duration_ms = duration_ms
            slot.agent_message = f"{type(exc).__name__}: {exc}"
            slot.result = {
                "status": "failed",
                "agent": handle.name,
                "index": slot.index,
                "report": slot.agent_message,
                "summary": slot.agent_message,
                "duration_ms": duration_ms,
                "model": slot.model,
                "objective": slot.objective,
            }
            await emit_stream(
                streaming_callback,
                {
                    "agent": handle.name,
                    "phase": "fail",
                    "detail": slot.agent_message,
                    "model": slot.model or "",
                    "commission_id": slot.commission_id,
                    "status": "failed",
                    "duration_ms": duration_ms,
                    "index": slot.index,
                },
            )
            self.active_agent = None
            return {
                "status": "failed",
                "agent": handle.name,
                "index": slot.index,
                "report": slot.agent_message,
                "summary": slot.agent_message,
                "duration_ms": duration_ms,
                "model": slot.model,
            }
