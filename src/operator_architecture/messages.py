"""OpenAI chat-completions compatible message helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


Message = dict[str, Any]


class Messages:
    """Mutable list of OpenAI-shaped chat messages.

    Compatible with chat.completions payloads: roles ``system``, ``user``,
    ``assistant``, ``tool`` plus optional ``tool_calls`` / ``tool_call_id``.
    """

    def __init__(self, messages: list[Message] | None = None) -> None:
        self._messages: list[Message] = list(messages or [])

    def __iter__(self):
        return iter(self._messages)

    def __len__(self) -> int:
        return len(self._messages)

    def __getitem__(self, index: int) -> Message:
        return self._messages[index]

    def to_list(self) -> list[Message]:
        return deepcopy(self._messages)

    def clear(self) -> None:
        self._messages.clear()

    def append(self, message: Message) -> Messages:
        self._messages.append(dict(message))
        return self

    def system(self, content: str) -> Messages:
        return self.append({"role": "system", "content": content})

    def user(self, content: str) -> Messages:
        return self.append({"role": "user", "content": content})

    def assistant(self, content: str, *, tool_calls: list[dict] | None = None) -> Messages:
        msg: Message = {"role": "assistant", "content": content}
        if tool_calls is not None:
            msg["tool_calls"] = tool_calls
        return self.append(msg)

    def tool(self, content: str, *, tool_call_id: str) -> Messages:
        return self.append(
            {"role": "tool", "content": content, "tool_call_id": tool_call_id}
        )

    def extend(self, messages: list[Message]) -> Messages:
        for m in messages:
            self.append(m)
        return self

    def replace(self, messages: list[Message]) -> Messages:
        self._messages = [dict(m) for m in messages]
        return self
