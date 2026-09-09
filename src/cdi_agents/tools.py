"""Lightweight tool layer: registry, schemas and audited dispatch.

Design notes:
- Tools are deterministic Python callables with a JSON Schema. The LLM
  may REQUEST a tool call (function calling), but execution always
  happens in code and every dispatch is logged (hard rule T3).
- The layer is additive: pipelines that never use tools are unaffected.
- A tool failure is returned to the LLM as an error payload instead of
  raising, so the reasoning loop can recover gracefully.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]          # JSON Schema object
    fn: Callable[..., Any]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    ok: bool
    result_preview: str
    ts: float = field(default_factory=time.time)


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self.call_log: list[ToolCallRecord] = []

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        if name not in self._tools:
            raise KeyError(f"unknown tool: {name}")
        return self._tools[name]

    def schemas(self) -> list[dict[str, Any]]:
        return [t.openai_schema() for t in self._tools.values()]

    def dispatch(self, name: str, arguments: dict[str, Any] | None) -> Any:
        arguments = arguments or {}
        try:
            tool = self.get(name)
            result = tool.fn(**arguments)
            payload = {"ok": True, "result": result}
        except Exception as err:  # noqa: BLE001 - reported to the LLM
            payload = {"ok": False,
                       "error": f"{type(err).__name__}: {err}"}
        self.call_log.append(ToolCallRecord(
            name=name, arguments=arguments, ok=payload["ok"],
            result_preview=json.dumps(payload, ensure_ascii=False,
                                      default=str)[:500]))
        return payload
