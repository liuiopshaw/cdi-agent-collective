"""Minimal OpenAI-compatible LLM client.

Implemented on plain requests so the framework stays dependency-light and
provider-agnostic (OpenAI, DeepSeek, Qwen, local vLLM, ...). A MockLLM is
provided for offline demos and tests; it returns deterministic scripted
replies and records every call for auditability.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import requests

from .config import ProviderConfig


@dataclass
class LLMResponse:
    content: str
    model: str
    usage: dict[str, Any] = field(default_factory=dict)
    latency_s: float = 0.0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


def _normalize_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize OpenAI-format tool calls into {id, name, arguments}."""
    normalized = []
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments", "")}
        normalized.append({"id": tc.get("id", f"call_{len(normalized)}"),
                           "name": fn.get("name", ""),
                           "arguments": args})
    return normalized


class ChatLLM:
    """Blocking chat client for one provider."""

    def __init__(self, provider: ProviderConfig, timeout_s: int = 120,
                 max_retries: int = 4):
        self.provider = provider
        self.timeout_s = timeout_s
        self.max_retries = max_retries

    def complete(self, system: str,
                 messages: list[dict[str, str]],
                 tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        api_key = self.provider.api_key()
        if not api_key:
            raise RuntimeError(
                f"missing API key: set env {self.provider.api_key_env}")
        url = self.provider.base_url.rstrip("/") + "/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}",
                   "Content-Type": "application/json"}
        payload_messages = ([{"role": "system", "content": system}]
                            if system else []) + messages
        body: dict[str, Any] = {"model": self.provider.model,
                                "messages": payload_messages}
        if tools:
            body["tools"] = tools
        backoff = 2.0
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            started = time.time()
            try:
                resp = requests.post(url, headers=headers, json=body,
                                     timeout=self.timeout_s)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise RuntimeError(f"transient HTTP {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                message = data["choices"][0]["message"]
                return LLMResponse(
                    content=message.get("content") or "",
                    model=data.get("model", self.provider.model),
                    usage=data.get("usage", {}),
                    latency_s=time.time() - started,
                    tool_calls=_normalize_tool_calls(message),
                )
            except Exception as err:  # noqa: BLE001 - retried below
                last_err = err
                if attempt < self.max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
        raise RuntimeError(f"LLM call failed after retries: {last_err}")


class MockLLM:
    """Deterministic offline stand-in used by demos and tests.

    ``script`` maps a substring of the latest user message to a reply. A
    reply may be a plain string or a dict of the form
    ``{"content": str, "tool_calls": [{"id", "name", "arguments"}]}`` to
    emulate function calling. The first matching key wins; otherwise
    ``default`` is returned. Every call is appended to ``call_log``.
    """

    def __init__(self, script: dict[str, Any] | None = None,
                 default: Any = "MOCK_REPLY",
                 on_call: Callable[[str, list[dict[str, str]]], None] | None = None):
        self.script = script or {}
        self.default = default
        self.on_call = on_call
        self.call_log: list[dict[str, Any]] = []
        self.provider = type("P", (), {"model": "mock-model"})()

    def complete(self, system: str,
                 messages: list[dict[str, str]],
                 tools: list[dict[str, Any]] | None = None) -> LLMResponse:
        # match against the last message; tool-result messages carry the
        # tool output JSON, which naturally misses the trigger keys and
        # falls through to ``default`` (the final answer), terminating
        # the tool loop in tests and demos.
        latest = messages[-1].get("content") or "" if messages else ""
        reply: Any = self.default
        for key, value in self.script.items():
            if key in latest:
                reply = value
                break
        content, tool_calls = reply, []
        if isinstance(reply, dict):
            content = reply.get("content", "")
            tool_calls = reply.get("tool_calls", [])
        self.call_log.append({"system": system, "messages": messages,
                              "reply": content, "tool_calls": tool_calls})
        if self.on_call:
            self.on_call(system, messages)
        return LLMResponse(content=content, model="mock-model",
                           tool_calls=tool_calls)


def dump_messages(messages: list[dict[str, str]]) -> str:
    return json.dumps(messages, ensure_ascii=False, indent=2)
