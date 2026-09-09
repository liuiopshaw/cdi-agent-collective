"""Base agent class: system prompt + LLM backend + memory + handler.

Design notes borrowed from published frameworks:
- each agent is self-contained and independently replaceable (Guo et al.);
- per-role temperature keeps deterministic roles cold and creative roles
  warm (PeroMAS);
- every reply is JSON-first: agents are instructed to answer with a
  machine-readable object so downstream code never parses prose.
"""

from __future__ import annotations

import json
import re
from typing import Any

from ..llm import ChatLLM, MockLLM
from ..state import Message, SharedState


class BaseAgent:
    role: str = "base"
    system_prompt: str = "You are a careful scientific assistant."

    def __init__(self, llm: ChatLLM | MockLLM, name: str | None = None):
        self.llm = llm
        self.name = name or self.role
        self.memory: list[Message] = []

    def call_llm(self, user_content: str,
                 extra_messages: list[dict[str, str]] | None = None) -> str:
        messages = list(extra_messages or [])
        messages.append({"role": "user", "content": user_content})
        return self.llm.complete(self.system_prompt, messages).content

    def call_with_tools(self, user_content: str, registry,
                        max_steps: int = 6,
                        extra_messages: list[dict[str, str]] | None = None
                        ) -> str:
        """Agentic tool loop (OpenAI function calling).

        The LLM may request tool calls; each request is executed
        deterministically by the registry and the result is fed back as a
        tool message. The loop ends when the LLM answers without tool
        calls. Every dispatch is recorded in registry.call_log (T3).
        """
        messages = list(extra_messages or [])
        messages.append({"role": "user", "content": user_content})
        for _ in range(max_steps):
            resp = self.llm.complete(self.system_prompt, messages,
                                     tools=registry.schemas())
            if not resp.tool_calls:
                return resp.content
            messages.append({"role": "assistant",
                             "content": resp.content or ""})
            for tc in resp.tool_calls:
                result = registry.dispatch(tc["name"],
                                           tc.get("arguments"))
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "content": json.dumps(result, ensure_ascii=False,
                                          default=str)})
        raise RuntimeError(
            f"tool loop exceeded max_steps={max_steps}")

    def handle(self, msg: Message, state: SharedState) -> Message:
        reply = self.call_llm(msg.content)
        return Message(sender=self.name, content=reply,
                       payload=msg.payload)

    @staticmethod
    def extract_json(text: str) -> Any:
        """Pull the first JSON object/array out of an LLM reply."""
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        candidates = fenced if fenced else [text]
        for cand in candidates:
            cand = cand.strip()
            for opener, closer in (("{", "}"), ("[", "]")):
                start = cand.find(opener)
                end = cand.rfind(closer)
                if start != -1 and end > start:
                    try:
                        return json.loads(cand[start:end + 1])
                    except json.JSONDecodeError:
                        continue
        raise ValueError(f"no JSON found in reply: {text[:200]}")
