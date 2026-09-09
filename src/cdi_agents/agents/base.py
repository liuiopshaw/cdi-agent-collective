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
