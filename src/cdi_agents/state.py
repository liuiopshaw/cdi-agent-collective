"""Shared session state: the single source of truth for all agents.

Mirrors the design of Guo et al. (Science 2026, Supplementary Text):
one mutable state object holding the descriptor dataset, the device /
hypothesis plan, the ordered message history, channel logs and the ML
model cache. Everything an agent learns is written here; everything an
agent reads comes from here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class Message:
    sender: str
    content: str
    payload: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)


@dataclass
class HypothesisEntry:
    """One row of the hypothesis-evidence table (pipeline P3)."""

    hid: str
    statement: str
    falsifiable_traits: list[str] = field(default_factory=list)
    verdicts: dict[str, str] = field(default_factory=dict)  # obs_id -> pass/fail/na
    rationale: dict[str, str] = field(default_factory=dict)  # obs_id -> why
    evidence_quotes: list[dict[str, str]] = field(default_factory=list)
    status: str = "candidate"  # candidate | eliminated | survivor

    @property
    def n_fail(self) -> int:
        return sum(1 for v in self.verdicts.values() if v == "fail")


@dataclass
class SharedState:
    dataset_version: str = "unloaded"
    n_records: int = 0
    hypotheses: dict[str, HypothesisEntry] = field(default_factory=dict)
    observations: dict[str, dict[str, Any]] = field(default_factory=dict)
    residual_report: dict[str, Any] = field(default_factory=dict)
    model_cache: dict[str, Any] = field(default_factory=dict)
    history: list[Message] = field(default_factory=list)
    channel_logs: dict[str, list[dict[str, Any]]] = field(
        default_factory=lambda: {"data": [], "pred": [], "mech": [],
                                 "critic": [], "probe": []})
    confirmations: list[dict[str, Any]] = field(default_factory=list)

    def log_message(self, msg: Message) -> None:
        self.history.append(msg)

    def log_exchange(self, channel: str, query: str, reply: str) -> None:
        self.channel_logs.setdefault(channel, []).append(
            {"query": query, "reply": reply, "ts": time.time()})

    def require_confirmation(self, topic: str,
                             confirm_fn=None) -> bool:
        """Human confirmation point. ``confirm_fn`` defaults to always-True
        for headless runs; the CLI passes an interactive prompt."""
        if confirm_fn is None:
            approved = True
        else:
            approved = bool(confirm_fn(topic))
        self.confirmations.append(
            {"topic": topic, "approved": approved, "ts": time.time()})
        return approved

    def save(self, path: str | Path) -> None:
        def default(obj):
            if isinstance(obj, (Message, HypothesisEntry)):
                return asdict(obj)
            return str(obj)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps(asdict(self), default=default,
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
