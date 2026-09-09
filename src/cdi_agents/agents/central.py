"""A0 Central Agent: orchestration, planning and synthesis.

Responsibilities (Guo et al. two-round protocol):
- interpret the user objective and decompose it into pipeline tasks;
- consult downstream agents with at least ``min_rounds`` question-answer
  exchanges before synthesizing a final recommendation;
- the synthesis call is explicitly instructed NOT to copy subordinate
  replies verbatim;
- hard human confirmation points before any frozen artifact changes.
"""

from __future__ import annotations

from ..bus import AgentLike, MessageBus
from ..state import Message, SharedState
from .base import BaseAgent

CENTRAL_SYSTEM = (
    "You are the Central Agent of a materials-research multi-agent "
    "framework for capacitive deionization (CDI) electrodes of the "
    "support x active-center family. You orchestrate specialist agents "
    "(data, prediction, mechanism, critic), maintain the shared research "
    "state, and synthesize their outputs. Rules: (1) never copy a "
    "subordinate reply verbatim - re-derive the synthesis; (2) separate "
    "clearly what is known physics, what is model prediction, and what is "
    "hypothesis; (3) every quantitative claim must carry its evidence "
    "pointer; (4) when delegating, state the exact expected JSON output "
    "schema; (5) keep replies under 300 words."
)


class CentralAgent(BaseAgent):
    role = "central"
    system_prompt = CENTRAL_SYSTEM

    def __init__(self, llm, buses: dict[str, MessageBus],
                 agents: dict[str, AgentLike], min_rounds: int = 2):
        super().__init__(llm, name="central")
        self.buses = buses
        self.agents = agents
        self.min_rounds = min_rounds

    def consult(self, channel: str, question: str,
                state: SharedState) -> list[Message]:
        """Two-round consultation with one downstream agent."""
        bus = self.buses[channel]
        agent = self.agents[channel]
        replies = []
        current = question
        for round_idx in range(self.min_rounds):
            reply = bus.ask(agent, current)
            replies.append(reply)
            if round_idx < self.min_rounds - 1:
                current = (
                    "Round 2 follow-up. Critically review your previous "
                    "reply: check numeric consistency, evidence pointers "
                    "and missing caveats. Previous reply: "
                    + reply.content)
        return replies

    def synthesize(self, question: str,
                   consultations: dict[str, list[Message]]) -> str:
        digest = "\n\n".join(
            f"[{ch} final reply]\n{msgs[-1].content}"
            for ch, msgs in consultations.items() if msgs)
        return self.call_llm(
            "User objective: " + question
            + "\n\nSubordinate consultations:\n" + digest
            + "\n\nSynthesize a final recommendation. Do not copy the "
              "replies verbatim; state agreements, conflicts and the "
              "evidence pointer for each claim.")
