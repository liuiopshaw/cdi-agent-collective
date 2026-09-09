"""A5 Blind Probe: multi-model blind rediscovery (optional module).

Each participating model receives ONLY the observation list and the
falsifiable-trait table of the hypothesis library - never the group's
conclusion, never any directional hint. Each model independently ranks
the most plausible mechanism. The result is a consensus report.

Governance (pre-registered): if the models do not converge on the
claimed mechanism, the module output is archived but NOT cited in the
manuscript. This pre-commitment avoids publication-bias criticism.
"""

from __future__ import annotations

import json

from ..state import Message, SharedState
from .base import BaseAgent

BLIND_SYSTEM = (
    "You are an independent electrochemistry expert. You are given a set "
    "of experimental observations from an unpublished study and a list "
    "of candidate mechanisms with their falsifiable traits. You have no "
    "knowledge of the authors' conclusion and must not ask for it. Rank "
    "the candidate mechanisms from most to least plausible and justify "
    "each ranking step strictly from the observations. Reply JSON: "
    "{\"ranking\": [\"H-...\", ...], \"top_choice\": \"H-...\", "
    "\"reasoning\": \"...\"}."
)


class BlindProbeAgent(BaseAgent):
    role = "blind_probe"
    system_prompt = BLIND_SYSTEM

    def probe(self, state: SharedState, claimed_hid: str | None) -> dict:
        library = [
            {"hid": h.hid, "statement": h.statement,
             "falsifiable_traits": h.falsifiable_traits}
            for h in state.hypotheses.values()]
        observations = [
            {"obs_id": oid, "description": o["description"]}
            for oid, o in state.observations.items()]
        raw = self.call_llm(
            "Candidate mechanisms:\n" + json.dumps(library, indent=1)
            + "\n\nObservations:\n" + json.dumps(observations, indent=1)
            + "\n\nRank the mechanisms. JSON only.")
        try:
            result = self.extract_json(raw)
        except ValueError:
            result = {"ranking": [], "top_choice": None,
                      "reasoning": "unparseable"}
        result["matches_claim"] = (
            claimed_hid is not None
            and result.get("top_choice") == claimed_hid)
        return result

    def handle(self, msg: Message, state: SharedState) -> Message:
        claimed = msg.payload.get("claimed_hid")
        result = self.probe(state, claimed)
        return Message(sender=self.name, content=json.dumps(result),
                       payload=result)


def consensus_report(results: dict[str, dict],
                     claimed_hid: str | None) -> dict:
    """Aggregate blind probes from several backbones."""
    votes: dict[str, int] = {}
    for model, res in results.items():
        top = res.get("top_choice")
        if top:
            votes[top] = votes.get(top, 0) + 1
    n = len(results)
    converged_on_claim = (
        claimed_hid is not None
        and votes.get(claimed_hid, 0) == n and n > 0)
    return {"votes": votes, "n_models": n,
            "converged_on_claim": converged_on_claim,
            "pre_registered_rule": "archive-only if not converged"}
