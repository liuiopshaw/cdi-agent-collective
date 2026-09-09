"""A4 Critic Agent: cross-vendor verification and adversarial review.

Two jobs:
1. Re-judge every 'fail' verdict in the elimination matrix with a
   different-vendor backbone; disagreements are flagged for human
   arbitration (anti-sycophancy, anti common-mode bias).
2. Adversarial challenge: enumerate the most boring alternative
   explanations for the claimed novel mechanism (leaching, apparent CER
   artifact, scavenger adsorption on the support, thermal artifacts...)
   and demand a data-based rebuttal or a follow-up experiment for each.
"""

from __future__ import annotations

import json

from ..state import Message, SharedState
from .base import BaseAgent

CRITIC_SYSTEM = (
    "You are the Critic Agent, the harshest reviewer of this research "
    "group. You distrust every conclusion, especially ones your "
    "colleagues like. When re-judging a mechanism-observation pair you "
    "reply JSON: {\"agree\": true|false, \"verdict\": \"pass\"|\"fail\"|"
    "\"na\", \"reason\": \"...\"}. When generating adversarial "
    "challenges you prefer boring explanations (metal leaching, "
    "measurement artifacts, scavenger side-reactions, ohmic heating) "
    "over glamorous ones, and you demand the exact dataset that would "
    "refute each alternative. Never agree to be polite."
)

DEFAULT_CHALLENGES: list[str] = [
    "Performance gain comes from homogeneous Fenton chemistry of leached Fe ions, not interfacial catalysis.",
    "RCS signals are apparent artifacts of the chlorine evolution reaction on a non-selective anode.",
    "Scavengers are adsorbed by the support, so quenching results are unreliable.",
    "Probe molecules react directly with the support surface groups.",
    "Observed enhancement is due to ohmic heating at high current density.",
    "Disinfection improvement is caused by direct electron transfer contact killing, not diffusible species.",
]


class CriticAgent(BaseAgent):
    role = "critic"
    system_prompt = CRITIC_SYSTEM

    # ------------------------------------------------------------------
    def cross_check(self, state: SharedState) -> list[dict]:
        """Re-judge every recorded verdict; return disagreement list."""
        disagreements = []
        for hid, hyp in state.hypotheses.items():
            for obs_id, verdict in hyp.verdicts.items():
                obs = state.observations[obs_id]
                raw = self.call_llm(
                    "Mechanism: " + hyp.statement
                    + "\nFalsifiable traits: "
                    + json.dumps(hyp.falsifiable_traits)
                    + "\nObservation (" + obs_id + "): " + obs["description"]
                    + "\nA colleague judged this pair as \"" + verdict
                    + "\" with reason: " + hyp.rationale.get(obs_id, "")
                    + "\nIndependently re-judge. JSON only.")
                try:
                    re = self.extract_json(raw)
                except ValueError:
                    re = {"agree": True, "verdict": verdict,
                          "reason": "unparseable; kept"}
                if not re.get("agree", True) or re.get("verdict") != verdict:
                    disagreements.append({
                        "hid": hid, "obs_id": obs_id,
                        "first": verdict, "second": re.get("verdict"),
                        "second_reason": re.get("reason", "")})
        return disagreements

    # ------------------------------------------------------------------
    def adversarial_review(self, claim: str,
                           available_data: list[str]) -> list[dict]:
        raw = self.call_llm(
            "Claimed novel mechanism: " + claim
            + "\nAvailable datasets: " + json.dumps(available_data)
            + "\nKnown boring alternatives to consider: "
            + json.dumps(DEFAULT_CHALLENGES)
            + "\nFor each applicable alternative, reply with a JSON list "
              "of objects: {\"challenge\": \"...\", \"rebuttal_data\": "
              "\"<dataset name or EMPTY>\", \"follow_up\": \"...\"}")
        try:
            items = self.extract_json(raw)
            if isinstance(items, dict):
                items = [items]
        except ValueError:
            items = [{"challenge": c, "rebuttal_data": "EMPTY",
                      "follow_up": "manual review required"}
                     for c in DEFAULT_CHALLENGES[:3]]
        return items

    # ------------------------------------------------------------------
    def handle(self, msg: Message, state: SharedState) -> Message:
        raw = self.call_llm(msg.content)
        return Message(sender=self.name, content=raw)
