"""A3 Mechanism Agent: hypothesis library and falsification matrix.

Builds the library of all reported ROS/RCS generation mechanisms with
their falsifiable traits, then judges every mechanism against every
experimental observation. Every 'fail' verdict must cite the observation
id and a reason (hard rule T2). Verdicts are later cross-checked by the
Critic Agent running a different-vendor backbone.
"""

from __future__ import annotations

import json

from ..state import HypothesisEntry, Message, SharedState
from .base import BaseAgent

MECHANISM_SYSTEM = (
    "You are the Mechanism Agent for electrochemical water treatment. "
    "You maintain a library of all reported reactive oxygen / chlorine "
    "species (ROS/RCS) generation mechanisms at electrode interfaces, "
    "each with falsifiable experimental traits (expected EPR signature, "
    "characteristic scavengers and expected suppression, probe-molecule "
    "reactions, kinetic isotope effects, pH and potential dependence). "
    "When judging a mechanism against an observation, you reply only "
    "with JSON: {\"verdict\": \"pass\"|\"fail\"|\"na\", \"reason\": "
    "\"...\"}. A fail verdict must state which expected trait is "
    "contradicted. You never soften a contradiction."
)

DEFAULT_LIBRARY: list[dict] = [
    {"hid": "H-OH", "statement": "Hydroxyl radical (free OH) dominated oxidation",
     "falsifiable_traits": [
         "DMPO-EPR 1:2:2:1 quartet",
         "strong suppression by tert-butanol",
         "pH-dependent activity"]},
    {"hid": "H-O2", "statement": "Superoxide radical (O2-) pathway",
     "falsifiable_traits": [
         "suppression by p-benzoquinone",
         "DMPO-OOH signal",
         "oxygen dependence"]},
    {"hid": "H-1O2", "statement": "Singlet oxygen (1O2) non-radical pathway",
     "falsifiable_traits": [
         "TEMP-EPR 1:1:1 triplet",
         "suppression by L-histidine / NaN3",
         "D2O lifetime enhancement"]},
    {"hid": "H-FeIV", "statement": "High-valent iron-oxo (FeIV=O / FeV=O) pathway",
     "falsifiable_traits": [
         "methyl phenyl sulfoxide to sulfone selective oxidation",
         "no suppression by tert-butanol",
         "18O isotope labeling evidence"]},
    {"hid": "H-ClR", "statement": "Reactive chlorine radicals (Cl, Cl2-, ClO)",
     "falsifiable_traits": [
         "chloride dependence of activity",
         "chlorinated byproduct formation",
         "suppression by chloride removal"]},
    {"hid": "H-CER", "statement": "Chlorine evolution reaction (free Cl2/HOCl/OCl-)",
     "falsifiable_traits": [
         "free chlorine detected by DPD assay",
         "potential above CER equilibrium",
         "DSA-like selectivity pattern"]},
    {"hid": "H-DET", "statement": "Direct electron transfer at the anode surface",
     "falsifiable_traits": [
         "activity requires electrode contact",
         "no radical probe signals",
         "current-density correlation"]},
]


class MechanismAgent(BaseAgent):
    role = "mechanism"
    system_prompt = MECHANISM_SYSTEM

    # ------------------------------------------------------------------
    def build_library(self, state: SharedState,
                      extra: list[dict] | None = None) -> list[str]:
        entries = list(DEFAULT_LIBRARY) + list(extra or [])
        for e in entries:
            state.hypotheses[e["hid"]] = HypothesisEntry(
                hid=e["hid"], statement=e["statement"],
                falsifiable_traits=e.get("falsifiable_traits", []))
        return [e["hid"] for e in entries]

    # ------------------------------------------------------------------
    def judge(self, hid: str, obs_id: str, state: SharedState) -> dict:
        hyp = state.hypotheses[hid]
        obs = state.observations[obs_id]
        raw = self.call_llm(
            "Mechanism id: " + hid
            + "\nMechanism: " + hyp.statement
            + "\nFalsifiable traits: " + json.dumps(hyp.falsifiable_traits)
            + "\nObservation (" + obs_id + "): " + obs["description"]
            + "\nJudge consistency. JSON only.")
        try:
            verdict = self.extract_json(raw)
        except ValueError:
            verdict = {"verdict": "na", "reason": "unparseable reply"}
        hyp.verdicts[obs_id] = verdict.get("verdict", "na")
        hyp.rationale[obs_id] = verdict.get("reason", "")
        hyp.status = "eliminated" if hyp.n_fail > 0 else hyp.status
        return verdict

    # ------------------------------------------------------------------
    def run_matrix(self, state: SharedState) -> dict[str, dict[str, str]]:
        matrix: dict[str, dict[str, str]] = {}
        for hid in state.hypotheses:
            matrix[hid] = {}
            for obs_id in state.observations:
                self.judge(hid, obs_id, state)
                matrix[hid][obs_id] = state.hypotheses[hid].verdicts[obs_id]
        for hid, hyp in state.hypotheses.items():
            if hyp.n_fail == 0 and len(hyp.verdicts) == len(state.observations):
                hyp.status = "survivor"
        return matrix

    # ------------------------------------------------------------------
    def handle(self, msg: Message, state: SharedState) -> Message:
        raw = self.call_llm(msg.content)
        return Message(sender=self.name, content=raw)
