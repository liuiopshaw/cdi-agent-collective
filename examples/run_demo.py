"""Offline end-to-end demo of the CDI Agent Collective.

Runs the full loop without any API key:
- P2 anchored prediction on a synthetic literature dataset, where the
  held-out "own system" is planted as a true outlier;
- P3 mechanism falsification matrix with a scripted mock LLM, followed
  by critic cross-check and adversarial challenge;
- A5 blind probe consensus across three mock backbones.

Usage:
    python examples/run_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdi_agents.agents.blind_probe import BlindProbeAgent, consensus_report
from cdi_agents.agents.critic import CriticAgent
from cdi_agents.agents.mechanism import MechanismAgent
from cdi_agents.agents.prediction import PredictionAgent
from cdi_agents.llm import MockLLM
from cdi_agents.pipelines.p2_anchored import AnchoredPredictionPipeline
from cdi_agents.pipelines.p3_mechanism import MechanismPipeline
from cdi_agents.state import SharedState

RNG = np.random.default_rng(7)


def synthetic_literature(n: int = 120) -> pd.DataFrame:
    """Known physics: performance rises with support conductivity and
    active-center loading, saturates, plus noise."""
    conductivity = RNG.uniform(0.2, 1.0, n)
    loading = RNG.uniform(0.5, 6.0, n)
    size = RNG.uniform(0.5, 5.0, n)
    performance = (40 * conductivity + 8 * np.log1p(loading)
                   - 1.5 * size + RNG.normal(0, 2.0, n))
    family = np.where(conductivity > 0.6, "MXene", "carbon")
    return pd.DataFrame({
        "sp_conductivity": conductivity,
        "ac_loading_wt": loading,
        "ac_size_nm": size,
        "support_family": family,
        "rs_performance": performance})


def main() -> None:
    print("=" * 64)
    print("CDI Agent Collective - offline demo")
    print("=" * 64)

    # ---------------- P2 anchored prediction ----------------------
    df = synthetic_literature()
    features = ["sp_conductivity", "ac_loading_wt", "ac_size_nm"]
    pred_agent = PredictionAgent(MockLLM())
    state = SharedState()
    p2 = AnchoredPredictionPipeline(state, pred_agent, r2_threshold=0.6)

    # planted own system: strong conductor + moderate loading, but its
    # measured performance is far above known physics (new mechanism).
    anchor_X = np.array([[0.95, 2.3, 0.8]])
    anchor_y = 40 * 0.95 + 8 * np.log1p(2.3) - 1.5 * 0.8 + 18.0

    result = p2.run(df, features, "rs_performance", anchor_X, anchor_y,
                    anchor_label="Fe-cluster@MXene",
                    group_col="support_family")
    cal = result["calibration"]
    print("\n[P2] calibration:")
    print("  best model:", cal["best_model"],
          "| n =", cal["n_samples"])
    for name, m in cal["benchmark"].items():
        print(f"    {name:8s} r2_loo={m['r2_loo']:+.3f} "
              f"mae={m['mae_loo']:.2f} gap={m['overfit_gap']:+.3f}")
    if cal["family_holdout"]:
        print("  family holdout r2:",
              cal["family_holdout"]["r2_family_holdout"])
    blind = result["anchor_blind_test"]
    print("[P2] anchor blind test: mu={:.2f} sigma={:.2f} y={:.2f} "
          "inside95={}".format(blind["mu"], blind["sigma"],
                               blind["y_observed"],
                               blind["inside_interval"]))
    cert = result["outlier_certificate"]
    print("[P2] outlier certificate: Z={:.2f} p={:.4f} certified={}"
          .format(cert["z_score_point"], cert["p_value_point"],
                  cert["certified_outlier"]))
    print("     attributed new-mechanism delta: {:.2f}"
          .format(cert["attributed_new_mechanism_contribution"]))

    # ---------------- P3 mechanism certification ------------------
    state.observations["OBS-1"] = {
        "description": "EPR shows a TEMP 1:1:1 triplet and no DMPO-OH "
                       "quartet under operating potential",
        "source": "own lab, unpublished"}
    state.observations["OBS-2"] = {
        "description": "tert-butanol up to 100 mM does not suppress "
                       "activity, while L-histidine suppresses it by 80%",
        "source": "own lab, unpublished"}
    state.observations["OBS-3"] = {
        "description": "activity scales with chloride concentration but "
                       "no free chlorine is detected by DPD assay",
        "source": "own lab, unpublished"}

    mech_script = {
        "H-OH": json.dumps({"verdict": "fail",
                            "reason": "no DMPO-OH quartet and no "
                                      "tert-butanol suppression"}),
        "H-O2": json.dumps({"verdict": "fail",
                            "reason": "no superoxide probe response"}),
        "H-1O2": json.dumps({"verdict": "pass",
                             "reason": "TEMP triplet plus histidine "
                                       "suppression match"}),
        "H-FeIV": json.dumps({"verdict": "na",
                              "reason": "no sulfoxide probe data"}),
        "H-ClR": json.dumps({"verdict": "pass",
                             "reason": "chloride dependence without "
                                       "free chlorine is consistent"}),
        "H-CER": json.dumps({"verdict": "fail",
                             "reason": "DPD assay excludes free "
                                       "chlorine"}),
        "H-DET": json.dumps({"verdict": "fail",
                             "reason": "diffusible species detected "
                                       "in bulk solution"}),
    }

    class _ScriptedMock(MockLLM):
        def complete(self, system, messages):
            latest = messages[-1]["content"]
            for hid, reply in mech_script.items():
                if f"Mechanism id: {hid}" in latest:
                    self.call_log.append({"system": system,
                                          "messages": messages,
                                          "reply": reply})
                    from cdi_agents.llm import LLMResponse
                    return LLMResponse(content=reply, model="mock")
            return super().complete(system, messages)

    mechanism = MechanismAgent(_ScriptedMock())
    critic_script = {
        "novel": json.dumps([
            {"challenge": "Fe leaching homogeneous Fenton",
             "rebuttal_data": "ICP-MS leaching time series",
             "follow_up": "none"},
            {"challenge": "CER apparent artifact",
             "rebuttal_data": "EMPTY",
             "follow_up": "rotating ring-disk chlorine detection"}]),
    }
    critic = CriticAgent(MockLLM(
        script=critic_script,
        default=json.dumps({"agree": True, "verdict": "fail",
                            "reason": "independent re-judgment"})))

    p3 = MechanismPipeline(state, mechanism, critic, confirm_fn=None)
    out = p3.run(claim="concerted 1O2-RCS interfacial pathway",
                 available_data=["EPR", "quenching", "DPD", "ICP-MS"])
    print("\n[P3] elimination matrix:")
    for hid, row in out["elimination"].get("matrix", {}).items():
        print(f"    {hid:8s} {row}")
    print("  survivors:", out["elimination"].get("survivors"))
    print("  critic disagreements:",
          len(out["elimination"].get("critic_disagreements", [])))
    print("[P3] adversarial table:")
    for item in out["adversarial"]:
        print("   -", item.get("challenge"),
              "| rebuttal:", item.get("rebuttal_data"))

    # ---------------- A5 blind probe consensus --------------------
    results = {}
    for name in ("mock-A", "mock-B", "mock-C"):
        probe = BlindProbeAgent(MockLLM(default=json.dumps({
            "ranking": ["H-1O2", "H-ClR", "H-FeIV"],
            "top_choice": "H-1O2",
            "reasoning": "TEMP triplet and histidine suppression"})))
        results[name] = probe.probe(state, claimed_hid="H-1O2")
    consensus = consensus_report(results, claimed_hid="H-1O2")
    print("\n[A5] blind probe consensus:", json.dumps(consensus))

    print("\nDemo finished. Shared state has",
          len(state.history), "logged messages.")


if __name__ == "__main__":
    main()
