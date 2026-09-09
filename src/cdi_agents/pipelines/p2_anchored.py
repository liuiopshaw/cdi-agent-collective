"""P2 Anchored prediction pipeline.

Stage 1: calibrated baseline on literature data (full model-pool LOOCV
benchmark, optionally family holdout).
Stage 2: blind test of the held-out own system against the frozen model.
Stage 3: residual outlier certification with robustness checks.

The frozen literature model and the anchor data live in different
namespaces; this module is the only place they meet, and the meeting is
read-only in one direction (model predicts anchor, never trains on it).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..agents.prediction import PredictionAgent
from ..ml.residual import classical_detectors_baseline
from ..state import SharedState


class AnchoredPredictionPipeline:
    def __init__(self, state: SharedState, agent: PredictionAgent,
                 r2_threshold: float = 0.6):
        self.state = state
        self.agent = agent
        self.r2_threshold = r2_threshold

    def run(self, literature_df: pd.DataFrame, features: list[str],
            target: str, anchor_X: np.ndarray, anchor_y: float,
            anchor_label: str = "own-system",
            group_col: str | None = None,
            seed: int = 0) -> dict[str, Any]:
        report = self.agent.calibrate(literature_df, features, target,
                                      group_col=group_col, seed=seed)
        best = report["best_model"]
        calibration_ok = best["sigma"] >= 0 and \
            report["benchmark"][report["best_model_name"]]["r2_loo"] \
            >= self.r2_threshold

        anchor = self.agent.anchor_test(report, anchor_X, anchor_y)

        cert = self.agent.certify(
            np.array(best["residuals_loo"], dtype=float),
            anchor["mu"], anchor["sigma"], anchor_y, label=anchor_label)

        try:
            X_lit = literature_df[features].dropna().to_numpy(dtype=float)
            cert["classical_detectors"] = classical_detectors_baseline(
                X_lit, anchor_X)
        except Exception as err:  # noqa: BLE001
            cert["classical_detectors"] = {"error": str(err)}

        result = {
            "calibration": {
                "n_samples": report["n_samples"],
                "best_model": report["best_model_name"],
                "benchmark": report["benchmark"],
                "family_holdout": report.get("family_holdout"),
                "r2_threshold": self.r2_threshold,
                "calibration_ok": calibration_ok,
            },
            "anchor_blind_test": anchor,
            "outlier_certificate": cert,
        }
        self.state.residual_report = {
            "anchor": anchor, "certificate": cert,
            "calibration_ok": calibration_ok}
        return result
