"""A2 Prediction Agent: anchored prediction and residual certification.

The LLM part only extracts modeling intent (features, target) from natural
language; all numeric work runs in the sklearn-based ML module. Numeric
computation is never delegated to the LLM (Guo et al. design rule).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ..ml.model_pool import run_model_pool_benchmark
from ..ml.residual import certify_outlier
from ..state import Message, SharedState
from .base import BaseAgent

PREDICTION_SYSTEM = (
    "You are the Prediction Agent. You never compute numbers yourself. "
    "Given a modeling request and a data dictionary, you reply with JSON "
    "specifying: feature columns, target column, and a short rationale. "
    "The actual model training, cross-validation and residual analysis "
    "are executed by deterministic code."
)


class PredictionAgent(BaseAgent):
    role = "prediction"
    system_prompt = PREDICTION_SYSTEM

    # ------------------------------------------------------------------
    def plan_features(self, columns: list[str], request: str) -> dict:
        raw = self.call_llm(
            "Available columns: " + json.dumps(columns)
            + "\nRequest: " + request
            + "\nReply JSON: {\"features\": [...], \"target\": \"...\", "
              "\"rationale\": \"...\"}")
        try:
            return self.extract_json(raw)
        except ValueError:
            return {"features": columns[:-1], "target": columns[-1],
                    "rationale": "fallback: all-but-last column"}

    # ------------------------------------------------------------------
    def calibrate(self, df: pd.DataFrame, features: list[str],
                  target: str, group_col: str | None = None,
                  seed: int = 0) -> dict[str, Any]:
        """Stage 1: full model-pool LOOCV benchmark (all numbers public),
        plus family-holdout (L2) when a group column is provided."""
        report = run_model_pool_benchmark(df, features, target,
                                          group_col=group_col, seed=seed)
        return report

    # ------------------------------------------------------------------
    def anchor_test(self, report: dict[str, Any],
                    anchor_X: np.ndarray, anchor_y: float,
                    alpha: float = 0.05) -> dict[str, Any]:
        """Stage 2: blind prediction on the held-out own system."""
        best = report["best_model"]
        mu = float(best["predict_fn"](anchor_X)[0])
        sigma = float(best.get("sigma", np.nan))
        lo = mu - 1.96 * sigma if np.isfinite(sigma) else None
        hi = mu + 1.96 * sigma if np.isfinite(sigma) else None
        inside = (lo is not None and lo <= anchor_y <= hi)
        return {"mu": mu, "sigma": sigma, "y_observed": anchor_y,
                "ci95": [lo, hi], "inside_interval": inside}

    # ------------------------------------------------------------------
    def certify(self, literature_residuals: np.ndarray, mu: float,
                sigma: float, y_observed: float,
                label: str = "anchor") -> dict[str, Any]:
        """Stage 3: residual outlier certification."""
        return certify_outlier(literature_residuals, mu, sigma,
                               y_observed, label=label)

    # ------------------------------------------------------------------
    def handle(self, msg: Message, state: SharedState) -> Message:
        columns = msg.payload.get("columns", [])
        plan = self.plan_features(columns, msg.content)
        return Message(sender=self.name, content=json.dumps(plan),
                       payload=plan)
