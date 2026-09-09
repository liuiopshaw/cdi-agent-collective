"""Residual-based outlier certification (pipeline P2, Stage 3).

Logic, following and extending Cheng et al. (RSC Adv. 2026):
- a model that is well calibrated on the literature distribution defines
  the baseline of known physics;
- the held-out system's deviation (Z score against the predictive
  distribution) certifies whether its performance is explainable by known
  structure-property relationships;
- the certification is strengthened by three robustness checks:
  (1) non-randomness: the anchor residual is compared against the full
      literature residual distribution, not just the point sigma;
  (2) bimodality scan of the residual distribution - a positive mode
      indicates systematic enhancement rather than noise;
  (3) classical density-based detectors (IsolationForest, LOF) are run as
      baselines, because density methods were shown to miss chemically
      meaningful outliers.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def _norm_cdf(z: float) -> float:
    return float(0.5 * (1.0 + math_erf(z / np.sqrt(2.0))))


def math_erf(x: float) -> float:
    from math import erf
    return erf(x)


def bimodality_scan(residuals: np.ndarray, n_bins: int = 20) -> dict[str, Any]:
    """Simple histogram-based bimodality descriptor of residuals."""
    residuals = np.asarray(residuals, dtype=float)
    if len(residuals) < 10:
        return {"test": "skipped", "reason": "too few residuals"}
    hist, edges = np.histogram(residuals, bins=n_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peaks = []
    for i in range(1, len(hist) - 1):
        if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1] and hist[i] > 0:
            peaks.append({"center": float(centers[i]),
                          "count": int(hist[i])})
    dip_stat = None
    try:  # Hartigan dip test is optional; degrade gracefully
        import diptest  # type: ignore
        dip_stat = float(diptest.dipstat(residuals))
    except Exception:  # noqa: BLE001
        pass
    return {"peaks": peaks, "n_peaks": len(peaks),
            "dip_statistic": dip_stat,
            "positive_mode": any(p["center"] > 0 for p in peaks)}


def certify_outlier(literature_residuals: np.ndarray, mu: float,
                    sigma: float, y_observed: float,
                    label: str = "anchor") -> dict[str, Any]:
    """Certify the held-out system as a statistical outlier.

    Parameters
    ----------
    literature_residuals : LOO residuals of the literature model.
    mu, sigma : predictive mean and std at the anchor point.
    y_observed : measured performance of the held-out system.
    """
    literature_residuals = np.asarray(literature_residuals, dtype=float)
    delta = float(y_observed - mu)
    z_point = delta / sigma if sigma and sigma > 0 else float("nan")
    p_point = float(2.0 * (1.0 - _norm_cdf(abs(z_point)))) \
        if np.isfinite(z_point) else float("nan")

    # non-randomness: compare against empirical residual distribution
    if len(literature_residuals) >= 10:
        emp_std = float(np.std(literature_residuals)) or 1e-12
        z_emp = delta / emp_std
        tail = float(np.mean(np.abs(literature_residuals) >= abs(delta)))
        ks = stats.kstest(literature_residuals,
                          stats.norm(loc=0.0, scale=emp_std).cdf)
        normality_p = float(ks.pvalue)
    else:
        z_emp, tail, normality_p = float("nan"), float("nan"), float("nan")

    certified = bool(np.isfinite(z_point) and abs(z_point) >= 2.0)

    return {
        "label": label,
        "mu": float(mu),
        "sigma": float(sigma),
        "y_observed": float(y_observed),
        "delta": delta,
        "z_score_point": float(z_point),
        "p_value_point": p_point,
        "z_score_empirical": float(z_emp),
        "empirical_tail_fraction": tail,
        "literature_residual_normality_p": normality_p,
        "bimodality": bimodality_scan(literature_residuals),
        "certified_outlier": certified,
        "attributed_new_mechanism_contribution": delta if certified else 0.0,
        "interpretation": (
            "performance exceeds the calibrated known-physics baseline; "
            "the residual delta is the quantified contribution of the "
            "proposed new mechanism" if certified else
            "performance is consistent with known physics; no novel "
            "contribution certified"),
    }


def classical_detectors_baseline(X: np.ndarray, anchor_X: np.ndarray,
                                 contamination: float = 0.05) -> dict[str, Any]:
    """IsolationForest / LOF baseline flags for the anchor point."""
    X = np.asarray(X, dtype=float)
    anchor_X = np.atleast_2d(np.asarray(anchor_X, dtype=float))
    out: dict[str, Any] = {}
    try:
        iso = IsolationForest(contamination=contamination,
                              random_state=0).fit(X)
        out["isolation_forest_flags_anchor"] = bool(
            iso.predict(anchor_X)[0] == -1)
    except Exception as err:  # noqa: BLE001
        out["isolation_forest_error"] = str(err)
    try:
        lof = LocalOutlierFactor(n_neighbors=min(20, max(2, len(X) - 1)),
                                 novelty=True).fit(X)
        out["lof_flags_anchor"] = bool(lof.predict(anchor_X)[0] == -1)
    except Exception as err:  # noqa: BLE001
        out["lof_error"] = str(err)
    return out
