"""Model pool with LOOCV benchmark and family-holdout evaluation.

Mirrors the transparency practice of Guo et al. (SI section 7.5): every
candidate model is benchmarked with leave-one-out cross-validation and
ALL results are reported, including unflattering ones. The overfitting
gap train R2 minus LOO R2 is always shown.

L2 family holdout: when ``group_col`` is given, each group (e.g. one
support family) is excluded in turn and predicted from the remaining
families. This tests cross-family transfer, the real generalization
claim of the framework.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:  # optional dependency
    from xgboost import XGBRegressor
    _HAS_XGB = True
except Exception:  # noqa: BLE001
    _HAS_XGB = False


def build_pool(seed: int = 0) -> dict[str, Any]:
    pool: dict[str, Any] = {
        "GPR": GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(), normalize_y=True,
            random_state=seed),
        "Linear": LinearRegression(),
        "RidgeCV": RidgeCV(alphas=np.logspace(-4, 4, 25)),
        "SVR": SVR(C=10.0),
        "RF": RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                    random_state=seed),
        "ET": ExtraTreesRegressor(n_estimators=600, random_state=seed),
        "GB": GradientBoostingRegressor(random_state=seed),
    }
    if _HAS_XGB:
        pool["XGB"] = XGBRegressor(n_estimators=50, random_state=seed)
    return pool


def _fit_predict_with_sigma(model, X_train, y_train, X_test):
    model.fit(X_train, y_train)
    if hasattr(model, "predict"):
        try:
            pred, sigma = model.predict(X_test, return_std=True)
            return np.asarray(pred), np.asarray(sigma)
        except TypeError:
            pass
    pred = model.predict(X_test)
    resid_std = float(np.std(y_train - model.predict(X_train)))
    return np.asarray(pred), np.full(len(np.asarray(pred)), resid_std)


def loocv(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.arange(n) != i
        m = clone(model)
        m.fit(X[mask], y[mask])
        preds[i] = m.predict(X[i:i + 1])[0]
    train_pred = model.fit(X, y).predict(X)
    return {
        "r2_train": float(r2_score(y, train_pred)),
        "r2_loo": float(r2_score(y, preds)) if n > 2 else float("nan"),
        "mae_loo": float(mean_absolute_error(y, preds)),
        "overfit_gap": float(r2_score(y, train_pred) - r2_score(y, preds))
        if n > 2 else float("nan"),
        "loo_predictions": preds,
    }


def family_holdout(model, X: np.ndarray, y: np.ndarray,
                   groups: np.ndarray) -> dict[str, Any]:
    """Leave-one-family-out evaluation (L2 transfer test)."""
    results = {}
    preds_all = np.full(len(y), np.nan)
    for g in np.unique(groups):
        test = groups == g
        if test.sum() == 0 or (~test).sum() < 3:
            continue
        m = clone(model)
        m.fit(X[~test], y[~test])
        preds_all[test] = m.predict(X[test])
        results[str(g)] = {"n": int(test.sum())}
    valid = ~np.isnan(preds_all)
    r2 = float(r2_score(y[valid], preds_all[valid])) if valid.sum() > 2 \
        else float("nan")
    return {"r2_family_holdout": r2, "per_family": results,
            "predictions": preds_all}


def run_model_pool_benchmark(df: pd.DataFrame, features: list[str],
                             target: str, group_col: str | None = None,
                             seed: int = 0) -> dict[str, Any]:
    data = df[features + [target]].dropna().reset_index(drop=True)
    if len(data) < 5:
        raise ValueError(f"too few complete rows: {len(data)}")
    X = data[features].to_numpy(dtype=float)
    y = data[target].to_numpy(dtype=float)
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    pool = build_pool(seed)
    table: dict[str, dict[str, float]] = {}
    fitted: dict[str, tuple[Any, np.ndarray, np.ndarray]] = {}
    for name, model in pool.items():
        metrics = loocv(model, Xs, y)
        table[name] = {k: v for k, v in metrics.items()
                       if k != "loo_predictions"}
        fitted[name] = (model, metrics["loo_predictions"], y)

    best_name = max(table, key=lambda n: table[n]["r2_loo"])
    best_model = pool[best_name]
    best_model.fit(Xs, y)
    resid = y - fitted[best_name][1]

    def predict_fn(X_new: np.ndarray) -> np.ndarray:
        X_new = np.atleast_2d(np.asarray(X_new, dtype=float))
        return best_model.predict(scaler.transform(X_new))

    report: dict[str, Any] = {
        "n_samples": int(len(data)),
        "features": features,
        "target": target,
        "benchmark": table,
        "best_model_name": best_name,
        "best_model": {
            "name": best_name,
            "predict_fn": predict_fn,
            "sigma": float(np.std(resid)),
            "residuals_loo": resid.tolist(),
        },
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "seed": seed,
    }
    if group_col and group_col in df.columns:
        groups = df.loc[data.index, group_col].to_numpy() \
            if group_col in df.columns else None
        if groups is not None:
            fh = family_holdout(GaussianProcessRegressor(
                kernel=RBF() + WhiteKernel(), normalize_y=True,
                random_state=seed), Xs, y, groups)
            report["family_holdout"] = {
                k: v for k, v in fh.items() if k != "predictions"}
    return report
