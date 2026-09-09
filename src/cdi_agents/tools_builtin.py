"""Built-in tools exposed to agents via the tool layer.

Four starter tools, all deterministic and side-effect-light:
- corpus_search: keyword retrieval over a local folder of extracted
  publication texts (honest local retrieval; no fabricated web access);
- dataset_query: filter the versioned descriptor dataset by support
  family and / or active-center element;
- compute_factor: compute the normalized C_s / R_s factor of one record;
- train_model: run the full model-pool LOOCV benchmark on a CSV table.

Register only what an agent actually needs; least-privilege applies to
agents too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .descriptors.normalization import FactorConstants, cs_factor, rs_factor
from .descriptors.schema import MaterialRecord
from .tools import Tool, ToolRegistry


def make_corpus_search(text_dir: str | Path) -> Tool:
    root = Path(text_dir)

    def corpus_search(query: str, limit: int = 5) -> list[dict[str, str]]:
        hits: list[dict[str, str]] = []
        terms = [t.lower() for t in query.split() if t.strip()]
        if not root.exists():
            raise FileNotFoundError(f"corpus dir not found: {root}")
        for path in sorted(root.glob("*.txt")):
            text = path.read_text(encoding="utf-8", errors="ignore")
            lower = text.lower()
            if all(t in lower for t in terms):
                idx = lower.find(terms[0]) if terms else 0
                snippet = text[max(0, idx - 200):idx + 400]
                hits.append({"record_id": path.stem,
                             "snippet": snippet.strip()})
            if len(hits) >= limit:
                break
        return hits

    return Tool(
        name="corpus_search",
        description="Keyword search over the local corpus of extracted "
                    "publication texts. Returns matching record ids and "
                    "snippets.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "space-separated keywords, "
                                         "all must match"},
                "limit": {"type": "integer", "default": 5}},
            "required": ["query"]},
        fn=corpus_search)


def make_dataset_query(dataset_path: str | Path) -> Tool:
    path = Path(dataset_path)

    def dataset_query(support_family: str | None = None,
                      element: str | None = None,
                      limit: int = 5) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"dataset not found: {path}")
        db = json.loads(path.read_text(encoding="utf-8"))
        records = db.get("records", [])
        matched = []
        for rec in records:
            if support_family and \
                    rec.get("support", {}).get("family") != support_family:
                continue
            if element and element not in \
                    rec.get("active_center", {}).get("element", []):
                continue
            matched.append(rec)
        return {"version": db.get("version"),
                "n_matched": len(matched),
                "records": matched[:limit]}

    return Tool(
        name="dataset_query",
        description="Query the versioned descriptor dataset. Filter by "
                    "support family and/or active-center element.",
        parameters={
            "type": "object",
            "properties": {
                "support_family": {"type": "string",
                                   "description": "e.g. MXene, carbon"},
                "element": {"type": "string", "description": "e.g. Fe"},
                "limit": {"type": "integer", "default": 5}},
            "required": []},
        fn=dataset_query)


def make_compute_factor(constants: FactorConstants | None = None) -> Tool:
    const = constants or FactorConstants()

    def compute_factor(record: dict[str, Any],
                       factor: str = "cs") -> dict[str, Any]:
        rec = MaterialRecord.from_dict(record)
        if factor == "cs":
            value = cs_factor(rec, const)
        elif factor == "rs":
            value = rs_factor(rec)
        else:
            raise ValueError(f"unknown factor: {factor}")
        return {"factor": factor, "value": value,
                "constants_frozen": const.frozen}

    return Tool(
        name="compute_factor",
        description="Compute the normalized C_s or R_s performance "
                    "factor for one descriptor record.",
        parameters={
            "type": "object",
            "properties": {
                "record": {"type": "object",
                           "description": "descriptor record JSON"},
                "factor": {"type": "string", "enum": ["cs", "rs"],
                           "default": "cs"}},
            "required": ["record"]},
        fn=compute_factor)


def make_train_model() -> Tool:
    from .ml.model_pool import run_model_pool_benchmark

    def train_model(csv_path: str, features: list[str],
                    target: str, seed: int = 0) -> dict[str, Any]:
        df = pd.read_csv(csv_path)
        report = run_model_pool_benchmark(df, features, target, seed=seed)
        best = report["best_model_name"]
        return {
            "n_samples": report["n_samples"],
            "best_model": best,
            "best_metrics": report["benchmark"][best],
            "all_models": report["benchmark"],
            "family_holdout": report.get("family_holdout"),
        }

    return Tool(
        name="train_model",
        description="Train the full regression model pool on a CSV table "
                    "with LOOCV benchmarking. Returns metrics for ALL "
                    "models plus the best one.",
        parameters={
            "type": "object",
            "properties": {
                "csv_path": {"type": "string"},
                "features": {"type": "array",
                             "items": {"type": "string"}},
                "target": {"type": "string"},
                "seed": {"type": "integer", "default": 0}},
            "required": ["csv_path", "features", "target"]},
        fn=train_model)


def default_registry(corpus_dir: str | Path | None = None,
                     dataset_path: str | Path | None = None,
                     with_train: bool = True) -> ToolRegistry:
    """Convenience factory: register the tools whose data roots exist."""
    registry = ToolRegistry()
    if corpus_dir:
        registry.register(make_corpus_search(corpus_dir))
    if dataset_path:
        registry.register(make_dataset_query(dataset_path))
    registry.register(make_compute_factor())
    if with_train:
        registry.register(make_train_model())
    return registry
