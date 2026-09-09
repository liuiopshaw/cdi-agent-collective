#!/usr/bin/env python3
"""Phase 1 batch entry: PDF directory -> texts -> validated dataset.

One command turns a folder of PDFs into a versioned descriptor dataset:

    python scripts/run_p1.py --pdf-dir data/pdfs --config config.yaml

Dry-run offline (no API key, mock LLM, for pipeline debugging):

    python scripts/run_p1.py --pdf-dir data/pdfs --mock

Inputs
------
--pdf-dir DIR   folder with .pdf files (pypdf required)
--text-dir DIR  folder with pre-extracted .txt files (skips PDF step)

Outputs (created under --out, default ./results/p1_<timestamp>/)
------
texts/<record_id>.txt     extracted raw texts
dataset.json              versioned accepted records (D_v1 candidate)
quarantine.json           records that failed the quality gate
qa_report.md              human-readable QA summary for manual review

Record ids are derived from file names (non-alphanumerics -> "_"), so
name your PDFs by DOI slug, e.g. "10.1021_jacs.5b12345.pdf".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdi_agents.agents.data_agent import DataAgent
from cdi_agents.bus import MessageBus
from cdi_agents.config import load_config
from cdi_agents.llm import ChatLLM, MockLLM
from cdi_agents.pipelines.p1_corpus import CorpusPipeline
from cdi_agents.state import SharedState

MOCK_RECORD = json.dumps({
    "record_id": "mock",
    "active_center": {"element": ["Fe"], "form": "cluster"},
    "support": {"family": "MXene", "name": "Ti3C2Tx"},
    "performance": {"sac_mg_g": 45.0},
    "provenance": {"doi": "10.0000/mock",
                   "evidence_quote": "mock sentence for dry run"}})


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def pdf_to_text(pdf_path: Path) -> str:
    try:
        import pypdf
    except ImportError as err:
        raise SystemExit(
            "pypdf is required for PDF extraction: pip install pypdf"
        ) from err
    reader = pypdf.PdfReader(str(pdf_path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def collect_items(pdf_dir: Path | None, text_dir: Path | None,
                  out_texts: Path, limit: int | None) -> list[dict]:
    items: list[dict] = []
    if text_dir:
        for path in sorted(text_dir.glob("*.txt")):
            items.append({"record_id": slugify(path.stem),
                          "text": path.read_text(encoding="utf-8",
                                                 errors="ignore")})
    elif pdf_dir:
        out_texts.mkdir(parents=True, exist_ok=True)
        for path in sorted(pdf_dir.glob("*.pdf")):
            rid = slugify(path.stem)
            text = pdf_to_text(path)
            (out_texts / f"{rid}.txt").write_text(text, encoding="utf-8")
            items.append({"record_id": rid, "text": text})
    else:
        raise SystemExit("provide --pdf-dir or --text-dir")
    if limit:
        items = items[:limit]
    if not items:
        raise SystemExit("no input files found")
    return items


def build_agent(config_path: str | None, mock: bool) -> DataAgent:
    if mock:
        return DataAgent(MockLLM(default=MOCK_RECORD))
    cfg = load_config(config_path or "config.yaml")
    provider = cfg.provider_for("data")
    role = cfg.role("data")
    llm = ChatLLM(provider)
    # temperature is applied server-side by some providers; kept in config
    # for documentation purposes and future client support.
    _ = role.temperature
    return DataAgent(llm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf-dir", type=Path)
    parser.add_argument("--text-dir", type=Path)
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--out", type=Path,
                        default=Path("results")
                        / f"p1_{time.strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--mock", action="store_true",
                        help="offline dry run with a scripted mock LLM")
    parser.add_argument("--limit", type=int, default=None,
                        help="process at most N files (for pilots)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    items = collect_items(args.pdf_dir, args.text_dir,
                          args.out / "texts", args.limit)
    print(f"[p1] collected {len(items)} documents")

    state = SharedState()
    bus = MessageBus(state, "data")
    agent = build_agent(None if args.mock else args.config, args.mock)
    pipeline = CorpusPipeline(state, bus, agent, args.out / "dataset.json")

    report = pipeline.ingest_batch(items)
    (args.out / "quarantine.json").write_text(
        json.dumps(report["quarantine"], ensure_ascii=False, indent=1),
        encoding="utf-8")

    lines = [
        "# P1 extraction QA report", "",
        f"- input documents: {len(items)}",
        f"- accepted records: {report['n_accepted']}",
        f"- quarantined: {report['n_quarantined']}",
        f"- dataset version: {report['dataset_version']}", "",
        "## Quarantine reasons", "",
    ]
    for q in report["quarantine"]:
        issues = q["payload"].get("qa_issues", [])
        lines.append(f"- {q['record_id']}: {'; '.join(issues)}")
    lines += ["", "## Manual review checklist", "",
              "- [ ] spot-check >= 20% of accepted records against the "
              "source PDFs (focus: numeric fields and evidence quotes)",
              "- [ ] field-level evidence-quote missing rate < 15%",
              "- [ ] confirm no own-lab data entered the dataset (T1)"]
    (args.out / "qa_report.md").write_text("\n".join(lines),
                                           encoding="utf-8")

    state.save(args.out / "session_state.json")
    print(f"[p1] accepted={report['n_accepted']} "
          f"quarantined={report['n_quarantined']} "
          f"version={report['dataset_version']}")
    print(f"[p1] outputs in {args.out}")


if __name__ == "__main__":
    main()
