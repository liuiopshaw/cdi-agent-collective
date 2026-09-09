"""P1 Corpus pipeline: publications -> validated descriptor database.

Flow per publication:
    raw text -> A1 extraction -> schema validation -> issue report.
Records passing the quality gate are appended to the versioned dataset;
records with issues are quarantined for manual review. The user's own
data is never touched here (hard rule T1).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..agents.data_agent import DataAgent
from ..bus import MessageBus
from ..state import Message, SharedState


class CorpusPipeline:
    def __init__(self, state: SharedState, bus: MessageBus,
                 agent: DataAgent, dataset_path: str | Path):
        self.state = state
        self.bus = bus
        self.agent = agent
        self.dataset_path = Path(dataset_path)

    def ingest_text(self, record_id: str, text: str) -> dict[str, Any]:
        reply = self.bus.ask(
            self.agent, f"extract record {record_id}",
            payload={"text": text, "record_id": record_id})
        return reply.payload

    def ingest_batch(self, items: list[dict[str, str]]) -> dict[str, Any]:
        accepted, quarantined = [], []
        for item in items:
            payload = self.ingest_text(item["record_id"], item["text"])
            if payload.get("record") and not payload.get("qa_issues"):
                accepted.append(payload["record"])
            else:
                quarantined.append({"record_id": item["record_id"],
                                    "payload": payload})
        version = self._append(accepted)
        report = {
            "dataset_version": version,
            "n_accepted": len(accepted),
            "n_quarantined": len(quarantined),
            "quarantine": quarantined,
        }
        self.state.dataset_version = version
        self.state.n_records = self._count()
        return report

    def _load(self) -> dict:
        if self.dataset_path.exists():
            return json.loads(
                self.dataset_path.read_text(encoding="utf-8"))
        return {"version": 0, "records": []}

    def _append(self, records: list[dict]) -> str:
        db = self._load()
        db["records"].extend(records)
        db["version"] = int(db.get("version", 0)) + 1
        self.dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.dataset_path.write_text(
            json.dumps(db, ensure_ascii=False, indent=1),
            encoding="utf-8")
        return f"v{db['version']}"

    def _count(self) -> int:
        return len(self._load().get("records", []))
