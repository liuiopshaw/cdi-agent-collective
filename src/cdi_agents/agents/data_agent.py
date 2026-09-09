"""A1 Data Agent: literature-to-database extraction (pipeline P1).

Transforms unstructured publication text into MaterialRecord JSON that
conforms to the three-layer descriptor schema. Hard rules enforced in the
system prompt: never guess missing values (use NR), always attach a
verbatim evidence quote, never invent a DOI.
"""

from __future__ import annotations

import json

from ..descriptors.schema import MaterialRecord, validate_record
from ..state import Message, SharedState
from .base import BaseAgent

DATA_SYSTEM = (
    "You are the Data Agent. You convert publication text about "
    "capacitive deionization electrodes into structured JSON records "
    "following the three-layer descriptor schema (active_center, support, "
    "interface, conditions, performance, provenance). Hard rules: "
    "(1) if a field is not reported, output the string \"NR\" - never "
    "guess; (2) provenance.evidence_quote must be a verbatim sentence "
    "copied from the source text supporting the most important claims; "
    "(3) never invent identifiers, numbers or units; (4) output only "
    "valid JSON, no commentary."
)

SCHEMA_HINT = json.dumps({
    "record_id": "string",
    "active_center": {"element": ["Fe"], "form": "cluster",
                      "loading_wt": 0.0, "size_nm": 0.0,
                      "valence": "NR", "coordination": "NR",
                      "coordination_number": None},
    "support": {"family": "MXene", "name": "Ti3C2Tx",
                "conductivity": "NR", "surface_area_m2g": None,
                "pore": "NR", "termination": "NR",
                "interlayer_nm": None, "defect": "NR"},
    "interface": {"bond": "NR", "charge_transfer": "NR",
                  "msi_strength": "NR", "anchor": "NR"},
    "conditions": {"voltage_window": "NR", "nacl_mg_L": None,
                   "mode": "batch", "flow_rate": "NR", "ph": None,
                   "dissolved_oxygen": "NR", "oxidant_dose": "NR",
                   "bacteria": "NR"},
    "performance": {"sac_mg_g": None, "asar": None,
                    "charge_efficiency": None, "cycles": None,
                    "retention_pct": None, "energy": "NR",
                    "ros_type": "NR", "ros_yield": "NR",
                    "rcs_yield": "NR", "disinfection_log": None,
                    "disinfection_rate": "NR"},
    "provenance": {"doi": "NR", "evidence_quote": "string", "page": "NR"},
}, indent=1)


class DataAgent(BaseAgent):
    role = "data"
    system_prompt = DATA_SYSTEM

    def handle(self, msg: Message, state: SharedState) -> Message:
        text = msg.payload.get("text", msg.content)
        prompt = ("Extract one structured record from this publication "
                  "text. Follow this schema exactly:\n" + SCHEMA_HINT
                  + "\n\nTEXT:\n" + text[:12000])
        raw = self.call_llm(prompt)
        try:
            data = self.extract_json(raw)
            record = MaterialRecord.from_dict(data)
            issues = validate_record(record)
            payload = {"record": record.to_dict(), "qa_issues": issues}
        except Exception as err:  # noqa: BLE001
            payload = {"record": None,
                       "qa_issues": [f"extraction parse failure: {err}"]}
        return Message(sender=self.name, content=raw, payload=payload)
