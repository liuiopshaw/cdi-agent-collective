"""Three-layer descriptor schema for support x active-center CDI materials.

Layer 1: active center. Layer 2: support / ligand. Layer 3: interface.
Plus condition and performance layers. Every record carries provenance:
a verbatim evidence quote and a source pointer (hard rule T2). Missing
values are recorded as NR (not reported); the extractor is forbidden from
guessing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

NR = "NR"


@dataclass
class ActiveCenter:
    element: list[str] = field(default_factory=list)      # e.g. ["Fe"]
    form: str = NR           # single_atom | cluster | nanoparticle
    loading_wt: float | None = None
    size_nm: float | None = None
    valence: str = NR        # e.g. "Fe2+/Fe3+ = 0.6/0.4"
    coordination: str = NR   # e.g. "Fe-O-Ti"
    coordination_number: float | None = None


@dataclass
class Support:
    family: str = NR         # MXene | carbon | MOF | LDH | g-C3N4 | oxide
    name: str = NR           # e.g. "Ti3C2Tx"
    conductivity: str = NR
    surface_area_m2g: float | None = None
    pore: str = NR
    termination: str = NR    # e.g. "-O 0.5, -OH 0.3, -F 0.2"
    interlayer_nm: float | None = None
    defect: str = NR


@dataclass
class Interface:
    bond: str = NR           # e.g. "Ti-O-Fe"
    charge_transfer: str = NR
    msi_strength: str = NR   # weak | medium | strong
    anchor: str = NR


@dataclass
class Conditions:
    voltage_max: float | None = None  # numeric cut-off / window upper bound (V)
    protocol: str = NR     # free text: CC/CV hybrid, hold time, cut-off notes
    nacl_mg_L: float | None = None
    mode: str = NR           # batch | flow-by | flow-through
    flow_rate: str = NR
    ph: float | None = None
    dissolved_oxygen: str = NR
    oxidant_dose: str = NR   # H2O2 / PMS dosing, if any
    bacteria: str = NR


@dataclass
class Performance:
    sac_mg_g: float | None = None
    asar: float | None = None
    charge_efficiency: float | None = None  # fraction in [0, 1], not percent
    cycles: float | None = None
    retention_pct: float | None = None
    energy: str = NR
    ros_type: str = NR
    ros_yield: str = NR
    rcs_yield: str = NR
    disinfection_log: float | None = None
    disinfection_rate: str = NR


@dataclass
class Provenance:
    doi: str = NR
    evidence_quote: str = ""     # verbatim sentence(s), hard rule T2
    page: str = NR


@dataclass
class MaterialRecord:
    record_id: str
    active_center: ActiveCenter = field(default_factory=ActiveCenter)
    support: Support = field(default_factory=Support)
    interface: Interface = field(default_factory=Interface)
    conditions: Conditions = field(default_factory=Conditions)
    performance: Performance = field(default_factory=Performance)
    provenance: Provenance = field(default_factory=Provenance)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "MaterialRecord":
        rec = MaterialRecord(record_id=data.get("record_id", "unknown"))
        for section, cls in (("active_center", ActiveCenter),
                             ("support", Support),
                             ("interface", Interface),
                             ("conditions", Conditions),
                             ("performance", Performance),
                             ("provenance", Provenance)):
            values = data.get(section, {})
            if isinstance(values, dict):
                valid = {k: v for k, v in values.items()
                         if k in cls.__dataclass_fields__}
                setattr(rec, section, cls(**valid))
        return rec


def validate_record(rec: MaterialRecord) -> list[str]:
    """Quality gate for pipeline P1. Returns a list of issues."""
    issues: list[str] = []
    if not rec.provenance.evidence_quote.strip():
        issues.append("missing evidence quote (rule T2)")
    if rec.support.family == NR:
        issues.append("support family not reported")
    if not rec.active_center.element:
        issues.append("active center element missing")
    perf = rec.performance
    if all(v is None for v in (perf.sac_mg_g, perf.disinfection_log)) \
            and perf.ros_yield == NR and perf.rcs_yield == NR:
        issues.append("no quantitative performance field")
    ce = perf.charge_efficiency
    if ce is not None and not 0.0 <= ce <= 1.0:
        issues.append("charge efficiency out of range [0, 1]; "
                      "schema expects a fraction, not a percent")
    return issues
