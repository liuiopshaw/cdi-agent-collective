"""Normalized performance factors C_s and R_s.

Analogous to the F_s stability factor of Guo et al.: raw literature metrics
measured under heterogeneous conditions are compressed into comparable
normalized factors. The exact constants are calibrated on the P1 pilot
sample and then frozen (confirmation point); after freezing they must not
change, to prevent metric fitting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .schema import MaterialRecord


@dataclass(frozen=True)
class FactorConstants:
    v_ref: float = 1.2            # reference voltage window (V)
    c0_ref: float = 500.0         # reference initial NaCl (mg/L)
    c0_base: float = 100.0        # baseline for log drive correction
    sac_ref: float = 20.0         # median SAC placeholder, calibrate in P1
    eta_ref: float = 0.5          # median charge efficiency placeholder
    frozen: bool = False          # set True after P1 calibration


def _parse_voltage(window: str) -> float | None:
    """Extract a numeric voltage window like '1.2 V' or '0-1.4'."""
    if not window or window == "NR":
        return None
    nums = []
    token = ""
    for ch in window:
        if ch.isdigit() or ch == ".":
            token += ch
        elif token:
            nums.append(float(token))
            token = ""
    if token:
        nums.append(float(token))
    if not nums:
        return None
    return max(nums) - min(nums) if len(nums) > 1 else nums[0]


def cs_factor(rec: MaterialRecord, const: FactorConstants) -> float | None:
    """Normalized desalination factor. None if inputs are missing."""
    sac = rec.performance.sac_mg_g
    eta = rec.performance.charge_efficiency
    c0 = rec.conditions.nacl_mg_L
    v = _parse_voltage(rec.conditions.voltage_window)
    if sac is None:
        return None
    eta_term = (eta / const.eta_ref) if eta is not None else 1.0
    v_term = (v / const.v_ref) if v is not None else 1.0
    if c0 and c0 > const.c0_base:
        # SAC scales roughly with the log concentration drive; dividing
        # by it yields a concentration-independent efficiency factor.
        c_term = math.log(const.c0_ref / const.c0_base) / math.log(
            c0 / const.c0_base)
    else:
        c_term = 1.0
    return (sac / const.sac_ref) * eta_term * v_term * c_term


def rs_factor(rec: MaterialRecord) -> float | None:
    """Normalized reactive-species factor (qualitative-semi track).

    ROS/RCS yields are reported in too many incompatible units across the
    literature, so the primary track is a graded score; quantitative
    subsets can be modeled separately. Returns None when ungradable.
    """
    text = " ".join([rec.performance.ros_yield, rec.performance.rcs_yield,
                     rec.performance.ros_type]).lower()
    if not text.strip() or text.strip() == "nr nr nr":
        return None
    if any(k in text for k in ("high", "ultra", "exceptional")):
        return 3.0
    if any(k in text for k in ("moderate", "medium")):
        return 2.0
    if any(k in text for k in ("low", "weak", "trace")):
        return 1.0
    return None
