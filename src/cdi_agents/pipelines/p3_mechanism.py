"""P3 Mechanism certification pipeline.

M1 build hypothesis library -> M2 mechanism x observation falsification
matrix -> M2b critic cross-check -> M3 adversarial challenge table.
Optional M4 blind probe is handled by a separate module.

Human confirmation points (hard rule): the matrix is frozen only after
explicit confirmation; disagreements between A3 and A4 are listed for
human arbitration, never auto-resolved.
"""

from __future__ import annotations

from typing import Any

from ..agents.critic import CriticAgent
from ..agents.mechanism import MechanismAgent
from ..state import SharedState


class MechanismPipeline:
    def __init__(self, state: SharedState,
                 mechanism: MechanismAgent, critic: CriticAgent,
                 confirm_fn=None):
        self.state = state
        self.mechanism = mechanism
        self.critic = critic
        self.confirm_fn = confirm_fn

    # M1 ----------------------------------------------------------------
    def build_library(self, extra: list[dict] | None = None) -> list[str]:
        return self.mechanism.build_library(self.state, extra=extra)

    def register_observation(self, obs_id: str, description: str,
                             source: str) -> None:
        """Observations must be phenomenon-only text, no conclusions."""
        self.state.observations[obs_id] = {
            "description": description, "source": source}

    # M2 ----------------------------------------------------------------
    def run_matrix(self) -> dict[str, Any]:
        if not self.state.require_confirmation(
                "freeze elimination matrix inputs", self.confirm_fn):
            return {"status": "aborted_by_user"}
        matrix = self.mechanism.run_matrix(self.state)
        disagreements = self.critic.cross_check(self.state)
        survivors = [h.hid for h in self.state.hypotheses.values()
                     if h.status == "survivor"]
        return {"matrix": matrix, "survivors": survivors,
                "critic_disagreements": disagreements,
                "arbitration_required": len(disagreements) > 0}

    # M3 ----------------------------------------------------------------
    def adversarial_table(self, claim: str,
                          available_data: list[str]) -> list[dict]:
        return self.critic.adversarial_review(claim, available_data)

    # full ---------------------------------------------------------------
    def run(self, claim: str, available_data: list[str],
            extra_hypotheses: list[dict] | None = None) -> dict[str, Any]:
        self.build_library(extra=extra_hypotheses)
        matrix = self.run_matrix()
        challenges = self.adversarial_table(claim, available_data)
        return {"elimination": matrix, "adversarial": challenges}
