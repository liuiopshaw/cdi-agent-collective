"""P4 Evaluation pipeline (methodology borrowed from PeroMAS).

Four evaluation axes:
1. atomic task accuracy: extraction QA on human-labeled ground truth;
2. dual validity scoring: independent LLM judge vs expert score, with a
   consistency delta (judges run cold and double-blind by design);
3. multi-backbone robustness: re-run a key reasoning chain with several
   providers and check whether conclusions flip;
4. single-agent ablation: same tools, one generalist agent vs the full
   collective - quantifies the multi-agent gain that Guo et al. did not
   measure.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from ..agents.base import BaseAgent
from ..descriptors.schema import MaterialRecord, validate_record

JUDGE_SYSTEM = (
    "You are an independent scientific evaluator. Score the scientific "
    "validity of the given output from 0 to 10, focusing on logic, "
    "evidence grounding and calibrated claims. Reply JSON: "
    "{\"score\": <number>, \"justification\": \"...\"}. Be strict."
)


class EvaluationPipeline:
    def __init__(self, judge: BaseAgent,
                 backbone_factory: Callable[[str], Any]):
        """``backbone_factory(name)`` returns a fresh agent bundle for a
        provider name; used for multi-backbone robustness checks."""
        self.judge = judge
        self.backbone_factory = backbone_factory

    # axis 1 -------------------------------------------------------------
    def extraction_accuracy(self, predictions: list[dict],
                            ground_truth: list[dict]) -> dict[str, Any]:
        """Field-level accuracy of Data Agent records vs human labels."""
        assert len(predictions) == len(ground_truth)
        total, correct = 0, 0
        for pred, gold in zip(predictions, ground_truth):
            rec = MaterialRecord.from_dict(pred).to_dict()
            for section, fields in gold.items():
                for key, gold_val in fields.items():
                    total += 1
                    pred_val = rec.get(section, {}).get(key)
                    if pred_val == gold_val:
                        correct += 1
        return {"n_records": len(predictions), "n_fields": total,
                "field_accuracy": correct / total if total else 0.0}

    # axis 2 -------------------------------------------------------------
    def dual_validity(self, output_text: str,
                      expert_score: float) -> dict[str, Any]:
        raw = self.judge.call_llm(
            "Output to evaluate:\n" + output_text[:8000])
        try:
            judged = self.judge.extract_json(raw)
            llm_score = float(judged.get("score"))
        except Exception:  # noqa: BLE001
            llm_score, judged = float("nan"), {"error": "unparseable"}
        delta = abs(llm_score - expert_score) \
            if llm_score == llm_score else float("nan")
        return {"llm_score": llm_score, "expert_score": expert_score,
                "consistency_delta": delta, "judge_reply": judged}

    # axis 3 -------------------------------------------------------------
    def multi_backbone(self, task_fn: Callable[[Any], str],
                       backbone_names: list[str]) -> dict[str, Any]:
        """Run the same reasoning task on several backbones; report the
        set of distinct conclusions (flips signal fragility)."""
        outputs = {}
        for name in backbone_names:
            bundle = self.backbone_factory(name)
            outputs[name] = task_fn(bundle)
        distinct = len(set(outputs.values()))
        return {"outputs": outputs, "n_backbones": len(backbone_names),
                "n_distinct_conclusions": distinct,
                "conclusion_stable": distinct == 1}

    # axis 4 -------------------------------------------------------------
    def ablation(self, collective_fn: Callable[[], dict],
                 single_agent_fn: Callable[[], dict],
                 scorer: Callable[[dict], float]) -> dict[str, Any]:
        c_out = collective_fn()
        s_out = single_agent_fn()
        c_score, s_score = scorer(c_out), scorer(s_out)
        return {"collective_score": c_score, "single_agent_score": s_score,
                "multi_agent_gain": c_score - s_score,
                "collective_output": c_out, "single_output": s_out}
