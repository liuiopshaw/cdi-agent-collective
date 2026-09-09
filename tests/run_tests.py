"""Test suite for CDI Agent Collective (stdlib unittest, no pytest).

Run:
    python tests/run_tests.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdi_agents.agents.base import BaseAgent
from cdi_agents.agents.data_agent import DataAgent
from cdi_agents.agents.mechanism import MechanismAgent
from cdi_agents.bus import MessageBus
from cdi_agents.descriptors.normalization import (FactorConstants, cs_factor)
from cdi_agents.descriptors.schema import (MaterialRecord, NR,
                                           validate_record)
from cdi_agents.llm import MockLLM
from cdi_agents.ml.model_pool import loocv, run_model_pool_benchmark
from cdi_agents.ml.residual import certify_outlier
from cdi_agents.state import Message, SharedState


class TestSchema(unittest.TestCase):
    def test_roundtrip(self):
        rec = MaterialRecord.from_dict({
            "record_id": "r1",
            "active_center": {"element": ["Fe"], "form": "cluster",
                              "loading_wt": 2.3},
            "support": {"family": "MXene", "name": "Ti3C2Tx"},
            "performance": {"sac_mg_g": 45.0},
            "provenance": {"doi": "10.1/x", "evidence_quote": "q"}})
        self.assertEqual(rec.active_center.element, ["Fe"])
        self.assertEqual(rec.support.family, "MXene")
        self.assertEqual(rec.interface.bond, NR)  # untouched -> NR
        d = rec.to_dict()
        rec2 = MaterialRecord.from_dict(d)
        self.assertEqual(rec2.performance.sac_mg_g, 45.0)

    def test_validation_flags_missing_quote(self):
        rec = MaterialRecord(record_id="r2")
        issues = validate_record(rec)
        self.assertTrue(any("T2" in i for i in issues))

    def test_unknown_fields_ignored(self):
        rec = MaterialRecord.from_dict({
            "record_id": "r3",
            "support": {"family": "carbon", "bogus_field": 1}})
        self.assertEqual(rec.support.family, "carbon")


class TestNormalization(unittest.TestCase):
    def test_cs_factor(self):
        rec = MaterialRecord.from_dict({
            "record_id": "r4",
            "conditions": {"voltage_window": "1.2", "nacl_mg_L": 500.0},
            "performance": {"sac_mg_g": 20.0, "charge_efficiency": 0.5}})
        const = FactorConstants()
        val = cs_factor(rec, const)
        self.assertIsNotNone(val)
        self.assertAlmostEqual(val, 1.0, places=6)

    def test_cs_factor_missing_sac(self):
        rec = MaterialRecord(record_id="r5")
        self.assertIsNone(cs_factor(rec, FactorConstants()))


class TestML(unittest.TestCase):
    def _toy(self, n=40):
        rng = np.random.default_rng(1)
        x1 = rng.uniform(0, 1, n)
        x2 = rng.uniform(0, 1, n)
        y = 3 * x1 - x2 + rng.normal(0, 0.05, n)
        return pd.DataFrame({"f1": x1, "f2": x2, "y": y})

    def test_loocv_shapes(self):
        from sklearn.linear_model import LinearRegression
        df = self._toy()
        X = df[["f1", "f2"]].to_numpy()
        y = df["y"].to_numpy()
        res = loocv(LinearRegression(), X, y)
        self.assertEqual(len(res["loo_predictions"]), len(y))
        self.assertGreater(res["r2_loo"], 0.9)

    def test_pool_benchmark(self):
        df = self._toy()
        report = run_model_pool_benchmark(df, ["f1", "f2"], "y", seed=0)
        self.assertIn("GPR", report["benchmark"])
        best = report["best_model_name"]
        self.assertGreater(report["benchmark"][best]["r2_loo"], 0.5)
        pred = report["best_model"]["predict_fn"](np.array([[0.5, 0.5]]))
        self.assertEqual(pred.shape, (1,))

    def test_certify_outlier(self):
        rng = np.random.default_rng(2)
        resid = rng.normal(0, 1.0, 200)
        cert = certify_outlier(resid, mu=10.0, sigma=1.0,
                               y_observed=14.5, label="toy")
        self.assertTrue(cert["certified_outlier"])
        self.assertGreater(cert["z_score_point"], 2.0)
        ok = certify_outlier(resid, mu=10.0, sigma=1.0,
                             y_observed=10.5, label="toy2")
        self.assertFalse(ok["certified_outlier"])


class TestAgents(unittest.TestCase):
    def test_extract_json(self):
        text = 'prefix ```json {"a": 1} ``` suffix'
        self.assertEqual(BaseAgent.extract_json(text), {"a": 1})
        self.assertEqual(BaseAgent.extract_json('[1, 2]'), [1, 2])
        with self.assertRaises(ValueError):
            BaseAgent.extract_json("no json here")

    def test_data_agent_pipeline(self):
        reply_json = json.dumps({
            "record_id": "r6",
            "active_center": {"element": ["Fe"], "form": "cluster"},
            "support": {"family": "MXene", "name": "Ti3C2Tx"},
            "performance": {"sac_mg_g": 50.0},
            "provenance": {"doi": "10.1/y",
                           "evidence_quote": "The SAC reached 50 mg/g."}})
        agent = DataAgent(MockLLM(default=reply_json))
        state = SharedState()
        bus = MessageBus(state, "data")
        reply = bus.ask(agent, "extract", payload={"text": "some text"})
        self.assertIsNotNone(reply.payload["record"])
        self.assertEqual(reply.payload["qa_issues"], [])
        self.assertEqual(len(state.channel_logs["data"]), 1)

    def test_mechanism_matrix(self):
        script = {
            "H-OH": json.dumps({"verdict": "fail",
                                "reason": "no quartet"}),
            "H-1O2": json.dumps({"verdict": "pass",
                                 "reason": "triplet match"}),
        }

        class _M(MockLLM):
            def complete(self, system, messages):
                latest = messages[-1]["content"]
                for hid, rep in script.items():
                    if f"Mechanism id: {hid}" in latest:
                        from cdi_agents.llm import LLMResponse
                        return LLMResponse(content=rep, model="mock")
                return super().complete(system, messages)

        state = SharedState()
        agent = MechanismAgent(_M(default=json.dumps(
            {"verdict": "na", "reason": "unrelated"})))
        agent.build_library(state, extra=[{
            "hid": "H-NOVEL",
            "statement": "concerted interfacial pathway",
            "falsifiable_traits": ["trait a"]}])
        state.observations["OBS-1"] = {
            "description": "TEMP triplet observed, no DMPO-OH quartet",
            "source": "own lab"}
        matrix = agent.run_matrix(state)
        self.assertEqual(matrix["H-OH"]["OBS-1"], "fail")
        self.assertEqual(matrix["H-1O2"]["OBS-1"], "pass")
        self.assertEqual(state.hypotheses["H-OH"].status, "eliminated")
        self.assertIn("H-1O2",
                      [h.hid for h in state.hypotheses.values()
                       if h.status in ("survivor", "candidate")])


class TestStateBus(unittest.TestCase):
    def test_confirmation_log(self):
        state = SharedState()
        self.assertTrue(state.require_confirmation("t1"))
        self.assertFalse(state.require_confirmation("t2",
                                                    lambda t: False))
        self.assertEqual(len(state.confirmations), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
