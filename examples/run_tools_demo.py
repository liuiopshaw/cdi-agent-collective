"""Offline demo of the tool layer: an agent that retrieves before answering.

A Mechanism Agent is asked about chloride-dependent activity. The mock
LLM first requests the corpus_search tool, receives real snippets from a
local corpus folder, and then answers grounded in the retrieved text.
Every dispatch is auditable in registry.call_log.

Usage:
    python examples/run_tools_demo.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cdi_agents.agents.mechanism import MechanismAgent
from cdi_agents.llm import MockLLM
from cdi_agents.tools_builtin import default_registry

CORPUS = {
    "10.1002_a.txt": ("The MXene anode captured chloride ions with "
                      "reversible intercalation at 1.2 V, delivering "
                      "17.6 mg g-1 in asymmetric CDI."),
    "10.1002_b.txt": ("Carbon cloth showed no chloride dependence in "
                      "the same potential window."),
}


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for name, text in CORPUS.items():
            (root / name).write_text(text, encoding="utf-8")

        registry = default_registry(corpus_dir=root, with_train=False)
        print("registered tools:",
              [s["function"]["name"] for s in registry.schemas()])

        script = {
            "chloride-dependent": {
                "content": "",
                "tool_calls": [{
                    "id": "call_1",
                    "name": "corpus_search",
                    "arguments": {"query": "MXene chloride", "limit": 3}}],
            },
        }
        final_answer = ("Based on retrieved record 10.1002_a: MXene "
                        "anodes show reversible chloride intercalation "
                        "(17.6 mg g-1), supporting a chloride-storage "
                        "mechanism rather than free-chlorine chemistry.")
        agent = MechanismAgent(MockLLM(script=script,
                                       default=final_answer))

        answer = agent.call_with_tools(
            "What evidence exists for chloride-dependent activity on "
            "MXene anodes? chloride-dependent",
            registry)
        print("\nagent answer:\n ", answer)
        print("\ntool call log:")
        for rec in registry.call_log:
            print(f"  {rec.name}({rec.arguments}) ok={rec.ok}")
            print(f"    -> {rec.result_preview[:120]}...")


if __name__ == "__main__":
    main()
