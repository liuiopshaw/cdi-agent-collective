# CDI Agent Collective

A lightweight multi-agent LLM framework for mechanism-certification-driven
materials research, designed for capacitive deionization (CDI) electrodes of
the "support (ligand) x active center" family.

The framework inverts the usual screening paradigm: instead of predicting
many candidate materials, it uses literature-trained models as a calibrated
baseline of *known physics*, then certifies that a held-out experimental
system (e.g. Fe nanoclusters on MXene) is a statistical outlier whose
residual quantifies the contribution of a *new* mechanism. A dedicated
mechanism pipeline then systematically falsifies all known ROS/RCS
generation pathways against experimental observations, so that the proposed
novel mechanism survives as the only consistent hypothesis.

## Design references

- Guo et al., Science 2026 (four-agent framework for perovskite solar cells):
  role separation, two-round orchestration, human confirmation points.
- PeroMAS (KDD 2026): hierarchical agent evaluation, dual LLM/expert
  validity scoring, multi-backbone robustness, single-agent ablation.
- Cheng et al., RSC Adv. 2026: residual-based outlier isolation as a
  discovery instrument; non-random clustering of residuals as validation.

## Architecture

```
                    +------------------------------+
                    |   A0 Central Agent (planner) |
                    +-------+-----------+----------+
            +---------------+           |
            v               v           v
   +---------+--+  +--------+------+  +-+--------------+
   | A1 Data    |  | A2 Prediction |  | A3 Mechanism   |
   | extraction |  | anchored GPR  |  | falsification  |
   +-----+------+  +--------+------+  +--------+-------+
        |                   |                  |
        |                   |           +------v-------+
        |                   |           | A4 Critic    |
        |                   |           | adversarial  |
        |                   |           +------+-------+
        v                   v                  v
   +-----------------------------------------------------+
   | Shared state S = { dataset D, hypothesis table H,    |
   |   model cache Theta, residual report R, logs L }     |
   +-----------------------------------------------------+
                            |
              +-------------+-------------+
              v                           v
   Held-out own data (vault)     A5 Blind Probe (optional)
```

## Pipelines

| Pipeline | Purpose | Key outputs |
|---|---|---|
| P1 Corpus | Literature extraction into a three-layer descriptor database | versioned dataset, extraction QA report |
| P2 Anchored prediction | Calibrated baseline, anchor blind test, residual outlier certification | calibration report, outlier certificate |
| P3 Mechanism | Hypothesis library, mechanism x observation falsification matrix, adversarial Q&A | elimination matrix, challenge-response table |
| P4 Evaluation | Atomic task accuracy, dual judges, multi-backbone, single-agent ablation | evaluation report |

## Hard rules (enforced by design)

- T1 Anchor isolation: the user's own experimental data never enters any
  training set; it is used only for blind validation.
- T2 Evidence provenance: every database record and every falsification
  verdict must carry a verbatim evidence quote and a source pointer.
- T3 Auditability: all inter-agent messages, prompts, model versions and
  random seeds are logged and exportable for supplementary material.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Only `numpy`, `pandas`, `scikit-learn`, `scipy` and `requests` are required.
`xgboost` is optional and auto-detected.

## Configure

```bash
cp config.example.yaml config.yaml
# edit provider base_url / api_key / model names
export CDI_LLM_API_KEY=sk-...
```

Any OpenAI-compatible endpoint works (OpenAI, DeepSeek, Qwen, local vLLM).

## Quick start (offline demo, no API key)

```bash
python examples/run_demo.py
```

The demo runs the full A0-A4 loop with a deterministic mock LLM plus a real
sklearn model pool on synthetic data, and prints the calibration report,
outlier certificate and elimination matrix.

## Phase 1 batch (literature -> dataset)

```bash
# offline dry run (mock LLM, checks plumbing only)
python scripts/run_p1.py --pdf-dir data/pdfs --mock --limit 5

# real run: name PDFs by DOI slug, then
python scripts/run_p1.py --pdf-dir data/pdfs --config config.yaml
```

Outputs land in `results/p1_<timestamp>/`: extracted texts, the
versioned `dataset.json`, `quarantine.json` and a `qa_report.md` with a
manual-review checklist. Pre-extracted `.txt` folders are accepted via
`--text-dir`.

## Tool layer (optional, additive)

Agents can use OpenAI-style function calling through a lightweight
registry. Execution is always deterministic code; every dispatch is
logged (`registry.call_log`, hard rule T3); tool failures return an
error payload instead of raising so the reasoning loop can recover.

```python
from cdi_agents.tools_builtin import default_registry

registry = default_registry(corpus_dir="results/p1_xxx/texts",
                            dataset_path="results/p1_xxx/dataset.json")
answer = agent.call_with_tools("your question", registry)
```

Built-in tools: `corpus_search` (keyword retrieval over local extracted
texts), `dataset_query` (filter the descriptor dataset by family /
element), `compute_factor` (C_s / R_s), `train_model` (full model-pool
LOOCV benchmark on a CSV). Custom tools are three-line `Tool` objects;
register only what an agent needs (least privilege). Try it offline:

```bash
python examples/run_tools_demo.py
```

## Layout

```
src/cdi_agents/
  config.py            model assignment and temperatures
  llm.py               OpenAI-compatible client with retry/backoff
  state.py             shared session state (dataset, hypotheses, logs)
  bus.py               channel message bus with full logging
  agents/              A0 central, A1 data, A2 prediction,
                       A3 mechanism, A4 critic, A5 blind probe
  descriptors/         three-layer descriptor schema, C_s / R_s factors
  ml/                  model pool, LOOCV, family holdout, residual analysis
  pipelines/           P1 corpus, P2 anchored, P3 mechanism, P4 evaluation
  prompts/             versioned prompt templates
tests/run_tests.py     stdlib unittest suite
examples/run_demo.py   offline end-to-end demo
```

## Citation

If you use this framework, please cite the companion manuscript (in
preparation) and the three design references above.
