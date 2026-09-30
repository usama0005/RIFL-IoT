# RIFL-IoT — Reliability-aware Implicit Feedback Learning for IoT LLM Agents

Research prototype for the paper investigating **reliability-aware continual adaptation of
LLM-guided IoT decisions from uncertain implicit physical-world feedback**. A frozen LLM
interprets sensor observations and proposes candidate actions; an external Learning Automaton
selects the executed action and adapts online from noisy, delayed, missing, and conflicting
operational feedback. Full design spec: [`DESIGN.md`](DESIGN.md).

## Status

| Phase | State |
|---|---|
| 1 Environment + B0 vanilla LLM | done |
| 2 Validation / reproducibility | done -- `scripts/validate_env.py`, replay-hash assertion in every run |
| 3 B1 (history) + B2 (naive feedback) | done |
| 5 Learning Automaton variant selection + alpha-tuning | done -- Pursuit frozen, alpha=0.04 (`results/phase5_la_variant_selection`, `results/phase5b_alpha_grid`) |
| 6 B4 (RIFL-IoT) implementation | done |
| 7-9 Noise sweep, B5 blind-mask ablation, negative controls | done |
| 10 Delay sweep + diagnostics | done |
| 12 Final matrix: missing / conflict / drift+noise / drift+delay | done |
| Static benchmark (S0, IoT-LLM protocol) | done -- `scripts/s0_static_benchmark.py` |
| Final statistics (paired t-test, Wilcoxon, reliability AUROC/F1/ECE) | done |

All primary results use **frozen** hyperparameters (Pursuit, alpha=0.04, kappa=0.2, tau=0.5,
half-life=10, omega=0.5, p_min=0.01) selected on validation seeds 100-102 before any corruption
experiment was run, then evaluated on independent test seeds 0-4. See `DESIGN.md` and the
paper's Experimental Setup section for the full protocol.

## Compared methods

| ID | Method | Purpose |
|---|---|---|
| B0 | Vanilla LLM | No adaptation |
| B1 | LLM + history | In-context adaptation only, no persistent learner |
| B2 | Naive feedback | Feedback assumed reliable (rho=w=1) |
| B4 | **RIFL-IoT** | Reliability- and delay-weighted adaptation (proposed) |
| B5 | BlindMask | Content-blind random attenuation; isolates whether B4's gain is more than a step-size effect |

## Headline results (test seeds 0-4, decision accuracy %)

| Condition | B2 | B5 | B4 (RIFL-IoT) |
|---|---|---|---|
| Clean | 93.7 +/- 0.4 | 92.4 +/- 0.5 | 93.7 +/- 0.4 |
| Mixed noise 30% | 87.3 +/- 4.7 | 88.5 +/- 7.0 | **93.0 +/- 0.4** |
| Mixed noise 50% | 67.0 +/- 9.9 | 79.1 +/- 8.0 | **92.5 +/- 0.3** |
| Symmetric flip 100% (negative control) | 30.0 +/- 12.0 | 32.4 +/- 11.8 | 35.7 +/- 15.2 (flat, n.s.) |
| Coherent/correlated 30% (limitation) | 74.4 +/- 7.1 | **82.1 +/- 7.0** | 75.1 +/- 7.4 |

RIFL-IoT's reliability estimator rho_t separates corrupted from clean feedback well under
independent noise (AUROC 0.93, F1 0.91-0.92) and collapses to chance under coherent corruption
(AUROC 0.50) -- the mechanism succeeds exactly where it is designed to and fails exactly where
the paper's negative controls predict it should. Full statistics (paired t-tests, exact
Wilcoxon, 95% CIs) are in `results/*/summary_agg.csv` and the paper's Results section.

## Commands

```bash
pip install -r requirements.txt
python scripts/validate_env.py --config configs/smoke.yaml     # no LLM, structural checks only
python scripts/run.py --config configs/smoke.yaml              # synthetic data + stub LLM, ~seconds
bash scripts/download_uci_hydraulic.sh data/uci_hydraulic       # real data
python scripts/run.py --config configs/phase1_vanilla_uci.yaml  # needs a served OpenAI-compatible LLM
```

Every `configs/*.yaml` is a self-contained, reproducible experiment (method set, conditions,
seeds). `smoke*.yaml` configs use fabricated data for pipeline checks only -- the runner refuses
synthetic data outside `smoke_only: true`. Key experiment configs, in the order they were run:

- `phase5_la_variant_selection.yaml`, `phase5b_alpha_grid.yaml` -- LA selection and tuning (validation seeds)
- `phase6_clean_drift_b0_b2_b4.yaml`, `phase7b_noise_tuned.yaml` -- clean/drift checkpoint, noise sweep
- `phase8_negative_controls.yaml`, `phase9_b5_ablation.yaml`, `phase9b_symflip100.yaml` -- negative controls, B5 ablation
- `phase10_delay.yaml`, `ablation_a2_no_temporal.yaml` -- delay sweep and temporal-credit ablation
- `phase12_final_matrix.yaml` -- missing / conflict / drift+noise / drift+delay

Analysis scripts (`scripts/*_report.py`, `scripts/significance_test*.py`,
`scripts/reliability_quality_report.py`) reconstruct every table in the paper directly from the
raw per-run logs in `runs/`.

## Repository layout

```
rifl_iot/environment/   dataset.py (UCI loader, calibration split, window stats)
                        dynamics.py (phase reward tables, oracle)
                        trajectory.py (pre-sampled common-random-number streams)
                        env.py (step, cumulative feedback delivery)
                        calibration.py (reference ranges for the LLM prompt, calib-pool only)
rifl_iot/feedback/      generator.py (channels A/B; mixed/symmetric-flip/correlated corruption,
                        conflict, missingness, delay)
rifl_iot/llm/           prompts.py, parse.py, stub.py, openai_compat.py, cache.py
rifl_iot/agents/        base.py, vanilla.py (B0), history.py (B1), naive_feedback.py (B2),
                        rifl_iot_agent.py (B4), blind_mask.py (B5)
rifl_iot/learning/      automaton.py (LR-I / LR-P / LR-epsilonP / Pursuit), temporal_credit.py
rifl_iot/reliability/   estimator.py (rho = v_range * v_stuck * v_agree)
rifl_iot/evaluation/    metrics.py
rifl_iot/experiments/   runner.py
scripts/                run.py, validate_env.py, s0_static_benchmark.py, and all *_report.py /
                        *_diagnostic.py / significance_test*.py analysis scripts
```

## Data and external benchmark provenance

Environment sensor data: UCI *Condition Monitoring of Hydraulic Systems* (Helwig et al.).
Actions, costs, breakdown probabilities, and the corruption/drift mechanics are constructed --
see `DESIGN.md` for the exact real-vs-constructed boundary.

The static benchmark (`scripts/s0_static_benchmark.py`) reproduces the *baseline* protocol
(binary close-to-failure vs. full-efficiency classification) from Tuo An et al., **"IoT-LLM,"
Patterns** (2026), DOI 10.1016/j.patter.2025.101429, which uses the same dataset. This is a
same-protocol, different-model comparison (Qwen2.5-7B-Instruct-AWQ here vs. GPT-4o-mini in the
original) and is reported as such -- see the script's docstring and the paper's benchmark
section for exactly which parts of the protocol are and are not matched.

## Reproducibility notes

- Every run asserts that all compared methods received an identical environment trajectory for
  a given (seed, schedule) via a trajectory hash check (visible in each run's console output and
  `seeds.json`).
- LLM outputs are cached (`llm_cache.sqlite` per experiment); re-running an experiment with the
  same observations is close to free.
- All method hyperparameters were frozen on validation seeds 100-102 before test seeds 0-4 were
  touched; no parameter in the final configs was tuned against a corruption/delay/missing result.
