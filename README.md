# RIFL-IoT — research prototype (Phase 1: environment + vanilla LLM)

Spec: `DESIGN.md`. Constructed environment parameters: `configs/env_default.yaml`.

## Status
| Phase | State |
|---|---|
| 1 Environment + B0 vanilla | done (stub + OpenAI-compatible backends) |
| 2 Validation / reproducibility | `scripts/validate_env.py` + runner replay assertion |
| 3–8 | not started |

## Commands
```bash
pip install -r requirements.txt
python scripts/validate_env.py --config configs/smoke.yaml     # no LLM
python scripts/run.py --config configs/smoke.yaml              # synthetic data + stub LLM, ~seconds
bash scripts/download_uci_hydraulic.sh data/uci_hydraulic      # real data
python scripts/run.py --config configs/phase1_vanilla_uci.yaml # needs a served model
```
`smoke` numbers are code-path checks only; the runner refuses synthetic data unless `smoke_only: true`.

## Layout
```
rifl_iot/environment/  dataset.py (UCI loader, split, window stats) dynamics.py (phases, oracle)
                       trajectory.py (pre-sampled CRN streams) env.py (step, delivery)
rifl_iot/feedback/     generator.py (channels A/B, noise, conflict, missing, delay)
rifl_iot/llm/          prompts, parser, stub, OpenAI-compatible client, sqlite cache
rifl_iot/agents/       base.py, vanilla.py (B0)
rifl_iot/evaluation/   metrics.py
rifl_iot/experiments/  runner.py
```
