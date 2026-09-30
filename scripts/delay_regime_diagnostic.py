#!/usr/bin/env python
"""Does B4's rho encode temporal relevance, or only content consistency? Splits every used
feedback event by same-phase vs cross-phase (origin_phase vs the phase at actual delivery),
across delay 1/5/10/20 under drift, aggregated over seeds 0-4. Compares B4 (content-aware rho)
against B5 (blind-mask rho, same temporal weighting) at the effective-alpha level.
Usage: python scripts/delay_regime_diagnostic.py
"""
import json
import os
from collections import defaultdict

import numpy as np

DELAYS = [1, 5, 10, 20]
SEEDS = [0, 1, 2, 3, 4]
ROOT = "runs/phase10_delay"


def load_steps_phase(run_dir):
    return {json.loads(l)["t"]: json.loads(l)["phase"] for l in open(f"{run_dir}/steps.jsonl")}


def load(run_dir):
    fb = {json.loads(l)["origin_t"]: json.loads(l) for l in open(f"{run_dir}/feedback.jsonl")}
    proc = [json.loads(l) for l in open(f"{run_dir}/feedback_processing.jsonl") if json.loads(l)["used"]]
    phase = load_steps_phase(run_dir)
    rows = []
    for r in proc:
        g = fb.get(r["origin_t"])
        if g is None or r["origin_t"] not in phase or r["arrival_t"] not in phase:
            continue
        rows.append({
            "rho": r["rho"], "alpha_t": r["alpha_t"], "delta": r.get("delta", r["arrival_t"] - r["origin_t"]),
            "v_range": r.get("v_range"), "v_stuck": r.get("v_stuck"), "v_agree": r.get("v_agree"),
            "corrupted": g["corrupted_A"], "r_norm_true": g["r_norm_true"],
            "origin_phase": phase[r["origin_t"]], "arrival_phase": phase[r["arrival_t"]],
        })
    return rows


print(f"{'='*100}\nGRID: mean rho by delay x same/cross-phase (B4, aggregated over seeds 0-4)\n{'='*100}")
print(f"{'delay':<8}{'same_phase_rho':>18}{'n_same':>10}{'cross_phase_rho':>18}{'n_cross':>10}")
grid = {}
for d in DELAYS:
    all_rows = []
    for s in SEEDS:
        run_dir = f"{ROOT}/B4_rifl_iot/drift_d{d}/seed_{s}"
        if os.path.isdir(run_dir):
            all_rows += load(run_dir)
    grid[d] = all_rows
    same = [r for r in all_rows if r["origin_phase"] == r["arrival_phase"]]
    cross = [r for r in all_rows if r["origin_phase"] != r["arrival_phase"]]
    rs = np.array([r["rho"] for r in same]) if same else np.array([])
    rc = np.array([r["rho"] for r in cross]) if cross else np.array([])
    print(f"{d:<8}{(rs.mean() if len(rs) else float('nan')):>18.4f}{len(rs):>10}"
         f"{(rc.mean() if len(rc) else float('nan')):>18.4f}{len(rc):>10}")

print(f"\n{'='*100}\nCross-phase rho distribution and correlation with r_norm_true, by delay\n{'='*100}")
for d in DELAYS:
    all_rows = grid[d]
    cross = [r for r in all_rows if r["origin_phase"] != r["arrival_phase"]]
    if not cross:
        print(f"delay={d}: no cross-phase events")
        continue
    rho = np.array([r["rho"] for r in cross])
    r_true = np.array([r["r_norm_true"] for r in cross])
    vals, counts = np.unique(np.round(rho, 3), return_counts=True)
    corr = np.corrcoef(rho, r_true)[0, 1] if len(rho) > 2 else float("nan")
    print(f"delay={d}  n_cross={len(cross)}  rho_dist={dict(zip(vals.tolist(), counts.tolist()))}  "
         f"corr(rho, r_norm_true)={corr:.4f}")

print(f"\n{'='*100}\nComponent breakdown at d20 (same vs cross phase): which component drives rho?\n{'='*100}")
all_rows = grid[20]
for label, mask_fn in [("same-phase", lambda r: r["origin_phase"] == r["arrival_phase"]),
                       ("cross-phase", lambda r: r["origin_phase"] != r["arrival_phase"])]:
    sub = [r for r in all_rows if mask_fn(r)]
    if not sub:
        continue
    v_range = np.mean([r["v_range"] for r in sub])
    v_stuck = np.mean([r["v_stuck"] for r in sub])
    v_agree = np.mean([r["v_agree"] for r in sub])
    print(f"  {label} (n={len(sub)}): mean v_range={v_range:.4f}  v_stuck={v_stuck:.4f}  v_agree={v_agree:.4f}")

print(f"\n{'='*100}\nEffective alpha_t: B4 (content-aware) vs B5 (blind), same/cross phase, by delay\n{'='*100}")
print(f"{'delay':<8}{'B4_same_alpha':>16}{'B4_cross_alpha':>16}{'B5_same_alpha':>16}{'B5_cross_alpha':>16}")
for d in DELAYS:
    b4_rows = grid[d]
    b5_rows = []
    for s in SEEDS:
        run_dir = f"{ROOT}/B5_blind_mask/drift_d{d}/seed_{s}"
        if os.path.isdir(run_dir):
            b5_rows += load(run_dir)
    def m(rows, same):
        sub = [r["alpha_t"] for r in rows if (r["origin_phase"] == r["arrival_phase"]) == same]
        return np.mean(sub) if sub else float("nan")
    print(f"{d:<8}{m(b4_rows,True):>16.5f}{m(b4_rows,False):>16.5f}{m(b5_rows,True):>16.5f}{m(b5_rows,False):>16.5f}")
