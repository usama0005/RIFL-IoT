#!/usr/bin/env python
"""Paired significance tests across B2/B5/B4 for the full condition set (noise + all negative
controls). For each condition: D_s = B4(s) - B2(s) and D_s = B4(s) - B5(s), paired per seed
(absolute performance, not noise-clean deltas, since some conditions like symflip100 have no
natural "clean" analog pairing -- clean itself is included as a condition for reference).
Usage: python scripts/significance_test_b5.py
"""
import csv
from collections import defaultdict

import numpy as np
from scipy import stats

FILES = ["results/phase9_b5_ablation/summary.csv", "results/phase9b_symflip100/summary.csv"]
rows = []
for f in FILES:
    try:
        rows += list(csv.DictReader(open(f)))
    except FileNotFoundError:
        pass
by = {(r["method"], r["condition"], r["seed"]): r for r in rows}
seeds = sorted(set(r["seed"] for r in rows), key=int)
CONDS = ["clean", "noise10", "noise30", "noise50", "ctrl_symflip50", "ctrl_symflip100", "ctrl_corr30"]
METRICS = [("accuracy", "higher_better"), ("mean_r_norm", "higher_better"), ("exp_regret", "lower_better")]

for cond in CONDS:
    present = [(m, s) for m in ["B2_naive", "B5_blind_mask", "B4_rifl_iot"] for s in seeds if (m, cond, s) in by]
    if not present:
        continue
    print(f"\n{'='*90}\n{cond}\n{'='*90}")
    for metric, direction in METRICS:
        vals = {m: [] for m in ["B2_naive", "B5_blind_mask", "B4_rifl_iot"]}
        for s in seeds:
            ok = all((m, cond, s) in by for m in vals)
            if not ok:
                continue
            for m in vals:
                vals[m].append(float(by[(m, cond, s)][metric]))
        if not vals["B4_rifl_iot"]:
            continue
        b2, b5, b4 = map(np.array, (vals["B2_naive"], vals["B5_blind_mask"], vals["B4_rifl_iot"]))
        print(f"\n--- {metric} ({direction}) ---")
        print(f"  B2 mean={b2.mean():.4f}  B5 mean={b5.mean():.4f}  B4 mean={b4.mean():.4f}  (n={len(b2)})")
        for label, other in [("B4 vs B2", b2), ("B4 vs B5", b5)]:
            D = b4 - other
            if direction == "lower_better":
                D = -D  # flip so positive always means "B4 better"
            t_stat, t_p = stats.ttest_rel(b4, other)
            try:
                w_stat, w_p = stats.wilcoxon(b4 - other)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
            n = len(D)
            se = D.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            tcrit = stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
            ci = (D.mean() - tcrit * se, D.mean() + tcrit * se) if n > 1 else (float("nan"), float("nan"))
            print(f"  {label}: mean_diff(B4-better-positive)={D.mean():.4f} sd={D.std(ddof=1):.4f} "
                 f"95%CI=[{ci[0]:.4f},{ci[1]:.4f}]  t_p={t_p:.4f}  wilcoxon_p={w_p:.4f}")
