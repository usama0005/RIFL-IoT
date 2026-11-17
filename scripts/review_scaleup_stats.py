#!/usr/bin/env python
"""Paired B4 vs B2 and B4 vs B5 stats at n=50 seeds -- same metrics as before, now with
real statistical power (exact Wilcoxon floor at n=50 is 2/2^50, effectively zero, so a
p-value here is meaningful rather than hitting the n=5 floor of .0625).
Usage: python scripts/review_scaleup_stats.py
"""
import csv
from collections import defaultdict

import numpy as np
from scipy import stats

rows = list(csv.DictReader(open("results/review_seed_scaleup/summary.csv")))
by = {(r["method"], r["condition"], r["seed"]): r for r in rows}
seeds = sorted(set(r["seed"] for r in rows), key=int)
CONDS = ["clean", "noise10", "noise30", "noise50", "correlated30", "symflip100", "conflict30"]
METHODS = ["B2_naive", "B5_blind_mask", "B4_rifl_iot"]

print(f"{'condition':<14}{'B2':>14}{'B5':>14}{'B4':>14}")
for cond in CONDS:
    line = f"{cond:<14}"
    means = {}
    for m in METHODS:
        xs = [float(by[(m, cond, s)]["accuracy"]) for s in seeds if (m, cond, s) in by]
        means[m] = np.array(xs)
        line += f"{np.mean(xs)*100:>8.2f}+-{np.std(xs,ddof=1)*100:<4.2f}"
    print(line)

print(f"\n{'='*90}\nPaired stats (accuracy), n=50\n{'='*90}")
for cond in CONDS:
    b2 = np.array([float(by[("B2_naive", cond, s)]["accuracy"]) for s in seeds if ("B2_naive", cond, s) in by])
    b5 = np.array([float(by[("B5_blind_mask", cond, s)]["accuracy"]) for s in seeds if ("B5_blind_mask", cond, s) in by])
    b4 = np.array([float(by[("B4_rifl_iot", cond, s)]["accuracy"]) for s in seeds if ("B4_rifl_iot", cond, s) in by])
    if len(b4) == 0:
        continue
    for label, other in [("B4-B2", b2), ("B4-B5", b5)]:
        if len(other) != len(b4):
            continue
        d = b4 - other
        t_p = stats.ttest_rel(b4, other).pvalue
        try:
            w_p = stats.wilcoxon(d).pvalue
        except ValueError:
            w_p = float("nan")
        ci = stats.t.interval(0.95, len(d)-1, loc=d.mean(), scale=stats.sem(d)) if len(d) > 1 else (float("nan"),)*2
        print(f"{cond:<14}{label}: mean_diff={d.mean()*100:+.2f}pp  95%CI=[{ci[0]*100:.2f},{ci[1]*100:.2f}]  t_p={t_p:.2e}  wilcoxon_p={w_p:.2e}")
