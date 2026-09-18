#!/usr/bin/env python
"""Alpha-grid comparison table (Pursuit, validation seeds 100-102, clean+drift).
Usage: python scripts/phase5b_report.py
"""
import csv
from collections import defaultdict

import numpy as np

ALPHAS = ["a002", "a004", "a010", "a020", "a040"]
METRICS = ["accuracy", "mean_r_norm", "exp_regret", "final_performance",
           "before_drift", "after_drift", "adaptation_steps", "forgetting"]

rows = list(csv.DictReader(open("results/phase5b_alpha_grid/summary.csv")))
by = defaultdict(list)
for r in rows:
    alpha, sched = r["condition"].rsplit("_", 1)
    by[(alpha, sched)].append(r)

for sched in ["clean", "drift"]:
    print(f"\n{'='*95}\n{sched.upper()}\n{'='*95}")
    print(f"{'alpha':<10}" + "".join(f"{m:>15}" for m in METRICS))
    for a in ALPHAS:
        rs = by[(a, sched)]
        if not rs:
            continue
        vals = []
        for m in METRICS:
            xs = [float(r[m]) for r in rs if r[m] not in ("", "nan")]
            vals.append(f"{np.mean(xs):.3f}\u00b1{np.std(xs):.3f}" if len(xs) > 1 else (f"{xs[0]:.3f}" if xs else "n/a"))
        print(f"{a:<10}" + "".join(f"{v:>15}" for v in vals))
