#!/usr/bin/env python
"""Phase 5 LA-variant comparison table from results/phase5_la_variant_selection/summary.csv.
Usage: python scripts/phase5_report.py
"""
import csv
from collections import defaultdict

import numpy as np

VARIANTS = ["LR_I", "LR_P", "LR_epsilonP", "Pursuit"]
METRICS = ["accuracy", "mean_r_norm", "exp_regret", "final_performance",
           "before_drift", "after_drift", "adaptation_steps", "forgetting"]

rows = list(csv.DictReader(open("results/phase5_la_variant_selection/summary.csv")))
by_variant_sched = defaultdict(list)
for r in rows:
    variant, sched = r["condition"].rsplit("_", 1)
    by_variant_sched[(variant, sched)].append(r)

for sched in ["clean", "drift"]:
    print(f"\n{'='*90}\n{sched.upper()}\n{'='*90}")
    header = f"{'variant':<14}" + "".join(f"{m:>16}" for m in METRICS)
    print(header)
    for v in VARIANTS:
        rs = by_variant_sched[(v, sched)]
        if not rs:
            continue
        vals = []
        for m in METRICS:
            xs = [float(r[m]) for r in rs if r[m] not in ("", "nan")]
            if xs:
                vals.append(f"{np.mean(xs):.3f}\u00b1{np.std(xs):.3f}" if len(xs) > 1 else f"{xs[0]:.3f}")
            else:
                vals.append("n/a")
        print(f"{v:<14}" + "".join(f"{v:>16}" for v in vals))
    print(f"  seeds used: {sorted(set(r['seed'] for r in rows))}")
