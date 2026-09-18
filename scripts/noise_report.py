#!/usr/bin/env python
"""Delta_noise = perf_noise - perf_clean per method (matched per seed), and Delta_B4 - Delta_B2.
Usage: python scripts/noise_report.py [results/phase7_noise_b0_b2_b4/summary.csv]
"""
import csv
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/phase7_noise_b0_b2_b4/summary.csv"
rows = list(csv.DictReader(open(path)))
METHODS = ["B0_vanilla", "B2_naive", "B4_rifl_iot"]
NOISE_CONDS = ["noise10", "noise30", "noise50"]
METRICS = ["accuracy", "mean_r_norm", "exp_regret"]

by = {(r["method"], r["condition"], r["seed"]): r for r in rows}
seeds = sorted(set(r["seed"] for r in rows))

print(f"{'='*100}\nPer-metric summary (mean +/- sd across seeds)\n{'='*100}")
header = f"{'method':<14}{'condition':<12}" + "".join(f"{m:>16}" for m in METRICS)
print(header)
for cond in ["clean"] + NOISE_CONDS:
    for method in METHODS:
        vals = [by.get((method, cond, s)) for s in seeds]
        vals = [v for v in vals if v]
        line = f"{method:<14}{cond:<12}"
        for m in METRICS:
            xs = [float(v[m]) for v in vals]
            line += f"{np.mean(xs):>9.3f}+-{np.std(xs):<5.3f}"
        print(line)
    print()

print(f"{'='*100}\nDelta_noise = perf(noise) - perf(clean), per seed then aggregated\n{'='*100}")
for m in METRICS:
    print(f"\n--- metric: {m} ---")
    header = f"{'condition':<12}" + "".join(f"{meth:>20}" for meth in METHODS) + f"{'  B4-B2 delta gap':>22}"
    print(header)
    for cond in NOISE_CONDS:
        deltas = {meth: [] for meth in METHODS}
        for s in seeds:
            for meth in METHODS:
                c, n = by.get((meth, "clean", s)), by.get((meth, cond, s))
                if c and n:
                    deltas[meth].append(float(n[m]) - float(c[m]))
        line = f"{cond:<12}"
        for meth in METHODS:
            d = deltas[meth]
            line += f"{np.mean(d):>13.3f}+-{np.std(d):<5.3f}" if d else f"{'n/a':>20}"
        if deltas["B4_rifl_iot"] and deltas["B2_naive"]:
            gap = np.array(deltas["B4_rifl_iot"]) - np.array(deltas["B2_naive"])
            line += f"{np.mean(gap):>15.3f}+-{np.std(gap):<5.3f}"
        print(line)

print(f"\n{'='*100}\nNote: for accuracy/reward, B4-B2 delta gap > 0 means B4 degrades LESS than B2 (good for B4).")
print("For exp_regret, B4-B2 delta gap < 0 means B4's regret grows LESS than B2's (good for B4).")
