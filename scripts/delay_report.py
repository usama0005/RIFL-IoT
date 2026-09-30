#!/usr/bin/env python
"""Delay sweep report: accuracy/reward/regret per method x condition, plus drift-only
adaptation_steps/forgetting (the metrics most relevant to the delay x drift hypothesis).
Usage: python scripts/delay_report.py [results/phase10_delay/summary.csv]
"""
import csv
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/phase10_delay/summary.csv"
rows = list(csv.DictReader(open(path)))
METHODS = ["B2_naive", "B5_blind_mask", "B4_rifl_iot"]
STAT_CONDS = ["stat_d1", "stat_d5", "stat_d10", "stat_d20"]
DRIFT_CONDS = ["drift_d0", "drift_d1", "drift_d5", "drift_d10", "drift_d20"]
METRICS = ["accuracy", "mean_r_norm", "exp_regret"]
DRIFT_METRICS = ["adaptation_steps", "forgetting", "before_drift", "after_drift"]

by = defaultdict(list)
for r in rows:
    by[(r["method"], r["condition"])].append(r)

def show(conds, extra_metrics=None):
    ms = METRICS + (extra_metrics or [])
    print(f"{'condition':<12}{'method':<16}" + "".join(f"{m:>16}" for m in ms))
    for cond in conds:
        for method in METHODS:
            rs = by.get((method, cond), [])
            if not rs:
                continue
            line = f"{cond:<12}{method:<16}"
            for m in ms:
                xs = [float(r[m]) for r in rs if r.get(m) not in (None, "", "nan")]
                line += f"{np.mean(xs):>9.3f}+-{np.std(xs):<5.3f}" if xs else f"{'n/a':>16}"
            print(line)
        print()

print(f"{'='*100}\nSTATIONARY (expect B4 approx= B2 throughout -- delay w/o drift only discards info)\n{'='*100}")
show(STAT_CONDS)
print(f"\n{'='*100}\nDRIFT (expect growing B4 advantage as delay increases)\n{'='*100}")
show(DRIFT_CONDS, DRIFT_METRICS)
