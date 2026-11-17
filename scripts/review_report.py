#!/usr/bin/env python
"""Generic accuracy/reward table for any review-decision experiment's summary.csv.
Usage: python scripts/review_report.py results/<experiment_id>/summary.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np

path = sys.argv[1]
rows = list(csv.DictReader(open(path)))
by = defaultdict(list)
for r in rows:
    by[(r["method"], r["condition"])].append(r)

methods = sorted(set(r["method"] for r in rows))
conditions = sorted(set(r["condition"] for r in rows))
print(f"{'condition':<22}{'method':<16}{'accuracy':>14}{'reward':>14}{'regret':>14}")
for cond in conditions:
    for m in methods:
        rs = by.get((m, cond), [])
        if not rs:
            continue
        acc = np.array([float(r["accuracy"]) for r in rs])
        rew = np.array([float(r["mean_r_norm"]) for r in rs])
        reg = np.array([float(r["exp_regret"]) for r in rs])
        print(f"{cond:<22}{m:<16}{acc.mean()*100:>9.1f}+-{acc.std(ddof=1)*100:<4.1f}"
             f"{rew.mean():>10.4f}+-{rew.std(ddof=1):<3.4f}{reg.mean():>9.2f}+-{reg.std(ddof=1):<4.2f}")
