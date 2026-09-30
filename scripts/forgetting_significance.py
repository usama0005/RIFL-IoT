#!/usr/bin/env python
"""Paired significance test on 'forgetting' specifically -- the mechanism-targeted outcome for
the temporal-credit hypothesis -- across drift_d1/d5/d10/d20. B4 vs B2 and B4 vs B5, paired per
seed. Also prints the B4-B2 diff trend across delay for a monotonicity check.
Usage: python scripts/forgetting_significance.py [results/phase10_delay/summary.csv]
"""
import csv
import sys

import numpy as np
from scipy import stats

path = sys.argv[1] if len(sys.argv) > 1 else "results/phase10_delay/summary.csv"
rows = list(csv.DictReader(open(path)))
by = {(r["method"], r["condition"], r["seed"]): r for r in rows}
seeds = sorted(set(r["seed"] for r in rows), key=int)
CONDS = ["drift_d1", "drift_d5", "drift_d10", "drift_d20"]
METHODS = ["B2_naive", "B5_blind_mask", "B4_rifl_iot"]

trend = []
for cond in CONDS:
    print(f"\n{'='*90}\n{cond}  (metric: forgetting)\n{'='*90}")
    vals = {m: [] for m in METHODS}
    for s in seeds:
        if all((m, cond, s) in by for m in METHODS):
            for m in METHODS:
                v = by[(m, cond, s)]["forgetting"]
                vals[m].append(float(v) if v not in ("", None) else float("nan"))
    b2, b5, b4 = (np.array(vals[m]) for m in METHODS)
    print(f"  per-seed: B2={np.round(b2,4).tolist()}  B5={np.round(b5,4).tolist()}  B4={np.round(b4,4).tolist()}")
    print(f"  mean: B2={b2.mean():.4f}  B5={b5.mean():.4f}  B4={b4.mean():.4f}  (n={len(b2)})")

    for label, other in [("B4 vs B2", b2), ("B4 vs B5", b5)]:
        D = other - b4  # positive = B4 has LOWER forgetting = B4 better (forgetting: lower is better)
        t_stat, t_p = stats.ttest_rel(other, b4)
        try:
            w_stat, w_p = stats.wilcoxon(D, mode="exact")
        except Exception:
            try:
                w_stat, w_p = stats.wilcoxon(D)
            except ValueError:
                w_stat, w_p = float("nan"), float("nan")
        n = len(D)
        se = D.std(ddof=1) / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        ci = (D.mean() - tcrit * se, D.mean() + tcrit * se)
        print(f"  {label}: mean_diff(positive=B4 better)={D.mean():.4f}  sd={D.std(ddof=1):.4f}  "
             f"95%CI=[{ci[0]:.4f},{ci[1]:.4f}]  t_p={t_p:.4f}  wilcoxon_p={w_p:.4f}")
    trend.append((cond, (b2 - b4).mean(), (b2 - b4).std(ddof=1)))

print(f"\n{'='*90}\nTREND: B2-B4 forgetting diff (positive = B4 lower/better) across delay\n{'='*90}")
for cond, m, sd in trend:
    print(f"  {cond:<12} mean_diff={m:.4f}  sd={sd:.4f}")
