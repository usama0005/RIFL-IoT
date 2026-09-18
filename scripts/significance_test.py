#!/usr/bin/env python
"""Paired B4-vs-B2 significance test on per-seed noise-clean deltas (DESIGN.md's protocol).
D_s = Delta_B4(s) - Delta_B2(s), where Delta_m(s) = m_noise(s) - m_clean(s).
Reports paired t-test, Wilcoxon signed-rank, mean/SD/95% CI, and the raw per-seed values.
Usage: python scripts/significance_test.py [results/phase7b_noise_tuned/summary.csv]
"""
import csv
import sys

import numpy as np
from scipy import stats

path = sys.argv[1] if len(sys.argv) > 1 else "results/phase7b_noise_tuned/summary.csv"
rows = list(csv.DictReader(open(path)))
by = {(r["method"], r["condition"], r["seed"]): r for r in rows}
seeds = sorted(set(r["seed"] for r in rows), key=int)
NOISE_CONDS = ["noise10", "noise30", "noise50"]
METRICS = [("accuracy", "higher_better"), ("mean_r_norm", "higher_better"), ("exp_regret", "lower_better")]

for cond in NOISE_CONDS:
    print(f"\n{'='*80}\n{cond}\n{'='*80}")
    for metric, direction in METRICS:
        deltas = {"B2_naive": [], "B4_rifl_iot": []}
        for s in seeds:
            for meth in deltas:
                c, n = by.get((meth, "clean", s)), by.get((meth, cond, s))
                deltas[meth].append(float(n[metric]) - float(c[metric]))
        d_b2 = np.array(deltas["B2_naive"])
        d_b4 = np.array(deltas["B4_rifl_iot"])
        D = d_b4 - d_b2  # positive = B4 degraded less, EXCEPT for exp_regret where negative = B4 degraded less

        t_stat, t_p = stats.ttest_rel(d_b4, d_b2)
        try:
            w_stat, w_p = stats.wilcoxon(D)
        except ValueError as e:
            w_stat, w_p = float("nan"), float("nan")
        mean_D, sd_D = D.mean(), D.std(ddof=1)
        n = len(D)
        se = sd_D / np.sqrt(n)
        tcrit = stats.t.ppf(0.975, df=n - 1)
        ci_lo, ci_hi = mean_D - tcrit * se, mean_D + tcrit * se

        good = ">0 (B4 degrades less)" if direction == "higher_better" else "<0 (B4 degrades less)"
        print(f"\n--- {metric} ({direction}; D_s {good}) ---")
        print(f"  per-seed: B2_delta={np.round(d_b2,4).tolist()}  B4_delta={np.round(d_b4,4).tolist()}")
        print(f"  D_s (B4-B2) per seed: {np.round(D,4).tolist()}")
        print(f"  mean D = {mean_D:.4f}, sd D = {sd_D:.4f}, n = {n}")
        print(f"  95% CI = [{ci_lo:.4f}, {ci_hi:.4f}]")
        print(f"  paired t-test: t={t_stat:.4f}, p={t_p:.4f}")
        print(f"  Wilcoxon signed-rank: W={w_stat}, p={w_p:.4f}" if w_p == w_p else "  Wilcoxon: undefined (n too small / all-zero diffs)")
