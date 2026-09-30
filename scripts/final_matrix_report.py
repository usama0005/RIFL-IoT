#!/usr/bin/env python
"""Final matrix report: tables A (accuracy), B (reward), C (adaptation/forgetting for the two
drift conditions), D (paired statistics B4 vs B2, B4 vs B5), plus per-seed dump (E).
Usage: python scripts/final_matrix_report.py [results/phase12_final_matrix/summary.csv]
"""
import csv
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

path = sys.argv[1] if len(sys.argv) > 1 else "results/phase12_final_matrix/summary.csv"
rows = list(csv.DictReader(open(path)))
by = {(r["method"], r["condition"], r["seed"]): r for r in rows}
seeds = sorted(set(r["seed"] for r in rows), key=int)
CONDS = ["missing10", "missing30", "missing50", "conflict30", "drift_noise30", "drift_delay10"]
DRIFT_CONDS = ["drift_noise30", "drift_delay10"]
METHODS = ["B2_naive", "B5_blind_mask", "B4_rifl_iot"]


def get_vals(cond, metric):
    out = {m: [] for m in METHODS}
    for s in seeds:
        if all((m, cond, s) in by for m in METHODS):
            for m in METHODS:
                v = by[(m, cond, s)].get(metric)
                out[m].append(float(v) if v not in (None, "", "nan") else float("nan"))
    return out


print(f"\n{'='*100}\nTABLE A: Accuracy mean +/- SD\n{'='*100}")
print(f"{'condition':<16}{'B2':>14}{'B5':>14}{'B4':>14}")
for cond in CONDS:
    v = get_vals(cond, "accuracy")
    line = f"{cond:<16}"
    for m in METHODS:
        arr = np.array(v[m])
        line += f"{arr.mean()*100:>7.1f}+-{arr.std(ddof=1)*100:<5.1f}"
    print(line)

print(f"\n{'='*100}\nTABLE B: Reward (mean_r_norm) mean +/- SD\n{'='*100}")
print(f"{'condition':<16}{'B2':>14}{'B5':>14}{'B4':>14}")
for cond in CONDS:
    v = get_vals(cond, "mean_r_norm")
    line = f"{cond:<16}"
    for m in METHODS:
        arr = np.array(v[m])
        line += f"{arr.mean():>10.4f}+-{arr.std(ddof=1):<6.4f}"
    print(line)

print(f"\n{'='*100}\nTABLE C: Adaptation / forgetting (drift conditions only)\n{'='*100}")
for cond in DRIFT_CONDS:
    print(f"\n--- {cond} ---")
    for metric in ["before_drift", "after_drift", "adaptation_gain", "adaptation_steps", "forgetting"]:
        v = get_vals(cond, metric)
        line = f"  {metric:<20}"
        for m in METHODS:
            arr = np.array(v[m])
            line += f"{arr.mean():>10.3f}+-{arr.std(ddof=1):<6.3f}"
        print(line)

print(f"\n{'='*100}\nTABLE D: Paired statistics (B4 vs B2, B4 vs B5) -- accuracy, reward, regret\n{'='*100}")
for cond in CONDS:
    print(f"\n--- {cond} ---")
    for metric, direction in [("accuracy", "higher_better"), ("mean_r_norm", "higher_better"),
                              ("exp_regret", "lower_better")]:
        v = get_vals(cond, metric)
        b2, b5, b4 = np.array(v["B2_naive"]), np.array(v["B5_blind_mask"]), np.array(v["B4_rifl_iot"])
        for label, other in [("B4-B2", b2), ("B4-B5", b5)]:
            D = (b4 - other) if direction == "higher_better" else (other - b4)
            t_stat, t_p = stats.ttest_rel(b4, other)
            try:
                w_stat, w_p = stats.wilcoxon(b4 - other, mode="exact")
            except Exception:
                try:
                    w_stat, w_p = stats.wilcoxon(b4 - other)
                except ValueError:
                    w_stat, w_p = float("nan"), float("nan")
            n = len(D)
            se = D.std(ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            tcrit = stats.t.ppf(0.975, df=n - 1) if n > 1 else float("nan")
            ci = (D.mean() - tcrit * se, D.mean() + tcrit * se) if n > 1 else (float("nan"),) * 2
            print(f"  {metric:<14} {label}: diff(B4-better-positive)={D.mean():.4f}  "
                 f"95%CI=[{ci[0]:.4f},{ci[1]:.4f}]  t_p={t_p:.4f}  wilcoxon_p={w_p:.4f}")

print(f"\n{'='*100}\nTABLE E: Per-seed accuracy\n{'='*100}")
for cond in CONDS:
    v = get_vals(cond, "accuracy")
    print(f"{cond}: B2={np.round(v['B2_naive'],4).tolist()}  B5={np.round(v['B5_blind_mask'],4).tolist()}  "
         f"B4={np.round(v['B4_rifl_iot'],4).tolist()}")

print(f"\n{'='*100}\nHash check\n{'='*100}")
hashes = defaultdict(set)
for r in rows:
    hashes[(r["condition"], r["seed"])].add(r["trajectory_hash"])
bad = {k: v for k, v in hashes.items() if len(v) > 1}
print("all (condition,seed) pairs have a single shared trajectory hash across methods:" if not bad
     else f"MISMATCH in: {bad}")
