#!/usr/bin/env python
"""AUROC/F1/ECE of rho_t on the HARDER corruption suite (review decision 3), same method
as reliability_quality_report.py but pointed at review_mixed_hard logs, and also broken down
by corruption SUB-TYPE (inrange_shift, jitter_stuck, channelB_indep, replace) so we can see
exactly which new fault types the detector catches or misses.
Usage: python scripts/review_reliability_hard.py
"""
import json

import numpy as np

ROOT = "runs/review_mixed_hard/B4_rifl_iot"
CONDS = {"10%": "hard_noise10", "30%": "hard_noise30", "50%": "hard_noise50"}
SEEDS = [0, 1, 2, 3, 4]


def load(cond):
    y_true, rho_all, ctype_all = [], [], []
    for s in SEEDS:
        run_dir = f"{ROOT}/{cond}/seed_{s}"
        fb = {json.loads(l)["origin_t"]: json.loads(l) for l in open(f"{run_dir}/feedback.jsonl")}
        proc = [json.loads(l) for l in open(f"{run_dir}/feedback_processing.jsonl") if json.loads(l)["used"]]
        for r in proc:
            g = fb.get(r["origin_t"])
            if g is None:
                continue
            y_true.append(int(g["corrupted_A"]))
            rho_all.append(r["rho"])
            ctype_all.append(g["corruption_type"])
    return np.array(y_true), np.array(rho_all), np.array(ctype_all)


def auroc(y_true, score):
    pos, neg = score[y_true == 1], score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = score[order]
    i, n = 0, len(score)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    u = ranks[y_true == 1].sum() - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def f1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


print(f"{'noise':<8}{'n':>8}{'AUROC':>10}{'F1':>10}")
for label, cond in CONDS.items():
    y_true, rho, ctype = load(cond)
    a = auroc(y_true, 1 - rho)
    f = f1(y_true, (rho < 1.0).astype(int))
    print(f"{label:<8}{len(y_true):>8}{a:>10.4f}{f:>10.4f}")

print("\nPer-corruption-subtype detection (at 50% noise, pooled seeds):")
y_true, rho, ctype = load("hard_noise50")
for sub in ["inrange_shift", "jitter_stuck", "channelB_indep", "replace", "none"]:
    mask = ctype == sub
    if mask.sum() == 0:
        continue
    n = mask.sum()
    mean_rho = rho[mask].mean()
    flagged_rate = (rho[mask] < 1.0).mean()
    if sub == "none":
        print(f"  {sub:<18} n={n:<6} mean_rho={mean_rho:.3f}  false_positive_rate={flagged_rate:.3f}")
    else:
        print(f"  {sub:<18} n={n:<6} mean_rho={mean_rho:.3f}  detected_rate={flagged_rate:.3f}")
