#!/usr/bin/env python
"""AUROC / F1 / ECE of rho_t as a corrupted-feedback detector, pooled across all 5 test
seeds. Prediction score for "corrupted" = 1 - rho_t. Threshold for F1: rho_t < 1 (any
downweighting) predicts corrupted. ECE bins by rho_t's discrete values (rho_t in {0,kappa,1}
in this estimator), comparing mean(1-rho) in each bin against the true corrupted fraction.
Usage: python scripts/reliability_quality_report.py
"""
import glob
import json
import os

import numpy as np

ROOT = "runs/phase9_b5_ablation/B4_rifl_iot"
CONDS = {"Mixed noise 30%": "noise30", "Mixed noise 50%": "noise50", "Coherent 30%": "ctrl_corr30"}
SEEDS = [0, 1, 2, 3, 4]


def load_condition(cond):
    y_true, rho_all = [], []
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
    return np.array(y_true), np.array(rho_all)


def auroc(y_true, score):
    pos = score[y_true == 1]
    neg = score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    sorted_scores = score[order]
    i = 0
    n = len(score)
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        ranks[order[i:j + 1]] = avg_rank
        i = j + 1
    r_pos_sum = ranks[y_true == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    u = r_pos_sum - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def f1(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def ece(y_true, rho):
    conf = 1 - rho
    bins = np.unique(rho)
    n = len(y_true)
    total = 0.0
    for b in bins:
        mask = rho == b
        if mask.sum() == 0:
            continue
        bin_conf = conf[mask].mean()
        bin_acc = y_true[mask].mean()
        total += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return total


print(f"{'Condition':<18}{'n':>8}{'AUROC':>10}{'F1':>10}{'ECE':>10}")
for label, cond in CONDS.items():
    y_true, rho = load_condition(cond)
    score = 1 - rho
    pred = (rho < 1.0).astype(int)
    a = auroc(y_true, score)
    f = f1(y_true, pred)
    e = ece(y_true, rho)
    print(f"{label:<18}{len(y_true):>8}{a:>10.4f}{f:>10.4f}{e:>10.4f}")
