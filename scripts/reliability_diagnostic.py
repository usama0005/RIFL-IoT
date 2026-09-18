#!/usr/bin/env python
"""Does rho actually track ground-truth corruption (corrupted_A in feedback.jsonl),
or is B4's noise-condition improvement explained by an incidental step-size reduction
that would help regardless of whether rho is correct? F1/AUROC of rho vs corrupted_A.
Usage: python scripts/reliability_diagnostic.py <run_dir>
  e.g. runs/phase7_noise_b0_b2_b4/B4_rifl_iot/noise50/seed_2
"""
import json
import sys

import numpy as np

run_dir = sys.argv[1]
fb = {json.loads(l)["origin_t"]: json.loads(l) for l in open(f"{run_dir}/feedback.jsonl")}
proc = [json.loads(l) for l in open(f"{run_dir}/feedback_processing.jsonl") if json.loads(l)["used"]]

y_true, rho_vals = [], []
for r in proc:
    origin = r["origin_t"]
    if origin in fb:
        y_true.append(fb[origin]["corrupted_A"])
        rho_vals.append(r["rho"])

y_true = np.array(y_true, dtype=int)
rho_vals = np.array(rho_vals)
pred_corrupted = rho_vals < 1.0  # any downweighting at all
tp = np.sum((pred_corrupted == 1) & (y_true == 1))
fp = np.sum((pred_corrupted == 1) & (y_true == 0))
fn = np.sum((pred_corrupted == 0) & (y_true == 1))
tn = np.sum((pred_corrupted == 0) & (y_true == 0))
precision = tp / (tp + fp) if (tp + fp) else float("nan")
recall = tp / (tp + fn) if (tp + fn) else float("nan")
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

print(f"{run_dir}")
print(f"  n events: {len(y_true)}, ground-truth corrupted: {y_true.sum()} ({y_true.mean():.1%})")
print(f"  rho<1 predicted corrupted: {pred_corrupted.sum()} ({pred_corrupted.mean():.1%})")
print(f"  TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"  precision={precision:.3f} recall={recall:.3f} F1={f1:.3f}")
print(f"  mean rho | corrupted=True:  {rho_vals[y_true==1].mean():.3f}")
print(f"  mean rho | corrupted=False: {rho_vals[y_true==0].mean():.3f}")
