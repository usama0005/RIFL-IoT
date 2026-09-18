#!/usr/bin/env python
"""Does B4's rho carry information about eventual reward even when it carries none about
corruption correctness? Joins feedback.jsonl (ground truth) with feedback_processing.jsonl
(rho and its components) on origin_t, for every USED event in one B4 run.
Usage: python scripts/anomaly_investigation.py <run_dir>
  e.g. runs/phase9_b5_ablation/B4_rifl_iot/ctrl_symflip50/seed_0
"""
import json
import sys

import numpy as np

run_dir = sys.argv[1]
fb = {json.loads(l)["origin_t"]: json.loads(l) for l in open(f"{run_dir}/feedback.jsonl")}
proc = [json.loads(l) for l in open(f"{run_dir}/feedback_processing.jsonl") if json.loads(l)["used"]]

rows = []
for r in proc:
    g = fb.get(r["origin_t"])
    if g is None:
        continue
    rows.append({
        "y_A": g["y_A"], "b": g["b"], "rho": r["rho"], "v_range": r.get("v_range"),
        "v_stuck": r.get("v_stuck"), "v_agree": r.get("v_agree"),
        "corrupted": g["corrupted_A"], "r_norm_true": g["r_norm_true"], "fail_true": g["fail_true"],
    })

y_A = np.array([x["y_A"] for x in rows])
rho = np.array([x["rho"] for x in rows])
corrupted = np.array([x["corrupted"] for x in rows], dtype=bool)
r_true = np.array([x["r_norm_true"] for x in rows])

print(f"{run_dir}  (n={len(rows)} used events)")
print(f"  corr(rho, y_A)       = {np.corrcoef(rho, y_A)[0,1]:.4f}")
print(f"  corr(rho, abs(y_A))  = {np.corrcoef(rho, np.abs(y_A))[0,1]:.4f}")
print(f"  corr(rho, r_norm_true) = {np.corrcoef(rho, r_true)[0,1]:.4f}")
print(f"  mean(rho | corrupted=True)  = {rho[corrupted].mean():.4f}  (n={corrupted.sum()})")
print(f"  mean(rho | corrupted=False) = {rho[~corrupted].mean():.4f}  (n={(~corrupted).sum()})")
med = np.median(r_true)
hi, lo = r_true >= med, r_true < med
print(f"  mean(rho | r_norm_true >= median) = {rho[hi].mean():.4f}")
print(f"  mean(rho | r_norm_true <  median) = {rho[lo].mean():.4f}")
print(f"\n  --- STRATIFIED: is the correlation driven by the genuinely-clean subset,")
print(f"      or does it hold even WITHIN the corrupted subset (the real anomaly test)? ---")
for label, mask in [("corrupted=True (should be ~informationless)", corrupted),
                    ("corrupted=False (genuinely clean)", ~corrupted)]:
    if mask.sum() > 2:
        c1 = np.corrcoef(rho[mask], r_true[mask])[0, 1]
        c2 = np.corrcoef(rho[mask], y_A[mask])[0, 1]
        print(f"  [{label}, n={mask.sum()}] corr(rho, r_norm_true)={c1:.4f}  corr(rho, y_A)={c2:.4f}")
vals, counts = np.unique(np.round(rho, 3), return_counts=True)
print(f"  rho distribution: {dict(zip(vals.tolist(), counts.tolist()))}")
v_agree = np.array([x["v_agree"] for x in rows])
print(f"  mean(v_agree | corrupted=True)  = {v_agree[corrupted].mean():.4f}")
print(f"  mean(v_agree | corrupted=False) = {v_agree[~corrupted].mean():.4f}")
print(f"  corr(v_agree, r_norm_true) = {np.corrcoef(v_agree, r_true)[0,1]:.4f}")
