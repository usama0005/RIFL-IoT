#!/usr/bin/env python
"""Rebuild summary.csv / summary_agg.csv for an experiment from the per-run metrics.json
files on disk (which the runner never overwrites) -- recovers from a partial re-run
clobbering the aggregated CSVs. Usage: python scripts/rebuild_summary.py <experiment_id>
"""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rifl_iot.evaluation.metrics import aggregate  # noqa: E402
from rifl_iot.experiments.runner import SCALAR_KEYS  # noqa: E402

experiment_id = sys.argv[1]
paths = sorted(glob.glob(f"runs/{experiment_id}/*/*/seed_*/metrics.json"))
rows = [json.load(open(p)) for p in paths]
print(f"found {len(rows)} runs for {experiment_id}")

res_dir = f"results/{experiment_id}"
os.makedirs(res_dir, exist_ok=True)
with open(os.path.join(res_dir, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "condition", "seed", "trajectory_hash"] + SCALAR_KEYS,
                       extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

methods = sorted(set(r["method"] for r in rows))
conditions = sorted(set(r["condition"] for r in rows))
agg_rows = []
for cond in conditions:
    for method in methods:
        sub = [r for r in rows if r["method"] == method and r["condition"] == cond]
        if not sub:
            continue
        for k in SCALAR_KEYS:
            a = aggregate([r.get(k) for r in sub])
            agg_rows.append({"method": method, "condition": cond, "metric": k, **a})
with open(os.path.join(res_dir, "summary_agg.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "condition", "metric", "mean", "sd", "ci95", "n"])
    w.writeheader()
    w.writerows(agg_rows)

print(f"rebuilt {res_dir}/summary.csv, summary_agg.csv")
