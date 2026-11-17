#!/usr/bin/env python
"""Review decision 4: compute RIFL-IoT's empirical mean effective alpha_t from EXISTING
logs (no new run needed), to set as the fixed step size for a tuned-constant-attenuation
baseline. Pools B4's alpha_t across all 5 test seeds for the given condition.
Usage: python scripts/compute_tuned_alpha.py [condition] [experiment_root]
"""
import glob
import json
import sys

import numpy as np

cond = sys.argv[1] if len(sys.argv) > 1 else "noise50"
root = sys.argv[2] if len(sys.argv) > 2 else "runs/phase9_b5_ablation"

alpha_ts = []
for path in sorted(glob.glob(f"{root}/B4_rifl_iot/{cond}/seed_*/feedback_processing.jsonl")):
    for line in open(path):
        r = json.loads(line)
        if r.get("used"):
            alpha_ts.append(r["alpha_t"])

alpha_ts = np.array(alpha_ts)
print(f"condition: {cond}  n_events={len(alpha_ts)}")
print(f"mean alpha_t = {alpha_ts.mean():.5f}")
print(f"median alpha_t = {np.median(alpha_ts):.5f}")
