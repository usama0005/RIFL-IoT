#!/usr/bin/env python
"""Review decision 8: feasibility check for a second task using valve/pump/accumulator
labels (profile.txt columns 1-3), BEFORE committing to building it. No simulation run here.
Usage: python scripts/valve_pump_feasibility.py [data_root]
"""
import os
import sys

import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "data/uci_hydraulic"

prof = np.loadtxt(os.path.join(root, "profile.txt"))
names = ["cooler (%)", "valve (%)", "pump leakage (0/1/2)", "accumulator (bar)", "stable flag"]
for col, name in enumerate(names):
    vals, counts = np.unique(np.round(prof[:, col], 1), return_counts=True)
    print(f"{name}: {dict(zip(vals.tolist(), counts.tolist()))}")

print(f"\ntotal cycles: {prof.shape[0]}")
print("\nRaw sensor files present in data dir (beyond TS1-4/CE/CP already loaded):")
expected_extra = ["VS1", "PS1", "PS2", "PS3", "PS4", "PS5", "PS6", "SE", "FS1", "FS2", "EPS1"]
for name in expected_extra:
    path = os.path.join(root, f"{name}.txt")
    present = os.path.exists(path)
    shape = np.loadtxt(path).shape if present else None
    print(f"  {name}.txt: {'present shape='+str(shape) if present else 'MISSING'}")
