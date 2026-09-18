#!/usr/bin/env python
"""Run an experiment from a YAML config.  python scripts/run.py --config configs/smoke.yaml"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rifl_iot.experiments.runner import run_experiment  # noqa: E402
from rifl_iot.utils.config import load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--seeds", type=int, nargs="*")
ap.add_argument("--conditions", nargs="*")
args = ap.parse_args()
cfg = load_config(args.config)
if args.seeds:
    cfg["seeds"] = args.seeds
if args.conditions:
    cfg["conditions"] = {k: cfg["conditions"][k] for k in args.conditions}
run_experiment(cfg)
