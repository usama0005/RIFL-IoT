#!/usr/bin/env python
"""Assemble the per-context Learning Automaton evidence table from existing logs
(steps.jsonl + feedback_processing.jsonl) -- no new instrumentation needed, since
agent_state, action, oracle_action, r_norm, exp_regret (steps.jsonl) and beta/action
(feedback_processing.jsonl) already carry everything this needs.

Usage: python scripts/analyze_la.py --run runs/<experiment>/<method>/<condition>/seed_<s>
"""
import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="path to a seed_<s> run directory")
args = ap.parse_args()

steps = [json.loads(l) for l in open(os.path.join(args.run, "steps.jsonl"))]
fb_path = os.path.join(args.run, "feedback_processing.jsonl")
fb = [json.loads(l) for l in open(fb_path)] if os.path.exists(fb_path) else []
CONTEXTS = ["healthy", "degraded", "critical"]
ACTIONS = ["A0_CONTINUE", "A1_DERATE", "A2_SCHEDULE_MAINTENANCE", "A3_SHUTDOWN_REPAIR"]

print(f"run: {args.run}  ({len(steps)} steps)\n")

for c, cname in enumerate(CONTEXTS):
    cs = [s for s in steps if s["true_context"] == c]
    if not cs:
        continue
    print(f"=== context: {cname} (n={len(cs)}, oracle={Counter(s['oracle_action'] for s in cs)}) ===")
    print("  selections:", Counter(s["action"] for s in cs))
    print("  accuracy:", round(np.mean([s["correct"] for s in cs]), 3))
    print("  cumulative reward (r_norm sum):", round(sum(s["r_norm"] for s in cs), 2))
    print("  cumulative regret (exp_regret sum):", round(sum(s["exp_regret"] for s in cs), 2))
    last_state = None
    for s in reversed(steps):
        st = s.get("agent_state", {})
        if cname in st:
            last_state = st[cname]
            break
    if last_state:
        print("  final LA probabilities [A0,A1,A2,A3]:", last_state)
    fb_by_action = defaultdict(list)
    for r in fb:
        if r.get("context") == cname and r.get("used"):
            fb_by_action[r["action"]].append(r["beta"])
    if fb_by_action:
        print("  mean feedback (beta) per action:")
        for a in sorted(fb_by_action):
            v = fb_by_action[a]
            print(f"    {ACTIONS[a]}: n={len(v)} mean={np.mean(v):.3f}")
    print()
