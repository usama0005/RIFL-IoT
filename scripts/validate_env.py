#!/usr/bin/env python
"""Phase 2 checks: oracle table, action gaps, feedback-threshold margins, replay determinism,
realised context mix and feedback-corruption rates. No LLM involved."""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rifl_iot.environment.dataset import load_data  # noqa: E402
from rifl_iot.environment.dynamics import Dynamics  # noqa: E402
from rifl_iot.environment.env import MaintenanceEnv  # noqa: E402
from rifl_iot.environment.trajectory import generate_trajectory  # noqa: E402
from rifl_iot.utils.config import apply_overrides, load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--min_gap", type=float, default=0.10)
args = ap.parse_args()
cfg = load_config(args.config)
ok = True

dyn = Dynamics(cfg["env"], "drift")
print("== oracle table (expected reward, * = oracle) ==")
for name, ph in dyn.phases.items():
    for c, cn in enumerate(dyn.contexts):
        e = ph.expected_reward(c)
        cells = " ".join(f"{v:+.2f}{'*' if i == ph.oracle(c) else ' '}" for i, v in enumerate(e))
        flag = "" if ph.gap(c) >= args.min_gap - 1e-9 else "  <-- GAP TOO SMALL"
        ok &= flag == ""
        print(f"{name} {cn:<9} {cells}  gap={ph.gap(c):.2f}{flag}")
changed = [[dyn.segment_phase(i - 1).oracle(c) != dyn.segment_phase(i).oracle(c) for c in range(dyn.C)]
           for i in range(1, len(dyn.schedule))]
print("oracle changes at each drift boundary (per context):", changed)
ok &= all(any(x) for x in changed[:-1])

sig, tau = cfg["feedback"]["sigma_A"], cfg["feedback"]["tau_fail"]
fail_max = max(dyn.normalize(ph.realized_reward(a, True)) for ph in dyn.phases.values() for a in range(dyn.K))
ok_min = min(dyn.normalize(ph.realized_reward(a, False)) for ph in dyn.phases.values() for a in range(dyn.K))
m = min(tau - fail_max, ok_min - tau) / sig
print(f"\n== feedback threshold == max fail score {fail_max:.3f} | tau {tau} | min ok score {ok_min:.3f} "
      f"| margin {m:.2f} sigma {'OK' if m >= 2.5 else '<-- TOO SMALL'}")
ok &= m >= 2.5

data = load_data(cfg["data"])
_, pools = data.split(cfg["data"]["calib_frac"], cfg["data"]["split_seed"])
print(f"\n== data == source={data.source} n={data.n} episode pool sizes={ {k: len(v) for k, v in pools.items()} }")
for c, cn in enumerate(dyn.contexts):
    ce = data.feat["cooling_efficiency_pct"][data.context == c, 0]
    print(f"  {cn:<9} CE mean: {ce.mean():6.2f} +- {ce.std():5.2f}  (n={len(ce)})")

print("\n== replay determinism ==")
for sched in ("stationary", "drift"):
    d = Dynamics(cfg["env"], sched)
    t1, h1 = generate_trajectory(d, pools, 0)
    _, h2 = generate_trajectory(d, pools, 0)
    _, h3 = generate_trajectory(d, pools, 1)
    same = h1 == h2 and h1 != h3
    ok &= same
    mix = np.bincount(t1["context"], minlength=3) / d.T
    print(f"  {sched:<10} seed0 {h1} == {h2}: {h1 == h2} | seed1 differs: {h1 != h3} | mix {mix.round(3)}")

print("\n== realised feedback stream (seed 0, fixed policy = oracle) ==")
for cond, ov in cfg.get("conditions", {}).items():
    ccfg = apply_overrides(cfg, ov)
    d = Dynamics(ccfg["env"], ccfg["run"]["schedule"])
    tr, _ = generate_trajectory(d, pools, 0)
    env = MaintenanceEnv(data, d, tr, ccfg["feedback"])
    for t in range(d.T):
        env.step(t, d.phase_at(t).oracle(int(tr["context"][t])))
    ev = env.events
    ahat = np.array([e.y_A < tau for e in ev])
    b = np.array([e.b for e in ev])
    print(f"  {cond:<12} delivered={np.mean([env.delivered(e) for e in ev]):.2f} "
          f"corruptA={np.mean([e.corrupted_A for e in ev]):.2f} "
          f"conflict={np.mean([e.conflict for e in ev]):.2f} "
          f"A/B disagree={np.mean(ahat != b):.3f} mean_delay={np.mean([e.delay for e in ev]):.1f}")

print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
