"""Fleet maintenance environment: real sensor windows, constructed consequences."""
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from rifl_iot.feedback.generator import FeedbackGenerator


@dataclass
class Observation:
    t: int
    cycle_id: int
    features: dict


class MaintenanceEnv:
    def __init__(self, data, dyn, traj, fb_cfg):
        self.data, self.dyn, self.tr = data, dyn, traj
        self.tagged = bool(fb_cfg["tagged"])
        self.fb = FeedbackGenerator(fb_cfg, dyn, traj)
        self._pending = defaultdict(list)
        self.events = []

    @property
    def T(self):
        return self.dyn.T

    def observe(self, t):
        cid = int(self.tr["cycle_id"][t])
        return Observation(t=t, cycle_id=cid, features=self.data.summary(cid))

    def deliver(self, t):
        """Events with arrival_t <= t not yet delivered (>= to flush, never silently drop).
        For delay=0 this means: available starting the NEXT decision, since a step's own
        feedback cannot causally be seen before that step's action is taken."""
        due = [k for k in self._pending if k <= t]
        evs = sorted((e for k in due for e in self._pending.pop(k)), key=lambda e: e.origin_t)
        return evs, [e.public(self.tagged) for e in evs]

    def step(self, t, a):
        c = int(self.tr["context"][t])
        ph = self.dyn.phase_at(t)
        fail = bool(self.tr["u_fail"][t] < ph.p_fail[c, a])
        r = ph.realized_reward(a, fail)
        exp_r = ph.expected_reward(c)
        ev = self.fb.emit(t, a, fail)
        self.events.append(ev)
        if not ev.missing:
            self._pending[ev.arrival_t].append(ev)
        return {
            "segment": int(self.tr["segment"][t]), "phase": ph.name, "true_context": c,
            "cycle_id": int(self.tr["cycle_id"][t]), "fail": fail, "r_true": r,
            "r_norm": self.dyn.normalize(r), "expected_rewards": exp_r.round(6).tolist(),
            "oracle_action": int(np.argmax(exp_r)),
            "exp_regret": float(exp_r.max() - exp_r[a]),
        }

    def delivered(self, ev):
        return (not ev.missing) and ev.arrival_t < self.T
