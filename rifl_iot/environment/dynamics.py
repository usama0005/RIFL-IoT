"""Constructed action->consequence model (DESIGN.md §3). Fully determined by config."""
from dataclasses import dataclass

import numpy as np


@dataclass
class PhaseModel:
    name: str
    context_mix: np.ndarray   # (C,)
    cost: np.ndarray          # (K,)
    p_fail: np.ndarray        # (C, K)
    fail_cost: float

    def expected_reward(self, c):
        return 1.0 - self.cost - self.fail_cost * self.p_fail[c]

    def realized_reward(self, a, fail):
        return 1.0 - self.cost[a] - self.fail_cost * float(fail)

    def oracle(self, c):
        return int(np.argmax(self.expected_reward(c)))

    def gap(self, c):
        e = np.sort(self.expected_reward(c))
        return float(e[-1] - e[-2])


class Dynamics:
    def __init__(self, env_cfg, schedule_name):
        self.actions = list(env_cfg["actions"])
        self.contexts = list(env_cfg["contexts"])
        self.K, self.C = len(self.actions), len(self.contexts)
        self.phases = {}
        for name, p in env_cfg["phases"].items():
            self.phases[name] = PhaseModel(
                name=name,
                context_mix=np.asarray(p["context_mix"], float),
                cost=np.asarray(p["downtime_cost"], float),
                p_fail=np.asarray([p["p_fail"][c] for c in self.contexts], float),
                fail_cost=float(p["fail_cost"]),
            )
            assert abs(self.phases[name].context_mix.sum() - 1) < 1e-9, name
        self.schedule_name = schedule_name
        self.schedule = [(str(n), int(L)) for n, L in env_cfg["schedules"][schedule_name]]
        self.T = sum(L for _, L in self.schedule)
        self.segment_of_t = np.concatenate([np.full(L, i) for i, (_, L) in enumerate(self.schedule)])
        self.boundaries = list(np.cumsum([L for _, L in self.schedule])[:-1])
        used = [self.phases[n] for n in env_cfg["phases"]]
        self.r_max = 1.0 - min(p.cost.min() for p in used)
        self.r_min = 1.0 - max(p.cost.max() for p in used) - max(p.fail_cost for p in used)

    def segment_phase(self, s):
        return self.phases[self.schedule[s][0]]

    def phase_at(self, t):
        return self.segment_phase(int(self.segment_of_t[t]))

    def normalize(self, r):
        return (r - self.r_min) / (self.r_max - self.r_min)
