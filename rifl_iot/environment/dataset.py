"""Real sensor windows from the UCI hydraulic condition-monitoring dataset.

Only observation distributions and cooler-condition labels come from here.
Actions and consequences are constructed (see DESIGN.md §0).
"""
import os
import warnings

import numpy as np

COOLER_LEVEL_TO_CONTEXT = {100: 0, 20: 1, 3: 2}
SENSORS = ["temperature_C", "cooling_efficiency_pct", "cooling_power_kW"]
STATS = ["mean", "std", "min", "max", "slope"]
N_CYCLES, N_SAMPLES = 2205, 60


def _window_stats(x):
    t = np.arange(x.shape[1], dtype=float)
    tc = t - t.mean()
    slope = (x - x.mean(1, keepdims=True)) @ tc / (tc @ tc)
    return np.stack([x.mean(1), x.std(1), x.min(1), x.max(1), slope], axis=1)


class HydraulicCycles:
    def __init__(self, ts, ce, cp, cooler, stable, source):
        self.source = source
        self.context = np.array([COOLER_LEVEL_TO_CONTEXT[int(round(c))] for c in cooler])
        self.stable = np.asarray(stable).astype(int)
        self.feat = {
            "temperature_C": _window_stats(ts),
            "cooling_efficiency_pct": _window_stats(ce),
            "cooling_power_kW": _window_stats(cp),
        }
        self.n = len(self.context)

    @classmethod
    def from_uci(cls, root):
        def load(name):
            arr = np.loadtxt(os.path.join(root, name))
            if arr.shape[0] != N_CYCLES:
                raise ValueError(f"{name}: expected {N_CYCLES} rows, got {arr.shape}")
            return arr

        ts = np.mean([load(f"TS{i}.txt") for i in range(1, 5)], axis=0)
        ce, cp, prof = load("CE.txt"), load("CP.txt"), load("profile.txt")
        for name, a in [("TS", ts), ("CE", ce), ("CP", cp)]:
            if a.shape != (N_CYCLES, N_SAMPLES):
                raise ValueError(f"{name}: expected {(N_CYCLES, N_SAMPLES)}, got {a.shape}")
        levels = set(np.round(prof[:, 0]).astype(int))
        if levels != set(COOLER_LEVEL_TO_CONTEXT):
            raise ValueError(f"unexpected cooler levels {levels}")
        return cls(ts, ce, cp, prof[:, 0], prof[:, 4], "uci_hydraulic")

    @classmethod
    def synthetic_smoke(cls, n_per=150, seed=0):
        """Made-up Gaussians for code-path testing ONLY. Never used for results."""
        warnings.warn("synthetic_smoke data: NOT a result source", stacklevel=2)
        rng = np.random.default_rng(seed)
        mus = {100: (35.0, 45.0, 2.2), 20: (40.0, 30.0, 1.6), 3: (48.0, 20.0, 1.2)}
        ts, ce, cp, cooler = [], [], [], []
        for lvl, (m_t, m_e, m_p) in mus.items():
            ts.append(m_t + rng.normal(0, 2, (n_per, 1)) + rng.normal(0, 0.3, (n_per, N_SAMPLES)))
            ce.append(m_e + rng.normal(0, 4, (n_per, 1)) + rng.normal(0, 0.5, (n_per, N_SAMPLES)))
            cp.append(m_p + rng.normal(0, 0.2, (n_per, 1)) + rng.normal(0, 0.05, (n_per, N_SAMPLES)))
            cooler += [lvl] * n_per
        return cls(np.vstack(ts), np.vstack(ce), np.vstack(cp), np.array(cooler),
                   np.zeros(len(cooler)), "synthetic_smoke")

    def split(self, calib_frac, seed, stable_only=False):
        """Stratified calibration/episode split. Returns (calib_idx, episode_pools[ctx])."""
        rng = np.random.default_rng(seed)
        calib, pools = [], {}
        for c in range(3):
            idx = np.where((self.context == c) & ((self.stable == 0) if stable_only else True))[0]
            idx = rng.permutation(idx)
            n_cal = int(round(calib_frac * len(idx)))
            calib.append(idx[:n_cal])
            pools[c] = np.sort(idx[n_cal:])
        return np.sort(np.concatenate(calib)), pools

    def summary(self, i, decimals=2):
        return {s: {st: round(float(self.feat[s][i, j]), decimals if st != "slope" else 4)
                    for j, st in enumerate(STATS)} for s in SENSORS}


def load_data(data_cfg):
    src = data_cfg["source"]
    if src == "uci_hydraulic":
        return HydraulicCycles.from_uci(data_cfg["root"])
    if src == "synthetic_smoke":
        return HydraulicCycles.synthetic_smoke()
    raise ValueError(f"unknown data source {src}")
