"""Pre-sampled exogenous randomness (common random numbers).

Every random draw the environment will ever need is sampled here, per step, from
independent named streams, regardless of condition. Changing a condition parameter
(eta, D, q, m) never shifts any other stream, and every method replays the same draws.
"""
import hashlib

import numpy as np

STREAM_IDS = {"context": 1, "cycle": 2, "fail": 3, "eps_A": 4, "err_B": 5,
              "corrupt": 6, "corrupt_aux": 7, "conflict": 8, "missing": 9, "delay": 10}


def _rng(seed, name):
    return np.random.default_rng(np.random.SeedSequence([int(seed), STREAM_IDS[name]]))


def generate_trajectory(dyn, pools, seed):
    T = dyn.T
    mix = np.stack([dyn.segment_phase(s).context_mix for s in range(len(dyn.schedule))])[dyn.segment_of_t]
    u = _rng(seed, "context").random(T)
    context = np.minimum((u[:, None] >= np.cumsum(mix, axis=1)).sum(1), dyn.C - 1)
    u_cyc = _rng(seed, "cycle").random(T)
    cycle_id = np.array([pools[c][int(uc * len(pools[c]))] for c, uc in zip(context, u_cyc)])
    r_cor, r_aux, r_con = _rng(seed, "corrupt"), _rng(seed, "corrupt_aux"), _rng(seed, "conflict")
    traj = {
        "segment": dyn.segment_of_t.astype(np.int64),
        "context": context.astype(np.int64),
        "cycle_id": cycle_id.astype(np.int64),
        "u_fail": _rng(seed, "fail").random(T),
        "eps_A": _rng(seed, "eps_A").standard_normal(T),
        "u_err_B": _rng(seed, "err_B").random(T),
        "u_corrupt": r_cor.random(T),
        "corrupt_type": r_cor.integers(0, 4, T),
        "u_replace": r_aux.random(T),
        "u_spike": r_aux.random(T),
        "spike_sign": r_aux.integers(0, 2, T),
        "u_conflict": r_con.random(T),
        "conflict_channel": r_con.integers(0, 2, T),
        "u_missing": _rng(seed, "missing").random(T),
        "u_delay": _rng(seed, "delay").random(T),
    }
    return traj, trajectory_hash(traj)


def trajectory_hash(traj):
    h = hashlib.sha256()
    for k in sorted(traj):
        h.update(k.encode())
        h.update(np.ascontiguousarray(traj[k]).tobytes())
    return h.hexdigest()[:16]
