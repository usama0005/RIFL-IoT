"""Environment-side implicit feedback emission and corruption (DESIGN.md §4).

Ground-truth flags are kept on the event for evaluation; agents only receive `public()`.
"""
import math
from dataclasses import asdict, dataclass

CORRUPTION_TYPES = ("flip", "spike", "stuck", "replace")


@dataclass
class FeedbackEvent:
    origin_t: int
    delay: int
    arrival_t: int
    missing: bool
    y_A: float
    b: int
    # ground truth (never shown to agents)
    fail_true: int
    r_norm_true: float
    corrupted_A: bool
    corruption_type: str
    conflict: bool
    faulty_channel: str

    def public(self, tagged):
        return {"origin_t": self.origin_t if tagged else None, "arrival_t": self.arrival_t,
                "y_A": self.y_A, "b": self.b}

    def record(self):
        return asdict(self)


def sample_delay(D, u):
    if D <= 0:
        return 0
    lo, hi = math.ceil(D / 2), math.floor(3 * D / 2)
    return lo + min(int(u * (hi - lo + 1)), hi - lo)


class FeedbackGenerator:
    def __init__(self, fb_cfg, dyn, traj):
        self.cfg, self.dyn, self.tr = fb_cfg, dyn, traj
        self.sigma = float(fb_cfg["sigma_A"])
        self.e_B = float(fb_cfg["e_B"])
        self.noise_model = fb_cfg["noise"]["model"]
        self.eta = float(fb_cfg["noise"]["eta"])
        self.q = float(fb_cfg["conflict_q"])
        self.m = float(fb_cfg["missing"]["rate"])
        if fb_cfg["missing"]["model"] != "mcar":
            raise NotImplementedError("only MCAR missingness in v1")
        self.D = int(fb_cfg["delay"])
        self._last_y = None

    def _score(self, t, a, fail):
        ph = self.dyn.phase_at(t)
        return self.dyn.normalize(ph.realized_reward(a, fail)) + self.sigma * self.tr["eps_A"][t]

    def emit(self, t, a, fail):
        tr = self.tr
        fail = int(fail)
        r_norm = self.dyn.normalize(self.dyn.phase_at(t).realized_reward(a, fail))
        y = self._score(t, a, fail)
        b = fail ^ int(tr["u_err_B"][t] < self.e_B)
        corrupted, ctype, conflict, faulty = False, "none", False, "none"

        if tr["u_corrupt"][t] < self.eta:
            corrupted = True
            if self.noise_model == "mixed":
                ctype = CORRUPTION_TYPES[int(tr["corrupt_type"][t])]
                if ctype == "stuck" and self._last_y is None:
                    ctype = "replace"
            elif self.noise_model == "symmetric_flip":
                # Zero-information control: BOTH channels replaced with independent draws
                # unrelated to ground truth. Neither v_range nor v_agree has real signal to
                # detect here -- distinct from "correlated", where both channels are wrong
                # but still consistent with each other (undetectable by redundancy specifically).
                ctype = "symmetric_flip"
                y = float(tr["u_replace"][t])
                b = int(tr["spike_sign"][t])
            elif self.noise_model == "correlated":
                ctype = "correlated_flip"
                b = 1 - b
            else:
                raise ValueError(self.noise_model)
            if ctype in ("flip", "correlated_flip"):
                y = self._score(t, a, 1 - fail)
            elif ctype == "spike":
                mag = 1.5 + tr["u_spike"][t]
                y = mag if tr["spike_sign"][t] else -mag
            elif ctype == "stuck":
                y = self._last_y
            elif ctype == "replace":
                y = float(tr["u_replace"][t])

        if tr["u_conflict"][t] < self.q:
            conflict = True
            if int(tr["conflict_channel"][t]) == 0:
                faulty = "A"
                y = self._score(t, a, 1 - fail)
                corrupted = True
                ctype = "conflict_A" if ctype == "none" else ctype + "+conflict_A"
            else:
                faulty = "B"
                b = 1 - b

        self._last_y = float(y)
        d = sample_delay(self.D, tr["u_delay"][t])
        return FeedbackEvent(
            origin_t=t, delay=d, arrival_t=t + d, missing=bool(tr["u_missing"][t] < self.m),
            y_A=float(y), b=int(b), fail_true=fail, r_norm_true=float(r_norm),
            corrupted_A=corrupted, corruption_type=ctype, conflict=conflict, faulty_channel=faulty)
