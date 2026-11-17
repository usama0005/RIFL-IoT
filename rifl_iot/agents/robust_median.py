"""B6: robust-statistics baseline (review ask: median/Huber estimator). Same LA/init/act
as B2, but the reward estimate d_hat(a) is the median of the last W y_A values assigned to
(context, action) rather than an EMA, with a FIXED step size alpha (no reliability or
temporal weighting) -- tests whether a generic robust estimator, not content-aware
reliability, is sufficient to handle corruption."""
import numpy as np
from collections import deque, defaultdict

from rifl_iot.agents.naive_feedback import NaiveFeedbackAgent
from rifl_iot.agents.misattribution import resolve_origin


class RobustMedianAgent(NaiveFeedbackAgent):
    name = "B6_robust_median"

    def __init__(self, llm, gen, system_prompt, seed, la_cfg, robust_cfg=None, misattribution_cfg=None):
        super().__init__(llm, gen, system_prompt, seed, la_cfg, misattribution_cfg)
        robust_cfg = robust_cfg or {}
        self.window = int(robust_cfg.get("window", 11))
        self.buffers = defaultdict(lambda: deque(maxlen=self.window))

    def receive_feedback(self, events, t):
        recs = []
        for e in events:
            origin = e["origin_t"]
            rec = {"arrival_t": t, "origin_t": origin, "used": False}
            if origin is not None and origin in self.history:
                true_delta = e["arrival_t"] - origin
                resolved = resolve_origin(origin, true_delta, self.history, self.misattr_rate, self.misattr_rng)
                cname, a = self.history.pop(resolved)
                y_A = float(np.clip(e["y_A"], 0.0, 1.0))
                key = (cname, a)
                self.buffers[key].append(y_A)
                d_hat_robust = float(np.median(self.buffers[key]))
                la = self.ca.get(cname)
                # Move toward whichever action currently has the highest robust median
                # among actions seen so far in this context; untouched actions default to
                # the live estimate just computed (only one changes per event).
                best_a, best_val = a, d_hat_robust
                for other_a in range(len(la.p)):
                    ok = (cname, other_a)
                    if ok in self.buffers and len(self.buffers[ok]) > 0:
                        v = float(np.median(self.buffers[ok]))
                        if v > best_val:
                            best_a, best_val = other_a, v
                onehot = np.zeros(len(la.p)); onehot[best_a] = 1.0
                la.p[:] = (1 - self.alpha) * la.p + self.alpha * onehot
                la.p[:] = np.maximum(la.p, la.p_min); la.p[:] /= la.p.sum()
                rec.update({"used": True, "rho": 1.0, "w": 1.0, "alpha_t": self.alpha,
                           "beta": d_hat_robust, "context": cname, "action": a})
            recs.append(rec)
        return recs
