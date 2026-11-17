"""B4: RIFL-IoT. Same LA/init/act as B2 (NaiveFeedbackAgent), but alpha_t = alpha*rho*w(Delta)
instead of B2's fixed rho=w=1. Architecturally guarantees B4 - reliability - temporal_credit == B2.

rho_floor (default 0.0, preserving original frozen behavior exactly): alpha_t uses
max(rho, rho_floor) instead of rho directly -- the step-size-floor + reliability-gate hybrid
(review decision 9), tested specifically to see whether it closes the B5>B4 gap under
correlated corruption without eroding the noise-robustness headline result.

Also supports the same optional origin-misattribution simulation as B2/B5 (decision 2)."""
import numpy as np

from rifl_iot.agents.misattribution import resolve_origin
from rifl_iot.agents.naive_feedback import NaiveFeedbackAgent
from rifl_iot.learning.temporal_credit import temporal_weight
from rifl_iot.reliability.estimator import ReliabilityEstimator


class ReliableFeedbackAgent(NaiveFeedbackAgent):
    name = "B4_rifl_iot"

    def __init__(self, llm, gen, system_prompt, seed, la_cfg, reliability_cfg=None,
                temporal_cfg=None, misattribution_cfg=None):
        super().__init__(llm, gen, system_prompt, seed, la_cfg, misattribution_cfg)
        reliability_cfg = reliability_cfg or {}
        temporal_cfg = temporal_cfg or {}
        components = reliability_cfg.get("components", ("range", "stuck", "agree"))
        self.reliability = ReliabilityEstimator(tau=float(reliability_cfg.get("tau", 0.5)),
                                                kappa=float(reliability_cfg.get("kappa", 0.2)),
                                                components=tuple(components))
        self.half_life = float(temporal_cfg.get("half_life", 10.0))
        self.rho_floor = float(reliability_cfg.get("rho_floor", 0.0))

    def receive_feedback(self, events, t):
        recs = []
        for e in events:
            origin = e["origin_t"]
            rec = {"arrival_t": t, "origin_t": origin, "used": False}
            if origin is not None and origin in self.history:
                true_delta = e["arrival_t"] - origin
                resolved = resolve_origin(origin, true_delta, self.history, self.misattr_rate, self.misattr_rng)
                cname, a = self.history.pop(resolved)
                y_A, b = e["y_A"], e["b"]
                beta = float(np.clip(y_A, 0.0, 1.0))
                rho, comps = self.reliability.compute(y_A, b)
                delta = e["arrival_t"] - resolved
                w = temporal_weight(delta, self.half_life)
                rho_eff = max(rho, self.rho_floor)
                alpha_t = self.alpha * rho_eff * w
                self.ca.get(cname).update(a, beta, alpha_t, rho_w=rho_eff * w)
                rec.update({"used": True, "rho": rho, "rho_eff": rho_eff, "w": w, "alpha_t": alpha_t,
                           "beta": beta, "context": cname, "action": a, "delta": delta,
                           "origin_true": origin, "origin_resolved": resolved, **comps})
            recs.append(rec)
        return recs
