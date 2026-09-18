"""B4: RIFL-IoT. Same LA/init/act as B2 (NaiveFeedbackAgent), but alpha_t = alpha*rho*w(Delta)
instead of B2's fixed rho=w=1. Architecturally guarantees B4 - reliability - temporal_credit == B2
(same class, same LA, only receive_feedback's weighting differs)."""
import numpy as np

from rifl_iot.agents.naive_feedback import NaiveFeedbackAgent
from rifl_iot.learning.temporal_credit import temporal_weight
from rifl_iot.reliability.estimator import ReliabilityEstimator


class ReliableFeedbackAgent(NaiveFeedbackAgent):
    name = "B4_rifl_iot"

    def __init__(self, llm, gen, system_prompt, seed, la_cfg, reliability_cfg=None, temporal_cfg=None):
        super().__init__(llm, gen, system_prompt, seed, la_cfg)
        reliability_cfg = reliability_cfg or {}
        temporal_cfg = temporal_cfg or {}
        self.reliability = ReliabilityEstimator(tau=float(reliability_cfg.get("tau", 0.5)),
                                                kappa=float(reliability_cfg.get("kappa", 0.2)))
        self.half_life = float(temporal_cfg.get("half_life", 10.0))

    def receive_feedback(self, events, t):
        recs = []
        for e in events:
            origin = e["origin_t"]
            rec = {"arrival_t": t, "origin_t": origin, "used": False}
            if origin is not None and origin in self.history:
                cname, a = self.history.pop(origin)
                y_A, b = e["y_A"], e["b"]
                beta = float(np.clip(y_A, 0.0, 1.0))
                rho, comps = self.reliability.compute(y_A, b)
                delta = e["arrival_t"] - origin
                w = temporal_weight(delta, self.half_life)
                alpha_t = self.alpha * rho * w
                self.ca.get(cname).update(a, beta, alpha_t, rho_w=rho * w)
                rec.update({"used": True, "rho": rho, "w": w, "alpha_t": alpha_t, "beta": beta,
                           "context": cname, "action": a, "delta": delta, **comps})
            recs.append(rec)
        return recs
