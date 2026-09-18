"""B5: blind random-mask ablation. Identical LA/init/act/temporal-credit as B4, but rho is an
independent 50/50 coin flip (0.2 or 1.0) drawn from a dedicated RNG stream, blind to feedback
content. Isolates whether B4's advantage is content-dependent reliability detection or just
the incidental effect of occasionally shrinking the step size regardless of correctness."""
import numpy as np

from rifl_iot.agents.naive_feedback import NaiveFeedbackAgent
from rifl_iot.learning.temporal_credit import temporal_weight


class BlindMaskAgent(NaiveFeedbackAgent):
    name = "B5_blind_mask"

    def __init__(self, llm, gen, system_prompt, seed, la_cfg, mask_cfg=None, temporal_cfg=None):
        super().__init__(llm, gen, system_prompt, seed, la_cfg)
        mask_cfg = mask_cfg or {}
        temporal_cfg = temporal_cfg or {}
        self.rho_low = float(mask_cfg.get("rho_low", 0.2))
        self.rho_high = float(mask_cfg.get("rho_high", 1.0))
        self.p_low = float(mask_cfg.get("p_low", 0.5))
        self.half_life = float(temporal_cfg.get("half_life", 10.0))
        # Dedicated stream (seed, 777) -- distinct from action-selection rng (seed, 999) and
        # every environment stream -- so the mask never sees feedback content or LLM output.
        self.mask_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 777]))

    def receive_feedback(self, events, t):
        recs = []
        for e in events:
            origin = e["origin_t"]
            rec = {"arrival_t": t, "origin_t": origin, "used": False}
            if origin is not None and origin in self.history:
                cname, a = self.history.pop(origin)
                beta = float(np.clip(e["y_A"], 0.0, 1.0))
                rho = self.rho_low if self.mask_rng.random() < self.p_low else self.rho_high
                delta = e["arrival_t"] - origin
                w = temporal_weight(delta, self.half_life)
                alpha_t = self.alpha * rho * w
                self.ca.get(cname).update(a, beta, alpha_t, rho_w=rho * w)
                rec.update({"used": True, "rho": rho, "w": w, "alpha_t": alpha_t, "beta": beta,
                           "context": cname, "action": a, "delta": delta})
            recs.append(rec)
        return recs
