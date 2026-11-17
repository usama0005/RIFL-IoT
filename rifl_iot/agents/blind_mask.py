"""B5: blind random-mask ablation. Identical LA/init/act/temporal-credit as B4, but rho is an
independent 50/50 coin flip (0.2 or 1.0) drawn from a dedicated RNG stream, blind to feedback
content. Also supports the same optional origin-misattribution simulation as B2/B4."""
import numpy as np

from rifl_iot.agents.misattribution import resolve_origin
from rifl_iot.agents.naive_feedback import NaiveFeedbackAgent
from rifl_iot.learning.temporal_credit import temporal_weight


class BlindMaskAgent(NaiveFeedbackAgent):
    name = "B5_blind_mask"

    def __init__(self, llm, gen, system_prompt, seed, la_cfg, mask_cfg=None, temporal_cfg=None,
                misattribution_cfg=None):
        super().__init__(llm, gen, system_prompt, seed, la_cfg, misattribution_cfg)
        mask_cfg = mask_cfg or {}
        temporal_cfg = temporal_cfg or {}
        self.rho_low = float(mask_cfg.get("rho_low", 0.2))
        self.rho_high = float(mask_cfg.get("rho_high", 1.0))
        self.p_low = float(mask_cfg.get("p_low", 0.5))
        self.half_life = float(temporal_cfg.get("half_life", 10.0))
        self.mask_rng = np.random.default_rng(np.random.SeedSequence([int(seed), 777]))

    def receive_feedback(self, events, t):
        recs = []
        for e in events:
            origin = e["origin_t"]
            rec = {"arrival_t": t, "origin_t": origin, "used": False}
            if origin is not None and origin in self.history:
                true_delta = e["arrival_t"] - origin
                resolved = resolve_origin(origin, true_delta, self.history, self.misattr_rate, self.misattr_rng)
                cname, a = self.history.pop(resolved)
                beta = float(np.clip(e["y_A"], 0.0, 1.0))
                rho = self.rho_low if self.mask_rng.random() < self.p_low else self.rho_high
                delta = e["arrival_t"] - resolved
                w = temporal_weight(delta, self.half_life)
                alpha_t = self.alpha * rho * w
                self.ca.get(cname).update(a, beta, alpha_t, rho_w=rho * w)
                rec.update({"used": True, "rho": rho, "w": w, "alpha_t": alpha_t, "beta": beta,
                           "context": cname, "action": a, "delta": delta,
                           "origin_true": origin, "origin_resolved": resolved})
            recs.append(rec)
        return recs
