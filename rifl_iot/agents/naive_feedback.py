"""B2: naive implicit feedback. Same LA/interface as B4, with rho=1, w=1 fixed
(alpha_t = alpha), and credit applied to the tagged origin action (DESIGN.md §8)."""
import numpy as np

from rifl_iot.agents.base import Agent, Decision
from rifl_iot.agents.vanilla import call_record
from rifl_iot.learning.automaton import ContextualAutomaton
from rifl_iot.llm.parse import K, parse_decision
from rifl_iot.llm.prompts import vanilla_messages

CONTEXT_NAMES = ["healthy", "degraded", "critical"]


def ctx_name(llm_context):
    return CONTEXT_NAMES[llm_context] if llm_context in (0, 1, 2) else "unknown"


class NaiveFeedbackAgent(Agent):
    name = "B2_naive"

    def __init__(self, llm, gen, system_prompt, seed, la_cfg):
        self.llm, self.gen, self.system_prompt = llm, gen, system_prompt
        self.rng = np.random.default_rng(np.random.SeedSequence([int(seed), 999]))
        self.alpha = float(la_cfg["alpha"])
        extra = {}
        if "epsilon" in la_cfg:
            extra["epsilon"] = float(la_cfg["epsilon"])
        if "pursuit_eta" in la_cfg:
            extra["eta"] = float(la_cfg["pursuit_eta"])
        self.ca = ContextualAutomaton(
            K, variant=la_cfg.get("variant", "LR_I"), omega=float(la_cfg.get("omega", 0.5)),
            p_min=float(la_cfg.get("p_min", 0.01)), **extra)
        self.history = {}  # origin_t -> (context_name, action)

    def act(self, obs, t):
        msgs = vanilla_messages(obs.features, self.system_prompt)
        resp = self.llm.complete(msgs, **self.gen)
        p = parse_decision(resp.text)
        cname = ctx_name(p.context)
        la = self.ca.get(cname, init_action=p.ranking[0])
        probs = la.p.copy()
        action = int(self.rng.choice(K, p=probs))
        self.history[t] = (cname, action)
        return Decision(action, p.context, p.ranking, probs.tolist(), [call_record(t, msgs, resp, p)])

    def receive_feedback(self, events, t):
        recs = []
        for e in events:
            origin = e["origin_t"]
            rec = {"arrival_t": t, "origin_t": origin, "used": False}
            if origin is not None and origin in self.history:
                cname, a = self.history.pop(origin)
                beta = float(np.clip(e["y_A"], 0.0, 1.0))
                self.ca.get(cname).update(a, beta, self.alpha)
                rec.update({"used": True, "rho": 1.0, "w": 1.0, "alpha_t": self.alpha,
                           "beta": beta, "context": cname, "action": a})
            recs.append(rec)
        return recs

    def state_snapshot(self):
        return self.ca.state_snapshot()
