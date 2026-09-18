"""B1: LLM + last-H interaction history. No persistent learner, no reliability weighting,
no phase/oracle leakage -- only observation, chosen action, and feedback events that have
arrived so far, exactly as DESIGN.md §8 specifies. Action = LLM's top-ranked action, same
as B0; the only difference from B0 is what the prompt contains."""
from collections import deque

from rifl_iot.agents.base import Agent, Decision
from rifl_iot.agents.vanilla import call_record
from rifl_iot.llm.parse import K, parse_decision
from rifl_iot.llm.prompts import compact_obs, history_messages

CONTEXT_NAMES = ["healthy", "degraded", "critical"]


class HistoryLLMAgent(Agent):
    name = "B1_history"

    def __init__(self, llm, gen, system_prompt, H=20):
        self.llm, self.gen, self.system_prompt, self.H = llm, gen, system_prompt, H
        self.records = deque(maxlen=H)
        self.by_t = {}  # t -> record dict, pruned once it falls out of the H-window

    def act(self, obs, t):
        msgs = history_messages(obs.features, self.system_prompt, list(self.records))
        resp = self.llm.complete(msgs, **self.gen)
        p = parse_decision(resp.text)
        action = p.ranking[0]
        rec = {"t": t, "llm_condition": CONTEXT_NAMES[p.context] if p.context in (0, 1, 2) else "unknown",
               "obs_compact": compact_obs(obs.features), "action": action, "feedback": []}
        if len(self.records) == self.H:
            del self.by_t[self.records[0]["t"]]
        self.records.append(rec)
        self.by_t[t] = rec
        probs = [1.0 if i == action else 0.0 for i in range(K)]
        return Decision(action, p.context, p.ranking, probs, [call_record(t, msgs, resp, p)])

    def receive_feedback(self, events, t):
        recs = []
        for e in events:
            origin = e["origin_t"]
            used = origin is not None and origin in self.by_t
            if used:
                self.by_t[origin]["feedback"].append({"y_A": e["y_A"], "b": e["b"]})
            recs.append({"arrival_t": t, "origin_t": origin, "used": used})
        return recs

    def state_snapshot(self):
        return {}
