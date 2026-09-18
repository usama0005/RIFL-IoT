"""B0: vanilla LLM. Top-ranked action; feedback is received and ignored."""
import hashlib

from rifl_iot.agents.base import Agent, Decision
from rifl_iot.llm.parse import K, parse_decision
from rifl_iot.llm.prompts import vanilla_messages


def call_record(t, messages, resp, parsed):
    return {"t": t, "prompt_hash": hashlib.sha256(str(messages).encode()).hexdigest()[:16],
            "cached": resp.cached, "tokens_in": resp.tokens_in, "tokens_out": resp.tokens_out,
            "latency_ms": resp.latency_ms, "parse_ok": parsed.ok,
            "full_ranking": parsed.full_ranking, "raw": resp.text}


class VanillaLLMAgent(Agent):
    name = "B0_vanilla"

    def __init__(self, llm, gen, system_prompt):
        self.llm, self.gen, self.system_prompt = llm, gen, system_prompt

    def act(self, obs, t):
        msgs = vanilla_messages(obs.features, self.system_prompt)
        resp = self.llm.complete(msgs, **self.gen)
        p = parse_decision(resp.text)
        a = p.ranking[0]
        probs = [1.0 if i == a else 0.0 for i in range(K)]
        return Decision(a, p.context, p.ranking, probs, [call_record(t, msgs, resp, p)])
