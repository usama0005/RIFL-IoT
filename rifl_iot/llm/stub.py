"""Deterministic rule-based stand-in for smoke tests. Not a baseline."""
import json
import re
import time

from rifl_iot.llm.base import LLMClient, LLMResponse

RANKINGS = {"healthy": ["A0", "A1", "A2", "A3"],
            "degraded": ["A1", "A2", "A0", "A3"],
            "critical": ["A3", "A2", "A1", "A0"]}


class StubLLM(LLMClient):
    def __init__(self, ce_healthy_min, ce_critical_max, model_id="stub-v0"):
        self.hi, self.lo, self.model_id = ce_healthy_min, ce_critical_max, model_id

    def complete(self, messages, **gen):
        t0 = time.perf_counter()
        user = messages[-1]["content"]
        m = re.search(r"cooling_efficiency_pct: mean=([-\d.]+)", user)
        ce = float(m.group(1)) if m else 0.0
        cond = "healthy" if ce >= self.hi else ("critical" if ce <= self.lo else "degraded")
        text = json.dumps({"condition": cond, "ranking": RANKINGS[cond], "rationale": "stub rule"})
        n_in = sum(len(x["content"].split()) for x in messages)
        return LLMResponse(text, n_in, len(text.split()), (time.perf_counter() - t0) * 1e3)
