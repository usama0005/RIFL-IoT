"""SQLite response cache keyed by (model, messages, generation params).

Valid only for deterministic (temperature 0), history-free prompts (B0/B2/B4).
"""
import hashlib
import json
import os
import sqlite3

from rifl_iot.llm.base import LLMClient, LLMResponse


def prompt_key(model_id, messages, gen):
    blob = json.dumps({"m": model_id, "msg": messages, "gen": gen}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class CachedLLM(LLMClient):
    def __init__(self, client, path):
        self.client, self.model_id = client, client.model_id
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, text TEXT, "
                        "tokens_in INT, tokens_out INT, latency_ms REAL)")

    def complete(self, messages, **gen):
        if gen.get("temperature", 0.0) != 0.0:
            return self.client.complete(messages, **gen)
        key = prompt_key(self.model_id, messages, gen)
        row = self.db.execute("SELECT text, tokens_in, tokens_out, latency_ms FROM cache WHERE key=?",
                              (key,)).fetchone()
        if row:
            return LLMResponse(row[0], row[1], row[2], row[3], cached=True)
        r = self.client.complete(messages, **gen)
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?,?,?)",
                        (key, r.text, r.tokens_in, r.tokens_out, r.latency_ms))
        self.db.commit()
        return r


def build_llm(llm_cfg):
    backend = llm_cfg["backend"]
    if backend == "stub":
        s = llm_cfg.get("stub", {})
        from rifl_iot.llm.stub import StubLLM
        client = StubLLM(s["ce_healthy_min"], s["ce_critical_max"], llm_cfg["model"])
    elif backend == "openai_compat":
        from rifl_iot.llm.openai_compat import OpenAICompatLLM
        client = OpenAICompatLLM(llm_cfg["base_url"], llm_cfg["model"], llm_cfg.get("api_key_env"))
    else:
        raise ValueError(backend)
    return CachedLLM(client, llm_cfg["cache_path"]) if llm_cfg.get("cache_path") else client
