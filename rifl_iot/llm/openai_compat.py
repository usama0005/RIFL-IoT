"""OpenAI-compatible chat endpoint (vLLM, llama.cpp server, OpenAI). stdlib only."""
import json
import os
import time
import urllib.request

from rifl_iot.llm.base import LLMClient, LLMResponse


class OpenAICompatLLM(LLMClient):
    def __init__(self, base_url, model, api_key_env=None, timeout=120, retries=3):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model_id = model
        self.key = os.environ.get(api_key_env) if api_key_env else None
        self.timeout, self.retries = timeout, retries

    def complete(self, messages, temperature=0.0, max_tokens=256, seed=None):
        body = {"model": self.model_id, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens}
        if seed is not None:
            body["seed"] = seed
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        last = None
        for attempt in range(self.retries):
            t0 = time.perf_counter()
            try:
                req = urllib.request.Request(self.url, json.dumps(body).encode(), headers)
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    data = json.loads(r.read())
                usage = data.get("usage", {})
                return LLMResponse(data["choices"][0]["message"]["content"] or "",
                                   usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0),
                                   (time.perf_counter() - t0) * 1e3)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"LLM call failed after {self.retries} attempts: {last}")
