from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cached: bool = False


class LLMClient:
    model_id = "base"

    def complete(self, messages, **gen):
        raise NotImplementedError
