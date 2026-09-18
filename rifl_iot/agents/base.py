from dataclasses import dataclass, field


@dataclass
class Decision:
    action: int
    llm_context: int               # -1 = unknown / not applicable
    llm_ranking: list
    selection_probs: list          # distribution the action was drawn from
    llm_calls: list = field(default_factory=list)   # per-call records for llm_calls.jsonl


class Agent:
    name = "base"

    def act(self, obs, t) -> Decision:
        raise NotImplementedError

    def receive_feedback(self, events, t):
        """events: public dicts. Returns per-event processing records (logged)."""
        return [{"arrival_t": t, "origin_t": e["origin_t"], "used": False} for e in events]

    def state_snapshot(self):
        """Learner state logged every step (e.g. all automaton probabilities)."""
        return {}
