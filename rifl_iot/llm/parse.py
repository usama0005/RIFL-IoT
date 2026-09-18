import json
import re
from dataclasses import dataclass, field

CONTEXTS = ("healthy", "degraded", "critical")
K = 4


@dataclass
class ParsedDecision:
    context: int = -1               # -1 = unknown
    ranking: list = field(default_factory=lambda: list(range(K)))
    ok: bool = False
    full_ranking: bool = False


def parse_decision(text):
    out = ParsedDecision()
    m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    cond = str(obj.get("condition", "")).strip().lower()
    out.context = CONTEXTS.index(cond) if cond in CONTEXTS else -1
    ranking = []
    for item in obj.get("ranking", []) or []:
        mm = re.match(r"\s*A([0-3])", str(item).upper())
        if mm and int(mm.group(1)) not in ranking:
            ranking.append(int(mm.group(1)))
    out.full_ranking = len(ranking) == K
    out.ok = out.context >= 0 and len(ranking) > 0
    if ranking:
        out.ranking = ranking + [a for a in range(K) if a not in ranking]
    return out
