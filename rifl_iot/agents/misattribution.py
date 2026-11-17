"""Origin-misattribution simulation (review decision 2): with no ground-truth origin
inference implemented, this SIMULATES ambiguous origin matching by redirecting credit
to a random OTHER pending decision with probability that grows with elapsed delay --
min(1, rate_per_step * delta). rate_per_step=0 (default) reproduces the original
exact-tagged-origin behavior exactly, so no previously frozen result changes unless a
condition explicitly sets a nonzero rate. This is a simulated stress test, not a real
origin-inference mechanism -- described as such in the paper."""


def resolve_origin(origin, delta, history, rate_per_step, rng):
    if rate_per_step <= 0 or origin not in history:
        return origin
    p = min(1.0, rate_per_step * delta)
    if rng.random() < p:
        candidates = [k for k in history if k != origin]
        if candidates:
            return candidates[int(rng.random() * len(candidates))]
    return origin
