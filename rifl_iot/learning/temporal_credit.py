"""Temporal credit weight w(Delta) = exp(-lambda*Delta), lambda = ln2/half_life (DESIGN.md §6)."""
import math


def temporal_weight(delta, half_life=10.0):
    if half_life <= 0:
        return 1.0
    lam = math.log(2) / half_life
    return math.exp(-lam * delta)
