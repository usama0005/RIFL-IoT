"""Rule-based feedback reliability rho = v_range * v_stuck * v_agree (DESIGN.md §5).
v_stuck compares against the last RECEIVED y_A on the physical channel-A stream
(stuck-sensor detection), independent of context or action."""


class ReliabilityEstimator:
    def __init__(self, tau=0.5, kappa=0.2, y_lo=-0.2, y_hi=1.2):
        self.tau, self.kappa, self.y_lo, self.y_hi = tau, kappa, y_lo, y_hi
        self.last_y = None

    def compute(self, y_A, b):
        v_range = 1.0 if (self.y_lo <= y_A <= self.y_hi) else 0.0
        v_stuck = 0.0 if (self.last_y is not None and y_A == self.last_y) else 1.0
        a_hat = 1 if y_A < self.tau else 0
        v_agree = 1.0 if a_hat == b else self.kappa
        rho = v_range * v_stuck * v_agree
        self.last_y = y_A
        return rho, {"v_range": v_range, "v_stuck": v_stuck, "v_agree": v_agree, "a_hat": a_hat}
