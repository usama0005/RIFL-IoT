"""Learning Automaton update rules (DESIGN.md §7). Same interface for B2 and B4:
`update(a, beta, alpha_t, rho_w=1.0)` where beta = clip(y_A, 0, 1) and
alpha_t = alpha * rho_t * w(Delta_t) (B2 fixes rho=w=1, so alpha_t = alpha).
`rho_w` is only consumed by Pursuit's estimator learning rate (see DESIGN.md §7.3).
"""
import numpy as np


class LearningAutomaton:
    def __init__(self, K, p_min=0.01, init_p=None):
        self.K, self.p_min = K, p_min
        self.p = np.array(init_p, dtype=float) if init_p is not None else np.full(K, 1.0 / K)
        self._project()

    def _project(self):
        self.p = np.maximum(self.p, self.p_min)
        self.p /= self.p.sum()

    def update(self, a, beta, alpha_t, rho_w=1.0):
        raise NotImplementedError


class LR_I(LearningAutomaton):
    """Linear reward-inaction (S-model). p <- p + alpha_t*beta*(e_a - p)."""

    def update(self, a, beta, alpha_t, rho_w=1.0):
        e_a = np.zeros(self.K)
        e_a[a] = 1.0
        self.p = self.p + alpha_t * beta * (e_a - self.p)
        self._project()


class LR_P(LearningAutomaton):
    """Reward-penalty / epsilon-penalty (S-model). epsilon=1.0 -> LR_P, epsilon<<1 -> LR_epsilonP."""

    def __init__(self, K, epsilon=1.0, **kw):
        super().__init__(K, **kw)
        self.epsilon = float(epsilon)

    def update(self, a, beta, alpha_t, rho_w=1.0):
        e_a = np.zeros(self.K)
        e_a[a] = 1.0
        self.p = self.p + alpha_t * beta * (e_a - self.p)
        self._project()
        b_t = self.epsilon * alpha_t * (1 - beta)
        if b_t > 0:
            new_p = self.p * (1 - b_t)
            new_p += b_t / (self.K - 1)
            new_p[a] = self.p[a] * (1 - b_t)
            self.p = new_p
            self._project()


class Pursuit(LearningAutomaton):
    """EMA reward estimator + pursuit of its current argmax (DESIGN.md §7.3)."""

    def __init__(self, K, eta=0.1, **kw):
        super().__init__(K, **kw)
        self.eta = float(eta)
        self.d_hat = np.full(K, 0.5)

    def update(self, a, beta, alpha_t, rho_w=1.0):
        self.d_hat[a] = self.d_hat[a] + self.eta * rho_w * (beta - self.d_hat[a])
        best = int(np.argmax(self.d_hat))
        e_best = np.zeros(self.K)
        e_best[best] = 1.0
        self.p = self.p + alpha_t * (e_best - self.p)
        self._project()


VARIANTS = {"LR_I": LR_I, "LR_P": LR_P, "LR_epsilonP": LR_P, "Pursuit": Pursuit}
VARIANT_DEFAULTS = {"LR_P": {"epsilon": 1.0}, "LR_epsilonP": {"epsilon": 0.05}, "Pursuit": {"eta": 0.1}}
VARIANT_EXTRA_KEYS = {"LR_I": (), "LR_P": ("epsilon",), "LR_epsilonP": ("epsilon",), "Pursuit": ("eta",)}


class ContextualAutomaton:
    """One LearningAutomaton per LLM condition label (healthy/degraded/critical/unknown)."""

    def __init__(self, K, variant="LR_I", omega=0.5, p_min=0.01, **variant_kwargs):
        self.K, self.omega, self.p_min = K, float(omega), float(p_min)
        self.variant = variant
        merged = {**VARIANT_DEFAULTS.get(variant, {}), **variant_kwargs}
        self.kwargs = {k: merged[k] for k in VARIANT_EXTRA_KEYS[variant] if k in merged}
        self.automata = {}

    def _init_p(self, init_action):
        if init_action is None:
            return None
        p = np.full(self.K, (1 - self.omega) / self.K)
        p[init_action] += self.omega
        return p

    def get(self, context_name, init_action=None):
        if context_name not in self.automata:
            cls = VARIANTS[self.variant]
            self.automata[context_name] = cls(self.K, p_min=self.p_min,
                                              init_p=self._init_p(init_action), **self.kwargs)
        return self.automata[context_name]

    def state_snapshot(self):
        return {c: la.p.round(4).tolist() for c, la in self.automata.items()}
