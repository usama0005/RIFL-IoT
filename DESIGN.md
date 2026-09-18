# RIFL-IoT — Design Specification (v0.1, pre-implementation)

Scope of the claim: **reliable learning from implicit physical-world feedback** (noise, delay,
missingness, sensor conflict, drift). Closed-loop LLM–IoT adaptation itself is NOT claimed as novel.
The environment is NOT a contribution; it is a transparent, reproducible testbed.

---

## 0. Data grounding — what is real and what is constructed

| Component | Source |
|---|---|
| Sensor windows given to the LLM (temperature, cooling efficiency, cooling power) | **Real**: UCI "Condition Monitoring of Hydraulic Systems" (Helwig et al., 2015), 2205 cycles × 60 s @ 1 Hz, files `TS1–TS4.txt`, `CE.txt`, `CP.txt`, `profile.txt` |
| Hidden unit condition (healthy / degraded / critical) | **Real labels**: cooler condition 100 % / 20 % / 3 % from `profile.txt` col 0 |
| Actions, breakdown probabilities, downtime costs, drift phases | **Constructed** (the dataset has no actions or consequences). All numbers are in `configs/env_default.yaml` |
| Feedback channels and corruption processes | **Constructed**, parameters in config |

**To verify before citing:** that this is the exact dataset IoT-LLM uses for its industrial anomaly
detection task (the sensor triple matches). The static IoT-LLM benchmark is used only for
static perception validation (does the LLM classify the cooler condition?), never as a
source of action→consequence data.

Data split (stratified by condition, `split_seed` fixed): 50 % *calibration pool* (thresholds,
no-LLM classifier, prompt development) / 50 % *episode pool* (observations during episodes). No overlap.

---

## 1. State

Setting: **fleet contextual bandit with delayed, corrupted feedback.** At each step `t` one unit
from a fleet of hydraulic rigs requires a decision. (A single-machine MDP with degradation
dynamics is deferred to v2; the fleet setting makes the oracle unambiguous and keeps
observations identical across methods.)

Hidden (environment-only) state at `t`:

- `c_t ∈ {0: healthy, 1: degraded, 2: critical}` ~ Categorical(`context_mix[φ_t]`)
- `φ_t` = operating phase (P1, P2, P3, …) from the schedule
- `k_t` = real cycle index, uniform over episode-pool cycles with condition `c_t`

Observation given to the agent: `o_t` = for each of {temperature_C (mean of TS1–4),
cooling_efficiency_pct, cooling_power_kW}: mean, std, min, max, slope over the 60 s window of
cycle `k_t`. **No** `t`, cycle id, phase, or label is exposed.

Because `c_t` and `k_t` are exogenous (independent of actions), every method sees the
**identical observation sequence** for a given seed.

## 2. Action space

`K = 4`, identical for all methods and all contexts:

| id | name | meaning |
|---|---|---|
| A0 | CONTINUE | normal operation |
| A1 | DERATE | reduce load; lower breakdown risk, lost throughput |
| A2 | SCHEDULE_MAINTENANCE | service at next window; moderate downtime |
| A3 | SHUTDOWN_REPAIR | stop and repair now; highest downtime, no breakdown |

The LLM returns a condition label and a **full ranking** of all 4 actions ("candidates" = ranked
set). The full set is always selectable, so a learner is never locked out of the post-drift optimum.

## 3. Reward

Per phase φ: `cost_φ(a)` (downtime fraction), `p_fail_φ(c, a)`, `C_f,φ` (breakdown downtime).

- Breakdown: `fail_t = 1[u_t < p_fail_φ(c_t, a_t)]`, `u_t ~ U(0,1)` pre-sampled
  (common random numbers → monotone coupling across actions and across methods).
- True reward: `R_true = 1 − cost_φ(a) − C_f,φ · fail`
- Normalised: `R_norm = (R_true − r_min)/(r_max − r_min)`,
  `r_max = 1 − min cost`, `r_min = 1 − max cost − max C_f` over all phases (= −0.70 by default).
- Oracle: `a*_t = argmax_a E[R_true | c_t, a, φ_t] = argmax_a 1 − cost_φ(a) − C_f p_fail_φ(c_t,a)`.

Default expected rewards (`C_f = 1`):

| Phase | healthy [A0..A3] | degraded [A0..A3] | critical [A0..A3] |
|---|---|---|---|
| P1 baseline | .99 .89 .80 .50 → **A0** (gap .10) | .40 .60 .75 .50 → **A2** (.15) | .05 .10 .30 .50 → **A3** (.20) |
| P2 parts shortage (A2/A3 downtime ↑) | .99 .89 .40 .30 → **A0** (.10) | .40 .60 .35 .30 → **A1** (.20) | .05 .10 −.10 .30 → **A3** (.20) |
| P3 cooler retrofit + high demand | .99 .74 .70 .50 → **A0** (.25) | .95 .72 .68 .50 → **A0** (.23) | .05 −.05 .20 .50 → **A3** (.30) |

Drift acts on the **degraded** context (A2 → A1 → A0); healthy/critical are stationary and serve
as forgetting probes. Gaps ≥ .10 were chosen for **learnability by any learner** under clean
feedback (Phase 2 checks this empirically), not to favour RIFL-IoT.

## 4. Implicit feedback

The agent never sees `R_true`. Each step emits one feedback event with two channels:

- **Channel A (production log / throughput meter):** `y_A = R_norm + σ_A ε`, `ε ~ N(0,1)`,
  `σ_A = 0.03`, **not clipped** (so exact repeats are a genuine stuck-sensor cue).
- **Channel B (independent breakdown indicator):** `b = fail XOR 1[v < e_B]`, `e_B = 0.02`,
  symmetric FP/FN (asymmetric rates bias agreement-based weighting; tested as sensitivity).
- Plant convention given to **all** methods: breakdown score threshold `τ = 0.5`
  (`â = 1[y_A < τ]`). Validation checks `τ` separates max-fail (.41) and min-no-fail (.59) scores
  by > 2.5 σ_A.
- Event tagging: v1 **tagged** (event carries origin step, like a work-order ID).
  Untagged mode is v2.

Corruption processes (ground-truth flags logged; draws pre-sampled per step, independent streams):

| Condition | Process |
|---|---|
| Noise η ∈ {.1,.3,.5} (`mixed`) | w.p. η, channel A corrupted with type ~ U{flip, spike, stuck, replace}: *flip* = score of the opposite breakdown outcome; *spike* = value in [1.5,2.5] or [−2.5,−1.5]; *stuck* = repeat last emitted `y_A`; *replace* = U(0,1) |
| Negative control `symmetric_flip` η=.5 | w.p. η flip → zero information; **no method should learn** |
| Negative control `correlated` | w.p. η flip A **and** B together → undetectable by redundancy |
| Delay D ∈ {0,1,5,10,20} | `Δ ~ U{⌈D/2⌉,…,⌊3D/2⌋}` (mean D); delivered at start of step `t+Δ`; events arriving after T are dropped and logged |
| Missing m ∈ {.1,.3,.5} (MCAR) | whole event dropped w.p. m |
| Conflict q = .3 | w.p. q exactly one channel is faulty (A or B, p = .5): faulty A → flip; faulty B → `b` negated |
| Drift | phase schedule P1(500) → P2(500) → P3(500) → P1(500) |

Note: 50 % `mixed` noise is still informative (not a coin flip); the symmetric-flip control
covers the information-theoretic limit.

## 5. Reliability (agent side, v1 rule-based)

`ρ_t = v_range · v_stuck · v_agree ∈ {0, κ, 1}`

- `v_range = 1[−0.2 ≤ y_A ≤ 1.2]` (clean values lie in [0,1] ± 6σ_A)
- `v_stuck = 0` if `y_A` exactly equals the previously **received** `y_A`, else 1
- `v_agree = 1` if `â == b`, else `κ` (default 0.2; tuned on validation seeds)

Ground truth for evaluation: `corrupted_A` (the reward-bearing value is wrong).
Metrics: F1 of `ρ < 1` vs `corrupted_A`, AUROC of `−ρ`, ECE treating `ρ` as P(clean), all
reported per corruption type. Expected: *replace* corruptions with `â == b` are undetectable
(≈ ½ of replace events) — reported, not hidden.

Deliberately excluded from v1: plausibility against the learner's own reward estimate
(it suppresses exactly the surprising feedback needed after drift).

## 6. Temporal credit (v1, tagged events)

`w(Δ) = exp(−λΔ)`, `λ = ln 2 / h`, half-life `h` tuned on validation seeds (default 10).
Update target = automaton/action recorded at the event's origin step.

Honest expectation: with tagged events and stationary dynamics, `w < 1` only discards
information → neutral or harmful. It can help only when **delay and drift co-occur**
(stale feedback from a previous phase). Hence the extra `drift + delay10` cell.
v2 (untagged): lag kernel over a window of past actions; compared against a trivial
"shift by nominal D" baseline.

## 7. Learning Automaton

Contextual: one automaton per LLM condition label ∈ {healthy, degraded, critical, unknown}.
Selection: `a_t ~ p_{ĉ_t}`. Init on first visit: `p = (1−ω)·uniform + ω·onehot(LLM top action)`, ω = 0.5 (ablated).
Probability floor `p_min = 0.01` after every update (non-absorbing under drift).

**Weights modulate the step size, not the reward value.** Literal `R_eff = ρ·w·R_raw` would
turn unreliable feedback into a *penalty* under reward–penalty schemes — a silent bug.
With `β = clip(y_A, 0, 1)` and `α_t = α·ρ_t·w(Δ_t)`:

1. **L_R-I (S-model):** `p ← p + α_t β (e_a − p)`
2. **L_R-P / L_R-εP (S-model):** reward part `p ← p + α_t β (e_a − p)`;
   penalty part with `b_t = ε·α_t(1−β)`: `p_a ← (1−b_t)p_a`, `p_j ← (1−b_t)p_j + b_t/(K−1)`
   (ε = 1 → L_R-P, ε ≪ 1 → L_R-εP)
3. **Pursuit (EMA estimator):** `d̂_a ← d̂_a + η·ρ_t·w_t (β − d̂_a)`;
   `p ← p + α_t (e_{argmax d̂} − p)`

Default for Phases 6–8 is chosen by the Phase-5 comparison on `clean` and `drift` only
(before any noise experiment is run), and frozen.

## 8. Baselines (all see the same observations and the same feedback stream)

| ID | Definition |
|---|---|
| B0 Vanilla | LLM top-ranked action. Feedback received, ignored. |
| B1 History | Same prompt + last H = 20 records (obs summary, action, feedback events arrived so far: origin step, `y_A`, `b`). T = 0. |
| B2 Naive | Same contextual LA and update rule as B4 with **ρ ≡ 1, w ≡ 1**, credit to tagged origin. Uses `y_A` only. Variant B2-latest: credit to current action. |
| B3 ReAct | Thought/Action/Observation; tools `get_recent_feedback(n≤50)`, `get_recent_history(n≤50)`; ≤ 3 tool calls then `Final: A?`. No persistent learner. |
| B4 RIFL-IoT | Contextual LA + ρ + w(Δ). |
| R-Oracle | true `c_t`, `φ_t` → `a*_t` (upper bound) |
| R-Random | uniform action (lower bound) |
| R-LA-noLLM | B4 with context from a CE threshold classifier fit on the calibration pool — **tests whether the LLM is needed at all** |

By construction, ablation "Full − reliability − temporal credit" == B2. Extra ablation "Full − LA"
replaces the LA with ε-greedy on the same weighted EMA estimates.

Tuning fairness: every learner's α (and λ, κ, η for B4) tuned on **validation seeds 100–102**,
`clean` + `drift` only, same grid size per method. Test seeds 0–4 (B1/B3) and 0–19 (cached methods).

LLM cost: B0/B2/B4/R-LA-noLLM call the LLM with observation-only prompts at temperature 0 →
cached per cycle (≤ 2205 unique calls total per model). B1/B3 are history-dependent and uncached.

## 9. Experiment matrix

Stationary schedule: P1 × 1000 steps. Drift schedule: P1/P2/P3/P1 × 500 = 2000 steps.

| Condition | Params | Schedule |
|---|---|---|
| clean | — | stationary |
| noise10/30/50 | mixed η | stationary |
| delay1/5/10/20 | D (delay0 = clean) | stationary |
| missing10/30/50 | MCAR m | stationary |
| conflict30 | q = .3 | stationary |
| drift | — | drift |
| drift_noise30, drift_delay10 | combined | drift |
| ctrl_symflip50, ctrl_corr30 | negative controls | stationary |

Main-table columns: Clean, 10/30/50 % Noise, **Delay = delay10**, **Missing = missing30**,
Conflict = conflict30, Drift = drift (full curves in figures).
Methods: B0–B4 × all conditions; B1/B3 may be restricted to
{clean, noise30, noise50, delay10, missing30, conflict30, drift} if compute-bound (stated in paper).

Pre-registered expectations (reported whether or not they hold):

| Condition | Expected B4 vs B2 |
|---|---|
| clean, delay (stationary), missing (MCAR) | ≈ equal (no mechanism for gain) |
| noise (mixed), conflict | B4 better, gap grows with η |
| drift_delay10 | B4 better (staleness discount) |
| ctrl_symflip50, ctrl_corr30 | ≈ equal (no detectable signal) |

Primary test: degradation `acc(noiseη) − acc(clean)` for B4 vs B2, paired over seeds
(same trajectories), paired t-test + Wilcoxon, mean ± sd and 95 % CI.

## 10. Outputs

`runs/<experiment>/<method>/<condition>/seed_<s>/`

| File | Content |
|---|---|
| `config_resolved.yaml` | full merged config for this run |
| `seeds.json` | seed, stream ids, split seed, trajectory hash |
| `env_trajectory.npz` | every pre-sampled draw (contexts, cycles, u_fail, noise, corruption, delay, missing) |
| `steps.jsonl` | per step: phase, true context, cycle id, LLM label + ranking, action, selection probs, all automaton probs, oracle action, expected rewards, fail, R_true, R_norm |
| `feedback.jsonl` | per event: origin, delay, arrival, delivered, missing, corrupted, type, conflict, y_A, b, ground truth |
| `feedback_processing.jsonl` | per delivered event: ρ, its components, w, step size, update target |
| `llm_calls.jsonl` | prompt hash, cached, tokens in/out, latency, parse ok, raw response |
| `metrics.json` | per-run metrics |

`results/<experiment>/summary.csv` (one row per run) and `summary_agg.csv`
(mean, sd, 95 % CI per method × condition × metric). Identical `(seed, schedule)` ⇒ identical
trajectory hash across all methods and conditions; the runner asserts this.

## Metric definitions (windows W = 100 steps)

- Decision accuracy: `mean 1[a_t = a*_t]` (overall, per context).
- Task success: mean `R_norm`; expected regret `Σ E[R|a*] − E[R|a_t]`.
- Before drift: accuracy in last W steps before each boundary; after drift: first W after;
  final: last W of the episode.
- Adaptation steps: steps after a boundary until the rolling accuracy over the last 30 visits
  to contexts whose oracle changed reaches 0.8 (censored at segment length, flag logged).
- Adaptation gain: acc(last W) − acc(first W) of a segment.
- Forgetting (P1 recurs): acc(last W of first P1) − acc(first W of recurring P1); recovery = its adaptation steps.
- System: logical/unique LLM calls, tokens, latency, non-LLM overhead per step.
