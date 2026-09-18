"""Per-run metrics (DESIGN.md, 'Metric definitions') and seed aggregation."""
import math

import numpy as np

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 14: 2.145, 19: 2.093, 24: 2.064, 29: 2.045}


def _t975(df):
    if df in T975:
        return T975[df]
    keys = [k for k in T975 if k <= df]
    return T975[max(keys)] if df <= 29 else 1.96


def _acc(x):
    return float(np.mean(x)) if len(x) else float("nan")


def adaptation_steps(steps, start, end, contexts, w, thr):
    """Env steps from `start` until rolling acc over last w visits to `contexts` >= thr."""
    hits = [(s["t"], s["correct"]) for s in steps[start:end] if s["true_context"] in contexts]
    for i in range(w - 1, len(hits)):
        if np.mean([c for _, c in hits[i - w + 1:i + 1]]) >= thr:
            return hits[i][0] - start + 1, False
    return end - start, True


def summarize_run(steps, dyn, events, delivered_flags, mcfg):
    W = int(mcfg["window"])
    corr = np.array([s["correct"] for s in steps], float)
    out = {
        "T": len(steps),
        "accuracy": _acc(corr),
        "mean_r_norm": float(np.mean([s["r_norm"] for s in steps])),
        "exp_regret": float(np.sum([s["exp_regret"] for s in steps])),
        "fail_rate": float(np.mean([s["fail"] for s in steps])),
    }
    for c, name in enumerate(dyn.contexts):
        out[f"acc_{name}"] = _acc([s["correct"] for s in steps if s["true_context"] == c])
    llm_ctx = [s["llm_context"] for s in steps]
    if any(x != -2 for x in llm_ctx):
        out["perception_acc"] = _acc([s["llm_context"] == s["true_context"] for s in steps])
        out["unknown_context_rate"] = _acc([s["llm_context"] == -1 for s in steps])

    # segments / drift
    seg_bounds = [0] + dyn.boundaries + [dyn.T]
    segs = []
    for i in range(len(seg_bounds) - 1):
        a, b = seg_bounds[i], seg_bounds[i + 1]
        segs.append({"phase": dyn.schedule[i][0], "acc": _acc(corr[a:b]),
                     "acc_first_W": _acc(corr[a:a + W]), "acc_last_W": _acc(corr[b - W:b])})
    out["segments"] = segs
    out["final_performance"] = segs[-1]["acc_last_W"]
    if len(segs) > 1:
        out["before_drift"] = float(np.mean([segs[i]["acc_last_W"] for i in range(len(segs) - 1)]))
        out["after_drift"] = float(np.mean([segs[i]["acc_first_W"] for i in range(1, len(segs))]))
        out["adaptation_gain"] = float(np.mean([s["acc_last_W"] - s["acc_first_W"] for s in segs[1:]]))
        ad = []
        for i in range(1, len(segs)):
            prev, cur = dyn.segment_phase(i - 1), dyn.segment_phase(i)
            changed = [c for c in range(dyn.C) if prev.oracle(c) != cur.oracle(c)]
            if changed:
                n, cens = adaptation_steps(steps, seg_bounds[i], seg_bounds[i + 1], changed,
                                           int(mcfg["adapt_window_visits"]), float(mcfg["adapt_threshold"]))
                ad.append({"boundary": seg_bounds[i], "contexts": changed, "steps": n, "censored": cens})
        out["adaptation"] = ad
        out["adaptation_steps"] = float(np.mean([x["steps"] for x in ad])) if ad else float("nan")
        forg = []
        for j in range(len(segs)):
            for i in range(j):
                if segs[i]["phase"] == segs[j]["phase"] and i < j - 1:
                    forg.append(segs[i]["acc_last_W"] - segs[j]["acc_first_W"])
                    break
        out["forgetting"] = float(np.mean(forg)) if forg else float("nan")

    # feedback stream (ground truth, method-independent up to action-dependent values)
    n = len(events)
    out["fb_delivered_frac"] = float(np.mean(delivered_flags)) if n else float("nan")
    out["fb_corrupted_frac"] = float(np.mean([e.corrupted_A for e in events])) if n else float("nan")
    out["fb_mean_delay"] = float(np.mean([e.delay for e in events])) if n else float("nan")
    return out


def aggregate(values):
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return {"mean": float("nan"), "sd": float("nan"), "ci95": float("nan"), "n": 0}
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    ci = _t975(len(v) - 1) * sd / math.sqrt(len(v)) if len(v) > 1 else float("nan")
    return {"mean": float(v.mean()), "sd": sd, "ci95": ci, "n": int(len(v))}
