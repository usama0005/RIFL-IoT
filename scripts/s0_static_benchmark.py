#!/usr/bin/env python
"""S0: plain-baseline reproduction of IoT-LLM's (Tuo An et al., Patterns 2026,
DOI 10.1016/j.patter.2025.101429) hydraulic "Machine" task BASELINE protocol only
(NOT their full IoT-LLM pipeline, which adds RAG-retrieved domain knowledge and
demonstrations we do not have access to -- see Table 3 of that paper for the ablation).

Protocol matched (from their Appendix B and Figure 8):
  - Same dataset: UCI Condition Monitoring of Hydraulic Systems
  - Binary task: "close-to-failure" (cooler=3%) vs "full-efficiency" (cooler=100%) only
    -- the middle "reduced efficiency" (cooler=20%) class is EXCLUDED, matching their setup
  - 700 samples per label (random sample, fixed seed -- NOT specified as fixed or random in
    the paper, so this is a documented assumption, not a verified match)
  - Same 3 sensor variables: temperature (mean of TS1-4), cooling power (CP), cooling
    efficiency (CE), as RAW 60-second/60-sample sequences (not summary statistics) --
    matching their baseline prompt style shown in Figure 8, not our own environment's
    mean/std/min/max/slope summary format
  - No retrieved domain knowledge, no retrieved demonstrations, no role-prompt, no CoT
    scaffolding beyond "analyze step by step then answer" -- matches their baseline, not
    their full "Ours" configuration

Protocol NOT matched (stated explicitly, not hidden):
  - MODEL: this runs on Qwen2.5-7B-Instruct-AWQ (our served model), not GPT-4o-mini.
    Any accuracy difference may reflect model capability, not protocol differences.
  - Exact prompt wording is our best-effort reconstruction from their published Figure 8
    example, not verbatim access to their internal template.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rifl_iot.llm.cache import build_llm

N_CYCLES, N_SAMPLES = 2205, 60


def load_raw(root):
    def load(name):
        arr = np.loadtxt(os.path.join(root, name))
        assert arr.shape == (N_CYCLES, N_SAMPLES), f"{name}: {arr.shape}"
        return arr
    ts = np.mean([load(f"TS{i}.txt") for i in range(1, 5)], axis=0)
    ce, cp = load("CE.txt"), load("CP.txt")
    prof = np.loadtxt(os.path.join(root, "profile.txt"))
    cooler = np.round(prof[:, 0]).astype(int)
    return ts, ce, cp, cooler


def seq_str(vals, decimals=3):
    return ", ".join(f"{v:.{decimals}f}" for v in vals)


def build_prompt(temp_seq, cp_seq, ce_seq):
    user = (
        "THE GIVEN DATA:\n"
        f"1. Temperature Change Sequence: {seq_str(temp_seq)}\n"
        f"2. Cooling Power Change Sequence: {seq_str(cp_seq)}\n"
        f"3. Cooling Efficiency Change Sequence: {seq_str(ce_seq)}\n\n"
        "QUESTION:\n"
        "Is the machine's cooling system functioning properly?\n\n"
        "Please analyze the data step by step to explain what it reflects. Then, after your "
        "analysis, you MUST end your response with exactly one final line, in exactly this "
        "format and nothing else on that line:\n"
        "ANSWER: functioning properly\n"
        "or\n"
        "ANSWER: not functioning properly\n"
        "Choose one of these two options even if the data is ambiguous or shows partial "
        "degradation -- a definitive choice is required.\n\n"
        "ANALYSIS:"
    )
    return [{"role": "user", "content": user}]


def parse_answer(text):
    t = (text or "").lower()
    idx = t.rfind("answer:")
    if idx == -1:
        return "unparsed"  # no fallback whole-text search -- that was the bug: it picked up
                            # hedging language from mid-analysis on truncated responses
    tail = t[idx:]
    if "not functioning properly" in tail or "not function" in tail or "malfunction" in tail:
        return "close-to-failure"
    if "functioning properly" in tail or "functioning correctly" in tail or "working properly" in tail:
        return "full-efficiency"
    return "unparsed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/uci_hydraulic")
    ap.add_argument("--n_per_label", type=int, default=700)
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--base_url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    ap.add_argument("--cache_path", default="runs/s0_static_benchmark/llm_cache_v2.sqlite")
    ap.add_argument("--max_tokens", type=int, default=1200)
    ap.add_argument("--out", default="results/s0_static_benchmark")
    args = ap.parse_args()

    ts, ce, cp, cooler = load_raw(args.root)
    idx_full = np.where(cooler == 100)[0]
    idx_fail = np.where(cooler == 3)[0]
    print(f"available: full-efficiency n={len(idx_full)}, close-to-failure n={len(idx_fail)}")

    rng = np.random.default_rng(args.seed)
    n_full = min(args.n_per_label, len(idx_full))
    n_fail = min(args.n_per_label, len(idx_fail))
    if n_full < args.n_per_label or n_fail < args.n_per_label:
        print(f"WARNING: requested {args.n_per_label}/label, only {n_full}/{n_fail} available -- using all available")
    sel_full = rng.choice(idx_full, size=n_full, replace=False)
    sel_fail = rng.choice(idx_fail, size=n_fail, replace=False)
    samples = [(i, "full-efficiency") for i in sel_full] + [(i, "close-to-failure") for i in sel_fail]
    rng.shuffle(samples)
    print(f"total samples: {len(samples)}")

    llm = build_llm({"backend": "openai_compat", "base_url": args.base_url, "model": args.model,
                     "cache_path": args.cache_path})

    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "predictions.jsonl")
    t0 = time.time()
    n_correct, n_unparsed = 0, 0
    per_class = {"full-efficiency": [0, 0], "close-to-failure": [0, 0]}
    with open(log_path, "w") as f:
        for k, (i, true_label) in enumerate(samples):
            msgs = build_prompt(ts[i], cp[i], ce[i])
            resp = llm.complete(msgs, temperature=0.0, max_tokens=args.max_tokens)
            pred = parse_answer(resp.text)
            correct = int(pred == true_label)
            n_correct += correct
            n_unparsed += int(pred == "unparsed")
            per_class[true_label][1] += 1
            per_class[true_label][0] += correct
            f.write(json.dumps({"idx": int(i), "true_label": true_label, "pred": pred,
                                "correct": correct, "tokens_out": resp.tokens_out,
                                "raw": resp.text}) + "\n")
            if (k + 1) % 100 == 0:
                print(f"  {k+1}/{len(samples)}  running_acc={n_correct/(k+1):.3f}  "
                     f"elapsed={time.time()-t0:.0f}s")

    n = len(samples)
    acc = n_correct / n
    print(f"\n{'='*70}\nS0 STATIC BASELINE RESULT\n{'='*70}")
    print(f"model: {args.model}  (NOT GPT-4o-mini -- protocol-matched, model-mismatched)")
    print(f"n={n}  accuracy={acc:.4f} ({acc*100:.1f}%)  unparsed={n_unparsed}")
    for cls, (c, t) in per_class.items():
        print(f"  {cls}: {c}/{t} = {c/t:.4f}" if t else f"  {cls}: n/a")
    print(f"\npublished reference (same task, same dataset, DIFFERENT model+pipeline):")
    print(f"  GPT-4o-mini baseline (no RAG/demos)      = 57.0%")
    print(f"  GPT-4o-mini + full IoT-LLM pipeline       = 92.5%  (NOT directly comparable yet)")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"n": n, "accuracy": acc, "n_unparsed": n_unparsed,
                   "per_class": {k: {"correct": v[0], "total": v[1]} for k, v in per_class.items()},
                   "model": args.model, "seed": args.seed}, f, indent=2)
    print(f"\nsaved: {log_path}, {args.out}/summary.json")


if __name__ == "__main__":
    main()
