"""Experiment runner: methods x conditions x seeds, full trajectory logging."""
import csv
import os
import time

import numpy as np
import yaml

from rifl_iot.agents.blind_mask import BlindMaskAgent
from rifl_iot.agents.history import HistoryLLMAgent
from rifl_iot.agents.naive_feedback import NaiveFeedbackAgent
from rifl_iot.agents.rifl_iot_agent import ReliableFeedbackAgent
from rifl_iot.agents.vanilla import VanillaLLMAgent
from rifl_iot.environment.calibration import compute_calibration_summary
from rifl_iot.environment.dataset import load_data
from rifl_iot.environment.dynamics import Dynamics
from rifl_iot.environment.env import MaintenanceEnv
from rifl_iot.environment.trajectory import STREAM_IDS, generate_trajectory
from rifl_iot.evaluation.metrics import aggregate, summarize_run
from rifl_iot.llm.cache import build_llm
from rifl_iot.utils.config import apply_overrides, config_hash
from rifl_iot.utils.io import JsonlWriter, write_json

SCALAR_KEYS = ["accuracy", "mean_r_norm", "exp_regret", "fail_rate", "acc_healthy", "acc_degraded",
               "acc_critical", "perception_acc", "unknown_context_rate", "final_performance",
               "before_drift", "after_drift", "adaptation_gain", "adaptation_steps", "forgetting",
               "fb_delivered_frac", "fb_corrupted_frac", "fb_mean_delay",
               "llm_calls", "llm_calls_uncached", "tokens_in", "tokens_out", "llm_latency_ms",
               "parse_fail_rate", "overhead_ms_per_step"]


def build_agent(method, cfg, llm, system_prompt, seed):
    if method == "B0_vanilla":
        return VanillaLLMAgent(llm, cfg["llm"]["gen"], system_prompt)
    if method == "B2_naive":
        return NaiveFeedbackAgent(llm, cfg["llm"]["gen"], system_prompt, seed, cfg["la"])
    if method == "B1_history":
        return HistoryLLMAgent(llm, cfg["llm"]["gen"], system_prompt, H=cfg.get("b1", {}).get("H", 20))
    if method == "B4_rifl_iot":
        return ReliableFeedbackAgent(llm, cfg["llm"]["gen"], system_prompt, seed, cfg["la"],
                                     cfg.get("reliability"), cfg.get("temporal_credit"))
    if method == "B5_blind_mask":
        return BlindMaskAgent(llm, cfg["llm"]["gen"], system_prompt, seed, cfg["la"],
                              cfg.get("blind_mask"), cfg.get("temporal_credit"))
    raise NotImplementedError(f"{method} is scheduled for a later phase")


def run_single(cfg, method, cond, seed, data, pools, llm, system_prompt):
    out_dir = os.path.join(cfg["output_root"], cfg["experiment_id"], method, cond, f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_resolved.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    dyn = Dynamics(cfg["env"], cfg["run"]["schedule"])
    traj, thash = generate_trajectory(dyn, pools, seed)
    np.savez_compressed(os.path.join(out_dir, "env_trajectory.npz"), **traj)
    write_json(os.path.join(out_dir, "seeds.json"), {
        "seed": seed, "stream_ids": STREAM_IDS, "split_seed": cfg["data"]["split_seed"],
        "trajectory_hash": thash, "schedule": dyn.schedule_name, "config_hash": config_hash(cfg)})

    env = MaintenanceEnv(data, dyn, traj, cfg["feedback"])
    agent = build_agent(method, cfg, llm, system_prompt, seed)
    w_steps = JsonlWriter(os.path.join(out_dir, "steps.jsonl"))
    w_proc = JsonlWriter(os.path.join(out_dir, "feedback_processing.jsonl"))
    w_llm = JsonlWriter(os.path.join(out_dir, "llm_calls.jsonl"))

    steps, llm_recs, t_llm_total, t_wall0 = [], [], 0.0, time.perf_counter()
    for t in range(dyn.T):
        _, public = env.deliver(t)
        for rec in agent.receive_feedback(public, t):
            w_proc.write(rec)
        obs = env.observe(t)
        t0 = time.perf_counter()
        d = agent.act(obs, t)
        t_act = time.perf_counter() - t0
        info = env.step(t, d.action)
        for r in d.llm_calls:
            w_llm.write(r)
            llm_recs.append(r)
            t_llm_total += 0.0 if r["cached"] else r["latency_ms"] / 1e3
        rec = {"t": t, **info, "llm_context": d.llm_context, "llm_ranking": d.llm_ranking,
               "action": d.action, "selection_probs": d.selection_probs,
               "agent_state": agent.state_snapshot(),
               "correct": int(d.action == info["oracle_action"]), "act_time_s": t_act}
        w_steps.write(rec)
        steps.append(rec)
    wall = time.perf_counter() - t_wall0

    delivered = []
    w_fb = JsonlWriter(os.path.join(out_dir, "feedback.jsonl"))
    for ev in env.events:
        dl = env.delivered(ev)
        delivered.append(dl)
        w_fb.write({**ev.record(), "delivered": dl})
    w_fb.close()
    for w in (w_steps, w_proc, w_llm):
        w.close()

    m = summarize_run(steps, dyn, env.events, delivered, cfg["metrics"])
    m.update({
        "method": method, "condition": cond, "seed": seed, "trajectory_hash": thash,
        "llm_calls": len(llm_recs),
        "llm_calls_uncached": sum(not r["cached"] for r in llm_recs),
        "tokens_in": sum(r["tokens_in"] for r in llm_recs),
        "tokens_out": sum(r["tokens_out"] for r in llm_recs),
        "llm_latency_ms": float(np.mean([r["latency_ms"] for r in llm_recs])) if llm_recs else 0.0,
        "parse_fail_rate": float(np.mean([not r["parse_ok"] for r in llm_recs])) if llm_recs else 0.0,
        "overhead_ms_per_step": 1e3 * max(wall - t_llm_total, 0.0) / dyn.T,
    })
    write_json(os.path.join(out_dir, "metrics.json"), m)
    return m


def run_experiment(cfg):
    if cfg["data"]["source"] == "synthetic_smoke" and not cfg.get("smoke_only"):
        raise RuntimeError("synthetic_smoke data is only allowed with smoke_only: true")
    data = load_data(cfg["data"])
    calib_idx, pools = data.split(cfg["data"]["calib_frac"], cfg["data"]["split_seed"],
                                  cfg["data"].get("stable_only", False))
    from rifl_iot.llm.prompts import build_system_prompt
    system_prompt = build_system_prompt(compute_calibration_summary(data, calib_idx))
    llm = build_llm(cfg["llm"])

    rows, hashes = [], {}
    for cond, overrides in cfg["conditions"].items():
        ccfg = apply_overrides(cfg, overrides)
        for method in cfg["methods"]:
            for seed in cfg["seeds"]:
                m = run_single(ccfg, method, cond, seed, data, pools, llm, system_prompt)
                key = (seed, ccfg["run"]["schedule"])
                if hashes.setdefault(key, m["trajectory_hash"]) != m["trajectory_hash"]:
                    raise AssertionError(f"trajectory mismatch for {key}: replay is broken")
                rows.append(m)
                print(f"{method:<14} {cond:<12} seed={seed}  acc={m['accuracy']:.3f}  "
                      f"r={m['mean_r_norm']:.3f}  perc={m.get('perception_acc', float('nan')):.3f}  "
                      f"fb_deliv={m['fb_delivered_frac']:.2f}  fb_corr={m['fb_corrupted_frac']:.2f}  "
                      f"hash={m['trajectory_hash']}")

    res_dir = os.path.join(cfg["results_root"], cfg["experiment_id"])
    os.makedirs(res_dir, exist_ok=True)
    existing_path = os.path.join(res_dir, "summary.csv")
    merged = {}
    if os.path.exists(existing_path):
        for r in csv.DictReader(open(existing_path)):
            r = {k: (None if v == "" else v) for k, v in r.items()}
            merged[(r["method"], r["condition"], r["seed"])] = r
    for r in rows:
        merged[(r["method"], r["condition"], str(r["seed"]))] = r
    rows = list(merged.values())

    with open(existing_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "condition", "seed", "trajectory_hash"] + SCALAR_KEYS,
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    agg_rows = []
    all_methods = sorted(set(r["method"] for r in rows))
    all_conditions = sorted(set(r["condition"] for r in rows))
    for cond in all_conditions:
        for method in all_methods:
            sub = [r for r in rows if r["method"] == method and r["condition"] == cond]
            if not sub:
                continue
            for k in SCALAR_KEYS:
                a = aggregate([r.get(k) for r in sub])
                agg_rows.append({"method": method, "condition": cond, "metric": k, **a})
    with open(os.path.join(res_dir, "summary_agg.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["method", "condition", "metric", "mean", "sd", "ci95", "n"])
        w.writeheader()
        w.writerows(agg_rows)
    print(f"\nreplay check passed: {len(hashes)} unique (seed, schedule) trajectories, "
          f"identical across all methods/conditions")
    print(f"results -> {res_dir}/summary.csv, summary_agg.csv")
    return rows
