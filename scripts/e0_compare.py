import csv, numpy as np
def load(p):
    return {(r["method"], r["condition"], int(r["seed"])): r for r in csv.DictReader(open(p))}
new = load("results/e0_repro/summary.csv")
old = load("results/v1_submitted/results/review_seed_scaleup/summary.csv")
common = sorted(set(new) & set(old))
print("rows new/old:", len(new), len(old), " keys only in one:", len(set(new) ^ set(old)))
bad = []
maxd = {"accuracy": 0.0, "mean_r_norm": 0.0, "exp_regret": 0.0}
for k in common:
    n, o = new[k], old[k]
    for m in maxd:
        maxd[m] = max(maxd[m], abs(float(n[m]) - float(o[m])))
    if n["trajectory_hash"] != o["trajectory_hash"] or abs(float(n["accuracy"]) - float(o["accuracy"])) > 1e-9:
        bad.append((k, n["accuracy"], o["accuracy"], n["trajectory_hash"], o["trajectory_hash"]))
print("mismatches (accuracy or hash):", len(bad), "of", len(common))
print("max abs diff:", maxd)
for b in bad[:10]:
    print("  ", b)
print("\ne0_repro mean+-SD accuracy (%), n=50:")
for cond in ["clean", "noise10", "noise30", "noise50", "correlated30", "symflip100", "conflict30"]:
    row = []
    for m in ["B0_vanilla", "B2_naive", "B5_blind_mask", "B4_rifl_iot"]:
        v = np.array([float(new[(m, cond, s)]["accuracy"]) for s in range(50) if (m, cond, s) in new]) * 100
        row.append(f"{m[:2]} {v.mean():6.2f}+-{v.std(ddof=1):5.2f}")
    print(f"{cond:<13}", "  ".join(row))
