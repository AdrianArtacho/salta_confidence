#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import dtw

# =================================================
# CLI
# =================================================

parser = argparse.ArgumentParser(
    description="Modality-level internal consistency and confidence"
)

parser.add_argument("--project", type=str, help="Project name (skip interactive selection)")
parser.add_argument("--dtw-downsample", type=int, default=100)
parser.add_argument("--global-plot", action="store_true")
parser.add_argument("--no-normalize", action="store_true")

args = parser.parse_args()

INPUT_DIR = "INPUT"
OUTPUT_DIR = "OUTPUT"

# =================================================
# Helpers
# =================================================

def log(msg):
    print(msg, flush=True)

def normalise(x):
    s = np.sum(x)
    return x / s if s > 0 else x

def corr_to_sim(r):
    return (r + 1) / 2 if np.isfinite(r) else 0.5

def dtw_to_sim(d, tau):
    if tau <= 0:
        return 0.0
    return np.exp(-d / tau)

# =================================================
# Project selection
# =================================================

projects = [
    d for d in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, d))
]

if not projects:
    raise RuntimeError(f"No project folders found in {INPUT_DIR}")

if args.project:
    if args.project not in projects:
        raise ValueError(f"Project '{args.project}' not found. Available: {projects}")
    project = args.project
    log(f"[PROJECT] Using project (CLI): {project}")
else:
    print("\nAvailable projects:")
    for i, p in enumerate(projects, 1):
        print(f"[{i}] {p}")
    while True:
        try:
            idx = int(input("Select project: ")) - 1
            if 0 <= idx < len(projects):
                project = projects[idx]
                break
        except ValueError:
            pass
        print("Invalid selection.")
    log(f"[PROJECT] Using project (interactive): {project}")

P_IN = os.path.join(INPUT_DIR, project)
P_OUT = os.path.join(OUTPUT_DIR, project)
os.makedirs(P_OUT, exist_ok=True)

# =================================================
# Load modality PDFs
# =================================================

modalities = {}

for csv in glob.glob(os.path.join(P_IN, "*.csv")):
    if os.path.basename(csv).startswith("weights"):
        continue

    name = os.path.splitext(os.path.basename(csv))[0]
    log(f"[LOAD] {name}")

    df = pd.read_csv(csv)[["x_axis", "y_axis", "feature"]]
    wdf = pd.read_csv(os.path.join(P_IN, f"weights_{name}.csv"))
    weights = dict(zip(wdf["feature_name"], wdf["weight_value"]))

    pivot = (
        df
        .pivot_table(
            index="x_axis",
            columns="feature",
            values="y_axis",
            aggfunc="mean"   # or "sum", see below
        )
        .fillna(0)
    )
    for col in pivot.columns:
        pivot[col] *= weights.get(col, 1.0)

    pdf = normalise(pivot.sum(axis=1).values)
    t = pivot.index.values

    modalities[name] = (t, pdf)

    plt.figure(figsize=(8, 3))
    plt.plot(t, pdf)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(P_OUT, f"{name}_pdf.png"))
    plt.close()

# =================================================
# Global overlay plot
# =================================================

def plot_global(mods, normalised=True):
    t_min = max(t.min() for t, _ in mods.values())
    t_max = min(t.max() for t, _ in mods.values())
    ref = max(mods, key=lambda k: len(mods[k][0]))
    t_ref = mods[ref][0]
    t_ref = t_ref[(t_ref >= t_min) & (t_ref <= t_max)]

    stack = []
    plt.figure(figsize=(10, 4))

    for name, (t, p) in mods.items():
        pi = np.interp(t_ref, t, p)
        if normalised:
            pi = normalise(pi)
        stack.append(pi)
        plt.plot(t_ref, pi, alpha=0.6, label=name)

    plt.plot(t_ref, np.mean(stack, axis=0), color="black", lw=3, label="mean")
    plt.legend()
    plt.title("Global modality overlay" + (" (norm)" if normalised else " (raw)"))
    plt.tight_layout()

    suffix = "norm" if normalised else "raw"
    plt.savefig(os.path.join(P_OUT, f"global_overlay_{suffix}.png"))
    plt.close()

if args.global_plot and len(modalities) > 1:
    plot_global(modalities, normalised=True)
    if args.no_normalize:
        plot_global(modalities, normalised=False)

# =================================================
# Pairwise modality consistency
# =================================================

rows = []
keys = list(modalities.keys())
DS = max(1, args.dtw_downsample)

log("\n[MODALITY] Computing cross-modality consistency")

for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        m1, m2 = keys[i], keys[j]
        log(f"[MODALITY] {m1} vs {m2}")

        t1, p1 = modalities[m1]
        t2, p2 = modalities[m2]

        t_min = max(t1.min(), t2.min())
        t_max = min(t1.max(), t2.max())
        if t_max <= t_min:
            continue

        t_ref = t1[(t1 >= t_min) & (t1 <= t_max)]
        p1i = np.interp(t_ref, t1, p1)
        p2i = np.interp(t_ref, t2, p2)

        corr = np.corrcoef(p1i, p2i)[0, 1]
        d = dtw(p1i[::DS], p2i[::DS])

        rows.append({
            "modality_A": m1,
            "modality_B": m2,
            "corr": corr,
            "dtw": d,
        })

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(P_OUT, "modality_internal_consistency.csv"), index=False)

if dfm.empty:
    log("[MODALITY] Only one modality — confidence undefined.")
    log("✓ Modality-level analysis done")
    exit(0)

tau = max(dfm["dtw"].median(), 1e-9)
agree = [
    0.5 * corr_to_sim(r["corr"]) + 0.5 * dtw_to_sim(r["dtw"], tau)
    for _, r in dfm.iterrows()
]

confidence = 100 * np.mean(agree)
pd.DataFrame([{"confidence_0_100": confidence}]) \
  .to_csv(os.path.join(P_OUT, "confidence_summary.csv"), index=False)

log(f"\nConfidence: {confidence:.2f}/100")
log("✓ Modality-level analysis done")
