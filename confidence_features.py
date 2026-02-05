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
    description="Feature-level internal consistency + heatmaps"
)

parser.add_argument("--project", type=str, help="Project name (skip interactive selection)")
parser.add_argument("--dtw-downsample", type=int, default=5)

args = parser.parse_args()

INPUT_DIR = "INPUT"
OUTPUT_DIR = "OUTPUT"

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
    print(f"[PROJECT] Using project (CLI): {project}")
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
    print(f"[PROJECT] Using project (interactive): {project}")

P_IN = os.path.join(INPUT_DIR, project)
P_OUT = os.path.join(OUTPUT_DIR, project)
os.makedirs(P_OUT, exist_ok=True)

DS = max(1, args.dtw_downsample)

# =================================================
# Feature consistency
# =================================================

for csv in glob.glob(os.path.join(P_IN, "*.csv")):
    if os.path.basename(csv).startswith("weights"):
        continue

    modality = os.path.splitext(os.path.basename(csv))[0]
    print(f"\n[FEATURE] {modality}")

    df = pd.read_csv(csv)[["x_axis", "y_axis", "feature"]]
    pivot = df.pivot(index="x_axis", columns="feature", values="y_axis").fillna(0)

    X = pivot.values
    feats = pivot.columns.astype(str).tolist()
    F = len(feats)

    rows = []
    for i in range(F):
        for j in range(i + 1, F):
            rows.append({
                "feature_A": feats[i],
                "feature_B": feats[j],
                "corr": np.corrcoef(X[:, i], X[:, j])[0, 1],
                "dtw": dtw(X[::DS, i], X[::DS, j]),
            })

    dfc = pd.DataFrame(rows)
    out_csv = os.path.join(P_OUT, f"feature_internal_consistency_{modality}.csv")
    dfc.to_csv(out_csv, index=False)
    print(f"[FEATURE] written {out_csv}")

    # ---- heatmaps ----
    feats_all = sorted(set(dfc["feature_A"]).union(dfc["feature_B"]))
    idx = {f: i for i, f in enumerate(feats_all)}
    N = len(feats_all)

    tau = max(dfc["dtw"].median(), 1e-9)

    mats = {
        "corr": np.eye(N),
        "dtw": np.eye(N),
        "agree": np.eye(N),
    }

    for _, r in dfc.iterrows():
        i, j = idx[r["feature_A"]], idx[r["feature_B"]]
        s_corr = (r["corr"] + 1) / 2
        s_dtw = np.exp(-r["dtw"] / tau)
        s_agree = 0.5 * s_corr + 0.5 * s_dtw
        mats["corr"][i, j] = mats["corr"][j, i] = s_corr
        mats["dtw"][i, j] = mats["dtw"][j, i] = s_dtw
        mats["agree"][i, j] = mats["agree"][j, i] = s_agree

    for name, M in mats.items():
        plt.figure(figsize=(7, 6))
        plt.imshow(M, vmin=0, vmax=1, cmap="viridis")
        plt.colorbar()
        plt.xticks(range(N), feats_all, rotation=90)
        plt.yticks(range(N), feats_all)
        plt.title(f"{modality} – {name}")
        plt.tight_layout()
        plt.savefig(os.path.join(P_OUT, f"feature_heatmap_{modality}_{name}.png"))
        plt.close()

print("\n✓ Feature-level consistency + heatmaps done")
