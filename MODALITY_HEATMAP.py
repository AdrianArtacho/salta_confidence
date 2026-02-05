import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CLI ----------------

parser = argparse.ArgumentParser(
    description="Generate modality-level agreement heat maps"
)

parser.add_argument(
    "csv",
    help="modality_internal_consistency.csv"
)

parser.add_argument(
    "--lambda-corr",
    type=float,
    default=0.5,
    help="Weight for correlation in combined agreement"
)

args = parser.parse_args()
LAMBDA = args.lambda_corr

# ---------------- Helpers ----------------

def corr_to_sim(r):
    if not np.isfinite(r):
        return 0.5
    return (r + 1.0) / 2.0

def dtw_to_sim(d, tau):
    if not np.isfinite(d) or d < 0:
        return 0.0
    return np.exp(-d / tau)

# ---------------- Load ----------------

df = pd.read_csv(args.csv)

modalities = sorted(
    set(df["modality_A"]).union(df["modality_B"])
)

idx = {m: i for i, m in enumerate(modalities)}
N = len(modalities)

# DTW scale
tau = df["dtw"].median()
if not np.isfinite(tau) or tau <= 0:
    tau = df["dtw"].mean()
if not np.isfinite(tau) or tau <= 0:
    tau = 1.0

matrices = {
    "corr": np.zeros((N, N)),
    "dtw": np.zeros((N, N)),
    "agree": np.zeros((N, N)),
}

for M in matrices.values():
    np.fill_diagonal(M, 1.0)

for _, row in df.iterrows():
    i = idx[row["modality_A"]]
    j = idx[row["modality_B"]]

    s_corr = corr_to_sim(row["corr"])
    s_dtw = dtw_to_sim(row["dtw"], tau)
    s_agree = LAMBDA * s_corr + (1 - LAMBDA) * s_dtw

    matrices["corr"][i, j] = matrices["corr"][j, i] = s_corr
    matrices["dtw"][i, j] = matrices["dtw"][j, i] = s_dtw
    matrices["agree"][i, j] = matrices["agree"][j, i] = s_agree

# ---------------- Plot ----------------

outdir = os.path.dirname(args.csv)

for name, M in matrices.items():
    plt.figure(figsize=(6, 5))
    im = plt.imshow(M, cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks(range(N), modalities, rotation=45, ha="right")
    plt.yticks(range(N), modalities)

    plt.title(f"Modality agreement — {name}")
    plt.tight_layout()

    outfile = os.path.join(outdir, f"modality_heatmap_{name}.png")
    plt.savefig(outfile, dpi=200)
    plt.close()

    print(f"✓ written {outfile}")
