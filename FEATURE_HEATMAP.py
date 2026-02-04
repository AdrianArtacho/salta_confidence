import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# CLI arguments
# -------------------------------------------------

parser = argparse.ArgumentParser(
    description="Generate feature-level consistency heat maps."
)

parser.add_argument(
    "csv",
    help="feature_internal_consistency_<MODALITY>.csv file"
)

parser.add_argument(
    "--metric",
    choices=["corr", "dtw", "agree"],
    default="agree",
    help="Metric to visualise (default: agree)"
)

parser.add_argument(
    "--lambda_corr",
    type=float,
    default=0.5,
    help="Weight for correlation when metric=agree"
)

parser.add_argument(
    "--out",
    default=None,
    help="Output directory (default: same as CSV)"
)

args = parser.parse_args()

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def corr_to_sim(r):
    return (r + 1.0) / 2.0

def dtw_to_sim(d, tau):
    return np.exp(-d / tau)

# -------------------------------------------------
# Load data
# -------------------------------------------------

df = pd.read_csv(args.csv)

features = sorted(
    set(df["feature_A"]).union(df["feature_B"])
)

idx = {f: i for i, f in enumerate(features)}
N = len(features)

# initialise matrix
M = np.zeros((N, N))
np.fill_diagonal(M, 1.0)

# DTW scaling
tau = df["dtw"].median()
if not np.isfinite(tau) or tau <= 0:
    tau = df["dtw"].mean()
if not np.isfinite(tau) or tau <= 0:
    tau = 1.0

# -------------------------------------------------
# Fill matrix
# -------------------------------------------------

for _, row in df.iterrows():
    i = idx[row["feature_A"]]
    j = idx[row["feature_B"]]

    if args.metric == "corr":
        val = corr_to_sim(row["corr"])
    elif args.metric == "dtw":
        val = dtw_to_sim(row["dtw"], tau)
    else:  # agree
        s_corr = corr_to_sim(row["corr"])
        s_dtw = dtw_to_sim(row["dtw"], tau)
        val = args.lambda_corr * s_corr + (1 - args.lambda_corr) * s_dtw

    M[i, j] = val
    M[j, i] = val

# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(figsize=(8, 7))
im = plt.imshow(M, cmap="viridis", vmin=0, vmax=1)
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.xticks(range(N), features, rotation=90)
plt.yticks(range(N), features)

title = os.path.splitext(os.path.basename(args.csv))[0]
plt.title(f"{title} — {args.metric}")

plt.tight_layout()

outdir = args.out or os.path.dirname(args.csv)
os.makedirs(outdir, exist_ok=True)

outfile = os.path.join(
    outdir,
    f"{title}_heatmap_{args.metric}.png"
)

plt.savefig(outfile, dpi=200)
plt.close()

print(f"✓ Heat map written to {outfile}")
