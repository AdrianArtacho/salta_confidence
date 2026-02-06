#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import pandas as pd
from tslearn.metrics import dtw

# =================================================
# CLI
# =================================================

parser = argparse.ArgumentParser(
    description="Compute time-resolved pairwise modality agreement"
)

parser.add_argument(
    "--project",
    required=True,
    help="Project name inside INPUT/"
)

parser.add_argument(
    "--window",
    type=float,
    default=2.0,
    help="Window size in seconds (total width)"
)

parser.add_argument(
    "--step",
    type=float,
    default=0.5,
    help="Step size in seconds"
)

parser.add_argument(
    "--dtw-downsample",
    type=int,
    default=5,
    help="Downsampling factor inside DTW windows"
)

args = parser.parse_args()

INPUT_DIR = "INPUT"
OUTPUT_DIR = "OUTPUT"

# =================================================
# Helpers
# =================================================

def normalise(x):
    s = np.sum(x)
    return x / s if s > 0 else x

def corr_to_sim(r):
    if not np.isfinite(r):
        return 0.5
    return (r + 1.0) / 2.0

def dtw_to_sim(d, tau):
    if not np.isfinite(d) or tau <= 0:
        return 0.0
    return np.exp(-d / tau)

# =================================================
# Load modality PDFs
# =================================================

P_IN = os.path.join(INPUT_DIR, args.project)
P_OUT = os.path.join(OUTPUT_DIR, args.project)
os.makedirs(P_OUT, exist_ok=True)

modalities = {}

for csv in glob.glob(os.path.join(P_IN, "*.csv")):
    if os.path.basename(csv).startswith("weights"):
        continue

    name = os.path.splitext(os.path.basename(csv))[0]
    print(f"[LOAD] {name}")

    df = pd.read_csv(csv)[["x_axis", "y_axis", "feature"]]

    # aggregate duplicates safely
    pivot = (
        df
        .pivot_table(
            index="x_axis",
            columns="feature",
            values="y_axis",
            aggfunc="mean"
        )
        .fillna(0)
    )

    pdf = normalise(pivot.sum(axis=1).values)
    t = pivot.index.values.astype(float)

    modalities[name] = (t, pdf)

if len(modalities) < 2:
    raise RuntimeError("Need at least two modalities")

# =================================================
# Common time grid
# =================================================

t_min = max(t.min() for t, _ in modalities.values())
t_max = min(t.max() for t, _ in modalities.values())

times = np.arange(t_min, t_max, args.step)

pairs = []
keys = list(modalities.keys())
for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        pairs.append((keys[i], keys[j]))

# precompute DTW scale globally
all_dtw = []

# =================================================
# Agreement time series
# =================================================

rows = []

half_w = args.window / 2
DS = max(1, args.dtw_downsample)

for tk in times:
    row = {"time": tk}

    for m1, m2 in pairs:
        t1, p1 = modalities[m1]
        t2, p2 = modalities[m2]

        # window masks
        m1_idx = (t1 >= tk - half_w) & (t1 <= tk + half_w)
        m2_idx = (t2 >= tk - half_w) & (t2 <= tk + half_w)

        if m1_idx.sum() < 5 or m2_idx.sum() < 5:
            row[f"{m1}__{m2}"] = np.nan
            continue

        s1 = p1[m1_idx]
        s2 = p2[m2_idx]

        # resample to same length
        n = min(len(s1), len(s2))
        s1 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s1)), s1)
        s2 = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s2)), s2)

        corr = np.corrcoef(s1, s2)[0, 1]
        d = dtw(s1[::DS], s2[::DS])

        all_dtw.append(d)

        row[f"{m1}__{m2}"] = (corr, d)

    rows.append(row)

# =================================================
# Convert to agreement values
# =================================================

tau = max(np.nanmedian(all_dtw), 1e-9)

out_rows = []

for r in rows:
    out = {"time": r["time"]}
    for k, v in r.items():
        if k == "time":
            continue
        if v is None or isinstance(v, float) and np.isnan(v):
            out[k] = np.nan
        else:
            corr, d = v
            out[k] = 0.5 * corr_to_sim(corr) + 0.5 * dtw_to_sim(d, tau)
    out_rows.append(out)

df_out = pd.DataFrame(out_rows)

out_csv = os.path.join(
    P_OUT,
    "modality_agreement_timeseries.csv"
)
df_out.to_csv(out_csv, index=False)

print(f"\n✓ Time-resolved agreement written to {out_csv}")
