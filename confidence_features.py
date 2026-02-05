import os
import glob
import argparse
import gc
import time
import numpy as np
import pandas as pd
from tslearn.metrics import dtw

INPUT_DIR = "INPUT"
OUTPUT_DIR = "OUTPUT"
MIN_POINTS = 10

parser = argparse.ArgumentParser(description="Feature-level consistency (memory-safe)")
parser.add_argument("--dtw-downsample", type=int, default=1)
args = parser.parse_args()
DS = max(1, args.dtw_downsample)

def log(msg):
    print(msg, flush=True)

log("\n=== FEATURE-LEVEL CONSISTENCY ===")
log(f"DTW downsample factor: {DS}")

projects = [d for d in os.listdir(INPUT_DIR)
            if os.path.isdir(os.path.join(INPUT_DIR, d))]

for i, p in enumerate(projects, 1):
    print(f"[{i}] {p}")

choice = int(input("Select project: ")) - 1
project = projects[choice]

P_IN = os.path.join(INPUT_DIR, project)
P_OUT = os.path.join(OUTPUT_DIR, project)
os.makedirs(P_OUT, exist_ok=True)

for csv in glob.glob(os.path.join(P_IN, "*.csv")):
    if os.path.basename(csv).startswith("weights"):
        continue

    name = os.path.splitext(os.path.basename(csv))[0]
    log(f"\n[FEATURE] Processing modality: {name}")

    df = pd.read_csv(csv)[["x_axis", "y_axis", "feature"]]
    pivot = df.pivot(index="x_axis", columns="feature", values="y_axis").fillna(0)

    X = pivot.values
    features = pivot.columns.astype(str).tolist()

    F = X.shape[1]
    log(f"[FEATURE] {F} features")

    rows = []
    t0 = time.time()

    for i in range(F):
        for j in range(i + 1, F):
            d = dtw(X[::DS, i], X[::DS, j])
            corr = np.corrcoef(X[:, i], X[:, j])[0, 1]

            rows.append({
                "feature_A": features[i],
                "feature_B": features[j],
                "corr": corr,
                "dtw": d
            })

        if i % 2 == 0:
            log(f"[FEATURE] feature {i+1}/{F}")

    out = os.path.join(P_OUT, f"feature_internal_consistency_{name}.csv")
    pd.DataFrame(rows).to_csv(out, index=False)

    log(f"[FEATURE] written {out}")
    log(f"[FEATURE] time: {time.time() - t0:.1f}s")

    del df, pivot, X
    gc.collect()

log("\n✓ Feature-level analysis done")
