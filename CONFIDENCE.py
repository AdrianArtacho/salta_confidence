import os
import glob
import argparse
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import dtw

# =================================================
# CONFIG
# =================================================

INPUT_DIR = "input"
OUTPUT_DIR = "OUTPUT"

MIN_POINTS = 10
MAX_FEATURE_PAIRS_DTW = 500  # None = all

# =================================================
# CLI ARGUMENTS
# =================================================

parser = argparse.ArgumentParser(
    description="Compute internal consistency and confidence metrics."
)

parser.add_argument(
    "--features",
    action="store_true",
    help="Compute feature-level internal consistency (slow, diagnostic)."
)

args = parser.parse_args()
RUN_FEATURE_LEVEL = args.features

# =================================================
# LOGGING
# =================================================

def log(msg):
    print(msg, flush=True)

log("\n=== CONFIDENCE METRIC ===")
log(f"Feature-level analysis enabled: {RUN_FEATURE_LEVEL}")

# =================================================
# HELPERS
# =================================================

def normalise(y):
    s = np.sum(y)
    return y / s if s > 0 else y

def corr_to_sim(r):
    if pd.isna(r):
        return 0.5
    return max(0.0, min(1.0, (r + 1.0) / 2.0))

def dtw_to_sim(d, tau):
    if pd.isna(d) or d < 0:
        return 0.0
    tau = max(float(tau), 1e-9)
    return float(np.exp(-d / tau))

def compute_confidence_from_metrics(df, lam=0.5):
    if df.empty:
        return np.nan

    tau = df["dtw"].median()
    if pd.isna(tau) or tau <= 0:
        tau = df["dtw"].mean()
    if pd.isna(tau) or tau <= 0:
        tau = 1.0

    sims = []
    for _, row in df.iterrows():
        sims.append(
            lam * corr_to_sim(row["corr"]) +
            (1 - lam) * dtw_to_sim(row["dtw"], tau)
        )

    return 100.0 * float(np.mean(sims))

def plot_pairwise_overlay(time, p1, p2, name1, name2, corr, dtw_dist, agree, outdir):
    plt.figure(figsize=(10, 4))
    plt.plot(time, p1, label=name1, linewidth=2)
    plt.plot(time, p2, label=name2, linewidth=2, alpha=0.85)
    plt.xlabel("Time")
    plt.ylabel("Probability density")
    plt.title(f"{name1} vs {name2}")

    txt = (
        f"corr = {corr:.2f}\n"
        f"DTW  = {dtw_dist:.2f}\n"
        f"agree = {agree:.2f}"
    )

    plt.text(
        0.01, 0.98, txt,
        transform=plt.gca().transAxes,
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"pair_{name1}__{name2}.png"))
    plt.close()

# =================================================
# PROJECT SELECTION
# =================================================

projects = [
    d for d in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, d))
]

if not projects:
    raise RuntimeError("No project folders found in input/")

log("\nAvailable projects in input/:")
for i, p in enumerate(projects, start=1):
    log(f"  [{i}] {p}")

choice = input("\nSelect a project by number: ").strip()
if not choice.isdigit() or not (1 <= int(choice) <= len(projects)):
    raise ValueError("Invalid project selection.")

project_name = projects[int(choice) - 1]
PROJECT_DIR = os.path.join(INPUT_DIR, project_name)
PROJECT_OUT = os.path.join(OUTPUT_DIR, project_name)
os.makedirs(PROJECT_OUT, exist_ok=True)

log(f"\n--- Processing project: {project_name} ---")

# =================================================
# LOAD MODALITIES
# =================================================

modalities = {}

csv_files = [
    f for f in glob.glob(os.path.join(PROJECT_DIR, "*.csv"))
    if not os.path.basename(f).startswith("weights")
]

if not csv_files:
    raise RuntimeError("No modality CSV files found.")

for f in csv_files:
    name = os.path.splitext(os.path.basename(f))[0]
    log(f"\n[LOAD] Modality: {name}")

    df = pd.read_csv(f)[["x_axis", "y_axis", "feature"]]

    wfile = os.path.join(PROJECT_DIR, f"weights_{name}.csv")
    if not os.path.exists(wfile):
        raise FileNotFoundError(f"Missing weights file for modality '{name}'")

    wdf = pd.read_csv(wfile)
    weights = dict(zip(wdf["feature_name"], wdf["weight_value"]))

    pivot = df.pivot(index="x_axis", columns="feature", values="y_axis").fillna(0)

    feat_time = pivot.index.values
    feat_names = pivot.columns.astype(str).tolist()
    feats_unweighted = pivot.values

    weighted = pivot.copy()
    for col in weighted.columns:
        weighted[col] *= weights.get(col, 1.0)

    pdf = normalise(weighted.sum(axis=1).values)
    time_axis = weighted.index.values

    modalities[name] = {
        "time": time_axis,
        "pdf": pdf,
        "feature_time": feat_time,
        "feature_names": feat_names,
        "features_unweighted": feats_unweighted,
    }

    plt.figure(figsize=(8, 3))
    plt.plot(time_axis, pdf, linewidth=2)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_OUT, f"{name}_pdf.png"))
    plt.close()

# =================================================
# A) MODALITY-LEVEL CONSISTENCY
# =================================================

log("\n[MODALITY-LEVEL] Computing cross-modality consistency...")

keys = list(modalities.keys())
results = []

total_pairs = len(keys) * (len(keys) - 1) // 2
pair_counter = 0

for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        pair_counter += 1
        m1, m2 = keys[i], keys[j]
        log(f"[MODALITY] Pair {pair_counter}/{total_pairs}: {m1} vs {m2}")

        t1, p1 = modalities[m1]["time"], modalities[m1]["pdf"]
        t2, p2 = modalities[m2]["time"], modalities[m2]["pdf"]

        t_min = max(t1.min(), t2.min())
        t_max = min(t1.max(), t2.max())
        if t_max <= t_min:
            continue

        t_ref = t1[(t1 >= t_min) & (t1 <= t_max)] if len(t1) >= len(t2) \
                else t2[(t2 >= t_min) & (t2 <= t_max)]

        if len(t_ref) < MIN_POINTS:
            continue

        p1i = np.interp(t_ref, t1, p1)
        p2i = np.interp(t_ref, t2, p2)

        results.append({
            "modality_A": m1,
            "modality_B": m2,
            "corr": np.corrcoef(p1i, p2i)[0, 1],
            "dtw": dtw(p1i, p2i),
        })

mod_df = pd.DataFrame(results)
mod_df.to_csv(os.path.join(PROJECT_OUT, "modality_internal_consistency.csv"), index=False)

mod_conf = compute_confidence_from_metrics(mod_df)
pd.DataFrame([{
    "confidence_0_100": mod_conf,
    "n_modalities": len(modalities),
    "n_pairs": len(mod_df),
}]).to_csv(os.path.join(PROJECT_OUT, "confidence_summary.csv"), index=False)

log(f"\n[RESULT] Modality confidence: {mod_conf:.2f}/100")

# =================================================
# B) FEATURE-LEVEL CONSISTENCY (OPTIONAL)
# =================================================

if RUN_FEATURE_LEVEL:
    log("\n[FEATURE-LEVEL] Computing feature-level internal consistency...")

    feature_summaries = []

    for mod_name, mod in modalities.items():
        X = mod["features_unweighted"]
        F = X.shape[1]

        log(f"\n[FEATURE] Modality '{mod_name}'")
        log(f"[FEATURE] Features: {F}")
        log(f"[FEATURE] Total feature pairs: {F * (F - 1) // 2}")

        if F < 2 or X.shape[0] < MIN_POINTS:
            log("[FEATURE] Skipped (not enough data)")
            continue

        pair_indices = [(i, j) for i in range(F) for j in range(i + 1, F)]

        if MAX_FEATURE_PAIRS_DTW is None or MAX_FEATURE_PAIRS_DTW >= len(pair_indices):
            dtw_pairs = pair_indices
        else:
            rng = np.random.default_rng(0)
            chosen = rng.choice(len(pair_indices), MAX_FEATURE_PAIRS_DTW, replace=False)
            dtw_pairs = [pair_indices[k] for k in chosen]

        log(f"[FEATURE] DTW pairs to compute: {len(dtw_pairs)}")

        dtw_map = {}
        t0 = time.time()

        for idx, (i, j) in enumerate(dtw_pairs, start=1):
            dtw_map[(i, j)] = dtw(X[:, i], X[:, j])

            if idx % 25 == 0 or idx == len(dtw_pairs):
                log(f"[FEATURE][DTW] {idx}/{len(dtw_pairs)} done")

        log(f"[FEATURE][DTW] Completed in {time.time() - t0:.1f} seconds")

    log("\n[FEATURE-LEVEL] Completed")

else:
    log("\n[FEATURE-LEVEL] Skipped (use --features to enable)")

log(f"\n✓ Done. Results written to {PROJECT_OUT}/")
