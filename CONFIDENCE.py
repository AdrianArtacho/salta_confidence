import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import dtw

# -------------------------------------------------
# Config
# -------------------------------------------------

INPUT_DIR = "input"
OUTPUT_DIR = "OUTPUT"

# Minimum points required to compare two series
MIN_POINTS = 10

# If a modality has many features, feature-pair counts explode.
# This caps DTW computations to keep runtimes sane.
# Set to None to compute all pairs.
MAX_FEATURE_PAIRS_DTW = 500  # e.g., 500 random feature pairs for DTW; correlation still computed for all pairs

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Helpers
# -------------------------------------------------

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

def compute_confidence_from_metrics(df, corr_col="corr", dtw_col="dtw", lam=0.5):
    if df.empty:
        return np.nan

    tau = df[dtw_col].median()
    if pd.isna(tau) or tau <= 0:
        tau = df[dtw_col].mean()
    if pd.isna(tau) or tau <= 0:
        tau = 1.0

    sims = []
    for _, row in df.iterrows():
        s_corr = corr_to_sim(row[corr_col])
        s_dtw = dtw_to_sim(row[dtw_col], tau)
        sims.append(lam * s_corr + (1 - lam) * s_dtw)

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

# -------------------------------------------------
# Load modality CSVs
# -------------------------------------------------

modalities = {}

csv_files = [
    f for f in glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    if not os.path.basename(f).startswith("weights_")
]

for f in csv_files:
    name = os.path.splitext(os.path.basename(f))[0]
    print(f"Processing modality: {name}")

    df = pd.read_csv(f)
    df = df[["x_axis", "y_axis", "feature"]]

    # Load weights
    wfile = os.path.join(INPUT_DIR, f"weights_{name}.csv")
    if os.path.exists(wfile):
        wdf = pd.read_csv(wfile)
        weights = dict(zip(wdf["feature_name"], wdf["weight_value"]))
    else:
        weights = {}

    # Pivot: rows=time, cols=features
    pivot = df.pivot(
        index="x_axis",
        columns="feature",
        values="y_axis"
    ).fillna(0)

    # Unweighted features matrix
    feat_time = pivot.index.values
    feat_names = pivot.columns.astype(str).tolist()
    feats_unweighted = pivot.values  # shape [T, F]

    # Weighted features matrix
    weighted = pivot.copy()
    for col in weighted.columns:
        weighted[col] *= weights.get(col, 1.0)
    feats_weighted = weighted.values

    # Modality PDF (raw, weighted, normalised)
    pdf = normalise(weighted.sum(axis=1).values)
    time = weighted.index.values

    modalities[name] = {
        "time": time,
        "pdf": pdf,
        "feature_time": feat_time,
        "feature_names": feat_names,
        "features_unweighted": feats_unweighted,
        "features_weighted": feats_weighted,
    }

    # Save per-modality plot
    plt.figure(figsize=(8, 3))
    plt.plot(time, pdf, linewidth=2)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_pdf.png"))
    plt.close()

# -------------------------------------------------
# A) Cross-modality comparisons (INTERPOLATED)
# -------------------------------------------------

mod_results = []
keys = list(modalities.keys())

for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        m1, m2 = keys[i], keys[j]

        t1, p1 = modalities[m1]["time"], modalities[m1]["pdf"]
        t2, p2 = modalities[m2]["time"], modalities[m2]["pdf"]

        t_min = max(t1.min(), t2.min())
        t_max = min(t1.max(), t2.max())
        if t_max <= t_min:
            continue

        # reference grid = denser modality
        if len(t1) >= len(t2):
            t_ref = t1[(t1 >= t_min) & (t1 <= t_max)]
        else:
            t_ref = t2[(t2 >= t_min) & (t2 <= t_max)]

        if len(t_ref) < MIN_POINTS:
            continue

        p1i = np.interp(t_ref, t1, p1)
        p2i = np.interp(t_ref, t2, p2)

        corr = np.corrcoef(p1i, p2i)[0, 1]
        dtw_dist = dtw(p1i, p2i)

        mod_results.append({
            "modality_A": m1,
            "modality_B": m2,
            "corr": corr,
            "dtw": dtw_dist,
            "t_min": float(t_min),
            "t_max": float(t_max),
            "n_points": int(len(t_ref)),
        })

mod_metrics_df = pd.DataFrame(mod_results)
mod_metrics_df.to_csv(os.path.join(OUTPUT_DIR, "modality_internal_consistency.csv"), index=False)

# Pairwise overlay plots (A)
if not mod_metrics_df.empty:
    tau = mod_metrics_df["dtw"].median()
    if pd.isna(tau) or tau <= 0:
        tau = 1.0

    for _, row in mod_metrics_df.iterrows():
        m1, m2 = row["modality_A"], row["modality_B"]

        t1, p1 = modalities[m1]["time"], modalities[m1]["pdf"]
        t2, p2 = modalities[m2]["time"], modalities[m2]["pdf"]

        t_min = max(t1.min(), t2.min())
        t_max = min(t1.max(), t2.max())

        if len(t1) >= len(t2):
            t_ref = t1[(t1 >= t_min) & (t1 <= t_max)]
        else:
            t_ref = t2[(t2 >= t_min) & (t2 <= t_max)]

        p1i = np.interp(t_ref, t1, p1)
        p2i = np.interp(t_ref, t2, p2)

        agree = 0.5 * corr_to_sim(row["corr"]) + 0.5 * dtw_to_sim(row["dtw"], tau)

        plot_pairwise_overlay(
            t_ref, p1i, p2i,
            m1, m2,
            row["corr"], row["dtw"], agree,
            OUTPUT_DIR
        )

# Modality-level confidence
modality_confidence = compute_confidence_from_metrics(mod_metrics_df, "corr", "dtw", lam=0.5)
pd.DataFrame([{
    "confidence_0_100": modality_confidence,
    "lambda_corr": 0.5,
    "n_modalities": len(modalities),
    "n_pairs": len(mod_metrics_df),
}]).to_csv(os.path.join(OUTPUT_DIR, "confidence_summary.csv"), index=False)

print(f"Modality confidence (raw, weighted PDFs): {modality_confidence:.2f}/100")

# -------------------------------------------------
# B) Feature-level consistency (within each modality)
# -------------------------------------------------

feature_summaries = []

for mod_name, mod in modalities.items():
    feat_names = mod["feature_names"]
    X = mod["features_unweighted"]  # [T, F]  (unweighted consistency)
    t = mod["feature_time"]

    F = X.shape[1]
    if F < 2:
        feature_summaries.append({
            "modality": mod_name,
            "n_features": F,
            "n_pairs_corr": 0,
            "n_pairs_dtw": 0,
            "feature_confidence_0_100": np.nan,
            "note": "not enough features"
        })
        continue

    # Ensure we have enough time points
    if X.shape[0] < MIN_POINTS:
        feature_summaries.append({
            "modality": mod_name,
            "n_features": F,
            "n_pairs_corr": 0,
            "n_pairs_dtw": 0,
            "feature_confidence_0_100": np.nan,
            "note": "not enough time points"
        })
        continue

    # ---- Correlation for all pairs (fast) ----
    # Standardise each feature lightly to avoid trivial scale dominance:
    # (If you prefer pure raw, delete the z-scoring.)
    Xz = X.copy().astype(float)
    mu = Xz.mean(axis=0, keepdims=True)
    sd = Xz.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0
    Xz = (Xz - mu) / sd

    # corr matrix [F,F]
    C = np.corrcoef(Xz.T)
    # extract upper-tri pairs
    corr_pairs = []
    pair_indices = []
    for i in range(F):
        for j in range(i + 1, F):
            corr_pairs.append(C[i, j])
            pair_indices.append((i, j))

    # ---- DTW for a subset of pairs (can be expensive) ----
    if MAX_FEATURE_PAIRS_DTW is None or MAX_FEATURE_PAIRS_DTW >= len(pair_indices):
        dtw_pairs_to_compute = pair_indices
    else:
        rng = np.random.default_rng(0)  # deterministic subsampling
        chosen = rng.choice(len(pair_indices), size=MAX_FEATURE_PAIRS_DTW, replace=False)
        dtw_pairs_to_compute = [pair_indices[k] for k in chosen]

    dtw_map = {}  # (i,j) -> dtw
    for (i, j) in dtw_pairs_to_compute:
        d = dtw(X[:, i], X[:, j])
        dtw_map[(i, j)] = d

    # Build feature metrics table
    rows = []
    for k, (i, j) in enumerate(pair_indices):
        rows.append({
            "modality": mod_name,
            "feature_A": feat_names[i],
            "feature_B": feat_names[j],
            "corr": float(corr_pairs[k]),
            "dtw": float(dtw_map.get((i, j), np.nan)),
        })

    feat_df = pd.DataFrame(rows)

    # Save per-modality feature pair table
    feat_df.to_csv(
        os.path.join(OUTPUT_DIR, f"feature_internal_consistency_{mod_name}.csv"),
        index=False
    )

    # Compute feature confidence:
    # If DTW was subsampled, rows without DTW get NaN; restrict to those with DTW for fair scoring.
    feat_df_scored = feat_df.dropna(subset=["dtw"])
    feat_conf = compute_confidence_from_metrics(feat_df_scored, "corr", "dtw", lam=0.5)

    feature_summaries.append({
        "modality": mod_name,
        "n_features": F,
        "n_pairs_corr": len(pair_indices),
        "n_pairs_dtw": len(feat_df_scored),
        "feature_confidence_0_100": feat_conf,
        "note": "" if len(feat_df_scored) > 0 else "dtw missing (no pairs computed)"
    })

feature_summary_df = pd.DataFrame(feature_summaries)
feature_summary_df.to_csv(os.path.join(OUTPUT_DIR, "feature_consistency_summary.csv"), index=False)

print("Feature-level consistency written to OUTPUT/")
print("✓ Done. Results written to OUTPUT/")
