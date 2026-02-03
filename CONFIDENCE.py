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

def plot_pairwise_overlay(
    time, p1, p2, name1, name2, corr, dtw_dist, agree, outdir
):
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

    pivot = df.pivot(
        index="x_axis",
        columns="feature",
        values="y_axis"
    ).fillna(0)

    weighted = pivot.copy()
    for col in weighted.columns:
        weighted[col] *= weights.get(col, 1.0)

    pdf = normalise(weighted.sum(axis=1).values)
    time = weighted.index.values

    modalities[name] = {"time": time, "pdf": pdf}

    plt.figure(figsize=(8, 3))
    plt.plot(time, pdf, linewidth=2)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_pdf.png"))
    plt.close()

# -------------------------------------------------
# Cross-modality comparisons (INTERPOLATED)
# -------------------------------------------------

results = []
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

        if len(t_ref) < 10:
            continue

        p1i = np.interp(t_ref, t1, p1)
        p2i = np.interp(t_ref, t2, p2)

        corr = np.corrcoef(p1i, p2i)[0, 1]
        dtw_dist = dtw(p1i, p2i)

        results.append({
            "modality_A": m1,
            "modality_B": m2,
            "corr": corr,
            "dtw": dtw_dist,
        })

metrics_df = pd.DataFrame(results)
metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "modality_internal_consistency.csv"),
    index=False
)

# -------------------------------------------------
# Pairwise overlay plots
# -------------------------------------------------

if not metrics_df.empty:
    tau = metrics_df["dtw"].median()
    if pd.isna(tau) or tau <= 0:
        tau = 1.0

    for _, row in metrics_df.iterrows():
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

# -------------------------------------------------
# Confidence score (RAW, INTERPOLATED)
# -------------------------------------------------

def compute_confidence(df, lam=0.5):
    if df.empty:
        return np.nan

    tau = df["dtw"].median()
    if pd.isna(tau) or tau <= 0:
        tau = df["dtw"].mean()
    if pd.isna(tau) or tau <= 0:
        tau = 1.0

    sims = []
    for _, row in df.iterrows():
        s_corr = corr_to_sim(row["corr"])
        s_dtw = dtw_to_sim(row["dtw"], tau)
        sims.append(lam * s_corr + (1 - lam) * s_dtw)

    return 100.0 * float(np.mean(sims))

confidence = compute_confidence(metrics_df)

summary = pd.DataFrame([{
    "confidence_0_100": confidence,
    "lambda_corr": 0.5,
    "n_modalities": len(modalities),
    "n_pairs": len(metrics_df),
}])

summary.to_csv(
    os.path.join(OUTPUT_DIR, "confidence_summary.csv"),
    index=False
)

print(f"Confidence (raw, weighted PDFs): {confidence:.2f}/100")
print("✓ Done. Results written to OUTPUT/")
