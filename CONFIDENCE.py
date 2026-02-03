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
    # r in [-1,1] -> [0,1]
    if pd.isna(r):
        return 0.5
    return max(0.0, min(1.0, (r + 1.0) / 2.0))

def dtw_to_sim(d, tau):
    # d >= 0 -> (0,1]
    if pd.isna(d) or d < 0:
        return 0.0
    tau = max(float(tau), 1e-9)
    return float(np.exp(-d / tau))

def plot_pairwise_overlay(
    time,
    p1,
    p2,
    name1,
    name2,
    corr,
    dtw_dist,
    agree,
    outdir
):
    plt.figure(figsize=(10, 4))

    plt.plot(time, p1, label=name1, linewidth=2)
    plt.plot(time, p2, label=name2, linewidth=2, alpha=0.85)

    plt.xlabel("Time")
    plt.ylabel("Probability density")
    plt.title(f"{name1}  vs  {name2}")

    txt = (
        f"corr = {corr:.2f}\n"
        f"DTW  = {dtw_dist:.2f}\n"
        f"agree = {agree:.2f}"
    )

    plt.text(
        0.01, 0.98,
        txt,
        transform=plt.gca().transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.legend()
    plt.tight_layout()

    fname = f"pair_{name1}__{name2}.png"
    plt.savefig(os.path.join(outdir, fname))
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

    # Required columns
    df = df[["x_axis", "y_axis", "feature"]]

    # Load weights
    wfile = os.path.join(INPUT_DIR, f"weights_{name}.csv")
    if os.path.exists(wfile):
        wdf = pd.read_csv(wfile)
        weights = dict(
            zip(
                wdf["feature_name"],
                wdf["weight_value"]
            )
        )
    else:
        weights = {}

    # Pivot: rows=time, cols=features
    pivot = df.pivot(
        index="x_axis",
        columns="feature",
        values="y_axis"
    ).fillna(0)

    # Apply weights
    weighted = pivot.copy()
    for col in weighted.columns:
        weighted[col] *= weights.get(col, 1.0)

    # Modality PDF (raw, weighted, normalised)
    pdf = normalise(weighted.sum(axis=1).values)
    time = weighted.index.values

    modalities[name] = {
        "time": time,
        "pdf": pdf
    }

    # Save per-modality plot
    plt.figure(figsize=(8, 3))
    plt.plot(time, pdf, linewidth=2)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_pdf.png"))
    plt.close()

# -------------------------------------------------
# Cross-modality comparisons
# -------------------------------------------------

results = []
keys = list(modalities.keys())

for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        m1, m2 = keys[i], keys[j]

        t1 = modalities[m1]["time"]
        t2 = modalities[m2]["time"]
        print(f"\nDEBUG {m1} vs {m2}")
        print(f"{m1}: dtype={t1.dtype}, min={t1.min()}, max={t1.max()}, len={len(t1)}")
        print(f"{m2}: dtype={t2.dtype}, min={t2.min()}, max={t2.max()}, len={len(t2)}")

        common_t = np.intersect1d(t1, t2)
        print(f"common_t: min={common_t.min()}, max={common_t.max()}, len={len(common_t)}")

        if len(common_t) < 10:
            continue

        def extract(mod):
            idx = np.isin(modalities[mod]["time"], common_t)
            return modalities[mod]["pdf"][idx]

        p1 = extract(m1)
        p2 = extract(m2)

        corr = np.corrcoef(p1, p2)[0, 1]
        dtw_dist = dtw(p1, p2)

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
        m1 = row["modality_A"]
        m2 = row["modality_B"]

        t1 = modalities[m1]["time"]
        t2 = modalities[m2]["time"]
        common_t = np.intersect1d(t1, t2)

        idx1 = np.isin(t1, common_t)
        idx2 = np.isin(t2, common_t)

        p1 = modalities[m1]["pdf"][idx1]
        p2 = modalities[m2]["pdf"][idx2]

        s_corr = corr_to_sim(row["corr"])
        s_dtw = dtw_to_sim(row["dtw"], tau)
        agree = 0.5 * s_corr + 0.5 * s_dtw

        plot_pairwise_overlay(
            time=common_t,
            p1=p1,
            p2=p2,
            name1=m1,
            name2=m2,
            corr=row["corr"],
            dtw_dist=row["dtw"],
            agree=agree,
            outdir=OUTPUT_DIR
        )


# -------------------------------------------------
# Confidence score (RAW ONLY)
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

confidence = compute_confidence(metrics_df, lam=0.5)

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
