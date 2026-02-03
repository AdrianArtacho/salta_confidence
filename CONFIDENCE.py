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
SMOOTH_WINDOW = 7  # tweak freely

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def smooth_signal(y, window):
    return (
        pd.Series(y)
        .rolling(window, center=True, min_periods=1)
        .mean()
        .values
    )

def normalise(y):
    s = np.sum(y)
    return y / s if s > 0 else y

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

    # Required columns (explicit, no guessing)
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

    # Modality PDFs
    pdf_raw = normalise(weighted.sum(axis=1).values)
    pdf_smooth = normalise(
        smooth_signal(pdf_raw, SMOOTH_WINDOW)
    )

    time = weighted.index.values

    modalities[name] = {
        "time": time,
        "raw": pdf_raw,
        "smooth": pdf_smooth
    }

    # Save per-modality plot
    plt.figure(figsize=(8, 3))
    plt.plot(time, pdf_raw, label="weighted (raw)", alpha=0.6)
    plt.plot(time, pdf_smooth, label="weighted + smoothed", linewidth=2)
    plt.title(name)
    plt.legend()
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

        # Align time axes (intersection)
        t1 = modalities[m1]["time"]
        t2 = modalities[m2]["time"]
        common_t = np.intersect1d(t1, t2)

        if len(common_t) < 10:
            continue

        def extract(mod, kind):
            idx = np.isin(modalities[mod]["time"], common_t)
            return modalities[mod][kind][idx]

        r1, r2 = extract(m1, "raw"), extract(m2, "raw")
        s1, s2 = extract(m1, "smooth"), extract(m2, "smooth")

        results.append({
            "modality_A": m1,
            "modality_B": m2,
            "corr_raw": np.corrcoef(r1, r2)[0, 1],
            "corr_smooth": np.corrcoef(s1, s2)[0, 1],
            "dtw_raw": dtw(r1, r2),
            "dtw_smooth": dtw(s1, s2),
        })

# Save metrics
metrics_df = pd.DataFrame(results)

metrics_df.to_csv(
    os.path.join(OUTPUT_DIR, "modality_internal_consistency.csv"),
    index=False
)

# -------------------------------------------------
# Confidence score
# -------------------------------------------------

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

def compute_confidence(df, corr_col, dtw_col, lam=0.5):
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

confidence_raw = compute_confidence(metrics_df, "corr_raw", "dtw_raw", lam=0.5)
confidence_smooth = compute_confidence(metrics_df, "corr_smooth", "dtw_smooth", lam=0.5)
confidence_final = 0.3 * confidence_raw + 0.7 * confidence_smooth

summary = pd.DataFrame([{
    "confidence_raw_0_100": confidence_raw,
    "confidence_smooth_0_100": confidence_smooth,
    "confidence_final_0_100": confidence_final,
    "lambda_corr": 0.5,
    "raw_weight": 0.3,
    "smooth_weight": 0.7,
    "n_modalities": len(modalities),
    "n_pairs": len(metrics_df),
}])

summary.to_csv(os.path.join(OUTPUT_DIR, "confidence_summary.csv"), index=False)

print(f"Confidence raw:    {confidence_raw:.2f}/100")
print(f"Confidence smooth: {confidence_smooth:.2f}/100")
print(f"Confidence final:  {confidence_final:.2f}/100")

# Optional: add per-pair agreement (smooth) + re-save metrics
if not metrics_df.empty:
    tau_s = metrics_df["dtw_smooth"].median()
    if pd.isna(tau_s) or tau_s <= 0:
        tau_s = 1.0

    metrics_df["sim_corr_smooth"] = metrics_df["corr_smooth"].apply(corr_to_sim)
    metrics_df["sim_dtw_smooth"] = metrics_df["dtw_smooth"].apply(lambda d: dtw_to_sim(d, tau_s))
    metrics_df["agree_smooth"] = 0.5 * metrics_df["sim_corr_smooth"] + 0.5 * metrics_df["sim_dtw_smooth"]

    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "modality_internal_consistency.csv"), index=False)

print("✓ Done. Results written to OUTPUT/")
