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
MIN_POINTS = 10
MAX_FEATURE_PAIRS_DTW = 500

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
        s_corr = corr_to_sim(row["corr"])
        s_dtw = dtw_to_sim(row["dtw"], tau)
        sims.append(lam * s_corr + (1 - lam) * s_dtw)

    return 100.0 * float(np.mean(sims))

# -------------------------------------------------
# Project selection (CLI)
# -------------------------------------------------

projects = [
    d for d in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, d))
]

if not projects:
    raise RuntimeError("No project folders found in input/")

print("\nAvailable projects in input/:\n")
for i, p in enumerate(projects, start=1):
    print(f"[{i}] {p}")

choice = input("\nSelect a project by number: ").strip()

if not choice.isdigit() or not (1 <= int(choice) <= len(projects)):
    raise ValueError("Invalid selection.")

project_name = projects[int(choice) - 1]
PROJECT_DIR = os.path.join(INPUT_DIR, project_name)
PROJECT_OUT = os.path.join(OUTPUT_DIR, project_name)
os.makedirs(PROJECT_OUT, exist_ok=True)

print(f"\nProcessing project: {project_name}\n")

# -------------------------------------------------
# Load modality CSVs
# -------------------------------------------------

modalities = {}

csv_files = [
    f for f in glob.glob(os.path.join(PROJECT_DIR, "*.csv"))
    if not os.path.basename(f).startswith("weights")
]

for f in csv_files:
    name = os.path.splitext(os.path.basename(f))[0]
    print(f"  Loading modality: {name}")

    df = pd.read_csv(f)[["x_axis", "y_axis", "feature"]]

    # Load weights
    wfile = os.path.join(PROJECT_DIR, f"weights_{name}.csv")
    if os.path.exists(wfile):
        wdf = pd.read_csv(wfile)
        weights = dict(zip(wdf["feature_name"], wdf["weight_value"]))
    else:
        raise FileNotFoundError(f"Missing weights file for modality '{name}'")

    pivot = df.pivot(index="x_axis", columns="feature", values="y_axis").fillna(0)

    # Unweighted features
    feat_time = pivot.index.values
    feat_names = pivot.columns.astype(str).tolist()
    feats_unweighted = pivot.values

    # Weighted
    weighted = pivot.copy()
    for col in weighted.columns:
        weighted[col] *= weights.get(col, 1.0)

    pdf = normalise(weighted.sum(axis=1).values)
    time = weighted.index.values

    modalities[name] = {
        "time": time,
        "pdf": pdf,
        "feature_time": feat_time,
        "feature_names": feat_names,
        "features_unweighted": feats_unweighted,
    }

    plt.figure(figsize=(8, 3))
    plt.plot(time, pdf, linewidth=2)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(PROJECT_OUT, f"{name}_pdf.png"))
    plt.close()

# -------------------------------------------------
# A) Modality-level consistency
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

print(f"\nModality confidence: {mod_conf:.2f}/100")
print("\n✓ Done. Results written to OUTPUT/{project_name}/")
