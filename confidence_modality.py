import os
import glob
import argparse
import gc
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import dtw

# ---------------- CONFIG ----------------

INPUT_DIR = "INPUT"
OUTPUT_DIR = "OUTPUT"
MIN_POINTS = 10

# ---------------- CLI ----------------

parser = argparse.ArgumentParser(description="Modality-level confidence (memory-safe)")
parser.add_argument("--dtw-downsample", type=int, default=1,
                    help="Downsample factor for DTW (>=1)")
parser.add_argument(
    "--global-plot",
    action="store_true",
    help="Generate global overlay plot of all modalities"
)

parser.add_argument(
    "--no-normalize",
    action="store_true",
    help="Plot raw (unnormalised) PDFs as well"
)
args = parser.parse_args()

DS = max(1, args.dtw_downsample)

def log(msg):
    print(msg, flush=True)

log("\n=== MODALITY-LEVEL CONFIDENCE ===")
log(f"DTW downsample factor: {DS}")

# ---------------- HELPERS ----------------

def normalise(x):
    s = np.sum(x)
    return x / s if s > 0 else x

def corr_to_sim(r):
    return (r + 1) / 2 if np.isfinite(r) else 0.5

def dtw_to_sim(d, tau):
    return np.exp(-d / tau)

# ---------------- GLOBAL PLOT HELPER
def plot_global_overlay(modalities, outdir, normalised=True):
    """
    modalities: dict {name: (time, pdf)}
    """
    # shared time grid
    t_min = max(t.min() for t, _ in modalities.values())
    t_max = min(t.max() for t, _ in modalities.values())

    ref_name = max(modalities, key=lambda k: len(modalities[k][0]))
    t_ref = modalities[ref_name][0]
    t_ref = t_ref[(t_ref >= t_min) & (t_ref <= t_max)]

    plt.figure(figsize=(10, 4))

    stack = []

    for name, (t, p) in modalities.items():
        pi = np.interp(t_ref, t, p)
        if normalised:
            pi = pi / np.sum(pi) if np.sum(pi) > 0 else pi
        stack.append(pi)

        plt.plot(t_ref, pi, alpha=0.6, linewidth=1.8, label=name)

    # mean profile
    mean_pdf = np.mean(stack, axis=0)
    plt.plot(t_ref, mean_pdf, color="black", linewidth=3, label="mean")

    plt.xlabel("Time")
    plt.ylabel("Probability density")
    title = "Global modality overlay"
    title += " (normalised)" if normalised else " (raw)"
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    suffix = "norm" if normalised else "raw"
    outfile = os.path.join(outdir, f"global_overlay_{suffix}.png")
    plt.savefig(outfile, dpi=200)
    plt.close()

    print(f"✓ written {outfile}")


# ---------------- PROJECT SELECTION ----------------

projects = [d for d in os.listdir(INPUT_DIR)
            if os.path.isdir(os.path.join(INPUT_DIR, d))]

for i, p in enumerate(projects, 1):
    print(f"[{i}] {p}")

choice = int(input("Select project: ")) - 1
project = projects[choice]

P_IN = os.path.join(INPUT_DIR, project)
P_OUT = os.path.join(OUTPUT_DIR, project)
os.makedirs(P_OUT, exist_ok=True)

log(f"\nProcessing project: {project}")

# ---------------- LOAD MODALITY PDFs ONLY ----------------

modalities = {}

for csv in glob.glob(os.path.join(P_IN, "*.csv")):
    if os.path.basename(csv).startswith("weights"):
        continue

    name = os.path.splitext(os.path.basename(csv))[0]
    log(f"[LOAD] {name}")

    df = pd.read_csv(csv)[["x_axis", "y_axis", "feature"]]

    wdf = pd.read_csv(os.path.join(P_IN, f"weights_{name}.csv"))
    weights = dict(zip(wdf["feature_name"], wdf["weight_value"]))

    pivot = df.pivot(index="x_axis", columns="feature", values="y_axis").fillna(0)

    for col in pivot.columns:
        pivot[col] *= weights.get(col, 1.0)

    pdf = normalise(pivot.sum(axis=1).values)
    time_axis = pivot.index.values

    modalities[name] = (time_axis, pdf)

    # plot
    plt.figure(figsize=(8, 3))
    plt.plot(time_axis, pdf)
    plt.title(name)
    plt.tight_layout()
    plt.savefig(os.path.join(P_OUT, f"{name}_pdf.png"))
    plt.close()

    del df, pivot
    gc.collect()

# ---------------- GLOBAL OVERLAY PLOT ----------------
if args.global_plot and len(modalities) > 1:
    plot_global_overlay(modalities, P_OUT, normalised=True)

    if not args.no_normalize:
        plot_global_overlay(modalities, P_OUT, normalised=False)


# ---------------- PAIRWISE MODALITY CONSISTENCY ----------------

log("\n[MODALITY] Computing cross-modality consistency")

keys = list(modalities.keys())
rows = []

for i in range(len(keys)):
    for j in range(i + 1, len(keys)):
        m1, m2 = keys[i], keys[j]
        log(f"[MODALITY] {m1} vs {m2}")

        t1, p1 = modalities[m1]
        t2, p2 = modalities[m2]

        t_min, t_max = max(t1.min(), t2.min()), min(t1.max(), t2.max())
        if t_max <= t_min:
            continue

        t_ref = t1[(t1 >= t_min) & (t1 <= t_max)]
        if len(t_ref) < MIN_POINTS:
            continue

        p1i = np.interp(t_ref, t1, p1)
        p2i = np.interp(t_ref, t2, p2)

        corr = np.corrcoef(p1i, p2i)[0, 1]

        d = dtw(p1i[::DS], p2i[::DS])

        # -------- Pairwise overlay plot (memory-safe) --------

        plt.figure(figsize=(10, 4))
        plt.plot(t_ref, p1i, label=m1, linewidth=2)
        plt.plot(t_ref, p2i, label=m2, linewidth=2, alpha=0.85)

        plt.xlabel("Time")
        plt.ylabel("Probability density")
        plt.title(f"{m1} vs {m2}")

        agree = (
            0.5 * corr_to_sim(corr) +
            0.5 * dtw_to_sim(d, tau if 'tau' in locals() else d)
        )

        txt = (
            f"corr = {corr:.2f}\n"
            f"DTW  = {d:.2f}\n"
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

        out_plot = os.path.join(
            P_OUT,
            f"pair_{m1}__{m2}.png"
        )

        plt.savefig(out_plot, dpi=200)
        plt.close()

        rows.append({
            "modality_A": m1,
            "modality_B": m2,
            "corr": corr,
            "dtw": d
        })

# ---------------- CONFIDENCE ----------------

dfm = pd.DataFrame(rows)
dfm.to_csv(os.path.join(P_OUT, "modality_internal_consistency.csv"), index=False)

tau = dfm["dtw"].median()
agree = [
    0.5 * corr_to_sim(r["corr"]) + 0.5 * dtw_to_sim(r["dtw"], tau)
    for _, r in dfm.iterrows()
]

confidence = 100 * np.mean(agree)

pd.DataFrame([{"confidence_0_100": confidence}]) \
  .to_csv(os.path.join(P_OUT, "confidence_summary.csv"), index=False)

log(f"\nConfidence: {confidence:.2f}/100")
log("✓ Modality-level analysis done")
