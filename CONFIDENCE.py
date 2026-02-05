#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse

# =================================================
# CLI
# =================================================

parser = argparse.ArgumentParser(
    description="Run full SALTA confidence pipeline"
)

parser.add_argument("--dtw-downsample-modality", type=int, default=100)
parser.add_argument("--dtw-downsample-features", type=int, default=5)
parser.add_argument("--global-plot", action="store_true")
parser.add_argument("--no-normalize", action="store_true")
parser.add_argument("--skip-modality-heatmaps", action="store_true")
parser.add_argument("--skip-features", action="store_true")

args = parser.parse_args()

INPUT_DIR = "INPUT"

# =================================================
# Project selection (ONCE)
# =================================================

projects = [
    d for d in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, d))
]

if not projects:
    raise RuntimeError("No projects found in INPUT/")

print("\nAvailable projects:")
for i, p in enumerate(projects, 1):
    print(f"[{i}] {p}")

while True:
    try:
        idx = int(input("Select project: ")) - 1
        if 0 <= idx < len(projects):
            project = projects[idx]
            break
    except ValueError:
        pass
    print("Invalid selection.")

print(f"\n[PIPELINE] Using project: {project}")

def run(cmd, label):
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)

# =================================================
# 1. Modality-level
# =================================================

cmd = [
    sys.executable,
    "confidence_modality.py",
    "--project", project,
    "--dtw-downsample", str(args.dtw_downsample_modality),
]

if args.global_plot:
    cmd.append("--global-plot")
if args.no_normalize:
    cmd.append("--no-normalize")

run(cmd, "[1/3] Modality-level confidence")

# =================================================
# 2. Modality heatmaps
# =================================================

if not args.skip_modality_heatmaps:
    csv = os.path.join("OUTPUT", project, "modality_internal_consistency.csv")
    if os.path.exists(csv):
        run(
            [sys.executable, "MODALITY_HEATMAP.py", csv],
            "[2/3] Modality agreement heatmaps"
        )

# =================================================
# 3. Feature-level
# =================================================

if not args.skip_features:
    run(
        [
            sys.executable,
            "confidence_features.py",
            "--project", project,
            "--dtw-downsample", str(args.dtw_downsample_features),
        ],
        "[3/3] Feature-level consistency + heatmaps"
    )

print("\n✓ SALTA confidence pipeline completed successfully")
