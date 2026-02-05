#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse

# =================================================
# CLI
# =================================================

parser = argparse.ArgumentParser(
    description="Run full SALTA confidence analysis pipeline"
)

parser.add_argument(
    "--dtw-downsample-modality",
    type=int,
    default=5,
    help="DTW downsampling factor for modality-level analysis"
)

parser.add_argument(
    "--dtw-downsample-features",
    type=int,
    default=5,
    help="DTW downsampling factor for feature-level analysis"
)

parser.add_argument(
    "--global-plot",
    action="store_true",
    help="Generate global modality overlay plots"
)

parser.add_argument(
    "--no-normalize",
    action="store_true",
    help="Also generate raw (unnormalised) global plots"
)

parser.add_argument(
    "--skip-modality-heatmaps",
    action="store_true",
    help="Skip modality agreement heatmap generation"
)

parser.add_argument(
    "--skip-features",
    action="store_true",
    help="Skip feature-level consistency and heatmaps"
)

args = parser.parse_args()

# =================================================
# Helpers
# =================================================

def run_step(cmd, label):
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    print("-" * 60)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR during step: {label}")
        sys.exit(e.returncode)

# =================================================
# Sanity checks
# =================================================

REQUIRED_SCRIPTS = [
    "confidence_modality.py",
    "MODALITY_HEATMAP.py",
    "confidence_features.py",
]

for s in REQUIRED_SCRIPTS:
    if not os.path.exists(s):
        print(f"✗ Required script not found: {s}")
        sys.exit(1)

# =================================================
# Pipeline
# =================================================

print("\n=== SALTA CONFIDENCE PIPELINE ===")

# -------------------------------------------------
# 1. Modality-level confidence
# -------------------------------------------------

cmd_modality = [
    sys.executable,
    "confidence_modality.py",
    "--dtw-downsample",
    str(args.dtw_downsample_modality),
]

if args.global_plot:
    cmd_modality.append("--global-plot")

if args.no_normalize:
    cmd_modality.append("--no-normalize")

run_step(cmd_modality, "[1/3] Modality-level confidence")

# -------------------------------------------------
# Read project marker written by confidence_modality.py
# -------------------------------------------------

PROJECT_FILE = os.path.join("OUTPUT", "_last_project.txt")

if not os.path.exists(PROJECT_FILE):
    print("✗ Could not find project marker (_last_project.txt).")
    print("✗ Cannot continue pipeline safely.")
    sys.exit(1)

with open(PROJECT_FILE) as f:
    project = f.read().strip()

print(f"\n[PIPELINE] Using project: {project}")

# -------------------------------------------------
# 2. Modality heatmaps
# -------------------------------------------------

if args.skip_modality_heatmaps:
    print("\n=== [2/3] Modality agreement heatmaps (skipped) ===")
else:
    csv_path = os.path.join(
        "OUTPUT",
        project,
        "modality_internal_consistency.csv"
    )

    if not os.path.exists(csv_path):
        print(f"✗ Modality CSV not found: {csv_path}")
        print("✗ Skipping modality heatmaps.")
    else:
        cmd_heatmap = [
            sys.executable,
            "MODALITY_HEATMAP.py",
            csv_path,
        ]
        run_step(cmd_heatmap, "[2/3] Modality agreement heatmaps")

# -------------------------------------------------
# 3. Feature-level consistency
# -------------------------------------------------

if args.skip_features:
    print("\n=== [3/3] Feature-level consistency (skipped) ===")
else:
    cmd_features = [
        sys.executable,
        "confidence_features.py",
        "--dtw-downsample",
        str(args.dtw_downsample_features),
    ]

    print(
        "\n=== [3/3] Feature-level consistency + heatmaps ===\n"
        "NOTE: Feature heatmaps are generated automatically."
    )

    run_step(cmd_features, "[3/3] Feature-level consistency + heatmaps")

# =================================================
# Done
# =================================================

print("\n✓ All requested confidence analyses completed successfully")
