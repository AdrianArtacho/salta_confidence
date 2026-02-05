#!/usr/bin/env python3

import subprocess
import sys
import os
import argparse

# -------------------------------------------------
# CLI
# -------------------------------------------------

parser = argparse.ArgumentParser(
    description="Run full SALTA confidence pipeline"
)

parser.add_argument(
    "--dtw-downsample-modality",
    type=int,
    default=100,
    help="DTW downsampling factor for modality-level analysis"
)

parser.add_argument(
    "--dtw-downsample-features",
    type=int,
    default=5,
    help="DTW downsampling factor for feature-level analysis"
)

parser.add_argument(
    "--lambda-corr",
    type=float,
    default=0.5,
    help="Weight of correlation vs DTW in agreement metrics"
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

args = parser.parse_args()

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def run_step(cmd, label):
    print(f"\n=== {label} ===")
    print(" ".join(cmd))
    print("-" * 60)

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR during step: {label}")
        sys.exit(e.returncode)

# -------------------------------------------------
# Sanity checks
# -------------------------------------------------

REQUIRED_SCRIPTS = [
    "confidence_modality.py",
    "confidence_features.py",
    "MODALITY_HEATMAP.py",
]

for s in REQUIRED_SCRIPTS:
    if not os.path.exists(s):
        print(f"✗ Required script not found: {s}")
        sys.exit(1)

# -------------------------------------------------
# Pipeline
# -------------------------------------------------

print("\n=== SALTA CONFIDENCE PIPELINE ===")

# 1. Modality-level confidence
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
# The modality script writes:
# OUTPUT/<project>/modality_internal_consistency.csv
# The project selection happens inside the script.
# -------------------------------------------------

# 2. Modality heatmaps
print("\n=== [2/3] Modality agreement heatmaps ===")
print(
    "NOTE: You will be asked again to select the same project.\n"
    "      (This is intentional to keep scripts independent.)"
)

# We cannot hardcode the CSV path without sharing state,
# so we re-run MODALITY_HEATMAP interactively.

cmd_heatmap = [
    sys.executable,
    "MODALITY_HEATMAP.py",
]

print(
    "Please paste the path to:\n"
    "  OUTPUT/<project>/modality_internal_consistency.csv\n"
    "when prompted."
)

subprocess.run(cmd_heatmap)

print(
    "\n=== [3/3] Feature-level consistency ===\n"
    "NOTE: Feature heatmaps will be generated automatically."
)


# 3. Feature-level consistency
cmd_features = [
    sys.executable,
    "confidence_features.py",
    "--dtw-downsample",
    str(args.dtw_downsample_features),
    "--lambda-corr",
    str(args.lambda_corr),
]

run_step(cmd_features, "[3/3] Feature-level consistency + heatmaps")

print("\n✓ All confidence analyses completed successfully")
