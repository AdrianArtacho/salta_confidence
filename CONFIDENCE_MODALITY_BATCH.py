#!/usr/bin/env python3

import os
import subprocess
import sys
from datetime import datetime

INPUT_DIR = "INPUT"
OUTPUT_DIR = "OUTPUT"
LOG_DIR = os.path.join(OUTPUT_DIR, "_batch_logs_modality")

os.makedirs(LOG_DIR, exist_ok=True)

# -------------------------------------------------
# Discover projects
# -------------------------------------------------

projects = [
    d for d in os.listdir(INPUT_DIR)
    if os.path.isdir(os.path.join(INPUT_DIR, d))
]

if not projects:
    print("No projects found in INPUT/")
    sys.exit(0)

print("\n=== SALTA CONFIDENCE – MODALITY BATCH MODE ===")
print(f"Found {len(projects)} projects:")
for p in projects:
    print(f" - {p}")

# -------------------------------------------------
# Batch processing
# -------------------------------------------------

results = []

for project in projects:
    print(f"\n=== Processing project: {project} ===")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        LOG_DIR,
        f"{project}_{timestamp}.log"
    )

    cmd = [
        sys.executable,
        "confidence_modality.py",
        "--project", project,
        "--dtw-downsample", "100",
        "--global-plot",
    ]

    with open(log_file, "w") as lf:
        try:
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=lf,
                check=False,   # do NOT raise on error
            )

            success = proc.returncode == 0
            results.append((project, success, log_file))

            if success:
                print(f"✓ {project} completed successfully")
            else:
                print(f"✗ {project} failed (see log)")

        except Exception as e:
            lf.write(f"\nFATAL ERROR: {e}\n")
            results.append((project, False, log_file))
            print(f"✗ {project} crashed (see log)")

# -------------------------------------------------
# Summary
# -------------------------------------------------

print("\n=== MODALITY BATCH SUMMARY ===")

ok = [r for r in results if r[1]]
fail = [r for r in results if not r[1]]

print(f"Successful projects: {len(ok)}")
print(f"Failed projects:     {len(fail)}")

if fail:
    print("\nFailures:")
    for project, _, log in fail:
        print(f" - {project} → {log}")

print("\nBatch modality-level confidence finished.")
