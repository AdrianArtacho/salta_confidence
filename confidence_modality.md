# Modality-Level Confidence Estimation

This script computes a **global confidence score** describing how consistently different **modalities agree with one another** over the duration of a project.

Unlike time-resolved analyses, this script collapses agreement into **one scalar per modality pair**, and then aggregates those into a **single confidence value** for the project.

It is intended as a **summary diagnostic** within the SALTA pipeline.

---

## What the script does

For a given project:

1. Loads one **PDF time series per modality**
2. Computes **pairwise agreement** between modalities using:

   * Pearson correlation (co-occurrence of events)
   * Dynamic Time Warping (temporal alignment)
3. Combines these measures into a single **pairwise agreement score**
4. Aggregates all modality pairs into a **global confidence value**
5. Writes detailed CSV outputs for inspection and visualization

---

### Expected input structure

```text
INPUT/
└── <project>/
    ├── modalityA.csv
    ├── modalityB.csv
    ├── modalityC.csv
    └── ...
```

Each modality CSV is expected to contain:

* `x_axis` → time (in tenths of a second)
* `y_axis` → PDF value
* `feature` → feature identifier

---

### How to run

Activate your virtual environment, then run:

```bash
python confidence_modality.py --project exp17
```

---

### Optional arguments

```bash
--dtw-downsample N
```

Downsamples the time series before DTW computation to reduce memory usage and runtime.

Example:

```bash
python confidence_modality.py \
  --project exp17 \
  --dtw-downsample 5
```

---

### Output

Results are written to:

```text
OUTPUT/<project>/
```

Key files include:

```text
modality_internal_consistency.csv
confidence_summary.csv
```

* `modality_internal_consistency.csv`
  Contains pairwise agreement metrics between all modality pairs

* `confidence_summary.csv`
  Contains the final global confidence score (0–100) and parameters used

---

### Interpretation

* **High confidence** → modalities exhibit consistent temporal structure
* **Low confidence** → modalities diverge or align inconsistently

The metric reflects **internal coherence**, not correctness or quality.

It is best interpreted comparatively:

* across different projects
* across different processing settings
* alongside time-resolved analyses

---

### Intended use

This script is designed for:

* quick dataset diagnostics
* reporting a single confidence value in publications
* comparing different experimental conditions
* complementing feature-level and time-resolved agreement analyses

It pairs naturally with:

* `confidence_features.py`
* `MODALITY_AGREEMENT_TIMESERIES.py`
* the interactive 3D Modality Agreement Explorer
