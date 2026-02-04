# SALTA – Internal Consistency & Confidence Metrics

This repository contains Python tools for computing and visualising **internal consistency**
and **confidence metrics** within the **SALTA (Segmentation Algorithm for Live Temporal Analysis)**
pipeline.

Rather than evaluating segmentation results against an external ground truth, these tools
quantify the **degree of convergence between multiple, independently derived temporal data
streams** (modalities and features). The resulting confidence values characterise how strongly
different sources of evidence support similar temporal structures.

The repository is intended for **analysis, diagnostics, and documentation** within artistic
research and choreomusical performance contexts.

---

## Conceptual Overview

The core idea is simple:

- Multiple **features** are extracted per **modality** (e.g. audio, motion, video).
- Features are combined into **modality-level probability density functions (PDFs)**.
- Agreement is evaluated:
  - **across modalities** (modality-level consistency)
  - **within modalities** (feature-level consistency, optional)
- A numerical **confidence score (0–100)** summarises cross-modal convergence.

Importantly:
- No modality is treated as privileged.
- No assumption of a single “correct” temporal structure is made.
- Confidence expresses **internal agreement**, not correctness.

---

## Repository Contents

```text
.
├── CONFIDENCE.py          # Main analysis script (CLI, interactive)
├── FEATURE_HEATMAP.py     # Visualisation of feature-level consistency
├── INPUT/                # Project folders (one folder = one dataset)
├── OUTPUT/               # Generated results (mirrors input structure)
└── README.md
````

---

## Input Data Structure

The script expects **project-based organisation**:

```text
INPUT/
├── project_A/
│   ├── AUDIO.csv
│   ├── MPIPE.csv
│   ├── weights_AUDIO.csv
│   ├── weights_MPIPE.csv
│
├── project_B/
│   ├── imu.csv
│   ├── video.csv
│   ├── weights_imu.csv
│   ├── weights_video.csv
```

### Modality CSV format

Each modality CSV must contain at least the following columns:

* `x_axis` – time (seconds or frames)
* `y_axis` – feature value / probability
* `feature` – feature identifier (string)

Additional columns (e.g. `tuples`) are ignored.

### Weights CSV format

Each modality has an accompanying weights file:

```text
feature_name,weight_value
```

Weights determine the contribution of individual features when constructing modality PDFs.

---

## Main Script: `CONFIDENCE.py`

### What it does

`CONFIDENCE.py` computes:

1. **Modality-level PDFs** (weighted feature aggregation)
2. **Cross-modality agreement**, using:

   * Pearson correlation
   * Dynamic Time Warping (DTW)
3. A **global confidence score (0–100)** summarising cross-modal consistency
4. **Pairwise overlay plots** explaining agreement visually
5. *(Optional)* **Feature-level internal consistency** within each modality

All comparisons are performed on **interpolated shared time grids** to avoid sampling artefacts.
No temporal smoothing is applied at the modality-PDF level.

---

### Running the script

From the repository root:

```bash
python CONFIDENCE.py
```

You will be prompted to select a project folder from `INPUT/`.

#### Optional: feature-level consistency (slow)

```bash
python CONFIDENCE.py --features
```

This enables feature–feature comparisons within each modality.
Because DTW is computationally expensive, this step is disabled by default.

---

### Terminal feedback

The script provides continuous, flushed terminal output, including:

* project selection
* modality loading
* modality-pair progress
* feature-pair DTW progress (throttled)
* timing information for long-running steps

This ensures the user always knows what is being processed.

---

### Output structure

For a project called `project_A`, results are written to:

```text
OUTPUT/project_A/
├── AUDIO_pdf.png
├── MPIPE_pdf.png
├── pair_AUDIO__MPIPE.png
├── modality_internal_consistency.csv
├── confidence_summary.csv
├── feature_internal_consistency_AUDIO.csv        (optional)
├── feature_internal_consistency_MPIPE.csv        (optional)
└── feature_consistency_summary.csv               (optional)
```

---

### Output files explained

* `*_pdf.png`
  Modality-level probability density functions

* `pair_*__*.png`
  Pairwise modality overlay plots with correlation, DTW, and agreement score

* `modality_internal_consistency.csv`
  Pairwise modality metrics (corr, DTW)

* `confidence_summary.csv`
  Global modality-level confidence (0–100)

* `feature_internal_consistency_*.csv`
  Feature–feature metrics within a modality (optional)

---

## Feature Heat Maps: `FEATURE_HEATMAP.py`

This script visualises **feature-level internal consistency** using heat maps.

It is intentionally separated from the main pipeline to keep analysis and visualisation decoupled.

---

### Usage

```bash
python FEATURE_HEATMAP.py \
    OUTPUT/project_A/feature_internal_consistency_MPIPE.csv
```

Optional arguments:

```bash
--metric corr     # correlation-based similarity
--metric dtw      # DTW-based similarity
--metric agree    # combined agreement (default)

--lambda_corr 0.5 # weight for correlation in combined agreement
--out PATH        # output directory
```

---

### Output

The script generates a symmetric **feature × feature heat map**:

```text
feature_internal_consistency_MPIPE_heatmap_agree.png
```

Values range from 0 (low agreement) to 1 (high agreement).

These plots reveal:

* clusters of mutually coherent features
* idiosyncratic or weakly aligned features
* internal structure within a modality

---

## Relation to the SALTA Pipeline

These tools form the **evaluation and diagnostic layer** of the SALTA pipeline:

* upstream stages handle feature extraction, smoothing, and weighting
* this repository evaluates **structural convergence** across those outputs

The confidence metric complements other measures such as EWMA-based smoothness by addressing a
different analytical question: **inter-stream agreement rather than intra-stream regularity**.

---

## Notes on Performance

* Modality-level analysis is fast and intended for routine use.
* Feature-level DTW analysis can be slow for large feature sets.

  * It is therefore optional (`--features`).
  * A configurable cap limits DTW computations per modality.

---

## License / Status

This repository is part of an ongoing artistic research project.
It is provided for research, documentation, and educational purposes.

---

## Contact

For questions related to SALTA, distributed performance, or choreomusical segmentation,
please contact the repository author.
