# SALTA – Internal Consistency & Confidence Metrics

This repository contains Python tools for computing and visualising **internal consistency**
and **confidence metrics** within the **SALTA (Segmentation Algorithm for Live Temporal Analysis)**
pipeline.

Rather than evaluating segmentation results against an external ground truth, these tools
quantify the **degree of convergence between multiple, independently derived temporal data
streams** (modalities and features). The resulting confidence values characterise how strongly
different sources of evidence support similar **temporal event structures**.

The repository is intended for **analysis, diagnostics, and documentation** within artistic
research and choreomusical performance contexts.

---

## Conceptual Overview

The core idea is simple:

- Multiple **features** are extracted per **modality** (e.g. audio, motion, video).
- Features are combined into **modality-level probability density functions (PDFs)**.
- Agreement is evaluated:
  - **across modalities** (modality-level consistency)
  - **within modalities** (feature-level consistency, diagnostic)
- A numerical **confidence score (0–100)** summarises cross-modal convergence.

Key principles:

- No modality is treated as privileged.
- No assumption of a single “correct” segmentation is made.
- Confidence expresses **internal agreement**, not correctness.
- Alignment is understood primarily as **temporal coincidence of salient events**, not
  similarity of amplitude or signal shape.

---

## Repository Contents

```text
.
├── confidence_modality.py     # Modality-level confidence + global plots
├── confidence_features.py     # Feature-level consistency + heatmaps
├── MODALITY_HEATMAP.py        # Modality agreement heatmaps (from CSV)
├── INPUT/                     # Project folders (one folder = one dataset)
├── OUTPUT/                    # Generated results (mirrors input structure)
└── README.md
````

---

## Input Data Structure

All scripts expect **project-based organisation**:

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

Each project is processed independently.

---

### Modality CSV format

Each modality CSV must contain at least the following columns:

* `x_axis` – time (seconds or frames)
* `y_axis` – feature value / probability
* `feature` – feature identifier (string)

Additional columns (e.g. `tuples`) are ignored.

---

### Weights CSV format

Each modality has an accompanying weights file:

```text
feature_name,weight_value
```

Weights determine the contribution of individual features when constructing modality PDFs.

---

## Script 1: `confidence_modality.py`

### What it does

`confidence_modality.py` computes **modality-level internal consistency**:

1. Weighted **modality PDFs**
2. Pairwise **cross-modality agreement**, using:

   * Pearson correlation (strict simultaneity)
   * Dynamic Time Warping (elastic temporal correspondence)
3. A **global confidence score (0–100)**
4. **Pairwise overlay plots** (diagnostic / explanatory)
5. *(Optional)* **global overlay plots** across all modalities
6. *(Optional)* **raw vs normalised** global representations

All comparisons are performed on **shared interpolated time grids** to avoid sampling artefacts.
No temporal smoothing is applied at the modality-PDF level.

---

### Running the script

From the repository root:

```bash
python confidence_modality.py
```

You will be prompted to select a project folder from `INPUT/`.

---

### Important CLI options

```bash
--dtw-downsample N
```

Downsample factor applied **only to DTW** (strongly recommended for long recordings).

Typical values:

* `50–100` for long performances
* `10–20` for shorter excerpts

```bash
--global-plot
```

Generate **global overlay plots** showing all modalities together.

```bash
--no-normalize
```

Also generate **raw (unnormalised)** global overlay plots in addition to the normalised version.

---

### Example

```bash
python confidence_modality.py \
  --dtw-downsample 100 \
  --global-plot
```

---

### Output (example)

```text
OUTPUT/project_A/
├── AUDIO_pdf.png
├── MPIPE_pdf.png
├── pair_AUDIO__MPIPE.png
├── global_overlay_norm.png
├── global_overlay_raw.png
├── modality_internal_consistency.csv
└── confidence_summary.csv
```

---

## Script 2: `confidence_features.py`

### What it does

`confidence_features.py` computes **feature-level internal consistency within each modality**.
This analysis is **diagnostic** and intended to reveal:

* redundant or highly similar features
* coherent feature clusters
* idiosyncratic or weakly aligned features

It also **automatically generates heat maps** from the computed tables.

---

### Running the script

```bash
python confidence_features.py
```

You will be prompted to select a project folder from `INPUT/`.

---

### Important CLI options

```bash
--dtw-downsample N
```

Downsample factor for feature-level DTW.

Recommended values:

* `2–5` for high-frequency features
* `5–10` for smoother features

```bash
--lambda-corr 0.5
```

Weight of correlation vs DTW when computing combined agreement.

---

### Example

```bash
python confidence_features.py \
  --dtw-downsample 5 \
  --lambda-corr 0.5
```

---

### Output (example)

```text
OUTPUT/project_A/
├── feature_internal_consistency_AUDIO.csv
├── feature_internal_consistency_AUDIO_heatmap_corr.png
├── feature_internal_consistency_AUDIO_heatmap_dtw.png
├── feature_internal_consistency_AUDIO_heatmap_agree.png
```

---

## Script 3: `MODALITY_HEATMAP.py`

This script visualises **cross-modality agreement** using heat maps.
It operates purely on previously generated CSV files.

---

### Usage

```bash
python MODALITY_HEATMAP.py \
  OUTPUT/project_A/modality_internal_consistency.csv
```

Optional arguments:

```bash
--lambda-corr 0.5
```

---

### Output

```text
OUTPUT/project_A/
├── modality_heatmap_corr.png
├── modality_heatmap_dtw.png
└── modality_heatmap_agree.png
```

These figures summarise **which modalities agree most strongly** with one another.

---

## Normalised vs Raw Representations

* **Normalised PDFs** emphasise *temporal alignment* and are used for confidence computation.
* **Raw PDFs** preserve expressive magnitude and are useful for interpretive comparison.

Both are supported to clearly separate **structural agreement** from **modality-specific
expressivity**.

---

## Relation to the SALTA Pipeline

These tools form the **evaluation and diagnostic layer** of the SALTA pipeline:

* upstream stages handle feature extraction, smoothing, and weighting
* this repository evaluates **structural convergence** across those outputs

The confidence metric complements EWMA-based smoothness measures by addressing a different
analytical question: **inter-stream agreement rather than intra-stream regularity**.

---

## Notes on Performance & Robustness

* Modality-level analysis is fast and intended for routine use.
* Feature-level DTW analysis can be computationally heavy:

  * downsampling is strongly recommended
  * results are typically stable across reasonable downsampling factors
* Stability under downsampling indicates that metrics capture **event-level structure**
  rather than sampling artefacts.

---

## License / Status

This repository is part of an ongoing artistic research project.
It is provided for research, documentation, and educational purposes.

---

## Contact

For questions related to SALTA, distributed performance, or choreomusical segmentation,
please contact the repository [author](www.artacho.at).

---

## [:memo: To-Do](https://trello.com/c/mWuPbiPu/124-confidence)