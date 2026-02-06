# Modality Agreement Time-Series Generator

This script generates **time-resolved pairwise agreement data** between modalities, suitable for **dynamic visualization** (e.g. the 3D Modality Agreement Explorer).

Instead of producing a single global agreement value per modality pair, it computes **moment-to-moment agreement** using a sliding window over the time axis.

The output is a CSV file where each column represents a **pair of modalities**, and each row represents agreement at a specific time point.

---

## What the script does

Given a project folder containing modality PDFs:

* loads one PDF time series per modality
* optionally downsamples the data for efficiency
* computes **pairwise agreement over time** using:

  * Pearson correlation (shape co-occurrence)
  * Dynamic Time Warping (temporal alignment)
* combines both into a single agreement value per time window
* writes a CSV file suitable for interactive visualization

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
* `feature` → feature name (used internally)

---

### How to run

Activate your virtual environment, then run:

```bash
python MODALITY_AGREEMENT_TIMESERIES.py --project exp17
```

#### Optional arguments

```bash
--dtw-downsample N
```

Downsample factor applied before DTW (recommended for large datasets).

Example:

```bash
python MODALITY_AGREEMENT_TIMESERIES.py \
  --project exp17 \
  --dtw-downsample 5
```

---

### Output

The script writes a CSV file to:

```text
OUTPUT/<project>/modality_agreement_timeseries.csv
```

The file has the form:

```csv
time,modA__modB,modA__modC,modB__modC
0.0,0.42,0.31,0.55
1.0,0.44,0.29,0.57
...
```

* `time` is in **tenths of a second**
* agreement values are normalized to `[0, 1]`
* each column represents one **pair of modalities**

---

### Intended use

This script is designed to feed:

* the **3D Modality Agreement Explorer** (`index.html`)
* exploratory analysis of **temporal coordination regimes**
* artistic-research workflows where *when* modalities align matters more than *how much overall*

It complements (rather than replaces) global confidence metrics.
