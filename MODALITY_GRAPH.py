#!/usr/bin/env python3

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os

# =================================================
# CLI
# =================================================

parser = argparse.ArgumentParser(
    description="Visualise modality agreement as a weighted graph"
)

parser.add_argument(
    "csv",
    help="Path to modality_internal_consistency.csv"
)

parser.add_argument(
    "--min-weight",
    type=float,
    default=0.0,
    help="Hide edges with agreement below this value (0–1)"
)

parser.add_argument(
    "--layout",
    choices=["spring", "circular"],
    default="spring",
    help="Graph layout"
)

args = parser.parse_args()

# =================================================
# Load data
# =================================================

df = pd.read_csv(args.csv)

required = {"modality_A", "modality_B"}
if not required.issubset(df.columns):
    raise ValueError("CSV must contain modality_A and modality_B columns")

# -------------------------------------------------
# Compute agreement from corr + dtw
# -------------------------------------------------

if not {"corr", "dtw"}.issubset(df.columns):
    raise ValueError("CSV must contain 'corr' and 'dtw' columns")

def corr_to_sim(r):
    if not np.isfinite(r):
        return 0.5
    return (r + 1.0) / 2.0

def dtw_to_sim(d, tau):
    if not np.isfinite(d) or tau <= 0:
        return 0.0
    return np.exp(-d / tau)

tau = max(df["dtw"].median(), 1e-9)

df["agree"] = [
    0.5 * corr_to_sim(r["corr"]) + 0.5 * dtw_to_sim(r["dtw"], tau)
    for _, r in df.iterrows()
]

weight_col = "agree"


# =================================================
# Build graph
# =================================================

G = nx.Graph()

for _, r in df.iterrows():
    w = float(r[weight_col])
    if w < args.min_weight:
        continue
    G.add_edge(r["modality_A"], r["modality_B"], weight=w)

if G.number_of_edges() == 0:
    raise RuntimeError("No edges above min-weight threshold")

# =================================================
# Node statistics
# =================================================

node_strength = {
    n: np.mean([d["weight"] for _, _, d in G.edges(n, data=True)])
    for n in G.nodes
}

# =================================================
# Layout
# =================================================

if args.layout == "spring":
    pos = nx.spring_layout(G, seed=42, weight="weight")
else:
    pos = nx.circular_layout(G)

# =================================================
# Draw
# =================================================

plt.figure(figsize=(8, 8))

# nodes
nx.draw_networkx_nodes(
    G,
    pos,
    node_size=[3000 * node_strength[n] for n in G.nodes],
    node_color=list(node_strength.values()),
    cmap="viridis"
)

# edges
weights = [d["weight"] for _, _, d in G.edges(data=True)]
nx.draw_networkx_edges(
    G,
    pos,
    width=[5 * w for w in weights],
    alpha=0.7
)

# labels
nx.draw_networkx_labels(G, pos, font_size=10)

plt.title("Modality Agreement Graph")
plt.axis("off")

# =================================================
# Save
# =================================================

out_dir = os.path.dirname(args.csv)
out_file = os.path.join(out_dir, "modality_agreement_graph.png")
plt.tight_layout()
plt.savefig(out_file, dpi=200)
plt.close()

print(f"✓ Graph written to {out_file}")
