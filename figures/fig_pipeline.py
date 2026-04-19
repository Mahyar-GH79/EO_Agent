"""
FIGURE 1 — Pipeline schematic
==============================
Renders the 3-agent pipeline as a clean vector diagram.

Blocks:
  NASA EO KG → GNN → ranked novel pairs
                        ↓
                 Agent 1 (GPT or Claude) — plausibility + novelty
                        ↓
                 Top-40 by combined score
                        ↓
                 Agent 2 (GPT or Claude) — hypothesis generation
                        ↓
                 Agent 3 (GPT or Claude) — blind + contextual judging

Outputs:
    ./neurips_figs/fig_pipeline.pdf
    ./neurips_figs/fig_pipeline.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "./neurips_figs"

OI = {
    "orange":  "#E69F00",
    "skyblue": "#56B4E9",
    "green":   "#009E73",
    "blue":    "#0072B2",
    "vermill": "#D55E00",
    "pink":    "#CC79A7",
}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 9

fig, ax = plt.subplots(figsize=(7.5, 4.5))

def box(x, y, w, h, text, color, ax, fontsize=9, fontweight="normal"):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.08",
                       facecolor=color, edgecolor="black", linewidth=1.2)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight)


def arrow(x1, y1, x2, y2, ax, text=None, text_offset_x=0, text_offset_y=0):
    ar = FancyArrowPatch((x1, y1), (x2, y2),
                         arrowstyle="-|>", mutation_scale=15,
                         color="black", linewidth=1.3)
    ax.add_patch(ar)
    if text:
        ax.text((x1 + x2) / 2 + text_offset_x,
                (y1 + y2) / 2 + text_offset_y,
                text, ha="center", va="center", fontsize=7.5,
                fontstyle="italic",
                bbox=dict(facecolor="white", edgecolor="none", pad=1))


# Layout: x from 0 to 10, y from 0 to 10
ax.set_xlim(0, 10); ax.set_ylim(0, 10)

# STAGES 1–2 block (left group)
box(0.2, 7.2, 2.8, 1.3,
    "NASA EO KG\n(150k nodes, 436k edges)",
    "#f5f5f5", ax, fontsize=8.5)

box(0.2, 4.8, 2.8, 1.3,
    "GNN (GraphSAGE-Hetero)\nSPECTER2 features\ndot + MLP scorers",
    OI["skyblue"], ax, fontsize=8.5)

box(0.2, 2.4, 2.8, 1.3,
    "Ranked novel pairs\n(top-5000 from 1M+)",
    OI["skyblue"], ax, fontsize=8.5)

# Downward arrows in left column
arrow(1.6, 7.15, 1.6, 6.15, ax, "train", text_offset_x=-0.5)
arrow(1.6, 4.75, 1.6, 3.75, ax, "score", text_offset_x=-0.5)

# Rightward arrow to pipeline
arrow(3.05, 3.05, 4.45, 3.05, ax)

# AGENT 1 (center)
box(4.5, 2.4, 2.5, 1.3,
    "Agent 1 (filter)\npair → plausibility + novelty\nGPT or Claude",
    OI["orange"], ax, fontsize=8.3)

# Downward to top-40
arrow(5.75, 2.35, 5.75, 1.35, ax, "rank + top-40", text_offset_x=0.9)

box(4.5, 0.1, 2.5, 1.1,
    "Top 40 candidate pairs\n(combined score)",
    "#fff3e0", ax, fontsize=8.5)

# Rightward to Agent 2
arrow(7.05, 3.05, 7.35, 3.05, ax)

# AGENT 2 (right center top)
box(7.4, 2.4, 2.5, 1.3,
    "Agent 2 (generator)\npair → hypothesis\nGPT or Claude",
    OI["green"], ax, fontsize=8.3)

# Downward to hypotheses
arrow(8.65, 2.35, 8.65, 1.35, ax)

box(7.4, 0.1, 2.5, 1.1,
    "Structured hypothesis\n(question, method, …)",
    "#e8f5e9", ax, fontsize=8.5)

# AGENT 3 (above agent 2)
box(7.4, 4.8, 2.5, 1.3,
    "Agent 3 (judge)\nhypothesis → scores\nGPT or Claude\n(blind + contextual)",
    OI["vermill"], ax, fontsize=8.3)

# Upward arrow from Agent 2 to Agent 3
arrow(8.65, 3.75, 8.65, 4.75, ax)

# AGENT 3 output (top right)
box(7.4, 7.2, 2.5, 1.3,
    "Scores: importance,\ntractability, novelty\n+ rationale",
    "#ffebee", ax, fontsize=8.5)

arrow(8.65, 6.15, 8.65, 7.15, ax)


# Dotted bounding box: "Stage 5: 2×2×2 factorial"
from matplotlib.patches import Rectangle
fb = Rectangle((4.35, 0.0), 5.7, 8.7, fill=False,
               edgecolor=OI["blue"], linewidth=1.5,
               linestyle=(0, (5, 2)), alpha=0.7)
ax.add_patch(fb)
ax.text(4.4, 8.8,
        "Stage 5: 2×2×2 factorial  (each agent ∈ {GPT-5.2, Claude Sonnet 4.6})",
        fontsize=9, color=OI["blue"], fontweight="bold", va="bottom")

# Dotted bounding box: "Stages 1–2: graph learning"
fb2 = Rectangle((0.05, 2.35), 3.1, 6.3, fill=False,
                edgecolor=OI["skyblue"], linewidth=1.5,
                linestyle=(0, (5, 2)), alpha=0.7)
ax.add_patch(fb2)
ax.text(0.1, 8.8, "Stages 1–2: graph learning",
        fontsize=9, color=OI["skyblue"], fontweight="bold", va="bottom")


# Title
ax.text(5.0, 9.5, "Figure 1. Hypothesis generation pipeline",
        ha="center", fontsize=11, fontweight="bold")

ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_pipeline.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig_pipeline.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"✅ {OUT_DIR}/fig_pipeline.pdf")