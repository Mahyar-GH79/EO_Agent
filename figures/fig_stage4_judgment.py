"""
STAGE 4 FIGURE (v2) — both judges
==================================
Rebuilds the Stage 4 judgment figure with BOTH GPT-5.2 and Claude Sonnet
4.6 on the same 800 pairs. Three panels in one wide row:

  Left:   Paired boxplots per stratum, GPT vs Claude side-by-side.
  Middle: Stratum A joint distribution (plaus x novel) — GPT.
  Right:  Stratum A joint distribution (plaus x novel) — Claude.

Uses Okabe-Ito palette, saves PDF + PNG to ./neurips_figs/.
"""

from __future__ import annotations
import csv
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "./neurips_figs"
FACT_DIR = f"{OUT_DIR}/stage5_factorial"

# Okabe-Ito palette
OI = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermill": "#D55E00", "pink": "#CC79A7",
}

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 9


# ──────────────────────────────────────────────────────────────
# Load both judges' scores
# ──────────────────────────────────────────────────────────────
def load_scores(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            r["plausibility"] = int(r["plausibility"])
            r["novelty"] = int(r["novelty"])
            rows.append(r)
    return rows


gpt_rows    = load_scores(f"{FACT_DIR}/agent1_gpt_scores.tsv")
claude_rows = load_scores(f"{FACT_DIR}/agent1_claude_scores.tsv")
print(f"Loaded GPT: {len(gpt_rows)}  Claude: {len(claude_rows)}")


# ──────────────────────────────────────────────────────────────
# Figure: 3 panels wide
# ──────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 4.5), constrained_layout=True)
gs = fig.add_gridspec(1, 3, width_ratios=[1.5, 1, 1])


# ═════════════════════════════════════════════════════════════
# Panel 1: Paired boxplots, GPT vs Claude per stratum
# ═════════════════════════════════════════════════════════════
ax_l = fig.add_subplot(gs[0, 0])

# Order: B, A, D, C for visual appeal (high to low plausibility expected)
strata = [
    ("Held-out real\n(B)",       "B"),
    ("Predicted novel\n(A)",      "A"),
    ("Hard negative\n(D)",        "D"),
    ("Random\n(C)",               "C"),
]

pos = np.arange(len(strata)) * 2.0  # wider spacing
width = 0.7

def box_at(ax, scores, position, color, label=None):
    bp = ax.boxplot(
        scores, positions=[position], widths=width, patch_artist=True,
        showfliers=False,
        boxprops=dict(facecolor=color, alpha=0.85, edgecolor="black", linewidth=1.0),
        whiskerprops=dict(color="black", linewidth=1.0),
        capprops=dict(color="black", linewidth=1.0),
        medianprops=dict(color="black", linewidth=1.5),
    )
    # Mean diamond
    m = np.mean(scores)
    ax.scatter([position], [m], marker="D", s=34,
               color="white", edgecolor="black", linewidth=1.2, zorder=5)
    return bp


for i, (label, stratum) in enumerate(strata):
    gpt_scores    = [r["plausibility"] for r in gpt_rows    if r["stratum"] == stratum]
    claude_scores = [r["plausibility"] for r in claude_rows if r["stratum"] == stratum]

    # Paired positions
    p_gpt    = pos[i] - 0.4
    p_claude = pos[i] + 0.4

    box_at(ax_l, gpt_scores,    p_gpt,    OI["skyblue"])
    box_at(ax_l, claude_scores, p_claude, OI["orange"])

    # Mean annotations below axis
    ax_l.text(p_gpt,    0.52, f"{np.mean(gpt_scores):.2f}",
              ha="center", fontsize=7.8, color=OI["blue"], fontweight="bold")
    ax_l.text(p_claude, 0.52, f"{np.mean(claude_scores):.2f}",
              ha="center", fontsize=7.8, color=OI["vermill"], fontweight="bold")

ax_l.set_xticks(pos)
ax_l.set_xticklabels([s[0] for s in strata], fontsize=9)
ax_l.set_ylabel("Plausibility score (1–5)", fontsize=9.5)
ax_l.set_ylim(0.3, 5.5)
ax_l.set_yticks([1, 2, 3, 4, 5])
ax_l.grid(axis="y", linestyle=":", alpha=0.4)
ax_l.set_axisbelow(True)
ax_l.set_title("Plausibility judgment per stratum, both judges", fontsize=10.5)

# Legend
# Legend — placed above the plot to avoid overlap with mean-value labels
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=OI["skyblue"], edgecolor="black", label="GPT-5.2"),
    plt.Rectangle((0, 0), 1, 1, facecolor=OI["orange"],  edgecolor="black", label="Claude Sonnet 4.6"),
]
ax_l.legend(
    handles=legend_handles,
    loc="upper center",
    bbox_to_anchor=(0.5, -0.15),
    ncol=2,
    frameon=True, framealpha=0.95, edgecolor="black",
)


# ═════════════════════════════════════════════════════════════
# Panel 2 & 3: Stratum A 2D distribution (plaus x novel), per judge
# ═════════════════════════════════════════════════════════════
def build_grid(rows_all):
    grid = np.zeros((5, 5), dtype=int)
    for r in rows_all:
        if r["stratum"] != "A": continue
        p = r["plausibility"]; n = r["novelty"]
        grid[5 - p, n - 1] += 1
    return grid


def draw_heatmap(ax, grid, title, cmap):
    vmax = grid.max()
    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=vmax * 1.05, aspect="auto")
    for i in range(5):
        for j in range(5):
            c = grid[i, j]
            if c > 0:
                color = "white" if c > vmax * 0.5 else "black"
                ax.text(j, i, str(c), ha="center", va="center",
                        fontsize=8.5, color=color, fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels([1, 2, 3, 4, 5])
    ax.set_yticks(range(5)); ax.set_yticklabels([5, 4, 3, 2, 1])
    ax.set_xlabel("Novelty (1–5)", fontsize=9.5)
    ax.set_ylabel("Plausibility (1–5)", fontsize=9.5)
    ax.set_title(title, fontsize=10.5)

    # Outline "hero zone": plaus >= 4 AND novelty >= 3
    # In grid coords: rows 0,1 (plaus 5,4), cols 2,3,4 (novelty 3,4,5)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle(
        (1.5, -0.5), 3.0, 2.0,
        fill=False, edgecolor=OI["vermill"], linewidth=2.0, linestyle="--",
    ))
    hero = grid[0:2, 2:5].sum()
    ax.text(4.45, -0.3, f"hero\nn={hero}", color=OI["vermill"],
            fontsize=8, ha="right", va="top", fontweight="bold")
    return im


ax_m = fig.add_subplot(gs[0, 1])
ax_r = fig.add_subplot(gs[0, 2])

grid_gpt    = build_grid(gpt_rows)
grid_claude = build_grid(claude_rows)

im_m = draw_heatmap(ax_m, grid_gpt,    "Stratum A: GPT-5.2",       plt.cm.Blues)
im_r = draw_heatmap(ax_r, grid_claude, "Stratum A: Claude Sonnet 4.6", plt.cm.Oranges)

# Per-panel colorbars (each judge's grid has different vmax)
plt.colorbar(im_m, ax=ax_m, label="count", shrink=0.85)
plt.colorbar(im_r, ax=ax_r, label="count", shrink=0.85)


plt.savefig(f"{OUT_DIR}/fig_stage4_judgment.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig_stage4_judgment.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"✅ {OUT_DIR}/fig_stage4_judgment.pdf")
print(f"✅ {OUT_DIR}/fig_stage4_judgment.png")


# ──────────────────────────────────────────────────────────────
# Print caption-ready stats for the paper
# ──────────────────────────────────────────────────────────────
print("\nCaption-ready stats:")
for label, stratum in strata:
    gp = [r["plausibility"] for r in gpt_rows    if r["stratum"] == stratum]
    cp = [r["plausibility"] for r in claude_rows if r["stratum"] == stratum]
    print(f"  {stratum}: GPT μ={np.mean(gp):.2f}  Claude μ={np.mean(cp):.2f}  "
          f"Δ={np.mean(gp) - np.mean(cp):+.2f}")

hero_gpt    = grid_gpt[0:2, 2:5].sum()
hero_claude = grid_claude[0:2, 2:5].sum()
print(f"\n  GPT hero zone:    {hero_gpt}/200 ({100*hero_gpt/200:.0f}%)")
print(f"  Claude hero zone: {hero_claude}/200 ({100*hero_claude/200:.0f}%)")