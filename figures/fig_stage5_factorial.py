"""
STAGE 5 FACTORIAL — Publication figures (colorful, Okabe-Ito)
"""

from __future__ import annotations
import csv
import os
import statistics as st
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

OUT_DIR = "./neurips_figs"
FACT_DIR = f"{OUT_DIR}/stage5_factorial"
EXP_DIR = f"{FACT_DIR}/experiments"

OI = {
    "black": "#000000", "orange": "#E69F00", "skyblue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermill": "#D55E00", "pink": "#CC79A7",
}
FACTOR_COLORS = [OI["skyblue"], OI["orange"], OI["vermill"], OI["green"], OI["pink"]]
FACTOR_HATCH  = ["", "//", "xx", "..", "\\\\"]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 9

rows = []
for fname in sorted(os.listdir(EXP_DIR)):
    if not fname.endswith("_validations.tsv"): continue
    if fname.startswith("control_"): continue
    parts = fname.replace("_validations.tsv", "").split("_")
    a1, a2, a3 = parts[0], parts[1], parts[2]
    with open(f"{EXP_DIR}/{fname}") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            r["agent1"] = a1; r["agent2"] = a2; r["agent3"] = a3
            r["importance"]   = int(r["importance"])
            r["tractability"] = int(r["tractability"])
            r["novelty"]      = int(r["novelty"])
            rows.append(r)
print(f"Loaded {len(rows)} validations")


def ss_total(vals):
    m = st.mean(vals); return sum((v - m) ** 2 for v in vals)


def ss_between(groups):
    all_vals = [v for g in groups.values() for v in g]
    gm = st.mean(all_vals); t = 0
    for gv in groups.values():
        if not gv: continue
        t += len(gv) * (st.mean(gv) - gm) ** 2
    return t


# ═══════════════════════════════════════════════════════════════
# Figure 3 — variance decomposition
# ═══════════════════════════════════════════════════════════════
axes_list = ["importance", "tractability", "novelty"]
factors = [
    ("Agent 1 (filter)",    "agent1"),
    ("Agent 2 (generator)", "agent2"),
    ("Agent 3 (judge)",     "agent3"),
    ("Condition",           "condition"),
    ("A3 × Condition",      "interaction"),
]

eta = np.zeros((3, 5))
for ai, axis in enumerate(axes_list):
    vals = [r[axis] for r in rows]; sst = ss_total(vals)
    if sst == 0: continue
    for fi, (_, factor) in enumerate(factors):
        if factor == "interaction":
            gs = defaultdict(list)
            for r in rows: gs[(r["agent3"], r["condition"])].append(r[axis])
            ssb = ss_between(gs)
            ss_a3 = ss_between({k: [r[axis] for r in rows if r["agent3"]==k]
                                for k in ["gpt", "claude"]})
            ss_c  = ss_between({k: [r[axis] for r in rows if r["condition"]==k]
                                for k in ["blind", "ctx"]})
            ssb = max(0, ssb - ss_a3 - ss_c)
        else:
            gs = defaultdict(list)
            for r in rows: gs[r[factor]].append(r[axis])
            ssb = ss_between(gs)
        eta[ai, fi] = ssb / sst

fig, ax = plt.subplots(figsize=(7.5, 3.4), constrained_layout=True)
x = np.arange(3); w = 0.16
for fi, (label, _) in enumerate(factors):
    off = (fi - 2) * w
    bars = ax.bar(x + off, eta[:, fi] * 100, w,
                  edgecolor="black", linewidth=0.8,
                  facecolor=FACTOR_COLORS[fi], hatch=FACTOR_HATCH[fi],
                  label=label)
    for b, v in zip(bars, eta[:, fi] * 100):
        if v > 3:
            ax.text(b.get_x() + b.get_width()/2, v + 0.4, f"{v:.1f}",
                    ha="center", fontsize=7.5, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([a.capitalize() for a in axes_list], fontsize=9.5)
ax.set_ylabel("Variance explained ($\\eta^2$, %)", fontsize=9.5)
ax.set_title("Factor contributions to score variance "
             "(2×2×2 factorial, N=640 judgments)", fontsize=10, loc="left")
ax.grid(axis="y", linestyle=":", alpha=0.4)
ax.set_axisbelow(True)
ax.legend(loc="upper right", ncol=2, frameon=True, framealpha=0.95,
          edgecolor="black")
ax.set_ylim(0, 31)
plt.savefig(f"{OUT_DIR}/fig_factorial_variance.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig_factorial_variance.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ fig_factorial_variance.pdf")


# ═══════════════════════════════════════════════════════════════
# Figure 4 — heatmap grid (cividis - colorblind-safe + print-safe)
# ═══════════════════════════════════════════════════════════════
row_keys = [("gpt","gpt"),("gpt","claude"),("claude","gpt"),("claude","claude")]
col_keys = [("gpt","blind"),("gpt","ctx"),("claude","blind"),("claude","ctx")]
row_labels = ["A1=gpt\nA2=gpt","A1=gpt\nA2=claude","A1=claude\nA2=gpt","A1=claude\nA2=claude"]
col_labels = ["A3=gpt\nblind","A3=gpt\nctx","A3=claude\nblind","A3=claude\nctx"]

fig, axs = plt.subplots(1, 3, figsize=(9.0, 4.5), constrained_layout=True)
cmap = plt.cm.cividis
for pi, axis in enumerate(axes_list):
    mat = np.zeros((4, 4))
    for ri, (a1, a2) in enumerate(row_keys):
        for ci, (a3, cond) in enumerate(col_keys):
            vs = [r[axis] for r in rows
                  if r["agent1"]==a1 and r["agent2"]==a2
                  and r["agent3"]==a3 and r["condition"]==cond]
            mat[ri, ci] = st.mean(vs) if vs else np.nan
    ax = axs[pi]
    im = ax.imshow(mat, cmap=cmap, vmin=2.0, vmax=5.0, aspect="auto")
    for ri in range(4):
        for ci in range(4):
            v = mat[ri, ci]
            if np.isnan(v): continue
            color = "white" if v < 3.5 else "black"
            ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=color, fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(col_labels, fontsize=7.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(row_labels, fontsize=7.5)
    ax.set_title(axis.capitalize(), fontsize=10.5)
    ax.axvline(x=1.5, color="white", linewidth=1.5)
    ax.axhline(y=1.5, color="white", linewidth=1.0, linestyle="--", alpha=0.7)
cb = fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.6, pad=0.12, aspect=40)
cb.set_label("Mean score (1–5)", fontsize=9.5)
fig.suptitle("Mean validator score for every factorial cell "
             "(rows: filter + generator; cols: judge + condition)",
             fontsize=10.5, y=1.02)
plt.savefig(f"{OUT_DIR}/fig_factorial_heatmap.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig_factorial_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ fig_factorial_heatmap.pdf")


# ═══════════════════════════════════════════════════════════════
# Figure 5 — inter-rater scatter, Okabe-Ito per axis
# ═══════════════════════════════════════════════════════════════
def pearson(xs, ys):
    mx=st.mean(xs); my=st.mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = (sum((x-mx)**2 for x in xs))**0.5
    dy = (sum((y-my)**2 for y in ys))**0.5
    return num/(dx*dy) if dx and dy else float("nan")

def spearman(xs, ys):
    def rk(vs):
        sv = sorted(enumerate(vs), key=lambda t: t[1])
        ranks=[0.0]*len(vs); i=0
        while i < len(sv):
            j=i
            while j+1<len(sv) and sv[j+1][1]==sv[i][1]: j+=1
            avg = (i+j)/2 + 1
            for k in range(i,j+1): ranks[sv[k][0]] = avg
            i=j+1
        return ranks
    return pearson(rk(xs), rk(ys))

AXIS_COLORS = {"importance": OI["vermill"], "tractability": OI["blue"], "novelty": OI["green"]}

fig, axs = plt.subplots(2, 3, figsize=(8.0, 5.2), constrained_layout=True)
for ci, axis in enumerate(axes_list):
    for ri, cond in enumerate(["blind", "ctx"]):
        paired = defaultdict(lambda: {"gpt": None, "claude": None})
        for r in rows:
            if r["condition"] != cond: continue
            key = (r["agent1"], r["agent2"], r["hyp_key"])
            paired[key][r["agent3"]] = r[axis]
        xs, ys = [], []
        for s in paired.values():
            if s["gpt"] is not None and s["claude"] is not None:
                xs.append(s["gpt"]); ys.append(s["claude"])
        ax = axs[ri, ci]
        rng = np.random.RandomState(42)
        jx = np.array(xs) + (rng.rand(len(xs)) - 0.5) * 0.28
        jy = np.array(ys) + (rng.rand(len(ys)) - 0.5) * 0.28
        ax.scatter(jx, jy, s=18, c=AXIS_COLORS[axis], alpha=0.45,
                   edgecolors="black", linewidths=0.4)
        ax.plot([0.5,5.5],[0.5,5.5], color="black", linewidth=0.9, alpha=0.8)
        ax.plot([0.5,5.5],[1.5,6.5], color="black", linewidth=0.5, alpha=0.35, linestyle="--")
        ax.plot([0.5,5.5],[-0.5,4.5], color="black", linewidth=0.5, alpha=0.35, linestyle="--")
        r_p = pearson(xs, ys); r_s = spearman(xs, ys)
        ax.text(0.05, 0.93, f"r={r_p:.2f}\nρ={r_s:.2f}\nn={len(xs)}",
                transform=ax.transAxes, va="top", ha="left", fontsize=7.5,
                bbox=dict(facecolor="white", edgecolor="black",
                          boxstyle="round,pad=0.3", linewidth=0.5))
        ax.set_xlim(0.5,5.5); ax.set_ylim(0.5,5.5)
        ax.set_xticks([1,2,3,4,5]); ax.set_yticks([1,2,3,4,5])
        ax.set_aspect("equal", adjustable="box")
        if ri == 1: ax.set_xlabel("GPT-5.2 score")
        if ci == 0: ax.set_ylabel(f"Claude Sonnet 4.6 score\n({cond})", fontsize=9)
        if ri == 0:
            ax.set_title(axis.capitalize(), fontsize=10.5,
                         color=AXIS_COLORS[axis], fontweight="bold")

fig.suptitle("Inter-rater agreement: GPT vs Claude on identical hypotheses",
             fontsize=10.5, y=1.02)
plt.savefig(f"{OUT_DIR}/fig_factorial_agreement.pdf", bbox_inches="tight")
plt.savefig(f"{OUT_DIR}/fig_factorial_agreement.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ fig_factorial_agreement.pdf")