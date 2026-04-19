"""
STAGE 0 — Unified Pipeline
==========================
Loads the NASA EO KG from ./nasa_eo_kg/ ONCE and produces every figure
and every LaTeX table for the paper's "Dataset" section from live data.

Replaces both neurips_viz.py (which used hardcoded numbers) and
stage0_analysis.py (which only did usage/temporal stats).

Outputs (all in ./neurips_figs/):
    Figures
        fig_graph_overview.pdf    — nodes / edges / pub-per-year / DAAC donut
        fig_schema_diagram.pdf    — KG schema with labeled arrows
        fig_usage_overview.pdf    — citation & co-usage degree distributions
        fig_temporal_split.pdf    — train / val / test per year
    Tables
        neurips_tables.tex        — compilable standalone doc
        neurips_tables_only.tex   — paste-into-paper version
    Stats & data
        full_stats.json           — everything numeric
        paper_dataset_edges.tsv
        cousage_edges_{train,val,test}.tsv
        dataset_features.tsv

Run:
    python3 stage0_pipeline.py

Requires: networkx, matplotlib, numpy
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from itertools import combinations

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch, Patch
import networkx as nx
import numpy as np


# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
SAVE_DIR = "./nasa_eo_kg"
OUT_DIR = "./neurips_figs"
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_MAX_YEAR = 2022
VAL_YEAR = 2023
TEST_YEAR = 2024
MIN_PAPERS_PER_DATASET = 5

# Sane bounds for temporal-extent fields (some NASA datasets have "2110" typos).
VALID_YEAR_MIN = 1970
VALID_YEAR_MAX = 2030


# ══════════════════════════════════════════════════════════════
# Style (matches original neurips_viz.py exactly)
# ══════════════════════════════════════════════════════════════
plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.3,
    "grid.linewidth":     0.5,
    "grid.color":         "#cccccc",
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

BLUE, LBLUE, TEAL = "#2166ac", "#74add1", "#1a9988"
ORANGE, GREEN, GRAY = "#d6604d", "#4dac26", "#888888"
DGRAY, GOLD, PURPLE = "#333333", "#b8860b", "#762a83"

NODE_COLORS = {
    "Publication":    BLUE,
    "Dataset":        TEAL,
    "ScienceKeyword": ORANGE,
    "Instrument":     GOLD,
    "Platform":       PURPLE,
    "Project":        GREEN,
    "DataCenter":     "#666666",
}
EDGE_COLORS = {
    "CITES":                  BLUE,
    "HAS_APPLIEDRESEARCHAREA":ORANGE,
    "USES_DATASET":           TEAL,
    "HAS_SCIENCEKEYWORD":     "#e08214",
    "HAS_PLATFORM":           PURPLE,
    "HAS_DATASET":            GRAY,
    "OF_PROJECT":             GREEN,
    "HAS_INSTRUMENT":         GOLD,
    "HAS_SUBCATEGORY":        "#aaaaaa",
}

# Human-readable DAAC expansions (for the detailed table).
DAAC_CANONICAL = {
    "ASDC":     "Atmospheric Science Data Center",
    "GES-DISC": "Goddard Earth Sciences DISC",
    "LP DAAC":  "Land Processes DAAC",
    "NSIDC":    "Natl.\\ Snow and Ice Data Center",
    "PODAAC":   "Physical Oceanography DAAC",
    "OB.DAAC":  "Ocean Biology DAAC",
    "GHRC":     "Global Hydrology Resource Ctr.",
}
NODE_ATTRS = {
    "Publication":    "doi, title, year, authors, abstract",
    "Dataset":        "shortName, longName, daac, cmrId, temporal extent",
    "ScienceKeyword": "name, subcategory hierarchy",
    "Instrument":     "shortName, longName",
    "Platform":       "shortName, longName, Type",
    "Project":        "shortName, longName",
    "DataCenter":     "shortName, longName, url",
}


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════
def find_graphml():
    for root, _, files in os.walk(SAVE_DIR):
        for f in files:
            if f.endswith(".graphml"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"No .graphml under {SAVE_DIR}/")


def clean_label(raw):
    return str(raw).strip().strip("[]:'\" ").strip()


def edge_type(data):
    return data.get("label") or data.get("type") or data.get("relationship") or ""


def parse_year(raw):
    """Parse a year from a raw field (used for publications)."""
    try:
        y = int(str(raw)[:4])
        if VALID_YEAR_MIN <= y <= VALID_YEAR_MAX:
            return y
    except (ValueError, TypeError):
        pass
    return None


# Earth observation satellites started with Landsat-1 (1972). NASA uses
# "1970-01-01" as a Unix-epoch placeholder for unknown start dates, and
# sometimes uses "2038-01-19" (Y2038) or "9999" placeholders for unknown
# end dates. Filter these out by tightening the accepted range for
# dataset temporal fields specifically.
DATASET_YEAR_MIN = 1972
DATASET_YEAR_MAX = 2027  # current year + small buffer for forecast products

# Explicit blacklist of known placeholder year values
PLACEHOLDER_YEARS = {1970, 1971, 2038, 2099, 2100, 9999}


def parse_dataset_year(raw):
    """Stricter parser for dataset temporal extent fields: rejects NASA
    placeholder values like 1970 (epoch) and 2038 (Y2038)."""
    try:
        y = int(str(raw)[:4])
    except (ValueError, TypeError):
        return None
    if y in PLACEHOLDER_YEARS:
        return None
    if DATASET_YEAR_MIN <= y <= DATASET_YEAR_MAX:
        return y
    return None


def canonicalise_daac(raw):
    """Map raw daac string (e.g. 'NASA/JPL/PODAAC', 'LAADS') to canonical short."""
    s = str(raw).upper()
    if "PODAAC" in s or "PO.DAAC" in s:
        return "PODAAC"
    if "GES-DISC" in s or "GES DISC" in s or "GESDISC" in s:
        return "GES-DISC"
    if "LP DAAC" in s or "LPDAAC" in s or "LPCLOUD" in s:
        return "LP DAAC"
    if "NSIDC" in s:
        return "NSIDC"
    if "OB.DAAC" in s or "OBDAAC" in s or "OB DAAC" in s:
        return "OB.DAAC"
    if "GHRC" in s:
        return "GHRC"
    if "ASDC" in s:
        return "ASDC"
    return "Other" if raw else "Unknown"


def log_hist(ax, values, color, xlabel, title, annotate_median=True):
    values = np.asarray([v for v in values if v > 0])
    if len(values) == 0:
        ax.set_title(title + " (empty)", fontweight="bold", fontsize=9)
        return
    bins = np.logspace(0, np.log10(values.max()) + 0.05, 30)
    ax.hist(values, bins=bins, color=color, edgecolor="white",
            linewidth=0.4, alpha=0.85)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Count", fontsize=8)
    ax.set_title(title, fontweight="bold", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    if annotate_median:
        med = np.median(values)
        ax.axvline(med, ls="--", color=DGRAY, lw=0.8, alpha=0.7)
        y_hi = ax.get_ylim()[1]
        ax.text(med * 1.15, y_hi * 0.3, f"median = {med:.0f}",
                fontsize=7, color=DGRAY)


# ══════════════════════════════════════════════════════════════
# PART 1 — Load graph and compute every stat we need
# ══════════════════════════════════════════════════════════════
def load_graph_and_compute():
    print("═" * 62)
    print("STAGE 0 — Loading graph and computing every statistic")
    print("═" * 62)

    G = nx.read_graphml(find_graphml())
    print(f"📊 Loaded: {G.number_of_nodes():,} nodes, "
          f"{G.number_of_edges():,} edges")

    # ── Index nodes by label ──────────────────────────────────
    node_label = {}
    nodes_by_label = defaultdict(list)
    for n, data in G.nodes(data=True):
        lbl = clean_label(data.get("labels", "UNKNOWN"))
        node_label[n] = lbl
        nodes_by_label[lbl].append(n)

    node_counts = {k: len(v) for k, v in nodes_by_label.items()}
    print("\nNode type distribution:")
    for k, v in sorted(node_counts.items(), key=lambda kv: -kv[1]):
        print(f"   {k:20s} {v:>10,}")

    # ── Edge type distribution + schema (src_label, edge, tgt_label) ──
    edge_counts = Counter()
    edge_schema_counts = Counter()
    for u, v, data in G.edges(data=True):
        et = edge_type(data)
        edge_counts[et] += 1
        src, tgt = node_label.get(u, "?"), node_label.get(v, "?")
        edge_schema_counts[(src, et, tgt)] += 1

    print("\nEdge type distribution:")
    for k, v in edge_counts.most_common():
        print(f"   {k:30s} {v:>10,}")

    # ── Publication years ─────────────────────────────────────
    paper_year = {
        n: parse_year(G.nodes[n].get("year"))
        for n in nodes_by_label.get("Publication", [])
    }
    year_counts = Counter(y for y in paper_year.values() if y is not None)
    pub_years = dict(sorted({
        y: c for y, c in year_counts.items() if 2015 <= y <= 2025
    }.items()))
    print(f"\nPublication year span: "
          f"{min(year_counts):d}–{max(year_counts):d} "
          f"(plotting 2016–2025)")

    # ── Dataset metadata ──────────────────────────────────────
    dataset_meta = {}
    for d in nodes_by_label.get("Dataset", []):
        dd = G.nodes[d]
        dataset_meta[d] = {
            "shortName":  dd.get("shortName", ""),
            "longName":   dd.get("longName", ""),
            "daac_raw":   dd.get("daac", ""),
            "daac":       canonicalise_daac(dd.get("daac", "")),
            "doi":        dd.get("doi", ""),
            "abstract":   dd.get("abstract", ""),
            "cmrId":      dd.get("cmrId", ""),
            "tempStart":  parse_dataset_year(dd.get("temporalExtentStart", "")),
            "tempEnd":    parse_dataset_year(dd.get("temporalExtentEnd", "")),
        }

    daac_counts = Counter(m["daac"] for m in dataset_meta.values())

    # Dataset temporal span (filtered to sane years)
    starts = [m["tempStart"] for m in dataset_meta.values() if m["tempStart"]]
    ends   = [m["tempEnd"]   for m in dataset_meta.values() if m["tempEnd"]]

    # Diagnostics: how many datasets had valid temporal metadata
    n_total_ds = len(dataset_meta)
    n_start_valid = len(starts)
    n_end_valid = len(ends)

    dataset_temporal = {
        "earliest_start": min(starts) if starts else None,
        "latest_end":     max(ends)   if ends   else None,
        "datasets_with_valid_start": n_start_valid,
        "datasets_with_valid_end":   n_end_valid,
        "datasets_total":            n_total_ds,
    }
    print(f"Dataset temporal span (sane): "
          f"{dataset_temporal['earliest_start']}–"
          f"{dataset_temporal['latest_end']}")
    print(f"   ({n_start_valid:,}/{n_total_ds:,} with valid start, "
          f"{n_end_valid:,}/{n_total_ds:,} with valid end "
          f"after filtering placeholders)")

    # ── Dataset → platform/instrument/keyword ─────────────────
    dataset_platforms = defaultdict(set)
    dataset_instruments = defaultdict(set)
    dataset_keywords = defaultdict(set)
    platform_instruments = defaultdict(set)
    for u, v, data in G.edges(data=True):
        et = edge_type(data)
        lu, lv = node_label.get(u), node_label.get(v)
        if et == "HAS_PLATFORM":
            if lu == "Dataset" and lv == "Platform":
                dataset_platforms[u].add(v)
            elif lv == "Dataset" and lu == "Platform":
                dataset_platforms[v].add(u)
        elif et == "HAS_SCIENCEKEYWORD":
            if lu == "Dataset":
                dataset_keywords[u].add(v)
            elif lv == "Dataset":
                dataset_keywords[v].add(u)
        elif et == "HAS_INSTRUMENT":
            if lu == "Platform" and lv == "Instrument":
                platform_instruments[u].add(v)
            elif lv == "Platform" and lu == "Instrument":
                platform_instruments[v].add(u)
    for d, plats in dataset_platforms.items():
        for p in plats:
            dataset_instruments[d] |= platform_instruments.get(p, set())

    # ── Paper → Dataset edges ─────────────────────────────────
    pd_edges = []
    for u, v, data in G.edges(data=True):
        if edge_type(data) != "USES_DATASET":
            continue
        if node_label.get(u) == "Publication" and node_label.get(v) == "Dataset":
            p, d = u, v
        elif node_label.get(v) == "Publication" and node_label.get(u) == "Dataset":
            p, d = v, u
        else:
            continue
        pd_edges.append((p, d, paper_year.get(p)))
    print(f"\nPaper→Dataset edges: {len(pd_edges):,} "
          f"({sum(1 for _,_,y in pd_edges if y):,} with valid year)")

    # ── Degree distributions ──────────────────────────────────
    papers_per_dataset = Counter(d for _, d, _ in pd_edges)
    datasets_per_paper = Counter(p for p, _, _ in pd_edges)

    # ── Co-usage graph ────────────────────────────────────────
    paper_to_datasets = defaultdict(set)
    for p, d, _ in pd_edges:
        paper_to_datasets[p].add(d)
    cousage_weight = Counter()
    for p, dsets in paper_to_datasets.items():
        if len(dsets) < 2:
            continue
        for di, dj in combinations(sorted(dsets), 2):
            cousage_weight[(di, dj)] += 1

    cousage_degree = Counter()
    for di, dj in cousage_weight:
        cousage_degree[di] += 1
        cousage_degree[dj] += 1

    # ── Temporal split ────────────────────────────────────────
    def split_of(year):
        if year is None:
            return None
        if year <= TRAIN_MAX_YEAR: return "train"
        if year == VAL_YEAR:       return "val"
        if year == TEST_YEAR:      return "test"
        return None

    split_pairs = {"train": Counter(), "val": Counter(), "test": Counter()}
    for p, dsets in paper_to_datasets.items():
        s = split_of(paper_year.get(p))
        if s is None or len(dsets) < 2:
            continue
        for di, dj in combinations(sorted(dsets), 2):
            split_pairs[s][(di, dj)] += 1

    train_set = set(split_pairs["train"])
    test_unseen = [pr for pr in split_pairs["test"] if pr not in train_set]

    # ── Slices ────────────────────────────────────────────────
    train_paper_count = Counter()
    for p, dsets in paper_to_datasets.items():
        if split_of(paper_year.get(p)) == "train":
            for d in dsets:
                train_paper_count[d] += 1

    def is_cold(d): return train_paper_count[d] < MIN_PAPERS_PER_DATASET
    def daac_of(d): return dataset_meta.get(d, {}).get("daac", "")
    def instr_of(d): return dataset_instruments.get(d, set())

    cold_start = [pr for pr in test_unseen if is_cold(pr[0]) or is_cold(pr[1])]
    cross_daac = [pr for pr in test_unseen
                  if daac_of(pr[0]) and daac_of(pr[1])
                  and daac_of(pr[0]) != daac_of(pr[1])]
    cross_instr = [pr for pr in test_unseen
                   if instr_of(pr[0]) and instr_of(pr[1])
                   and not (instr_of(pr[0]) & instr_of(pr[1]))]

    return {
        "G": G,
        "node_label": node_label,
        "nodes_by_label": nodes_by_label,
        "node_counts": node_counts,
        "edge_counts": dict(edge_counts),
        "edge_schema_counts": dict(edge_schema_counts),
        "paper_year": paper_year,
        "pub_years": pub_years,
        "dataset_meta": dataset_meta,
        "daac_counts": dict(daac_counts),
        "dataset_temporal": dataset_temporal,
        "dataset_platforms": dataset_platforms,
        "dataset_instruments": dataset_instruments,
        "dataset_keywords": dataset_keywords,
        "pd_edges": pd_edges,
        "paper_to_datasets": paper_to_datasets,
        "papers_per_dataset": papers_per_dataset,
        "datasets_per_paper": datasets_per_paper,
        "cousage_weight": cousage_weight,
        "cousage_degree": cousage_degree,
        "split_pairs": split_pairs,
        "train_paper_count": train_paper_count,
        "test_unseen": test_unseen,
        "cold_start": cold_start,
        "cross_daac": cross_daac,
        "cross_instr": cross_instr,
    }


# ══════════════════════════════════════════════════════════════
# FIGURE 1 — Graph overview (live data)
# ══════════════════════════════════════════════════════════════
def fig_graph_overview(S):
    print("\n▶ Rendering fig_graph_overview.pdf …")

    # Preserve the fixed display order from the original figure.
    node_order = ["Publication", "Dataset", "ScienceKeyword",
                  "Instrument", "Platform", "Project", "DataCenter"]
    edge_order = ["CITES", "HAS_APPLIEDRESEARCHAREA", "USES_DATASET",
                  "HAS_SCIENCEKEYWORD", "HAS_PLATFORM", "HAS_DATASET",
                  "OF_PROJECT", "HAS_INSTRUMENT", "HAS_SUBCATEGORY"]
    node_data = {k: S["node_counts"].get(k, 0) for k in node_order
                 if S["node_counts"].get(k, 0) > 0}
    edge_data = {k: S["edge_counts"].get(k, 0) for k in edge_order
                 if S["edge_counts"].get(k, 0) > 0}
    pub_years = S["pub_years"]

    # Top-7 DAACs + Other for the donut (matches the original visual).
    raw_daac = dict(S["daac_counts"])
    top_keys = ["ASDC", "GES-DISC", "LP DAAC", "NSIDC", "PODAAC",
                "OB.DAAC", "GHRC"]
    daac_display = {}
    other = 0
    for k, v in raw_daac.items():
        if k in top_keys:
            daac_display[k] = v
        else:
            other += v
    if other > 0:
        daac_display["Other"] = other
    # Reorder by the conventional top_keys order + Other at end
    daac_display = {k: daac_display[k] for k in top_keys + ["Other"]
                    if k in daac_display}

    fig = plt.figure(figsize=(7.2, 5.8))
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.38, wspace=0.80,
                           top=0.94, bottom=0.22,
                           left=0.13, right=0.97)

    # (a) Nodes
    ax1 = fig.add_subplot(gs[0, 0])
    labels = list(node_data.keys())
    counts = list(node_data.values())
    colors = [NODE_COLORS.get(l, GRAY) for l in labels]
    y = np.arange(len(labels))
    bars = ax1.barh(y, counts, color=colors, height=0.6,
                    edgecolor="white", linewidth=0.4)
    ax1.set_yticks(y); ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Number of Nodes", fontsize=8)
    ax1.set_title("(a) Node Type Distribution", fontweight="bold", fontsize=9)
    ax1.set_xscale("log")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(bars, counts):
        ax1.text(val * 1.05, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", fontsize=7, color=DGRAY)
    ax1.grid(axis="x", alpha=0.3); ax1.set_axisbelow(True)

    # (b) Edges
    ax2 = fig.add_subplot(gs[0, 1])
    elabels = list(edge_data.keys())
    ecounts = list(edge_data.values())
    ecolors = [EDGE_COLORS.get(e, GRAY) for e in elabels]
    ey = np.arange(len(elabels))
    ax2.barh(ey, ecounts, color=ecolors, height=0.6,
             edgecolor="white", linewidth=0.4)
    ax2.set_yticks(ey); ax2.set_yticklabels(elabels, fontsize=7.5)
    ax2.set_xlabel("Number of Edges", fontsize=8)
    ax2.set_title("(b) Edge Type Distribution", fontweight="bold", fontsize=9)
    ax2.set_xscale("log")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.grid(axis="x", alpha=0.3); ax2.set_axisbelow(True)

    # (c) Publication growth
    ax3 = fig.add_subplot(gs[1, 0])
    years = list(pub_years.keys()); pubs = list(pub_years.values())
    ax3.fill_between(years, pubs, alpha=0.18, color=BLUE)
    ax3.plot(years, pubs, color=BLUE, linewidth=1.8,
             marker="o", markersize=4, markerfacecolor="white",
             markeredgewidth=1.5, markeredgecolor=BLUE)
    ax3.set_xlabel("Year", fontsize=8); ax3.set_ylabel("Publications", fontsize=8)
    ax3.set_title("(c) Publication Growth", fontweight="bold", fontsize=9)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
    ax3.set_xticks(years[::2])
    if years:
        ax3.set_xlim(min(years) - 0.5, max(years) + 0.5)
    if pub_years:
        peak_x = max(pub_years, key=pub_years.get); peak_y = pub_years[peak_x]
        ax3.annotate(f"Peak: {peak_y:,}",
                     xy=(peak_x, peak_y),
                     xytext=(peak_x - 3.5, peak_y * 0.99),
                     fontsize=7, color=BLUE,
                     arrowprops=dict(arrowstyle="->", color=BLUE, lw=0.8))
    ax3.grid(alpha=0.3); ax3.set_axisbelow(True)

    # (d) DAAC donut
    ax4 = fig.add_subplot(gs[1, 1])
    dlabels = list(daac_display.keys()); dvals = list(daac_display.values())
    dcolors = [BLUE, TEAL, ORANGE, GREEN, PURPLE,
               GOLD, "#d95f02", GRAY][:len(dlabels)]
    wedges, _, autotexts = ax4.pie(
        dvals, labels=None, colors=dcolors,
        autopct=lambda p: f"{p:.1f}%" if p > 5 else "",
        pctdistance=0.78, startangle=140,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=0.8),
        textprops=dict(fontsize=7),
    )
    for at in autotexts:
        at.set_fontsize(6.5); at.set_color("white"); at.set_fontweight("bold")
    ax4.legend(wedges, [f"{l} ({v:,})" for l, v in zip(dlabels, dvals)],
               loc="lower left", bbox_to_anchor=(-0.35, -0.25),
               fontsize=6, framealpha=0.9, ncol=2,
               handlelength=1.0, handleheight=0.8)
    ax4.set_title("(d) Dataset Distribution by DAAC",
                  fontweight="bold", fontsize=9)

    path = f"{OUT_DIR}/fig_graph_overview.pdf"
    fig.savefig(path); plt.close(fig)
    print(f"   ✅ {path}")


# ══════════════════════════════════════════════════════════════
# FIGURE 2 — Schema diagram (live counts)
# ══════════════════════════════════════════════════════════════
def fig_schema_diagram(S):
    print("\n▶ Rendering fig_schema_diagram.pdf …")
    nc = S["node_counts"]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 5); ax.axis("off")

    nodes = {
        "Publication":    (1.8, 3.8),
        "Dataset":        (5.0, 3.8),
        "ScienceKeyword": (8.2, 3.8),
        "Instrument":     (5.0, 1.8),
        "Platform":       (3.0, 1.8),
        "Project":        (7.0, 1.8),
        "DataCenter":     (5.0, 0.3),
    }
    node_w, node_h = 1.55, 0.52

    def draw_node(label, xy, color):
        x, y = xy
        count = nc.get(label, 0)
        ax.add_patch(FancyBboxPatch(
            (x - node_w / 2, y - node_h / 2), node_w, node_h,
            boxstyle="round,pad=0.06", facecolor=color,
            edgecolor="white", linewidth=1.2, zorder=3))
        ax.text(x, y + 0.08, label, ha="center", va="center",
                fontsize=8, fontweight="bold", color="white", zorder=4)
        ax.text(x, y - 0.13, f"n={count:,}", ha="center", va="center",
                fontsize=6.5, color="white", alpha=0.88, zorder=4)

    for lbl, pos in nodes.items():
        draw_node(lbl, pos, NODE_COLORS[lbl])

    edges = [
        ("Publication",    "Publication",    "CITES",                    0.35,  BLUE,   ( 0.0,  0.35)),
        ("Publication",    "Dataset",        "USES_DATASET",             0.0,   TEAL,   ( 0.0,  0.18)),
        ("Publication",    "ScienceKeyword", "HAS_APPLIEDRESEARCHAREA",  0.18,  ORANGE, ( 0.0,  0.18)),
        ("Dataset",        "ScienceKeyword", "HAS_SCIENCEKEYWORD",       0.0,   ORANGE, ( 0.0,  0.18)),
        ("Dataset",        "Platform",       "HAS_PLATFORM",             0.0,   PURPLE, ( 0.0, -0.18)),
        ("Dataset",        "Instrument",     "HAS_INSTRUMENT",           0.0,   GOLD,   ( 0.0, -0.18)),
        ("Dataset",        "Project",        "OF_PROJECT",               0.0,   GREEN,  ( 0.0, -0.18)),
        ("DataCenter",     "Dataset",        "HAS_DATASET",              0.0,   GRAY,   (-0.3,  0.0)),
        ("ScienceKeyword", "ScienceKeyword", "HAS_SUBCATEGORY",          0.4,   GRAY,   ( 0.45, 0.2)),
    ]

    def border(center, target, w, h):
        cx, cy = center; tx, ty = target
        dx, dy = tx - cx, ty - cy
        if dx == 0 and dy == 0:
            return cx, cy + h / 2
        scale = min(abs(w / 2 / dx) if dx else 1e9,
                    abs(h / 2 / dy) if dy else 1e9)
        return cx + dx * scale, cy + dy * scale

    for src, tgt, lbl, curve, color, loff in edges:
        sp, tp = nodes[src], nodes[tgt]
        if src == tgt:
            x, y = sp
            ax.add_patch(mpatches.FancyArrowPatch(
                (x - 0.3, y + node_h / 2), (x + 0.3, y + node_h / 2),
                connectionstyle="arc3,rad=-1.2",
                arrowstyle="-|>", color=color,
                mutation_scale=10, linewidth=1.0, zorder=2))
            ax.text(x + loff[0], y + node_h / 2 + 0.38 + loff[1],
                    lbl, ha="center", fontsize=6, color=color, style="italic")
            continue
        sb = border(sp, tp, node_w, node_h)
        tb = border(tp, sp, node_w, node_h)
        ax.add_patch(mpatches.FancyArrowPatch(
            sb, tb, connectionstyle=f"arc3,rad={curve}",
            arrowstyle="-|>", color=color, mutation_scale=10,
            linewidth=1.0, zorder=2))
        mx = (sb[0] + tb[0]) / 2 + loff[0]
        my = (sb[1] + tb[1]) / 2 + loff[1]
        ax.text(mx, my, lbl, ha="center", fontsize=5.8,
                color=color, style="italic",
                bbox=dict(facecolor="white", alpha=0.7,
                          edgecolor="none", pad=1.0))

    ax.set_title("NASA EO Knowledge Graph — Schema",
                 fontweight="bold", fontsize=10, pad=8)
    path = f"{OUT_DIR}/fig_schema_diagram.pdf"
    fig.savefig(path); plt.close(fig)
    print(f"   ✅ {path}")


# ══════════════════════════════════════════════════════════════
# FIGURE 3 — Usage overview (citation & co-usage distributions)
# ══════════════════════════════════════════════════════════════
def fig_usage_overview(S):
    print("\n▶ Rendering fig_usage_overview.pdf …")
    fig = plt.figure(figsize=(7.2, 5.8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.38,
                           top=0.94, bottom=0.10, left=0.10, right=0.97)
    ax1 = fig.add_subplot(gs[0, 0])
    log_hist(ax1, list(S["papers_per_dataset"].values()), TEAL,
             "Papers citing a given dataset", "(a) Dataset citation distribution")
    ax2 = fig.add_subplot(gs[0, 1])
    log_hist(ax2, list(S["datasets_per_paper"].values()), BLUE,
             "Datasets cited by a given paper", "(b) Datasets-per-paper distribution")
    ax3 = fig.add_subplot(gs[1, 0])
    cw = list(S["cousage_weight"].values())
    log_hist(ax3, cw, ORANGE,
             "# of papers sharing a dataset pair", "(c) Co-usage edge weights")
    ax3.text(0.55, 0.85,
             f"{len(cw):,} unique pairs\n{sum(cw):,} total weight",
             transform=ax3.transAxes, fontsize=7, color=DGRAY,
             bbox=dict(facecolor="white", edgecolor="none", alpha=0.8))
    ax4 = fig.add_subplot(gs[1, 1])
    log_hist(ax4, list(S["cousage_degree"].values()), PURPLE,
             "# of co-used partners per dataset", "(d) Co-usage graph degree")
    path = f"{OUT_DIR}/fig_usage_overview.pdf"
    fig.savefig(path); plt.close(fig)
    print(f"   ✅ {path}")


# ══════════════════════════════════════════════════════════════
# FIGURE 4 — Temporal split
# ══════════════════════════════════════════════════════════════
def fig_temporal_split(S):
    print("\n▶ Rendering fig_temporal_split.pdf …")

    edges_by_year = Counter(y for _, _, y in S["pd_edges"] if y is not None)
    pairs_by_year = Counter()
    for p, dsets in S["paper_to_datasets"].items():
        y = S["paper_year"].get(p)
        if y is None or len(dsets) < 2:
            continue
        pairs_by_year[y] += len(list(combinations(dsets, 2)))

    years = sorted(y for y in set(edges_by_year) | set(pairs_by_year)
                   if 2015 <= y <= 2025)

    def split_color(y):
        if y <= TRAIN_MAX_YEAR: return BLUE
        if y == VAL_YEAR: return GOLD
        if y == TEST_YEAR: return ORANGE
        return GRAY

    colors = [split_color(y) for y in years]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(7.2, 2.8),
        gridspec_kw={"wspace": 0.35, "bottom": 0.22,
                     "top": 0.88, "left": 0.09, "right": 0.98},
    )
    axA.bar(years, [edges_by_year.get(y, 0) for y in years],
            color=colors, edgecolor="white", linewidth=0.5, width=0.75)
    axA.set_xlabel("Year", fontsize=8)
    axA.set_ylabel("USES_DATASET edges", fontsize=8)
    axA.set_title("(a) Paper→Dataset edges per year",
                  fontweight="bold", fontsize=9)
    axA.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
    axA.set_xticks(years); axA.tick_params(axis="x", rotation=45)
    legend = [
        Patch(facecolor=BLUE,   label=f"Train  (≤{TRAIN_MAX_YEAR})"),
        Patch(facecolor=GOLD,   label=f"Val    ({VAL_YEAR})"),
        Patch(facecolor=ORANGE, label=f"Test   ({TEST_YEAR})"),
        Patch(facecolor=GRAY,   label="Dropped (2025)"),
    ]
    axA.legend(handles=legend, loc="upper left", fontsize=6.5, frameon=False)

    axB.bar(years, [pairs_by_year.get(y, 0) for y in years],
            color=colors, edgecolor="white", linewidth=0.5, width=0.75)
    axB.set_xlabel("Year", fontsize=8)
    axB.set_ylabel("Co-usage weight", fontsize=8)
    axB.set_title("(b) Co-usage pairs per year",
                  fontweight="bold", fontsize=9)
    axB.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else f"{int(x)}"))
    axB.set_xticks(years); axB.tick_params(axis="x", rotation=45)

    path = f"{OUT_DIR}/fig_temporal_split.pdf"
    fig.savefig(path); plt.close(fig)
    print(f"   ✅ {path}")


# ══════════════════════════════════════════════════════════════
# TABLES — all from live data
# ══════════════════════════════════════════════════════════════
def generate_tables(S):
    print("\n▶ Writing LaTeX tables …")

    # Aggregate numbers
    node_order = ["Publication", "Dataset", "ScienceKeyword",
                  "Instrument", "Platform", "Project", "DataCenter"]
    node_counts = {k: S["node_counts"].get(k, 0) for k in node_order
                   if S["node_counts"].get(k, 0) > 0}
    total_nodes = sum(node_counts.values())
    total_edges = sum(S["edge_counts"].values())

    # Degree stats on the full graph (all edges)
    G = S["G"]
    degrees = [d for _, d in G.degree()]
    mean_deg = float(np.mean(degrees)) if degrees else 0.0
    max_deg = max(degrees) if degrees else 0

    # Publications with title/abstract (for the summary table)
    pub_nodes = S["nodes_by_label"].get("Publication", [])
    pub_with_content = sum(
        1 for n in pub_nodes
        if G.nodes[n].get("title") or G.nodes[n].get("abstract"))
    pct_pub_content = 100 * pub_with_content / max(1, len(pub_nodes))

    # Dataset temporal span from sane-filtered metadata
    dt = S["dataset_temporal"]
    temp_span = (f"{dt['earliest_start']}–{dt['latest_end']}"
                 if dt['earliest_start'] and dt['latest_end']
                 else "n/a")

    cites_count = S["edge_counts"].get("CITES", 0)
    uses_count  = S["edge_counts"].get("USES_DATASET", 0)

    lines = [
        r"% ════════════════════════════════════════════",
        r"% NASA EO Knowledge Graph — Paper Tables",
        r"% Generated by stage0_pipeline.py (from live GraphML)",
        r"% ════════════════════════════════════════════",
        r"% Required packages: booktabs, siunitx, xcolor",
        "",
    ]

    # ── Table 1: Graph summary ────────────────────────────────
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Summary statistics of the NASA Earth Observation Knowledge "
        r"Graph (NASA EO-KG), computed directly from the released GraphML.}",
        r"\label{tab:graph_stats}",
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"\textbf{Property} & \textbf{Value} & \textbf{Note} \\",
        r"\midrule",
        f"Total nodes       & \\num{{{total_nodes}}}   & {len(node_counts)} distinct types \\\\",
        f"Total edges       & \\num{{{total_edges}}}   & {len(S['edge_counts'])} relation types \\\\",
        f"Mean degree       & {mean_deg:.2f}           & Across all nodes \\\\",
        f"Max degree        & \\num{{{max_deg}}}       & High-citation node \\\\",
        f"Publications      & \\num{{{len(pub_nodes)}}} & {pct_pub_content:.1f}\\% with title/abstract \\\\",
        f"Datasets          & \\num{{{node_counts.get('Dataset', 0)}}} & 100\\% with CMR metadata \\\\",
        f"Dataset temporal span & {temp_span} & Earliest start -- latest end \\\\",
        f"CITES edges       & \\num{{{cites_count}}}   & Citation network \\\\",
        f"USES\\_DATASET     & \\num{{{uses_count}}}   & Pub.\\ $\\to$ Dataset links \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    # ── Table 2: Node types ───────────────────────────────────
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Node type distribution in NASA EO-KG with key attributes.}",
        r"\label{tab:node_types}",
        r"\small",
        r"\begin{tabular}{llrp{4.5cm}}",
        r"\toprule",
        r"\textbf{Node Type} & \textbf{Count} & \textbf{\%} & \textbf{Key Attributes} \\",
        r"\midrule",
    ]
    for lbl, cnt in node_counts.items():
        pct = 100 * cnt / total_nodes
        attrs = NODE_ATTRS.get(lbl, "--")
        lines.append(
            f"\\texttt{{{lbl}}} & \\num{{{cnt}}} & {pct:.1f}\\% & {attrs} \\\\"
        )
    lines += [
        r"\midrule",
        f"\\textbf{{Total}} & \\num{{{total_nodes}}} & 100\\% & \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    # ── Table 3: Edge schema (with live src→tgt resolution) ───
    # Pick the dominant src→tgt for each edge type
    schema_by_edge = {}
    for (src, edge, tgt), c in S["edge_schema_counts"].items():
        cur = schema_by_edge.get(edge)
        if cur is None or c > cur[2]:
            schema_by_edge[edge] = (src, tgt, c)

    edge_order = ["CITES", "HAS_APPLIEDRESEARCHAREA", "USES_DATASET",
                  "HAS_SCIENCEKEYWORD", "HAS_PLATFORM", "HAS_DATASET",
                  "OF_PROJECT", "HAS_INSTRUMENT", "HAS_SUBCATEGORY"]
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Edge (relation) types in NASA EO-KG with dominant source "
        r"and target node types.}",
        r"\label{tab:edge_types}",
        r"\small",
        r"\begin{tabular}{llll r}",
        r"\toprule",
        r"\textbf{Relation} & \textbf{Source} & \textbf{Target} & \textbf{Count} & \textbf{\%} \\",
        r"\midrule",
    ]
    for e in edge_order:
        if e not in schema_by_edge:
            continue
        src, tgt, _ = schema_by_edge[e]
        cnt = S["edge_counts"][e]
        pct = 100 * cnt / total_edges
        rel = e.replace("_", r"\_")
        lines.append(
            f"\\texttt{{{rel}}} & \\texttt{{{src}}} & \\texttt{{{tgt}}} "
            f"& \\num{{{cnt}}} & {pct:.1f}\\% \\\\"
        )
    lines += [
        r"\midrule",
        f"\\textbf{{Total}} & & & \\num{{{total_edges}}} & 100\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    # ── Table 4: DAAC distribution (live) ─────────────────────
    raw_daac = dict(S["daac_counts"])
    top_keys = ["ASDC", "GES-DISC", "LP DAAC", "NSIDC", "PODAAC",
                "OB.DAAC", "GHRC"]
    other_cnt = sum(v for k, v in raw_daac.items() if k not in top_keys)
    rows = [(k, raw_daac.get(k, 0)) for k in top_keys if raw_daac.get(k, 0)]
    if other_cnt > 0:
        rows.append(("Other", other_cnt))
    total_ds = sum(v for _, v in rows)

    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Distribution of NASA EO-KG datasets across Distributed "
        r"Active Archive Centers (DAACs).}",
        r"\label{tab:daac}",
        r"\small",
        r"\begin{tabular}{llr r}",
        r"\toprule",
        r"\textbf{DAAC} & \textbf{Full Name} & \textbf{Datasets} & \textbf{\%} \\",
        r"\midrule",
    ]
    for short, cnt in rows:
        pct = 100 * cnt / total_ds
        full = DAAC_CANONICAL.get(short, "Remaining DAACs")
        lines.append(
            f"\\texttt{{{short}}} & {full} & \\num{{{cnt}}} & {pct:.1f}\\% \\\\"
        )
    lines += [
        r"\midrule",
        f"\\textbf{{Total}} & & \\num{{{total_ds}}} & 100\\% \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    # ── Table 5: Co-usage statistics ──────────────────────────
    sp = S["split_pairs"]; cw = S["cousage_weight"]; cd = S["cousage_degree"]
    ppd = S["papers_per_dataset"]; dpp = S["datasets_per_paper"]
    lines += [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Co-usage statistics for the NASA EO-KG. Co-usage edges "
        r"are dataset pairs cited together in at least one publication. "
        r"Train/Val/Test split is temporal "
        f"(pre-{VAL_YEAR} / {VAL_YEAR} / {TEST_YEAR}).}}",
        r"\label{tab:cousage_stats}",
        r"\small",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"\textbf{Property} & \textbf{Value} \\",
        r"\midrule",
        f"Paper$\\to$Dataset edges (total)       & \\num{{{len(S['pd_edges'])}}} \\\\",
        f"Unique datasets cited                  & \\num{{{len(ppd)}}} \\\\",
        f"Datasets with $\\geq {MIN_PAPERS_PER_DATASET}$ papers         & "
        f"\\num{{{sum(1 for c in ppd.values() if c >= MIN_PAPERS_PER_DATASET)}}} \\\\",
        f"Papers citing $\\geq 2$ datasets       & "
        f"\\num{{{sum(1 for c in dpp.values() if c >= 2)}}} \\\\",
        r"\midrule",
        f"Co-usage graph nodes                   & \\num{{{len(cd)}}} \\\\",
        f"Co-usage graph edges (unique pairs)    & \\num{{{len(cw)}}} \\\\",
        f"Sum of co-usage weights                & \\num{{{sum(cw.values())}}} \\\\",
        f"Max pair weight                        & "
        f"\\num{{{max(cw.values()) if cw else 0}}} \\\\",
        f"Median / Max degree                    & "
        f"{int(np.median(list(cd.values()))) if cd else 0} / "
        f"\\num{{{max(cd.values()) if cd else 0}}} \\\\",
        r"\midrule",
        f"Train co-usage pairs                   & \\num{{{len(sp['train'])}}} \\\\",
        f"Val co-usage pairs                     & \\num{{{len(sp['val'])}}} \\\\",
        f"Test co-usage pairs                    & \\num{{{len(sp['test'])}}} \\\\",
        f"Test pairs unseen in train             & \\num{{{len(S['test_unseen'])}}} \\\\",
        r"\midrule",
        f"Cold-start test pairs                  & \\num{{{len(S['cold_start'])}}} \\\\",
        f"Cross-DAAC test pairs                  & \\num{{{len(S['cross_daac'])}}} \\\\",
        f"Cross-instrument test pairs            & \\num{{{len(S['cross_instr'])}}} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]

    # Standalone + paste versions
    preamble = [
        r"\documentclass{article}",
        r"\usepackage{booktabs}",
        r"\usepackage{siunitx}",
        r"\usepackage{xcolor}",
        r"\usepackage{microtype}",
        r"\usepackage[margin=1in]{geometry}",
        r"\sisetup{group-separator={,}, group-minimum-digits=4}",
        r"\begin{document}",
        "",
    ]
    ending = ["", r"\end{document}"]

    with open(f"{OUT_DIR}/neurips_tables.tex", "w") as f:
        f.write("\n".join(preamble + lines + ending))
    with open(f"{OUT_DIR}/neurips_tables_only.tex", "w") as f:
        f.write("\n".join(lines))
    print(f"   ✅ {OUT_DIR}/neurips_tables.tex")
    print(f"   ✅ {OUT_DIR}/neurips_tables_only.tex")


# ══════════════════════════════════════════════════════════════
# Saving reusable artifacts + stats.json
# ══════════════════════════════════════════════════════════════
def save_artifacts(S):
    print("\n▶ Saving reusable TSVs and full_stats.json …")

    def save_pairs(path, pairs_counter):
        with open(path, "w") as f:
            f.write("dataset_i\tdataset_j\tweight\n")
            for (di, dj), w in pairs_counter.items():
                f.write(f"{di}\t{dj}\t{w}\n")

    save_pairs(f"{OUT_DIR}/cousage_edges_train.tsv", S["split_pairs"]["train"])
    save_pairs(f"{OUT_DIR}/cousage_edges_val.tsv",   S["split_pairs"]["val"])
    save_pairs(f"{OUT_DIR}/cousage_edges_test.tsv",  S["split_pairs"]["test"])

    with open(f"{OUT_DIR}/paper_dataset_edges.tsv", "w") as f:
        f.write("paper\tdataset\tyear\n")
        for p, d, y in S["pd_edges"]:
            f.write(f"{p}\t{d}\t{y if y is not None else ''}\n")

    with open(f"{OUT_DIR}/dataset_features.tsv", "w") as f:
        f.write("dataset\tshortName\tdaac\tnum_papers_train\tnum_platforms\t"
                "num_instruments\tnum_keywords\thas_abstract\n")
        for d, meta in S["dataset_meta"].items():
            f.write("\t".join([
                d,
                meta["shortName"].replace("\t", " "),
                meta["daac"],
                str(S["train_paper_count"][d]),
                str(len(S["dataset_platforms"][d])),
                str(len(S["dataset_instruments"][d])),
                str(len(S["dataset_keywords"][d])),
                "1" if meta["abstract"] else "0",
            ]) + "\n")

    stats = {
        "graph": {
            "num_nodes": S["G"].number_of_nodes(),
            "num_edges": S["G"].number_of_edges(),
            "node_counts": S["node_counts"],
            "edge_counts": S["edge_counts"],
        },
        "publications": {"per_year": S["pub_years"]},
        "daac": S["daac_counts"],
        "dataset_temporal": S["dataset_temporal"],
        "paper_dataset_edges": {
            "total": len(S["pd_edges"]),
            "datasets_cited_atleast_once": len(S["papers_per_dataset"]),
            "datasets_with_5plus_papers":
                sum(1 for c in S["papers_per_dataset"].values()
                    if c >= MIN_PAPERS_PER_DATASET),
            "papers_citing_atleast_two_ds":
                sum(1 for c in S["datasets_per_paper"].values() if c >= 2),
            "max_papers_per_dataset":   max(S["papers_per_dataset"].values()),
            "median_papers_per_dataset":
                int(np.median(list(S["papers_per_dataset"].values()))),
            "max_datasets_per_paper":   max(S["datasets_per_paper"].values()),
            "median_datasets_per_paper":
                int(np.median(list(S["datasets_per_paper"].values()))),
        },
        "cousage_graph": {
            "unique_pairs":    len(S["cousage_weight"]),
            "total_weight":    sum(S["cousage_weight"].values()),
            "max_pair_weight": max(S["cousage_weight"].values()) if S["cousage_weight"] else 0,
            "nodes_with_edges":len(S["cousage_degree"]),
            "median_degree":   int(np.median(list(S["cousage_degree"].values())))
                                   if S["cousage_degree"] else 0,
            "max_degree":      max(S["cousage_degree"].values())
                                   if S["cousage_degree"] else 0,
        },
        "temporal_split": {
            "train_pairs":         len(S["split_pairs"]["train"]),
            "val_pairs":           len(S["split_pairs"]["val"]),
            "test_pairs":          len(S["split_pairs"]["test"]),
            "test_unseen_in_train":len(S["test_unseen"]),
        },
        "slices": {
            "cold_start":       len(S["cold_start"]),
            "cross_daac":       len(S["cross_daac"]),
            "cross_instrument": len(S["cross_instr"]),
        },
        "config": {
            "TRAIN_MAX_YEAR": TRAIN_MAX_YEAR,
            "VAL_YEAR": VAL_YEAR,
            "TEST_YEAR": TEST_YEAR,
            "MIN_PAPERS_PER_DATASET": MIN_PAPERS_PER_DATASET,
        },
    }
    with open(f"{OUT_DIR}/full_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(f"   ✅ {OUT_DIR}/full_stats.json")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    S = load_graph_and_compute()
    fig_graph_overview(S)
    fig_schema_diagram(S)
    fig_usage_overview(S)
    fig_temporal_split(S)
    generate_tables(S)
    save_artifacts(S)

    print("\n" + "═" * 62)
    print("STAGE 0 PIPELINE COMPLETE")
    print("═" * 62)
    print(f"All outputs in ./{OUT_DIR}/")
    print("  Figures: fig_graph_overview, fig_schema_diagram,")
    print("           fig_usage_overview, fig_temporal_split")
    print("  Tables:  neurips_tables.tex, neurips_tables_only.tex")
    print("  Data:    paper_dataset_edges.tsv, dataset_features.tsv,")
    print("           cousage_edges_{train,val,test}.tsv")
    print("  Stats:   full_stats.json")
    print("\nNext: Stage 1 — Baselines "
          "(Common Neighbors, MF, Content, Node2Vec).")