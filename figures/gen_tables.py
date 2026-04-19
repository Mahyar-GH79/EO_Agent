"""
TABLE GENERATOR — writes 4 LaTeX tables ready to \\input{}
===========================================================
Produces:
  tab_gnn_baselines.tex     — Table 1: Hits@10 for baselines + GNN
  tab_stage4_strata.tex     — Table 2: Stage 4 strata summary
  tab_inter_rater.tex       — Table 3: inter-rater agreement
  tab_flagship.tex          — Table 4: flagship hypotheses

All written to ./neurips_figs/tables/
Each uses booktabs (\\toprule \\midrule \\bottomrule).
"""

import csv
import os
import statistics as st
from collections import defaultdict

OUT_DIR = "./neurips_figs"
TAB_DIR = f"{OUT_DIR}/tables"
FACT_DIR = f"{OUT_DIR}/stage5_factorial"
EXP_DIR = f"{FACT_DIR}/experiments"
os.makedirs(TAB_DIR, exist_ok=True)


def latex_escape(s):
    """Escape LaTeX special characters in string cells."""
    if s is None: return ""
    s = str(s)
    replacements = [("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
                    ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
                    ("^", r"\textasciicircum{}"), ("~", r"\textasciitilde{}")]
    for old, new in replacements:
        s = s.replace(old, new)
    return s


def write_table(path, content):
    with open(path, "w") as f:
        f.write(content)
    print(f"✅ {path}")


# ══════════════════════════════════════════════════════════════
# Table 1 — GNN + baselines Hits@10
# ══════════════════════════════════════════════════════════════
# Hardcoded from your Stage 2.5b verification + baselines from Stage 1
# (Source: your running notes in the conversation summary)
rows_t1 = [
    # (method, all, cold_start, cross_daac)
    ("Popularity",                  0.140, 0.064, 0.130),
    ("MF-SVD",                      0.192, 0.274, 0.125),
    ("Common Neighbors",            0.349, 0.211, 0.228),
    ("Adamic-Adar",                 0.355, 0.225, 0.235),
    ("BGE-base (content)",          0.394, 0.484, 0.103),
    ("SPECTER2 (content)",          0.413, 0.477, 0.127),
    ("GNN-Homo (ours)",             0.448, 0.473, 0.146),
    ("GNN-Hetero (ours)",           0.468, 0.519, 0.175),
    (r"\textbf{GNN-Hetero+area (dot)}",  0.474, 0.524, 0.170),
    (r"\textbf{GNN-Hetero+area (MLP)}",  0.472, 0.375, 0.282),
]
# Bold the max per column
col_max = {
    1: max(r[1] for r in rows_t1),
    2: max(r[2] for r in rows_t1),
    3: max(r[3] for r in rows_t1),
}

def fmt_cell(v, is_max):
    s = f"{v:.3f}"
    return f"\\textbf{{{s}}}" if is_max else s


lines = [
    r"\begin{table}[h]",
    r"\centering",
    r"\small",
    r"\caption{Hits@10 on three test-set slices (pre-2023 train, 2024 test). "
    r"\textbf{Bold}: best per column. Content baselines fail cross-DAAC; "
    r"structural baselines fail cold-start; heterogeneous GNN with research-area "
    r"edges is the first method competitive everywhere. The two scorer heads "
    r"form a Pareto pair: dot for cold-start, MLP for cross-DAAC.}",
    r"\label{tab:gnn_baselines}",
    r"\begin{tabular}{lccc}",
    r"\toprule",
    r"Method & All & Cold-start & Cross-DAAC \\",
    r"\midrule",
]
for m, a, c, x in rows_t1:
    lines.append(
        f"{m} & {fmt_cell(a, a == col_max[1])} & "
        f"{fmt_cell(c, c == col_max[2])} & {fmt_cell(x, x == col_max[3])} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write_table(f"{TAB_DIR}/tab_gnn_baselines.tex", "\n".join(lines))


# ══════════════════════════════════════════════════════════════
# Table 2 — Stage 4 strata
# ══════════════════════════════════════════════════════════════
# Read from GPT agent1 scores (which is a superset of stage4_judgments.tsv)
a1_gpt_path = f"{FACT_DIR}/agent1_gpt_scores.tsv"
a1_claude_path = f"{FACT_DIR}/agent1_claude_scores.tsv"

stratum_stats = {"A": {}, "B": {}, "C": {}, "D": {}}
for judge, path in [("gpt", a1_gpt_path), ("claude", a1_claude_path)]:
    if not os.path.exists(path): continue
    grps = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            grps[r["stratum"]].append((int(r["plausibility"]), int(r["novelty"])))
    for s, vs in grps.items():
        p = [v[0] for v in vs]; n = [v[1] for v in vs]
        stratum_stats[s][f"plaus_{judge}"] = st.mean(p)
        stratum_stats[s][f"novel_{judge}"] = st.mean(n)
        stratum_stats[s]["n"] = len(vs)

stratum_labels = {
    "A": "Predicted novel (model top-ranked)",
    "B": "Held-out real co-usages (2024)",
    "C": "Random novel pairs",
    "D": "Hard negatives (same-DAAC novel)",
}

lines = [
    r"\begin{table}[h]",
    r"\centering",
    r"\small",
    r"\caption{Stage 4 pair-level judgments. Each of 800 pairs (200 per stratum) "
    r"was scored by both GPT-5.2 and Claude Sonnet 4.6 on plausibility and "
    r"novelty (1–5). Both judges order the strata identically "
    r"(B $\geq$ A $>$ D $>$ C on plausibility); Claude's scores show a tighter "
    r"distribution on novelty but a larger A$-$C plausibility gap.}",
    r"\label{tab:stage4_strata}",
    r"\begin{tabular}{llcccc}",
    r"\toprule",
    r"\multirow{2}{*}{Stratum} & \multirow{2}{*}{Description} & "
    r"\multicolumn{2}{c}{Plausibility} & \multicolumn{2}{c}{Novelty} \\",
    r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}",
    r"& & GPT-5.2 & Claude 4.6 & GPT-5.2 & Claude 4.6 \\",
    r"\midrule",
]
for s in ["B", "A", "D", "C"]:
    d = stratum_stats.get(s, {})
    lines.append(
        f"{s} & {stratum_labels[s]} & "
        f"{d.get('plaus_gpt', 0):.2f} & {d.get('plaus_claude', 0):.2f} & "
        f"{d.get('novel_gpt', 0):.2f} & {d.get('novel_claude', 0):.2f} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
write_table(f"{TAB_DIR}/tab_stage4_strata.tex", "\n".join(lines))


# ══════════════════════════════════════════════════════════════
# Table 3 — Inter-rater agreement
# ══════════════════════════════════════════════════════════════
inter_rater_path = f"{FACT_DIR}/analysis/inter_rater.tsv"
if os.path.exists(inter_rater_path):
    ir_rows = []
    with open(inter_rater_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ir_rows.append(r)

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{Inter-rater agreement between GPT-5.2 and Claude Sonnet 4.6 "
        r"on identical hypotheses. Judges agree best on tractability "
        r"(Pearson $r > 0.60$), less on novelty, and only weakly on importance "
        r"($r \approx 0.35$–$0.40$). Showing full dataset context improves "
        r"agreement on all three axes. $\kappa_q$ is Cohen's quadratic-weighted "
        r"kappa.}",
        r"\label{tab:inter_rater}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Axis & Condition & $n$ & Pearson $r$ & Spearman $\rho$ & "
        r"$\kappa_q$ & Exact \% \\",
        r"\midrule",
    ]
    for r in ir_rows:
        lines.append(
            f"{latex_escape(r['axis']).capitalize()} & {r['condition']} & "
            f"{r['n']} & {float(r['pearson']):+.3f} & "
            f"{float(r['spearman']):+.3f} & "
            f"{float(r['kappa_q']):+.3f} & "
            f"{float(r['exact_match']) * 100:.0f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table(f"{TAB_DIR}/tab_inter_rater.tex", "\n".join(lines))


# ══════════════════════════════════════════════════════════════
# Table 4 — Flagship hypotheses
# ══════════════════════════════════════════════════════════════
flagship_path = f"{FACT_DIR}/analysis/flagship_hypotheses.tsv"
if os.path.exists(flagship_path):
    fs_rows = []
    with open(flagship_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            r["imp_mean"] = float(r["imp_mean"])
            r["tract_mean"] = float(r["tract_mean"])
            r["novel_mean"] = float(r["novel_mean"])
            r["combined"] = float(r["combined"])
            r["max_disagreement"] = int(float(r["max_disagreement"]))
            fs_rows.append(r)
    fs_rows.sort(key=lambda r: -r["combined"])

    # Pick 5 flagship candidates: diversity across domains, top scoring,
    # low disagreement. Let's hand-pick from our earlier analysis
    # using recognizable pairs.
    # We'll just take the top-5 unique-by-(shortName_i, shortName_j) pairs.
    seen_pairs = set()
    picks = []
    for r in fs_rows:
        pair_key = r["pair_short"]
        if pair_key in seen_pairs:
            continue
        if r["max_disagreement"] > 2:
            continue
        seen_pairs.add(pair_key)
        picks.append(r)
        if len(picks) >= 5: break

    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\small",
        r"\caption{Five flagship hypotheses from the 2$\times$2$\times$2 factorial. "
        r"Each was generated by the (Agent 1, Agent 2) combination listed, then "
        r"rated by both judges under both conditions (blind and contextual); the "
        r"\textit{Imp}, \textit{Tract}, and \textit{Novel} columns report the "
        r"mean across these 4 validator judgments. \textit{Max $\Delta$} is the "
        r"largest disagreement on any axis among the 4 judgments; a low value "
        r"indicates the hypothesis is rated consistently across agents and "
        r"conditions.}",
        r"\label{tab:flagship}",
        r"\begin{tabular}{p{2.3cm}p{2.1cm}cccccp{4.2cm}}",
        r"\toprule",
        r"Dataset pair & Domain & Imp & Tract & Novel & Max $\Delta$ & "
        r"Source (A1→A2) & Research question (abbreviated) \\",
        r"\midrule",
    ]
    for r in picks:
        a1a2 = f"{r['agent1']}$\\to${r['agent2']}"
        pair = latex_escape(r["pair_short"])
        dom = latex_escape(r["domain"])
        rq = r["research_question"]
        # Truncate to ~110 chars
        if len(rq) > 110:
            rq = rq[:107].rstrip() + "…"
        rq_esc = latex_escape(rq)
        lines.append(
            f"{pair} & {dom} & "
            f"{r['imp_mean']:.2f} & {r['tract_mean']:.2f} & {r['novel_mean']:.2f} & "
            f"{r['max_disagreement']} & {a1a2} & {rq_esc} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_table(f"{TAB_DIR}/tab_flagship.tex", "\n".join(lines))


print(f"\n✅ All tables in {TAB_DIR}/")
print("   In the paper, \\input{tables/tab_gnn_baselines.tex} etc.")