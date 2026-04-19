"""
STAGE 5 FACTORIAL — Final analysis for paper
============================================
Three analyses on the completed 2x2x2 factorial:

  1. INTER-RATER AGREEMENT
     For each (hypothesis, condition) pair, do GPT and Claude judges
     agree on scores? Reports Pearson r, Spearman rho, exact-match %,
     within-1 %, and Cohen's kappa (weighted, ordinal).

  2. VARIANCE DECOMPOSITION
     Of the total variance in a score, how much is explained by
     Agent 1 / Agent 2 / Agent 3 / condition / their interactions?
     Reports sum of squares and R^2 contribution per factor.

  3. FLAGSHIP HYPOTHESES
     Find hypotheses that score HIGH across all 4 (generator x judge)
     conditions — i.e., robust to the choice of agent. These are the
     case studies the paper will feature.

Outputs:
    ./neurips_figs/stage5_factorial/analysis/inter_rater.tsv
    ./neurips_figs/stage5_factorial/analysis/variance_decomp.tsv
    ./neurips_figs/stage5_factorial/analysis/flagship_hypotheses.tsv
"""

from __future__ import annotations
import csv
import os
import statistics as st
from collections import defaultdict

OUT_DIR = "./neurips_figs"
FACT_DIR = f"{OUT_DIR}/stage5_factorial"
EXP_DIR = f"{FACT_DIR}/experiments"
ANALYSIS_DIR = f"{FACT_DIR}/analysis"
os.makedirs(ANALYSIS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════
# Load all validator data
# ═══════════════════════════════════════════════════════════════
def load_all_validations():
    rows = []
    for fname in sorted(os.listdir(EXP_DIR)):
        if not fname.endswith("_validations.tsv"):
            continue
        if fname.startswith("control_"):
            continue  # skip C-stratum controls for factorial
        # Parse experiment name from filename, e.g. "gpt_claude_claude_validations.tsv"
        parts = fname.replace("_validations.tsv", "").split("_")
        a1, a2, a3 = parts[0], parts[1], parts[2]
        path = f"{EXP_DIR}/{fname}"
        with open(path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                r["agent1"] = a1
                r["agent2"] = a2
                r["agent3"] = a3
                r["importance"] = int(r["importance"])
                r["tractability"] = int(r["tractability"])
                r["novelty"] = int(r["novelty"])
                rows.append(r)
    return rows


def load_hypotheses():
    """Hypotheses indexed by (agent1, agent2, orig_pair_key) for flagship selection."""
    hyps = {}
    for a1 in ["gpt", "claude"]:
        for a2 in ["gpt", "claude"]:
            path = f"{FACT_DIR}/hypotheses_{a1}_{a2}.tsv"
            if not os.path.exists(path):
                continue
            with open(path) as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    hyps[(a1, a2, r["orig_pair_key"])] = r
    return hyps


rows = load_all_validations()
hyps = load_hypotheses()
print(f"Loaded {len(rows)} validations, {len(hyps)} hypotheses")


# ═══════════════════════════════════════════════════════════════
# 1. INTER-RATER AGREEMENT
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("1. INTER-RATER AGREEMENT (GPT vs Claude on same hypothesis)")
print("═" * 72)

def pearson(xs, ys):
    n = len(xs)
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = (sum((x - mx) ** 2 for x in xs)) ** 0.5
    dy = (sum((y - my) ** 2 for y in ys)) ** 0.5
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def spearman(xs, ys):
    """Rank correlation. Simple implementation — ties get average ranks."""
    def rank(vs):
        sorted_vs = sorted(enumerate(vs), key=lambda t: t[1])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(sorted_vs):
            j = i
            while j + 1 < len(sorted_vs) and sorted_vs[j + 1][1] == sorted_vs[i][1]:
                j += 1
            avg_rank = (i + j) / 2 + 1  # 1-indexed
            for k in range(i, j + 1):
                ranks[sorted_vs[k][0]] = avg_rank
            i = j + 1
        return ranks
    return pearson(rank(xs), rank(ys))


def quadratic_weighted_kappa(xs, ys, min_rating=1, max_rating=5):
    """Cohen's quadratic weighted kappa for ordinal ratings."""
    n_cats = max_rating - min_rating + 1
    # Observed matrix
    O = [[0] * n_cats for _ in range(n_cats)]
    for x, y in zip(xs, ys):
        O[x - min_rating][y - min_rating] += 1
    # Marginals
    row_totals = [sum(row) for row in O]
    col_totals = [sum(O[i][j] for i in range(n_cats)) for j in range(n_cats)]
    N = len(xs)
    # Expected matrix
    E = [[row_totals[i] * col_totals[j] / N for j in range(n_cats)]
         for i in range(n_cats)]
    # Weights (quadratic)
    W = [[((i - j) ** 2) / ((n_cats - 1) ** 2) for j in range(n_cats)]
         for i in range(n_cats)]
    num = sum(W[i][j] * O[i][j] for i in range(n_cats) for j in range(n_cats))
    den = sum(W[i][j] * E[i][j] for i in range(n_cats) for j in range(n_cats))
    if den == 0: return float("nan")
    return 1 - num / den


# Build paired (GPT, Claude) for each (a1, a2, hyp_key, condition)
agreement_results = []
for axis in ["importance", "tractability", "novelty"]:
    for cond in ["blind", "ctx"]:
        paired = defaultdict(lambda: {"gpt": None, "claude": None})
        for r in rows:
            if r["condition"] != cond:
                continue
            key = (r["agent1"], r["agent2"], r["hyp_key"])
            paired[key][r["agent3"]] = r[axis]
        xs, ys = [], []
        for key, scores in paired.items():
            if scores["gpt"] is not None and scores["claude"] is not None:
                xs.append(scores["gpt"])
                ys.append(scores["claude"])
        if len(xs) < 5:
            continue
        r_p = pearson(xs, ys)
        r_s = spearman(xs, ys)
        kappa = quadratic_weighted_kappa(xs, ys)
        exact = sum(1 for x, y in zip(xs, ys) if x == y) / len(xs)
        near = sum(1 for x, y in zip(xs, ys) if abs(x - y) <= 1) / len(xs)
        agreement_results.append({
            "axis": axis, "condition": cond, "n": len(xs),
            "pearson": r_p, "spearman": r_s, "kappa_q": kappa,
            "exact_match": exact, "within_1": near,
        })
        print(f"  {axis:<14} [{cond:<5}]  n={len(xs):3d}  "
              f"Pearson={r_p:+.3f}  Spearman={r_s:+.3f}  "
              f"kappa_q={kappa:+.3f}  exact={exact:.0%}  within-1={near:.0%}")

# Save
with open(f"{ANALYSIS_DIR}/inter_rater.tsv", "w") as f:
    cols = list(agreement_results[0].keys())
    f.write("\t".join(cols) + "\n")
    for r in agreement_results:
        f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                          for c in cols) + "\n")
print(f"\n✅ {ANALYSIS_DIR}/inter_rater.tsv")


# ═══════════════════════════════════════════════════════════════
# 2. VARIANCE DECOMPOSITION
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("2. VARIANCE DECOMPOSITION (what factor explains score variance?)")
print("═" * 72)
print("   Reports eta-squared: sum_of_squares(factor) / total_sum_of_squares")
print()

def ss_total(vals):
    m = st.mean(vals)
    return sum((v - m) ** 2 for v in vals)


def ss_between(groups):
    """Sum of squares between groups, weighted by group size."""
    all_vals = [v for g in groups.values() for v in g]
    grand_mean = st.mean(all_vals)
    ss = 0
    for gvals in groups.values():
        if not gvals: continue
        gm = st.mean(gvals)
        ss += len(gvals) * (gm - grand_mean) ** 2
    return ss


var_decomp_rows = []
for axis in ["importance", "tractability", "novelty"]:
    vals = [r[axis] for r in rows]
    sst = ss_total(vals)
    if sst == 0:
        continue
    print(f"  [{axis}]  SST = {sst:.1f}  (total variance)")

    # Main effects
    for factor in ["agent1", "agent2", "agent3", "condition"]:
        groups = defaultdict(list)
        for r in rows:
            groups[r[factor]].append(r[axis])
        ssb = ss_between(groups)
        eta2 = ssb / sst
        print(f"     {factor:<12}: SS={ssb:6.1f}  eta^2={eta2:.4f}  ({100*eta2:4.1f}% of variance)")
        var_decomp_rows.append({
            "axis": axis, "factor": factor, "ss": ssb, "eta_squared": eta2,
        })

    # Two-way: agent3 x condition
    groups = defaultdict(list)
    for r in rows:
        groups[(r["agent3"], r["condition"])].append(r[axis])
    ssb = ss_between(groups)
    # Interaction = SS(a3*cond) - SS(a3) - SS(cond)
    ss_a3 = ss_between({k: [r[axis] for r in rows if r["agent3"] == k]
                        for k in ["gpt", "claude"]})
    ss_cond = ss_between({k: [r[axis] for r in rows if r["condition"] == k]
                          for k in ["blind", "ctx"]})
    ss_interaction = max(0, ssb - ss_a3 - ss_cond)
    eta2_int = ss_interaction / sst
    print(f"     agent3 x cond (interaction): SS={ss_interaction:6.1f}  "
          f"eta^2={eta2_int:.4f}  ({100*eta2_int:4.1f}%)")
    var_decomp_rows.append({
        "axis": axis, "factor": "agent3 x condition",
        "ss": ss_interaction, "eta_squared": eta2_int,
    })
    print()

with open(f"{ANALYSIS_DIR}/variance_decomp.tsv", "w") as f:
    cols = ["axis", "factor", "ss", "eta_squared"]
    f.write("\t".join(cols) + "\n")
    for r in var_decomp_rows:
        f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                          for c in cols) + "\n")
print(f"✅ {ANALYSIS_DIR}/variance_decomp.tsv")


# ═══════════════════════════════════════════════════════════════
# 3. FLAGSHIP HYPOTHESES
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("3. FLAGSHIP HYPOTHESES (robust high-quality across all conditions)")
print("═" * 72)
print("   For each hypothesis (a1, a2, hyp_key), compute mean score across")
print("   both judges x both conditions (4 validator scores total).")
print("   Top 15 shown by combined importance+tractability+novelty.")
print()

# Aggregate per hypothesis
hyp_scores = defaultdict(list)
for r in rows:
    key = (r["agent1"], r["agent2"], r["hyp_key"])
    hyp_scores[key].append(r)

flagship_rows = []
for key, vs in hyp_scores.items():
    if len(vs) < 4:  # need all 4 judge/cond combos
        continue
    a1, a2, hyp_key = key
    imp = st.mean([v["importance"] for v in vs])
    tra = st.mean([v["tractability"] for v in vs])
    nov = st.mean([v["novelty"] for v in vs])
    # Variance across conditions — low = robust, high = fragile
    imp_range = max(v["importance"] for v in vs) - min(v["importance"] for v in vs)
    tra_range = max(v["tractability"] for v in vs) - min(v["tractability"] for v in vs)
    nov_range = max(v["novelty"] for v in vs) - min(v["novelty"] for v in vs)
    combined = imp + tra + nov
    # Lookup hypothesis text
    h = hyps.get(key, {})
    flagship_rows.append({
        "agent1": a1, "agent2": a2, "hyp_key": hyp_key,
        "pair_short": f"{h.get('shortName_i','?')} + {h.get('shortName_j','?')}",
        "domain": h.get("domain", "?"),
        "imp_mean": imp, "tract_mean": tra, "novel_mean": nov,
        "combined": combined,
        "imp_range": imp_range, "tract_range": tra_range, "novel_range": nov_range,
        "max_disagreement": max(imp_range, tra_range, nov_range),
        "research_question": h.get("research_question", ""),
        "hypothesis": h.get("hypothesis", ""),
        "analysis_method": h.get("analysis_method", ""),
        "scientific_importance": h.get("scientific_importance", ""),
    })

# Sort by combined score, filter for robust (max_disagreement <= 2)
flagship_rows.sort(key=lambda r: -r["combined"])
robust_flagships = [r for r in flagship_rows if r["max_disagreement"] <= 2]

print(f"Total hypotheses scored: {len(flagship_rows)}")
print(f"Robust hypotheses (max disagreement across judges/conds <= 2): {len(robust_flagships)}")
print()
print("TOP 15 BY COMBINED SCORE (robust, max_disagreement <= 2):")
print("-" * 72)
for i, r in enumerate(robust_flagships[:15], 1):
    print(f"\n  #{i:2d}. [{r['agent1']}->{r['agent2']}] {r['pair_short']}")
    print(f"        domain: {r['domain']}")
    print(f"        scores: imp={r['imp_mean']:.2f}  tract={r['tract_mean']:.2f}  "
          f"novel={r['novel_mean']:.2f}  combined={r['combined']:.2f}")
    print(f"        max disagreement across 4 judgments: {r['max_disagreement']}")
    # Short snippet of research question
    rq = r["research_question"][:130]
    print(f"        Q: {rq}{'...' if len(r['research_question']) > 130 else ''}")

# Save full ranked list
cols = ["agent1", "agent2", "hyp_key", "pair_short", "domain",
        "imp_mean", "tract_mean", "novel_mean", "combined",
        "imp_range", "tract_range", "novel_range", "max_disagreement",
        "research_question", "hypothesis", "analysis_method",
        "scientific_importance"]
with open(f"{ANALYSIS_DIR}/flagship_hypotheses.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in flagship_rows:
        f.write("\t".join(
            f"{r[c]:.4f}" if isinstance(r[c], float) else str(r.get(c, "")).replace("\t", " ").replace("\n", " ")
            for c in cols
        ) + "\n")
print(f"\n✅ {ANALYSIS_DIR}/flagship_hypotheses.tsv  ({len(flagship_rows)} hypotheses ranked)")


# ═══════════════════════════════════════════════════════════════
# Quick summary
# ═══════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("SUMMARY FOR PAPER")
print("═" * 72)
print(f"""
  Inter-rater:          see {ANALYSIS_DIR}/inter_rater.tsv
  Variance decomp:      see {ANALYSIS_DIR}/variance_decomp.tsv
  Flagship hypotheses:  see {ANALYSIS_DIR}/flagship_hypotheses.tsv

  Ready to draft the paper. Top 3-5 flagship hypotheses above should
  become the qualitative case studies in the results section.
""")