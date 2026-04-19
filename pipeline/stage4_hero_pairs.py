"""
STAGE 4d — Hero pairs analysis
==============================
Extracts candidate "hero pairs" from Stratum A for qualitative discussion
in the paper. A hero pair is one that scores high on BOTH plausibility
and novelty — genuinely plausible, genuinely non-obvious combinations.

Selection logic (in order of preference):
    1. plausibility == 5 AND novelty >= 3      (high-plaus, moderately novel)
    2. plausibility == 4 AND novelty >= 4      (plausible, highly novel)
    3. plausibility == 5 AND novelty == 2      (strongly plausible, domain-standard)

Prints top candidates with full metadata + rationales, per scorer.
Writes a TSV of the top 20 candidates for easy browsing.
"""

from __future__ import annotations

import csv
from collections import defaultdict

JUDGMENTS_PATH = "./neurips_figs/stage4/stage4_judgments.tsv"
OUTPUT_PATH = "./neurips_figs/stage4/hero_pairs_candidates.tsv"


def load_judgments():
    rows = []
    with open(JUDGMENTS_PATH) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            r["plausibility"] = int(r["plausibility"])
            r["novelty"] = int(r["novelty"])
            rows.append(r)
    return rows


def combined_score(r):
    """Weight: plausibility slightly over novelty (paper emphasizes it)."""
    return 0.55 * r["plausibility"] + 0.45 * r["novelty"]


def print_pair(r, idx=None):
    prefix = f"  #{idx:2d}. " if idx is not None else "   "
    print(f"{prefix}[plaus={r['plausibility']}  novelty={r['novelty']}  "
          f"source={r['source_model']}]")
    print(f"        A: {r['shortName_i']}  ({r['daac_i']})")
    print(f"           {r['longName_i'][:95]}")
    print(f"        B: {r['shortName_j']}  ({r['daac_j']})")
    print(f"           {r['longName_j'][:95]}")
    print(f"        model scores:  dot={float(r['score_dot']):+.3f}  "
          f"mlp={float(r['score_mlp']):+.3f}")
    print(f"        GPT-5.2 rationale:")
    # Simple word-wrap for terminal
    rat = r["rationale"]
    words = rat.split()
    line = "          "
    for w in words:
        if len(line) + len(w) > 95:
            print(line); line = "          " + w
        else:
            line += (" " if line.strip() else "") + w
    if line.strip():
        print(line)
    print()


rows = load_judgments()
stratum_a = [r for r in rows if r["stratum"] == "A"]
print(f"Loaded {len(stratum_a)} Stratum A judgments\n")


# ══════════════════════════════════════════════════════════════
# Three tier lists
# ══════════════════════════════════════════════════════════════
tier_1 = [r for r in stratum_a if r["plausibility"] == 5 and r["novelty"] >= 3]
tier_2 = [r for r in stratum_a if r["plausibility"] == 4 and r["novelty"] >= 4]
tier_3 = [r for r in stratum_a if r["plausibility"] == 5 and r["novelty"] == 2]

tier_1.sort(key=combined_score, reverse=True)
tier_2.sort(key=combined_score, reverse=True)
tier_3.sort(key=combined_score, reverse=True)

print("═" * 70)
print("TIER 1 — plausibility=5 AND novelty>=3  (HIGHEST-VALUE HERO PAIRS)")
print("═" * 70)
print(f"Count: {len(tier_1)} pairs\n")
for i, r in enumerate(tier_1[:10], 1):
    print_pair(r, idx=i)

print("═" * 70)
print("TIER 2 — plausibility=4 AND novelty>=4  (NOVEL CROSS-DOMAIN PAIRS)")
print("═" * 70)
print(f"Count: {len(tier_2)} pairs\n")
for i, r in enumerate(tier_2[:10], 1):
    print_pair(r, idx=i)

print("═" * 70)
print("TIER 3 — plausibility=5 AND novelty=2  (STRONGLY PLAUSIBLE, DOMAIN-STANDARD)")
print("═" * 70)
print(f"Count: {len(tier_3)} pairs\n")
for i, r in enumerate(tier_3[:5], 1):
    print_pair(r, idx=i)


# ══════════════════════════════════════════════════════════════
# Split the hero candidates by source_model — do dot and MLP
# surface meaningfully different pairs?
# ══════════════════════════════════════════════════════════════
print("═" * 70)
print("HERO PAIRS BY SOURCE MODEL")
print("═" * 70)
by_src_tier = defaultdict(lambda: defaultdict(int))
for r in stratum_a:
    if r["plausibility"] == 5 and r["novelty"] >= 3:
        by_src_tier[r["source_model"]]["tier_1"] += 1
    if r["plausibility"] == 4 and r["novelty"] >= 4:
        by_src_tier[r["source_model"]]["tier_2"] += 1
    if r["plausibility"] == 5 and r["novelty"] == 2:
        by_src_tier[r["source_model"]]["tier_3"] += 1

for src in ["dot", "mlp"]:
    d = by_src_tier[src]
    print(f"  {src:<4}: tier_1={d['tier_1']:>3}  "
          f"tier_2={d['tier_2']:>3}  tier_3={d['tier_3']:>3}")


# ══════════════════════════════════════════════════════════════
# Save top 20 candidates to TSV for easier browsing
# ══════════════════════════════════════════════════════════════
all_heroes = tier_1 + tier_2 + tier_3
# Dedup (a pair can't be in more than one tier anyway but defensively)
seen = set()
heroes = []
for r in all_heroes:
    key = (r["dataset_i"], r["dataset_j"])
    if key in seen: continue
    seen.add(key)
    heroes.append(r)
heroes.sort(key=combined_score, reverse=True)

cols = ["source_model", "plausibility", "novelty",
        "shortName_i", "daac_i", "shortName_j", "daac_j",
        "longName_i", "longName_j",
        "score_dot", "score_mlp", "rank_dot", "rank_mlp",
        "rationale"]

with open(OUTPUT_PATH, "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in heroes[:25]:
        f.write("\t".join(str(r.get(c, "")).replace("\t", " ").replace("\n", " ")
                          for c in cols) + "\n")

print(f"\n✅ Saved top {min(25, len(heroes))} hero candidates to {OUTPUT_PATH}")
print(f"   Total hero-tier pairs found: {len(heroes)} out of 200")