import csv
from collections import defaultdict

by_source = defaultdict(list)
with open("./neurips_figs/stage4/stage4_judgments.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["stratum"] != "A":
            continue
        key = r["source_model"]
        by_source[key].append((int(r["plausibility"]), int(r["novelty"])))

print(f"{'source':<10} {'n':<5} {'plausibility':<15} {'novelty':<10}")
for src, vals in by_source.items():
    plaus = [v[0] for v in vals]
    novel = [v[1] for v in vals]
    print(f"{src:<10} {len(vals):<5} "
          f"{sum(plaus)/len(plaus):<15.3f} "
          f"{sum(novel)/len(novel):<10.3f}")

# Also cross-tab: how many pairs in each (plausibility, novelty) cell?
print("\n2D distribution of Stratum A (plausibility × novelty):")
grid = defaultdict(int)
for vals in by_source.values():
    for p, n in vals:
        grid[(p, n)] += 1
print("   plaus=5:", {n: grid[(5, n)] for n in range(1, 6)})
print("   plaus=4:", {n: grid[(4, n)] for n in range(1, 6)})
print("   plaus=3:", {n: grid[(3, n)] for n in range(1, 6)})
print("   plaus=2:", {n: grid[(2, n)] for n in range(1, 6)})
print("   plaus=1:", {n: grid[(1, n)] for n in range(1, 6)})

# The "sweet spot": high plausibility AND high novelty
sweet = sum(1 for vals in by_source.values() for p, n in vals if p >= 4 and n >= 4)
print(f"\nPairs with plaus>=4 AND novelty>=4 in Stratum A: {sweet} / 200")