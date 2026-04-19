"""
STAGE 4a v3 — shortName-level dedup + global cap
"""

import csv
import os
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np

SAVE_DIR = "./nasa_eo_kg"
OUT_DIR = "./neurips_figs"
FINAL_DIR = f"{OUT_DIR}/final_models"
STAGE4_DIR = f"{OUT_DIR}/stage4"

SEED = 42
TOP_K_PER_MODEL = 100
MAX_PER_SN = 2
CANDIDATE_POOL = 5000
np.random.seed(SEED)

print("═" * 62)
print("STAGE 4a v3 — shortName-level dedup + global cap")
print("═" * 62)

ds_list = [l.strip() for l in open(f"{FINAL_DIR}/dataset_ids.txt")]
ds_idx = {d: i for i, d in enumerate(ds_list)}
N = len(ds_list)

emb_dot = np.load(f"{FINAL_DIR}/final_dot.npy").astype(np.float32)
emb_mlp = np.load(f"{FINAL_DIR}/final_mlp.npy").astype(np.float32)
mlp_w = np.load(f"{FINAL_DIR}/final_mlp_scorer.npz")


def _load_pairs(path):
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            a, b, w = line.rstrip("\n").split("\t")
            out.append((a, b, int(w)))
    return out


def _load_pd(path):
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            out.append((parts[0], parts[1]))
    return out


train_pairs = _load_pairs(f"{OUT_DIR}/cousage_edges_train.tsv")
val_pairs = _load_pairs(f"{OUT_DIR}/cousage_edges_val.tsv")
test_pairs = _load_pairs(f"{OUT_DIR}/cousage_edges_test.tsv")
pd_edges = _load_pd(f"{OUT_DIR}/paper_dataset_edges.tsv")

ever_cooc = set()
for L in (train_pairs, val_pairs, test_pairs):
    for a, b, _ in L:
        if a in ds_idx and b in ds_idx:
            ever_cooc.add(tuple(sorted((a, b))))

paper_to_datasets = defaultdict(set)
for p, d in pd_edges:
    if d in ds_idx:
        paper_to_datasets[p].add(d)
for p, dsets in paper_to_datasets.items():
    if len(dsets) < 2:
        continue
    for di, dj in combinations(sorted(dsets), 2):
        ever_cooc.add(tuple(sorted((di, dj))))

print(f"   datasets:        {N:,}")
print(f"   ever-cooc pairs: {len(ever_cooc):,}")

mask = np.zeros((N, N), dtype=bool)
for a, b in ever_cooc:
    if a in ds_idx and b in ds_idx:
        i, j = ds_idx[a], ds_idx[b]
        mask[i, j] = True
        mask[j, i] = True


print("\n▶ Loading dataset metadata from GraphML …")
import networkx as nx

graphml_path = None
for root, _, files in os.walk(SAVE_DIR):
    for f in files:
        if f.endswith(".graphml"):
            graphml_path = os.path.join(root, f)
            break
    if graphml_path:
        break


def canonicalise_daac(raw):
    s = str(raw).upper()
    if "PODAAC" in s or "PO.DAAC" in s: return "PODAAC"
    if "GES-DISC" in s or "GESDISC" in s: return "GES-DISC"
    if "LP DAAC" in s or "LPDAAC" in s or "LPCLOUD" in s: return "LP DAAC"
    if "NSIDC" in s: return "NSIDC"
    if "OB.DAAC" in s or "OBDAAC" in s: return "OB.DAAC"
    if "GHRC" in s: return "GHRC"
    if "ASDC" in s: return "ASDC"
    return "Other" if raw else "Unknown"


G = nx.read_graphml(graphml_path)
ds_meta = {}
for d in ds_list:
    if d in G:
        data = G.nodes[d]
        ds_meta[d] = {
            "shortName": str(data.get("shortName", "")),
            "longName": str(data.get("longName", "")),
            "abstract": str(data.get("abstract", "")),
            "daac": canonicalise_daac(data.get("daac", "")),
        }
    else:
        ds_meta[d] = {"shortName": d, "longName": "", "abstract": "", "daac": "Unknown"}
del G


def canonical_sn(d):
    sn = ds_meta[d]["shortName"].strip()
    return sn if sn else d


sn_counter = Counter(canonical_sn(d) for d in ds_list)
n_dupes = sum(1 for sn, c in sn_counter.items() if c > 1)
print(f"   unique shortNames among {N} datasets: {len(sn_counter)}")
print(f"   shortNames shared by 2+ dataset IDs: {n_dupes}")


print("\n▶ Scoring all novel pairs …")
S_dot = emb_dot @ emb_dot.T
np.fill_diagonal(S_dot, -np.inf)
S_dot[mask] = -np.inf

W1, b1 = mlp_w["W1"], mlp_w["b1"]
W2, b2 = mlp_w["W2"], mlp_w["b2"]
S_mlp = np.full((N, N), -np.inf, dtype=np.float32)
CHUNK = 50
for i_s in range(0, N, CHUNK):
    i_e = min(i_s + CHUNK, N)
    for j_s in range(0, N, CHUNK):
        j_e = min(j_s + CHUNK, N)
        ei = np.repeat(emb_mlp[i_s:i_e], j_e - j_s, axis=0)
        ej = np.tile(emb_mlp[j_s:j_e], (i_e - i_s, 1))
        x = np.concatenate([ei, ej, ei * ej], axis=-1)
        h = np.maximum(0, x @ W1.T + b1)
        s = (h @ W2.T + b2).squeeze(-1)
        S_mlp[i_s:i_e, j_s:j_e] = s.reshape(i_e - i_s, j_e - j_s)
np.fill_diagonal(S_mlp, -np.inf)
S_mlp[mask] = -np.inf
S_mlp = (S_mlp + S_mlp.T) / 2.0
S_mlp[mask] = -np.inf


def ranked_upper_tri(matrix, pool_size):
    triu = np.triu(np.ones_like(matrix, dtype=bool), k=1)
    flat = matrix.copy()
    flat[~triu] = -np.inf
    flat_1d = flat.ravel()
    k = min(pool_size, int((flat_1d > -np.inf).sum()))
    idx = np.argpartition(flat_1d, -k)[-k:]
    idx_sorted = idx[np.argsort(flat_1d[idx])[::-1]]
    return [(int(v // N), int(v % N), float(flat_1d[v])) for v in idx_sorted]


def greedy_select(ranked, target_k, sn_count, seen_sn_pairs, max_per_sn, source_tag):
    admitted = []
    for i, j, s in ranked:
        sn_i = canonical_sn(ds_list[i])
        sn_j = canonical_sn(ds_list[j])
        if sn_i == sn_j:
            continue
        sn_pair = tuple(sorted((sn_i, sn_j)))
        if sn_pair in seen_sn_pairs:
            continue
        if sn_count[sn_i] >= max_per_sn or sn_count[sn_j] >= max_per_sn:
            continue
        admitted.append({
            "i": i, "j": j,
            "score_dot": float(S_dot[i, j]),
            "score_mlp": float(S_mlp[i, j]),
            "source_model": source_tag,
        })
        sn_count[sn_i] += 1
        sn_count[sn_j] += 1
        seen_sn_pairs.add(sn_pair)
        if len(admitted) >= target_k:
            break
    return admitted


print("\n▶ Selecting diverse picks with global shortName cap …")
pool_dot = ranked_upper_tri(S_dot, CANDIDATE_POOL)
pool_mlp = ranked_upper_tri(S_mlp, CANDIDATE_POOL)
print(f"   candidate pools: dot={len(pool_dot)}  mlp={len(pool_mlp)}")

sn_count = Counter()
seen_sn_pairs = set()

dot_admitted = greedy_select(pool_dot, TOP_K_PER_MODEL,
                             sn_count, seen_sn_pairs, MAX_PER_SN, "dot")
print(f"   dot admitted: {len(dot_admitted)}/{TOP_K_PER_MODEL}")

mlp_admitted = greedy_select(pool_mlp, TOP_K_PER_MODEL,
                             sn_count, seen_sn_pairs, MAX_PER_SN, "mlp")
print(f"   mlp admitted: {len(mlp_admitted)}/{TOP_K_PER_MODEL}")

stratum_A = dot_admitted + mlp_admitted

triu = np.triu(np.ones((N, N), dtype=bool), k=1)
dot_flat = np.sort(S_dot[triu])[::-1]
mlp_flat = np.sort(S_mlp[triu])[::-1]


def quick_rank(sorted_scores, value):
    return int(np.searchsorted(-sorted_scores, -value, side="left")) + 1


for e in stratum_A:
    e["rank_dot"] = quick_rank(dot_flat, e["score_dot"])
    e["rank_mlp"] = quick_rank(mlp_flat, e["score_mlp"])


sn_occ = Counter()
for e in stratum_A:
    sn_occ[canonical_sn(ds_list[e["i"]])] += 1
    sn_occ[canonical_sn(ds_list[e["j"]])] += 1
print(f"\n▶ New Stratum A diversity:")
print(f"   total pairs: {len(stratum_A)}")
print(f"   unique shortNames: {len(sn_occ)}")
print(f"   max occurrences of any shortName: {max(sn_occ.values())}  "
      f"(cap = {MAX_PER_SN})")

sn_pair_cnt = Counter()
for e in stratum_A:
    sn_i = canonical_sn(ds_list[e["i"]])
    sn_j = canonical_sn(ds_list[e["j"]])
    sn_pair_cnt[tuple(sorted((sn_i, sn_j)))] += 1
dupes = [(p, c) for p, c in sn_pair_cnt.items() if c > 1]
print(f"   shortName-pair duplicates: {len(dupes)}  (should be 0)")


print("\n▶ Loading current strata.tsv to preserve B/C/D …")
old_rows = []
with open(f"{STAGE4_DIR}/strata.tsv") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        old_rows.append(r)
preserved = [r for r in old_rows if r["stratum"] != "A"]
print(f"   preserved: {len(preserved)} rows (strata B/C/D)")


def safe_tab(x):
    if x is None: return ""
    return str(x).replace("\t", " ").replace("\n", " ").replace("\r", "")


new_A_rows = []
for k, e in enumerate(stratum_A):
    i, j = e["i"], e["j"]
    a, b = ds_list[i], ds_list[j]
    new_A_rows.append({
        "stratum": "A",
        "pair_idx": k,
        "dataset_i": a,
        "dataset_j": b,
        "shortName_i": ds_meta[a]["shortName"],
        "shortName_j": ds_meta[b]["shortName"],
        "longName_i": ds_meta[a]["longName"],
        "longName_j": ds_meta[b]["longName"],
        "daac_i": ds_meta[a]["daac"],
        "daac_j": ds_meta[b]["daac"],
        "abstract_i": ds_meta[a]["abstract"],
        "abstract_j": ds_meta[b]["abstract"],
        "score_dot": e["score_dot"],
        "score_mlp": e["score_mlp"],
        "rank_dot": e["rank_dot"],
        "rank_mlp": e["rank_mlp"],
        "source_model": e["source_model"],
    })

backup = f"{STAGE4_DIR}/strata_v2.tsv"
if os.path.exists(backup):
    os.remove(backup)
os.rename(f"{STAGE4_DIR}/strata.tsv", backup)

cols = ["stratum", "pair_idx", "dataset_i", "dataset_j",
        "shortName_i", "shortName_j", "longName_i", "longName_j",
        "daac_i", "daac_j", "abstract_i", "abstract_j",
        "score_dot", "score_mlp", "rank_dot", "rank_mlp",
        "source_model"]

with open(f"{STAGE4_DIR}/strata.tsv", "w") as f:
    f.write("\t".join(cols) + "\n")
    for r in new_A_rows + preserved:
        f.write("\t".join(safe_tab(r.get(c)) for c in cols) + "\n")

print(f"\n✅ Wrote new {STAGE4_DIR}/strata.tsv")
print(f"   previous (v2) backed up to {backup}")

print("\n" + "═" * 62)
print("NEW Stratum A — first 5 dot picks, first 5 mlp picks")
print("═" * 62)
for r in [x for x in new_A_rows if x["source_model"] == "dot"][:5]:
    sa, sb = r["shortName_i"][:30], r["shortName_j"][:30]
    print(f"   {sa:<32} ({r['daac_i']:<10}) + {sb:<32} ({r['daac_j']:<10})  "
          f"dot={float(r['score_dot']):+.3f}  mlp={float(r['score_mlp']):+.3f}")
for r in [x for x in new_A_rows if x["source_model"] == "mlp"][:5]:
    sa, sb = r["shortName_i"][:30], r["shortName_j"][:30]
    print(f"   {sa:<32} ({r['daac_i']:<10}) + {sb:<32} ({r['daac_j']:<10})  "
          f"dot={float(r['score_dot']):+.3f}  mlp={float(r['score_mlp']):+.3f}")

print("\nDAAC combinations (top 8):")
dc = Counter(tuple(sorted([r["daac_i"], r["daac_j"]])) for r in new_A_rows)
for (a, b), c in dc.most_common(8):
    print(f"   {a:<12} + {b:<12}: {c}")

print(f"\nFinal Stratum A: {len(new_A_rows)} pairs, {len(sn_occ)} unique products")