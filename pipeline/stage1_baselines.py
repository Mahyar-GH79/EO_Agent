"""
STAGE 1 — Baselines
===================
Runs five link-prediction baselines for the dataset co-usage task:
    1. Popularity              (score = log(deg_i) * log(deg_j))
    2. Common Neighbors (CN)   + Adamic-Adar (AA) on train co-usage graph
    3. Matrix Factorization    (truncated SVD of bipartite paper×dataset)
    4. SPECTER2                (cosine sim of dataset abstract embeddings)
    5. BGE-base                (cosine sim of dataset abstract embeddings)

Each baseline scores candidate dataset pairs; we evaluate using a
1:100 negative-sampling ranking protocol on three test pools:
    - all           : all test-unseen pairs
    - cold_start    : test pairs where ≥1 dataset has <5 train papers
    - cross_daac    : test pairs spanning different DAACs

Reports: Hits@10, Hits@50, MRR, AP, AUC per (baseline, pool).

Outputs:
    ./neurips_figs/stage1_results.tsv          tidy long-form results
    ./neurips_figs/tab_stage1_baselines.tex    paper-ready LaTeX table
    ./neurips_figs/embeddings/specter2.npy     cached node features
    ./neurips_figs/embeddings/bge.npy          cached node features
    ./neurips_figs/embeddings/dataset_ids.txt  row-order for the .npy files

Run:
    python3 stage1_baselines.py

Requirements:
    pip install networkx numpy scipy scikit-learn torch transformers \
                sentence-transformers adapters tqdm
"""

from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict

import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import roc_auc_score, average_precision_score


# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
SAVE_DIR = "./nasa_eo_kg"
OUT_DIR = "./neurips_figs"
EMB_DIR = f"{OUT_DIR}/embeddings"
os.makedirs(EMB_DIR, exist_ok=True)

N_NEGATIVES = 100         # per positive, for ranking eval
NEG_ALPHA = 0.75          # degree-biased negative sampling exponent
MIN_PAPERS_PER_DATASET = 5
SEED = 42
MF_RANK = 64              # truncated SVD rank
BATCH_EMB = 32            # embedding batch size
CUDA = None               # None = auto-detect

random.seed(SEED)
np.random.seed(SEED)


# ══════════════════════════════════════════════════════════════
# Load Stage 0 artifacts
# ══════════════════════════════════════════════════════════════
def load_pairs(path):
    """Load TSV with columns dataset_i, dataset_j, weight."""
    out = []
    with open(path) as f:
        next(f)  # header
        for line in f:
            a, b, w = line.rstrip("\n").split("\t")
            out.append((a, b, int(w)))
    return out


def load_paper_dataset_edges(path):
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            p, d = parts[0], parts[1]
            y = parts[2] if len(parts) > 2 else ""
            out.append((p, d, int(y) if y else None))
    return out


def load_dataset_features(path):
    """Returns dict: dataset_id -> {shortName, daac, num_papers_train, ...}."""
    out = {}
    with open(path) as f:
        header = next(f).rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            row = dict(zip(header, parts))
            out[row["dataset"]] = row
    return out


print("═" * 62)
print("STAGE 1 — Loading Stage 0 artifacts")
print("═" * 62)

train_pairs = load_pairs(f"{OUT_DIR}/cousage_edges_train.tsv")
val_pairs   = load_pairs(f"{OUT_DIR}/cousage_edges_val.tsv")
test_pairs  = load_pairs(f"{OUT_DIR}/cousage_edges_test.tsv")
pd_edges    = load_paper_dataset_edges(f"{OUT_DIR}/paper_dataset_edges.tsv")
ds_features = load_dataset_features(f"{OUT_DIR}/dataset_features.tsv")

print(f"   train pairs:      {len(train_pairs):>8,}")
print(f"   val pairs:        {len(val_pairs):>8,}")
print(f"   test pairs:       {len(test_pairs):>8,}")
print(f"   paper→dataset:    {len(pd_edges):>8,}")
print(f"   dataset features: {len(ds_features):>8,}")


# ══════════════════════════════════════════════════════════════
# Evaluation universe: datasets that appear in train pairs
# ══════════════════════════════════════════════════════════════
train_datasets = set()
for a, b, _ in train_pairs:
    train_datasets.add(a); train_datasets.add(b)

print(f"\n   datasets appearing in train pairs: {len(train_datasets):,}")
print(f"   evaluation universe (= candidates for negatives)")


# ══════════════════════════════════════════════════════════════
# Build indices & graph objects
# ══════════════════════════════════════════════════════════════
ds_list = sorted(train_datasets)
ds_idx = {d: i for i, d in enumerate(ds_list)}
N = len(ds_list)

train_paper_set = set()
# A paper is "train" if its year ≤ 2022 (set by Stage 0 config)
TRAIN_MAX_YEAR = 2022
for p, d, y in pd_edges:
    if y is not None and y <= TRAIN_MAX_YEAR:
        train_paper_set.add(p)

deg_train = Counter()
for p, d, y in pd_edges:
    if p in train_paper_set and d in ds_idx:
        deg_train[d] += 1

# Build train co-usage graph as networkx (for CN / AA)
print("\n▶ Building train co-usage graph …")
G_train = nx.Graph()
G_train.add_nodes_from(ds_list)
for a, b, w in train_pairs:
    if a in ds_idx and b in ds_idx:
        G_train.add_edge(a, b, weight=w)
print(f"   {G_train.number_of_nodes():,} nodes, "
      f"{G_train.number_of_edges():,} edges")

train_pair_set = {tuple(sorted((a, b))) for a, b, _ in train_pairs
                  if a in ds_idx and b in ds_idx}


# ══════════════════════════════════════════════════════════════
# Build test pools (all / cold_start / cross_daac)
# ══════════════════════════════════════════════════════════════
def daac_of(d):
    return ds_features.get(d, {}).get("daac", "")

def train_paper_count(d):
    try:
        return int(ds_features.get(d, {}).get("num_papers_train", 0))
    except (ValueError, TypeError):
        return 0


# keep only test pairs where both datasets are in the training universe
# (otherwise there's nothing to compare them against for structure-based methods)
eligible_test = [
    (a, b, w) for a, b, w in test_pairs
    if a in ds_idx and b in ds_idx and
    tuple(sorted((a, b))) not in train_pair_set
]
print(f"\n▶ Test pool construction")
print(f"   raw test pairs:                    {len(test_pairs):>8,}")
print(f"   eligible (both in train universe): {len(eligible_test):>8,}")

pool_all = eligible_test
pool_cold = [
    (a, b, w) for a, b, w in eligible_test
    if train_paper_count(a) < MIN_PAPERS_PER_DATASET
    or train_paper_count(b) < MIN_PAPERS_PER_DATASET
]
pool_cross_daac = [
    (a, b, w) for a, b, w in eligible_test
    if daac_of(a) and daac_of(b) and daac_of(a) != daac_of(b)
]

print(f"   pool 'all':          {len(pool_all):>6,}")
print(f"   pool 'cold_start':   {len(pool_cold):>6,}")
print(f"   pool 'cross_daac':   {len(pool_cross_daac):>6,}")


# ══════════════════════════════════════════════════════════════
# Negative sampling (degree-biased, excludes train + positives)
# ══════════════════════════════════════════════════════════════
print("\n▶ Building negative-sample pool …")

# Degree-biased sampling distribution
deg_arr = np.array([deg_train[d] + 1 for d in ds_list], dtype=np.float64)
prob = deg_arr ** NEG_ALPHA
prob /= prob.sum()

rng = np.random.default_rng(SEED)


def sample_negatives_for_pool(pool):
    """For each positive (a, b), return 100 negative (a, b') pairs."""
    pos_pair_set = {tuple(sorted((a, b))) for a, b, _ in pool}
    all_pos_set = pos_pair_set | train_pair_set

    result = []
    # Pre-sample a big chunk of random j's for efficiency
    total_needed = len(pool) * N_NEGATIVES * 2
    all_js = rng.choice(N, size=total_needed, p=prob, replace=True)
    cursor = 0

    for (a, b, _) in pool:
        a_idx = ds_idx[a]
        negs = []
        attempts = 0
        while len(negs) < N_NEGATIVES and attempts < N_NEGATIVES * 50:
            if cursor >= len(all_js):
                all_js = rng.choice(N, size=total_needed, p=prob, replace=True)
                cursor = 0
            j_idx = all_js[cursor]; cursor += 1; attempts += 1
            if j_idx == a_idx:
                continue
            cand_b = ds_list[j_idx]
            key = tuple(sorted((a, cand_b)))
            if key in all_pos_set:
                continue
            negs.append(cand_b)
        result.append((a, b, negs))
    return result


pool_all_neg   = sample_negatives_for_pool(pool_all)
pool_cold_neg  = sample_negatives_for_pool(pool_cold)
pool_cross_neg = sample_negatives_for_pool(pool_cross_daac)
print(f"   negatives sampled for all three pools ({N_NEGATIVES} per positive)")

# Persist the exact evaluation pools so Stage 2 (GNN) uses identical negatives
# for an apples-to-apples comparison with Stage 1 baselines.
def save_eval_pool(path, pool_with_negs):
    with open(path, "w") as f:
        f.write("a\tb_pos\tb_negs\n")
        for a, b_pos, b_negs in pool_with_negs:
            f.write(f"{a}\t{b_pos}\t{','.join(b_negs)}\n")


save_eval_pool(f"{OUT_DIR}/eval_pool_all.tsv",        pool_all_neg)
save_eval_pool(f"{OUT_DIR}/eval_pool_cold_start.tsv", pool_cold_neg)
save_eval_pool(f"{OUT_DIR}/eval_pool_cross_daac.tsv", pool_cross_neg)
print(f"   saved evaluation pools to {OUT_DIR}/eval_pool_*.tsv")


# ══════════════════════════════════════════════════════════════
# Ranking-based evaluation
# ══════════════════════════════════════════════════════════════
def evaluate(score_fn, pool_with_negs, name=""):
    """score_fn(a, b) -> float. pool: list of (a, b_pos, [b_negs])."""
    ranks = []
    all_scores, all_labels = [], []
    for a, b_pos, b_negs in pool_with_negs:
        candidates = [b_pos] + b_negs
        scores = np.array([score_fn(a, c) for c in candidates])
        # rank of the positive (1-indexed), ties broken randomly
        order = np.argsort(-scores + rng.uniform(-1e-12, 1e-12, size=scores.shape))
        rank = int(np.where(order == 0)[0][0]) + 1
        ranks.append(rank)
        all_scores.extend(scores.tolist())
        all_labels.extend([1] + [0] * len(b_negs))

    ranks = np.array(ranks)
    hits10 = float((ranks <= 10).mean())
    hits50 = float((ranks <= 50).mean())
    mrr    = float((1.0 / ranks).mean())
    auc    = float(roc_auc_score(all_labels, all_scores))
    ap     = float(average_precision_score(all_labels, all_scores))
    return {"Hits@10": hits10, "Hits@50": hits50, "MRR": mrr,
            "AUC": auc, "AP": ap, "N": len(ranks)}


# ══════════════════════════════════════════════════════════════
# Baseline 1 — Popularity
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("BASELINE 1 — Popularity")
print("═" * 62)


def pop_score(a, b):
    da = deg_train.get(a, 0) + 1
    db = deg_train.get(b, 0) + 1
    return np.log(da) * np.log(db)


# ══════════════════════════════════════════════════════════════
# Baseline 2 — Common Neighbors + Adamic-Adar
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("BASELINE 2 — Common Neighbors + Adamic-Adar")
print("═" * 62)


def cn_score(a, b):
    if a not in G_train or b not in G_train:
        return 0.0
    neigh_a = set(G_train.neighbors(a))
    neigh_b = set(G_train.neighbors(b))
    return float(len(neigh_a & neigh_b))


def aa_score(a, b):
    if a not in G_train or b not in G_train:
        return 0.0
    na = set(G_train.neighbors(a))
    nb = set(G_train.neighbors(b))
    common = na & nb
    s = 0.0
    for c in common:
        deg_c = G_train.degree(c)
        if deg_c > 1:
            s += 1.0 / np.log(deg_c)
    return s


# ══════════════════════════════════════════════════════════════
# Baseline 3 — Matrix Factorization (truncated SVD on paper×dataset)
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("BASELINE 3 — Matrix Factorization (SVD, rank 64)")
print("═" * 62)

train_paper_list = sorted(train_paper_set)
paper_idx = {p: i for i, p in enumerate(train_paper_list)}
print(f"   papers: {len(train_paper_list):,}, datasets: {N:,}")

rows, cols, vals = [], [], []
for p, d, y in pd_edges:
    if p in paper_idx and d in ds_idx:
        rows.append(paper_idx[p])
        cols.append(ds_idx[d])
        vals.append(1.0)
M = csr_matrix((vals, (rows, cols)),
               shape=(len(train_paper_list), N), dtype=np.float32)
print(f"   paper×dataset matrix: {M.shape}, nnz={M.nnz:,}")

print(f"   running truncated SVD (rank={MF_RANK}) …")
U, s, Vt = svds(M, k=MF_RANK)
# s is ascending, reverse to descending
s = s[::-1]; U = U[:, ::-1]; Vt = Vt[::-1, :]
ds_emb_mf = (Vt.T * np.sqrt(s))    # (N, MF_RANK)
# normalise for cosine
ds_emb_mf_n = ds_emb_mf / (np.linalg.norm(ds_emb_mf, axis=1, keepdims=True) + 1e-9)


def mf_score(a, b):
    if a not in ds_idx or b not in ds_idx:
        return 0.0
    return float(ds_emb_mf_n[ds_idx[a]] @ ds_emb_mf_n[ds_idx[b]])


# ══════════════════════════════════════════════════════════════
# Baseline 4/5 — SPECTER2 and BGE content embeddings
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("BASELINE 4/5 — Content embeddings (SPECTER2, BGE)")
print("═" * 62)

# Need dataset titles + abstracts → load from the GraphML
print("   loading GraphML to extract dataset titles and abstracts …")
graphml_path = None
for root, _, files in os.walk(SAVE_DIR):
    for f in files:
        if f.endswith(".graphml"):
            graphml_path = os.path.join(root, f); break
    if graphml_path:
        break
if graphml_path is None:
    raise FileNotFoundError(f"No .graphml under {SAVE_DIR}/")

G_full = nx.read_graphml(graphml_path)
ds_titles = []
ds_abstracts = []
for d in ds_list:
    if d in G_full:
        data = G_full.nodes[d]
        # Title = longName (fall back to shortName); abstract as-is
        title = str(data.get("longName", "") or data.get("shortName", ""))
        abstract = str(data.get("abstract", ""))
    else:
        title, abstract = d, ""
    ds_titles.append(title)
    ds_abstracts.append(abstract[:3000])  # truncate to keep under 512 subwords
del G_full

avg_title = np.mean([len(t) for t in ds_titles])
avg_abs = np.mean([len(a) for a in ds_abstracts])
print(f"   prepared {len(ds_list):,} datasets "
      f"(avg title {avg_title:.0f} chars, avg abstract {avg_abs:.0f} chars)")


def _device():
    import torch
    if CUDA is False:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def embed_specter2(cache_path):
    """SPECTER2 requires the adapters library to load the proximity adapter
    on top of allenai/specter2_base. Plain SentenceTransformer loading
    falls back to the base BERT and gives poor embeddings — don't do that."""
    if os.path.exists(cache_path):
        print(f"   ↻ cache hit: {cache_path}")
        return np.load(cache_path)

    import torch
    from transformers import AutoTokenizer
    try:
        from adapters import AutoAdapterModel
    except ImportError:
        raise ImportError(
            "SPECTER2 requires the `adapters` library. Install with:\n"
            "    pip install adapters\n"
            "(This is a separate package, not part of transformers.)")
    from tqdm import tqdm

    device = _device()
    print(f"   loading SPECTER2 (allenai/specter2_base + proximity adapter) on {device} …")
    tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
    model = AutoAdapterModel.from_pretrained("allenai/specter2_base")
    # Proximity adapter — the standard SPECTER2 embedding task.
    model.load_adapter("allenai/specter2", source="hf",
                       load_as="proximity", set_active=True)
    model.to(device); model.eval()
    # Belt-and-braces: explicitly activate after .to(device). Without this
    # step, the model may silently forward through the base BERT only
    # (warning: "There are adapters available but none are activated").
    model.set_active_adapters("proximity")

    # Verify the adapter is actually active. If this list is empty we
    # would be computing meaningless base-BERT embeddings.
    active = model.active_adapters
    print(f"   active adapters: {active}")
    if not active:
        raise RuntimeError(
            "SPECTER2 proximity adapter failed to activate. "
            "Embeddings would be meaningless — aborting.")

    # SPECTER2 expects title + [SEP] + abstract
    texts = [t + tok.sep_token + a for t, a in zip(ds_titles, ds_abstracts)]
    out = np.zeros((len(texts), 768), dtype=np.float32)
    bs = BATCH_EMB
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), bs), desc="SPECTER2"):
            batch = texts[i:i + bs]
            enc = tok(batch, padding=True, truncation=True,
                      max_length=512, return_tensors="pt",
                      return_token_type_ids=False)
            enc = {k: v.to(device) for k, v in enc.items()}
            o = model(**enc)
            # CLS pooling (SPECTER/SPECTER2 convention)
            emb = o.last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, dim=-1)
            out[i:i + bs] = emb.cpu().numpy()

    np.save(cache_path, out)
    print(f"   saved: {cache_path} ({out.shape})")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def embed_bge(cache_path):
    """BGE-base via sentence-transformers. For symmetric cosine similarity
    (as in our pair scoring) no instruction prefix is needed."""
    if os.path.exists(cache_path):
        print(f"   ↻ cache hit: {cache_path}")
        return np.load(cache_path)

    import torch
    from sentence_transformers import SentenceTransformer

    device = _device()
    print(f"   loading BGE-base (BAAI/bge-base-en-v1.5) on {device} …")
    model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    # Concatenate title and abstract, separated by a plain period
    texts = [f"{t}. {a}".strip() for t, a in zip(ds_titles, ds_abstracts)]
    emb = model.encode(
        texts, batch_size=BATCH_EMB, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    )
    np.save(cache_path, emb)
    print(f"   saved: {cache_path} ({emb.shape})")
    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return emb


ds_emb_specter = embed_specter2(f"{EMB_DIR}/specter2.npy")
ds_emb_bge = embed_bge(f"{EMB_DIR}/bge.npy")

# Save dataset ID order (used by Stage 2 as node feature row mapping)
with open(f"{EMB_DIR}/dataset_ids.txt", "w") as f:
    for d in ds_list:
        f.write(d + "\n")


def specter_score(a, b):
    if a not in ds_idx or b not in ds_idx:
        return 0.0
    return float(ds_emb_specter[ds_idx[a]] @ ds_emb_specter[ds_idx[b]])


def bge_score(a, b):
    if a not in ds_idx or b not in ds_idx:
        return 0.0
    return float(ds_emb_bge[ds_idx[a]] @ ds_emb_bge[ds_idx[b]])


# ══════════════════════════════════════════════════════════════
# Run evaluation
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("EVALUATION")
print("═" * 62)

baselines = [
    ("Popularity",        pop_score),
    ("CommonNeighbors",   cn_score),
    ("AdamicAdar",        aa_score),
    ("MF-SVD",            mf_score),
    ("SPECTER2",          specter_score),
    ("BGE-base",          bge_score),
]

pools = [
    ("all",        pool_all_neg),
    ("cold_start", pool_cold_neg),
    ("cross_daac", pool_cross_neg),
]

results = []
for pool_name, pool in pools:
    if not pool:
        continue
    print(f"\n▶ Pool: {pool_name}  (N = {len(pool):,})")
    print(f"   {'Baseline':<18} {'Hits@10':>8} {'Hits@50':>8} "
          f"{'MRR':>8} {'AUC':>8} {'AP':>8}")
    for bname, fn in baselines:
        m = evaluate(fn, pool, name=bname)
        row = {"pool": pool_name, "baseline": bname, **m}
        results.append(row)
        print(f"   {bname:<18} "
              f"{m['Hits@10']:>8.3f} {m['Hits@50']:>8.3f} "
              f"{m['MRR']:>8.3f} {m['AUC']:>8.3f} {m['AP']:>8.3f}")


# ══════════════════════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════════════════════
out_tsv = f"{OUT_DIR}/stage1_results.tsv"
with open(out_tsv, "w") as f:
    cols = ["pool", "baseline", "N", "Hits@10", "Hits@50", "MRR", "AUC", "AP"]
    f.write("\t".join(cols) + "\n")
    for r in results:
        f.write("\t".join(str(r[c]) for c in cols) + "\n")
print(f"\n✅ saved: {out_tsv}")


# LaTeX table, one row per (pool × baseline)
print("\n▶ Writing LaTeX results table …")
tex = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Stage~1 baselines on the co-usage link-prediction task "
    r"(1:100 negative sampling). Higher is better for all metrics.}",
    r"\label{tab:stage1_baselines}",
    r"\small",
    r"\begin{tabular}{llrrrrr}",
    r"\toprule",
    r"\textbf{Pool} & \textbf{Baseline} & "
    r"\textbf{Hits@10} & \textbf{Hits@50} & "
    r"\textbf{MRR} & \textbf{AUC} & \textbf{AP} \\",
    r"\midrule",
]
for pool_name, _ in pools:
    first = True
    for r in results:
        if r["pool"] != pool_name:
            continue
        pool_cell = (r"\multirow{6}{*}{\texttt{" + pool_name + r"}}"
                     if first else "")
        first = False
        tex.append(
            f"{pool_cell} & \\texttt{{{r['baseline']}}} & "
            f"{r['Hits@10']:.3f} & {r['Hits@50']:.3f} & "
            f"{r['MRR']:.3f} & {r['AUC']:.3f} & {r['AP']:.3f} \\\\"
        )
    tex.append(r"\midrule")
tex = tex[:-1]  # drop trailing midrule
tex += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
    "",
]
out_tex = f"{OUT_DIR}/tab_stage1_baselines.tex"
with open(out_tex, "w") as f:
    f.write("\n".join(tex))
print(f"✅ saved: {out_tex}")


print("\n" + "═" * 62)
print("STAGE 1 COMPLETE")
print("═" * 62)
print(f"Results:    {out_tsv}")
print(f"LaTeX:      {out_tex}")
print(f"Embeddings: {EMB_DIR}/specter2.npy, {EMB_DIR}/bge.npy")
print(f"Order:      {EMB_DIR}/dataset_ids.txt")
print("\nNext: Stage 2 — PyG heterogeneous GNN.")