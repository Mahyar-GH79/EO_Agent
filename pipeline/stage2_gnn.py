"""
STAGE 2 — GNN (Homogeneous + Heterogeneous), PyTorch Geometric
==============================================================
Trains two GNN variants for dataset co-usage link prediction:

    1. GNN-Homo:   2-layer GraphSAGE over the co-usage graph only.
                   Node features = SPECTER2 embeddings.
                   Tests: does learned structural smoothing beat fixed
                   SPECTER2 cosine?

    2. GNN-Hetero: 2-layer HeteroConv (SAGE per relation) over the full
                   KG (Dataset, ScienceKeyword, Instrument, Platform, Project).
                   Dataset nodes use SPECTER2 features; other node types
                   use learnable embeddings.
                   Tests: does the surrounding KG metadata add signal on
                   top of co-usage + content?

Both variants are evaluated on the SAME test pools + negatives that
Stage 1 saved to eval_pool_{all,cold_start,cross_daac}.tsv, so the
numbers are directly comparable with the baseline table.

Runs 3 seeds per variant, reports mean ± std.

Outputs:
    ./neurips_figs/stage2_results.tsv
    ./neurips_figs/tab_stage2_gnn.tex
    ./neurips_figs/gnn_{homo,hetero}_seed{0,1,2}.pt  (checkpoints)

Run:
    python3 stage2_gnn.py

Requirements:
    pip install torch torch-geometric numpy networkx scikit-learn
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
SAVE_DIR = "./nasa_eo_kg"
OUT_DIR = "./neurips_figs"
EMB_DIR = f"{OUT_DIR}/embeddings"
CKPT_DIR = f"{OUT_DIR}/checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

HIDDEN = 128
N_EPOCHS = 300
LR = 1e-3
WD = 1e-5
PATIENCE = 30          # early stop on val MRR
NEG_PER_POS_TRAIN = 5  # training-time negatives
NEG_ALPHA = 0.75       # match Stage 1
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# Load Stage 0/1 artifacts
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("STAGE 2 — Loading artifacts")
print("═" * 62)


def load_pairs(path):
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            a, b, w = line.rstrip("\n").split("\t")
            out.append((a, b, int(w)))
    return out


def load_eval_pool(path):
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            a, b_pos = parts[0], parts[1]
            b_negs = parts[2].split(",") if parts[2] else []
            out.append((a, b_pos, b_negs))
    return out


train_pairs = load_pairs(f"{OUT_DIR}/cousage_edges_train.tsv")
val_pairs   = load_pairs(f"{OUT_DIR}/cousage_edges_val.tsv")
eval_all    = load_eval_pool(f"{OUT_DIR}/eval_pool_all.tsv")
eval_cold   = load_eval_pool(f"{OUT_DIR}/eval_pool_cold_start.tsv")
eval_cross  = load_eval_pool(f"{OUT_DIR}/eval_pool_cross_daac.tsv")

print(f"   train co-usage pairs: {len(train_pairs):,}")
print(f"   val co-usage pairs:   {len(val_pairs):,}")
print(f"   eval pools:           all={len(eval_all):,}  "
      f"cold={len(eval_cold):,}  cross={len(eval_cross):,}")

# Load SPECTER2 embeddings (1,475 datasets, in dataset_ids.txt order)
ds_list = [l.strip() for l in open(f"{EMB_DIR}/dataset_ids.txt")]
ds_idx = {d: i for i, d in enumerate(ds_list)}
N_DS = len(ds_list)
specter = np.load(f"{EMB_DIR}/specter2.npy").astype(np.float32)
assert specter.shape[0] == N_DS, \
    f"SPECTER2 rows ({specter.shape[0]}) != dataset count ({N_DS})"
print(f"   SPECTER2 features: {specter.shape}")


# ══════════════════════════════════════════════════════════════
# Build homogeneous co-usage graph (PyG edge_index)
# ══════════════════════════════════════════════════════════════
def build_homo_edge_index():
    rows, cols = [], []
    for a, b, _ in train_pairs:
        if a in ds_idx and b in ds_idx:
            i, j = ds_idx[a], ds_idx[b]
            rows += [i, j]   # undirected
            cols += [j, i]
    return torch.tensor([rows, cols], dtype=torch.long)


homo_edge_index = build_homo_edge_index()
print(f"   homo edge_index: {homo_edge_index.shape} "
      f"({homo_edge_index.shape[1]//2} undirected edges)")


# ══════════════════════════════════════════════════════════════
# Build heterogeneous graph from the GraphML
# ══════════════════════════════════════════════════════════════
def build_hetero_data():
    """Build a PyG HeteroData with:
        Node types: Dataset, ScienceKeyword, Instrument, Platform, Project
        Edge types used:
            (Dataset, co_usage, Dataset)            -- from train_pairs
            (Dataset, has_keyword, ScienceKeyword)
            (Dataset, has_platform, Platform)
            (Platform, has_instrument, Instrument)
            (Dataset, of_project, Project)
            (ScienceKeyword, has_subcategory, ScienceKeyword)
       Plus reverse edges for message passing.
    """
    import networkx as nx
    from torch_geometric.data import HeteroData

    graphml_path = None
    for root, _, files in os.walk(SAVE_DIR):
        for f in files:
            if f.endswith(".graphml"):
                graphml_path = os.path.join(root, f); break
        if graphml_path:
            break
    print(f"   loading GraphML from {graphml_path} …")
    G = nx.read_graphml(graphml_path)

    def clean_label(raw):
        return str(raw).strip().strip("[]:'\" ").strip()

    def edge_type_of(data):
        return (data.get("label") or data.get("type")
                or data.get("relationship") or "")

    # Group nodes by type. We only keep datasets that are in ds_list;
    # for the other node types, we keep all nodes connected to those datasets.
    ds_set = set(ds_list)
    kept_datasets = set(ds_list)

    keep_by_type = {
        "ScienceKeyword": set(), "Instrument": set(),
        "Platform": set(), "Project": set(),
    }
    # First pass: find which ScienceKeyword/Instrument/Platform/Project
    # nodes connect (directly or one hop away through Platform) to a kept
    # dataset. Start by walking every edge and collecting touched nodes.
    node_label = {n: clean_label(d.get("labels", "UNKNOWN"))
                  for n, d in G.nodes(data=True)}

    # Direct: Dataset->{SK, Platform, Project}, and Platform->Instrument
    ds_platform = defaultdict(set)
    ds_keyword = defaultdict(set)
    ds_project = defaultdict(set)
    platform_instr = defaultdict(set)
    keyword_subcat = defaultdict(set)

    for u, v, data in G.edges(data=True):
        et = edge_type_of(data)
        lu, lv = node_label.get(u), node_label.get(v)
        # Normalise direction
        if et == "HAS_PLATFORM":
            if lu == "Dataset" and lv == "Platform":
                ds_platform[u].add(v)
            elif lv == "Dataset" and lu == "Platform":
                ds_platform[v].add(u)
        elif et == "HAS_SCIENCEKEYWORD":
            if lu == "Dataset" and lv == "ScienceKeyword":
                ds_keyword[u].add(v)
            elif lv == "Dataset" and lu == "ScienceKeyword":
                ds_keyword[v].add(u)
        elif et == "OF_PROJECT":
            if lu == "Dataset" and lv == "Project":
                ds_project[u].add(v)
            elif lv == "Dataset" and lu == "Project":
                ds_project[v].add(u)
        elif et == "HAS_INSTRUMENT":
            if lu == "Platform" and lv == "Instrument":
                platform_instr[u].add(v)
            elif lv == "Platform" and lu == "Instrument":
                platform_instr[v].add(u)
        elif et == "HAS_SUBCATEGORY":
            if lu == "ScienceKeyword" and lv == "ScienceKeyword":
                keyword_subcat[u].add(v)

    # Keep only things connected to a dataset in our universe
    kept_platforms = set()
    kept_keywords = set()
    kept_projects = set()
    for d in kept_datasets:
        kept_platforms |= ds_platform.get(d, set())
        kept_keywords |= ds_keyword.get(d, set())
        kept_projects |= ds_project.get(d, set())
    kept_instruments = set()
    for p in kept_platforms:
        kept_instruments |= platform_instr.get(p, set())

    # Stable ordering / index maps per node type
    sk_list = sorted(kept_keywords)
    inst_list = sorted(kept_instruments)
    plat_list = sorted(kept_platforms)
    proj_list = sorted(kept_projects)
    sk_idx = {n: i for i, n in enumerate(sk_list)}
    inst_idx = {n: i for i, n in enumerate(inst_list)}
    plat_idx = {n: i for i, n in enumerate(plat_list)}
    proj_idx = {n: i for i, n in enumerate(proj_list)}

    print(f"   nodes per type: Dataset={N_DS}, ScienceKeyword={len(sk_list)}, "
          f"Instrument={len(inst_list)}, Platform={len(plat_list)}, "
          f"Project={len(proj_list)}")

    def edge_tensor(pairs):
        if not pairs:
            return torch.empty((2, 0), dtype=torch.long)
        a = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        b = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        return torch.stack([a, b], dim=0)

    # Build edge tensors
    ds_plat_edges = [(ds_idx[d], plat_idx[p])
                     for d in kept_datasets for p in ds_platform.get(d, [])
                     if p in plat_idx]
    ds_kw_edges = [(ds_idx[d], sk_idx[k])
                   for d in kept_datasets for k in ds_keyword.get(d, [])
                   if k in sk_idx]
    ds_proj_edges = [(ds_idx[d], proj_idx[p])
                     for d in kept_datasets for p in ds_project.get(d, [])
                     if p in proj_idx]
    plat_inst_edges = [(plat_idx[p], inst_idx[i])
                       for p in kept_platforms for i in platform_instr.get(p, [])
                       if i in inst_idx]
    kw_sub_edges = [(sk_idx[u], sk_idx[v])
                    for u in kept_keywords for v in keyword_subcat.get(u, [])
                    if v in sk_idx]

    print(f"   edges: Dataset-Platform={len(ds_plat_edges)}, "
          f"Dataset-Keyword={len(ds_kw_edges)}, "
          f"Dataset-Project={len(ds_proj_edges)}, "
          f"Platform-Instrument={len(plat_inst_edges)}, "
          f"Keyword-Subcat={len(kw_sub_edges)}")

    data = HeteroData()
    data["Dataset"].x = torch.from_numpy(specter)
    data["ScienceKeyword"].num_nodes = len(sk_list)
    data["Instrument"].num_nodes = len(inst_list)
    data["Platform"].num_nodes = len(plat_list)
    data["Project"].num_nodes = len(proj_list)

    # Co-usage edges (both directions for message passing)
    cu_src, cu_dst = [], []
    for a, b, _ in train_pairs:
        if a in ds_idx and b in ds_idx:
            i, j = ds_idx[a], ds_idx[b]
            cu_src += [i, j]; cu_dst += [j, i]
    data["Dataset", "co_usage", "Dataset"].edge_index = \
        torch.tensor([cu_src, cu_dst], dtype=torch.long)

    # Typed edges + their reverses
    data["Dataset", "has_platform", "Platform"].edge_index = edge_tensor(ds_plat_edges)
    data["Platform", "rev_has_platform", "Dataset"].edge_index = \
        data["Dataset", "has_platform", "Platform"].edge_index.flip(0)

    data["Dataset", "has_keyword", "ScienceKeyword"].edge_index = edge_tensor(ds_kw_edges)
    data["ScienceKeyword", "rev_has_keyword", "Dataset"].edge_index = \
        data["Dataset", "has_keyword", "ScienceKeyword"].edge_index.flip(0)

    data["Dataset", "of_project", "Project"].edge_index = edge_tensor(ds_proj_edges)
    data["Project", "rev_of_project", "Dataset"].edge_index = \
        data["Dataset", "of_project", "Project"].edge_index.flip(0)

    data["Platform", "has_instrument", "Instrument"].edge_index = edge_tensor(plat_inst_edges)
    data["Instrument", "rev_has_instrument", "Platform"].edge_index = \
        data["Platform", "has_instrument", "Instrument"].edge_index.flip(0)

    data["ScienceKeyword", "has_subcategory", "ScienceKeyword"].edge_index = \
        edge_tensor(kw_sub_edges)
    data["ScienceKeyword", "rev_has_subcategory", "ScienceKeyword"].edge_index = \
        data["ScienceKeyword", "has_subcategory", "ScienceKeyword"].edge_index.flip(0)

    return data


# ══════════════════════════════════════════════════════════════
# Models
# ══════════════════════════════════════════════════════════════
from torch_geometric.nn import SAGEConv, HeteroConv


class HomoGNN(nn.Module):
    """2-layer GraphSAGE with SPECTER2 initial features."""

    def __init__(self, in_dim, hidden):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden)
        self.conv2 = SAGEConv(hidden, hidden)
        self.dropout = 0.2

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class HeteroGNN(nn.Module):
    """2-layer heterogeneous GNN. HeteroConv of SAGEConv per edge type.
    Dataset nodes start from SPECTER2; other node types from a learnable
    embedding per node."""

    def __init__(self, data, hidden):
        super().__init__()
        self.hidden = hidden

        # Input projection for datasets (SPECTER2 768 -> hidden)
        ds_in = data["Dataset"].x.size(1)
        self.ds_proj = nn.Linear(ds_in, hidden)

        # Learnable embeddings for other node types
        self.embs = nn.ModuleDict({
            "ScienceKeyword": nn.Embedding(data["ScienceKeyword"].num_nodes, hidden),
            "Instrument":     nn.Embedding(data["Instrument"].num_nodes, hidden),
            "Platform":       nn.Embedding(data["Platform"].num_nodes, hidden),
            "Project":        nn.Embedding(data["Project"].num_nodes, hidden),
        })
        for e in self.embs.values():
            nn.init.xavier_uniform_(e.weight)

        # Two HeteroConv layers
        def make_conv():
            return HeteroConv({
                ("Dataset", "co_usage", "Dataset"):            SAGEConv((hidden, hidden), hidden),
                ("Dataset", "has_platform", "Platform"):       SAGEConv((hidden, hidden), hidden),
                ("Platform", "rev_has_platform", "Dataset"):   SAGEConv((hidden, hidden), hidden),
                ("Dataset", "has_keyword", "ScienceKeyword"):  SAGEConv((hidden, hidden), hidden),
                ("ScienceKeyword", "rev_has_keyword", "Dataset"): SAGEConv((hidden, hidden), hidden),
                ("Dataset", "of_project", "Project"):          SAGEConv((hidden, hidden), hidden),
                ("Project", "rev_of_project", "Dataset"):      SAGEConv((hidden, hidden), hidden),
                ("Platform", "has_instrument", "Instrument"):  SAGEConv((hidden, hidden), hidden),
                ("Instrument", "rev_has_instrument", "Platform"): SAGEConv((hidden, hidden), hidden),
                ("ScienceKeyword", "has_subcategory", "ScienceKeyword"):     SAGEConv((hidden, hidden), hidden),
                ("ScienceKeyword", "rev_has_subcategory", "ScienceKeyword"): SAGEConv((hidden, hidden), hidden),
            }, aggr="sum")

        self.conv1 = make_conv()
        self.conv2 = make_conv()
        self.dropout = 0.2

    def forward(self, data):
        # Initial representations
        x_dict = {"Dataset": self.ds_proj(data["Dataset"].x)}
        for t, emb in self.embs.items():
            x_dict[t] = emb.weight

        edge_index_dict = {k: v.edge_index for k, v in data.edge_items()}

        h = self.conv1(x_dict, edge_index_dict)
        h = {k: F.relu(v) for k, v in h.items()}
        h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}
        h = self.conv2(h, edge_index_dict)
        return h


# ══════════════════════════════════════════════════════════════
# Training utilities
# ══════════════════════════════════════════════════════════════
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# Degree-biased sampling distribution for training negatives
deg_train = Counter()
for a, b, _ in train_pairs:
    if a in ds_idx and b in ds_idx:
        deg_train[a] += 1
        deg_train[b] += 1
deg_arr = np.array([deg_train[d] + 1 for d in ds_list], dtype=np.float64)
prob_train_neg = (deg_arr ** NEG_ALPHA)
prob_train_neg /= prob_train_neg.sum()
train_pair_set = {tuple(sorted((a, b))) for a, b, _ in train_pairs
                  if a in ds_idx and b in ds_idx}

# Positive edges (unique, undirected) for training
train_pos_ij = np.array(
    [(ds_idx[a], ds_idx[b]) for a, b, _ in train_pairs
     if a in ds_idx and b in ds_idx],
    dtype=np.int64)
print(f"   training positives: {len(train_pos_ij):,} unique pairs")


def sample_train_batch(rng, n_pos=None):
    """Return (src, pos_dst, neg_dst) all LongTensors."""
    if n_pos is None:
        n_pos = len(train_pos_ij)
    # shuffle positives
    perm = rng.permutation(len(train_pos_ij))[:n_pos]
    pos_ij = train_pos_ij[perm]
    src = pos_ij[:, 0]
    pos = pos_ij[:, 1]
    # Degree-biased negatives: one per positive (faster than NEG_PER_POS_TRAIN
    # for training — 5:1 typical in word2vec but here 1:1 is fine).
    neg = rng.choice(N_DS, size=n_pos, p=prob_train_neg)
    # Avoid accidental positives (cheap rejection)
    for _ in range(3):
        bad = np.array([
            tuple(sorted((ds_list[s], ds_list[n]))) in train_pair_set or s == n
            for s, n in zip(src, neg)
        ])
        if not bad.any():
            break
        new_neg = rng.choice(N_DS, size=int(bad.sum()), p=prob_train_neg)
        neg[bad] = new_neg
    return (torch.from_numpy(src).long(),
            torch.from_numpy(pos).long(),
            torch.from_numpy(neg).long())


# ══════════════════════════════════════════════════════════════
# Evaluation (uses Stage 1's exact pools/negatives)
# ══════════════════════════════════════════════════════════════
def score_pair(emb_ds, a, b):
    return float(emb_ds[ds_idx[a]] @ emb_ds[ds_idx[b]])


def evaluate_pool(emb_ds, pool):
    """emb_ds: torch tensor [N_DS, hidden], L2-normalised."""
    emb_np = emb_ds.detach().cpu().numpy()
    rng = np.random.default_rng(0)
    ranks = []
    all_scores, all_labels = [], []
    for a, b_pos, b_negs in pool:
        if a not in ds_idx or b_pos not in ds_idx:
            continue
        cands = [b_pos] + [n for n in b_negs if n in ds_idx]
        if len(cands) < 2:
            continue
        scores = np.array([score_pair(emb_np, a, c) for c in cands])
        order = np.argsort(-scores + rng.uniform(-1e-12, 1e-12, size=scores.shape))
        rank = int(np.where(order == 0)[0][0]) + 1
        ranks.append(rank)
        all_scores.extend(scores.tolist())
        all_labels.extend([1] + [0] * (len(cands) - 1))
    ranks = np.array(ranks)
    return {
        "N":       len(ranks),
        "Hits@10": float((ranks <= 10).mean()),
        "Hits@50": float((ranks <= 50).mean()),
        "MRR":     float((1.0 / ranks).mean()),
        "AUC":     float(roc_auc_score(all_labels, all_scores)),
        "AP":      float(average_precision_score(all_labels, all_scores)),
    }


# Val pairs used for early stopping — convert to the same format
val_pool = []
train_set = set(train_pair_set)
# Use a small negative set for val MRR (not the big 100-neg eval pool)
rng_val = np.random.default_rng(1234)
for a, b, _ in val_pairs:
    if a not in ds_idx or b not in ds_idx:
        continue
    key = tuple(sorted((a, b)))
    if key in train_set:
        continue
    # 20 negatives for quick val eval
    negs = []
    while len(negs) < 20:
        j = int(rng_val.choice(N_DS, p=prob_train_neg))
        cand = ds_list[j]
        kk = tuple(sorted((a, cand)))
        if kk in train_set or cand == a or cand == b:
            continue
        negs.append(cand)
    val_pool.append((a, b, negs))
print(f"   val pool for early stopping: {len(val_pool):,} pairs")


# ══════════════════════════════════════════════════════════════
# Training loops
# ══════════════════════════════════════════════════════════════
def train_homo(seed):
    set_seed(seed)
    model = HomoGNN(specter.shape[1], HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    x = torch.from_numpy(specter).to(DEVICE)
    edge_index = homo_edge_index.to(DEVICE)

    best_mrr = -1.0; best_state = None; stale = 0
    rng_np = np.random.default_rng(seed)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        h = model(x, edge_index)
        h = F.normalize(h, p=2, dim=-1)
        src, pos, neg = sample_train_batch(rng_np)
        src = src.to(DEVICE); pos = pos.to(DEVICE); neg = neg.to(DEVICE)
        pos_score = (h[src] * h[pos]).sum(dim=-1)
        neg_score = (h[src] * h[neg]).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]),
        )
        loss.backward()
        opt.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                h_eval = F.normalize(model(x, edge_index), p=2, dim=-1)
            m = evaluate_pool(h_eval, val_pool)
            if m["MRR"] > best_mrr + 1e-4:
                best_mrr = m["MRR"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale * 5 > PATIENCE:
                print(f"      epoch {epoch:3d}  loss={loss.item():.4f}  "
                      f"val MRR={m['MRR']:.3f}  [early stop]")
                break
            if epoch % 25 == 0:
                print(f"      epoch {epoch:3d}  loss={loss.item():.4f}  "
                      f"val MRR={m['MRR']:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final = F.normalize(model(x, edge_index), p=2, dim=-1)
    return model, final


def train_hetero(seed, hetero_data):
    set_seed(seed)
    data = hetero_data.to(DEVICE)
    model = HeteroGNN(data, HIDDEN).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    best_mrr = -1.0; best_state = None; stale = 0
    rng_np = np.random.default_rng(seed)

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        opt.zero_grad()
        h = model(data)
        h_ds = F.normalize(h["Dataset"], p=2, dim=-1)
        src, pos, neg = sample_train_batch(rng_np)
        src = src.to(DEVICE); pos = pos.to(DEVICE); neg = neg.to(DEVICE)
        pos_score = (h_ds[src] * h_ds[pos]).sum(dim=-1)
        neg_score = (h_ds[src] * h_ds[neg]).sum(dim=-1)
        loss = F.binary_cross_entropy_with_logits(
            torch.cat([pos_score, neg_score]),
            torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)]),
        )
        loss.backward()
        opt.step()

        if epoch % 5 == 0:
            model.eval()
            with torch.no_grad():
                h_eval = F.normalize(model(data)["Dataset"], p=2, dim=-1)
            m = evaluate_pool(h_eval, val_pool)
            if m["MRR"] > best_mrr + 1e-4:
                best_mrr = m["MRR"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale * 5 > PATIENCE:
                print(f"      epoch {epoch:3d}  loss={loss.item():.4f}  "
                      f"val MRR={m['MRR']:.3f}  [early stop]")
                break
            if epoch % 25 == 0:
                print(f"      epoch {epoch:3d}  loss={loss.item():.4f}  "
                      f"val MRR={m['MRR']:.3f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        final = F.normalize(model(data)["Dataset"], p=2, dim=-1)
    return model, final


# ══════════════════════════════════════════════════════════════
# Run all seeds × both variants
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("TRAINING")
print("═" * 62)

pools = [("all", eval_all), ("cold_start", eval_cold), ("cross_daac", eval_cross)]
metric_keys = ["Hits@10", "Hits@50", "MRR", "AUC", "AP"]

results_all = []  # rows: variant, seed, pool, metric -> value

# --- Homo ---
print("\n▶ Building homogeneous graph …")
for seed in SEEDS:
    print(f"\n▶ GNN-Homo  seed={seed}")
    t0 = time.time()
    model, final = train_homo(seed)
    print(f"   trained in {time.time()-t0:.1f}s")
    torch.save(model.state_dict(), f"{CKPT_DIR}/gnn_homo_seed{seed}.pt")
    for pool_name, pool in pools:
        m = evaluate_pool(final, pool)
        print(f"   {pool_name:<11} N={m['N']:5d}  "
              f"H@10={m['Hits@10']:.3f}  H@50={m['Hits@50']:.3f}  "
              f"MRR={m['MRR']:.3f}  AUC={m['AUC']:.3f}  AP={m['AP']:.3f}")
        results_all.append({"variant": "GNN-Homo", "seed": seed,
                            "pool": pool_name, **m})

# --- Hetero ---
print("\n▶ Building heterogeneous graph …")
hetero_data = build_hetero_data()

for seed in SEEDS:
    print(f"\n▶ GNN-Hetero  seed={seed}")
    t0 = time.time()
    model, final = train_hetero(seed, hetero_data)
    print(f"   trained in {time.time()-t0:.1f}s")
    torch.save(model.state_dict(), f"{CKPT_DIR}/gnn_hetero_seed{seed}.pt")
    for pool_name, pool in pools:
        m = evaluate_pool(final, pool)
        print(f"   {pool_name:<11} N={m['N']:5d}  "
              f"H@10={m['Hits@10']:.3f}  H@50={m['Hits@50']:.3f}  "
              f"MRR={m['MRR']:.3f}  AUC={m['AUC']:.3f}  AP={m['AP']:.3f}")
        results_all.append({"variant": "GNN-Hetero", "seed": seed,
                            "pool": pool_name, **m})


# ══════════════════════════════════════════════════════════════
# Aggregate + write results
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("AGGREGATE RESULTS (mean ± std over 3 seeds)")
print("═" * 62)

out_tsv = f"{OUT_DIR}/stage2_results.tsv"
with open(out_tsv, "w") as f:
    cols = ["variant", "seed", "pool", "N"] + metric_keys
    f.write("\t".join(cols) + "\n")
    for r in results_all:
        f.write("\t".join(str(r[c]) for c in cols) + "\n")
print(f"\n✅ raw results: {out_tsv}")

# Aggregate
agg = {}
for r in results_all:
    key = (r["variant"], r["pool"])
    agg.setdefault(key, {k: [] for k in metric_keys})
    for k in metric_keys:
        agg[key][k].append(r[k])

print(f"\n{'Variant':<12} {'Pool':<12} "
      f"{'H@10':>14} {'H@50':>14} {'MRR':>14} {'AUC':>14} {'AP':>14}")
for (variant, pool), vals in agg.items():
    row = f"{variant:<12} {pool:<12} "
    for k in metric_keys:
        vs = vals[k]
        row += f"{np.mean(vs):>7.3f}±{np.std(vs):>5.3f} "
    print(row)


# LaTeX: mean ± std per (variant × pool)
print("\n▶ Writing LaTeX table …")
tex = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Stage~2 GNN results on the co-usage link-prediction task "
    r"(1:100 negative sampling, same pools as Table~\ref{tab:stage1_baselines}). "
    r"Mean $\pm$ std over 3 seeds.}",
    r"\label{tab:stage2_gnn}",
    r"\small",
    r"\begin{tabular}{llrrrrr}",
    r"\toprule",
    r"\textbf{Variant} & \textbf{Pool} & \textbf{Hits@10} & "
    r"\textbf{Hits@50} & \textbf{MRR} & \textbf{AUC} & \textbf{AP} \\",
    r"\midrule",
]
for variant in ["GNN-Homo", "GNN-Hetero"]:
    for pool_name, _ in pools:
        vals = agg[(variant, pool_name)]
        cells = " & ".join(
            f"{np.mean(vals[k]):.3f}{{\\scriptsize$\\pm${np.std(vals[k]):.3f}}}"
            for k in metric_keys
        )
        tex.append(f"\\texttt{{{variant}}} & \\texttt{{{pool_name}}} & {cells} \\\\")
    tex.append(r"\midrule")
tex = tex[:-1]
tex += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
    "",
]
with open(f"{OUT_DIR}/tab_stage2_gnn.tex", "w") as f:
    f.write("\n".join(tex))
print(f"✅ LaTeX: {OUT_DIR}/tab_stage2_gnn.tex")

print("\n" + "═" * 62)
print("STAGE 2 COMPLETE")
print("═" * 62)
print("Next: Stage 4 — LLM-as-judge on the top novel predictions.")
print("(Stage 3 = just aggregating Stages 1 & 2 into the final results table;")
print(" we already have that in tab_stage1_baselines.tex + tab_stage2_gnn.tex.)")