"""
STAGE 2.5c — Save final model embeddings for Stage 4
====================================================
Retrains the two final configs once (seed 0) and saves their artifacts
so Stage 4 can load them directly without retraining.

Configs (both Hetero + pubs_area):
    A. Dot scorer          → final_dot.npy   (1,475 × 128)
    B. MLP scorer          → final_mlp.npy   (1,475 × 128)
                             final_mlp_scorer.npz (MLP weights)

Also saves:
    dataset_ids.txt           (row order, identical to the one saved by Stage 1)
    final_meta.json           (config metadata, 6-seed metrics, seed used here)

Run:
    python3 stage2p5c_save_final.py
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

SAVE_DIR = "./nasa_eo_kg"
OUT_DIR = "./neurips_figs"
EMB_DIR = f"{OUT_DIR}/embeddings"
FINAL_DIR = f"{OUT_DIR}/final_models"
os.makedirs(FINAL_DIR, exist_ok=True)

HIDDEN = 128
N_EPOCHS = 300
LR = 1e-3
WD = 1e-5
PATIENCE = 30
NEG_ALPHA = 0.75
SEED = 0                      # use seed 0; produces deterministic artifacts
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ───── Load artifacts ────────────────────────────────────────
def _load_pairs(path):
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            a, b, w = line.rstrip("\n").split("\t")
            out.append((a, b, int(w)))
    return out


def _load_pool(path):
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            a, b_pos = parts[0], parts[1]
            b_negs = parts[2].split(",") if len(parts) > 2 and parts[2] else []
            out.append((a, b_pos, b_negs))
    return out


train_pairs = _load_pairs(f"{OUT_DIR}/cousage_edges_train.tsv")
val_pairs   = _load_pairs(f"{OUT_DIR}/cousage_edges_val.tsv")
eval_all    = _load_pool(f"{OUT_DIR}/eval_pool_all.tsv")
eval_cold   = _load_pool(f"{OUT_DIR}/eval_pool_cold_start.tsv")
eval_cross  = _load_pool(f"{OUT_DIR}/eval_pool_cross_daac.tsv")

ds_list = [l.strip() for l in open(f"{EMB_DIR}/dataset_ids.txt")]
ds_idx = {d: i for i, d in enumerate(ds_list)}
N_DS = len(ds_list)
specter = np.load(f"{EMB_DIR}/specter2.npy").astype(np.float32)

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
train_pos_ij = np.array(
    [(ds_idx[a], ds_idx[b]) for a, b, _ in train_pairs
     if a in ds_idx and b in ds_idx], dtype=np.int64)


# ───── Build hetero graph with pubs_uses + pubs_area ─────────
def build_hetero():
    import networkx as nx
    from torch_geometric.data import HeteroData

    graphml_path = None
    for root, _, files in os.walk(SAVE_DIR):
        for f in files:
            if f.endswith(".graphml"):
                graphml_path = os.path.join(root, f); break
        if graphml_path:
            break
    G = nx.read_graphml(graphml_path)

    def _lbl(raw):
        return str(raw).strip().strip("[]:'\" ").strip()

    def _et(data):
        return (data.get("label") or data.get("type")
                or data.get("relationship") or "")

    node_label = {n: _lbl(d.get("labels", "UNKNOWN"))
                  for n, d in G.nodes(data=True)}

    kept_datasets = set(ds_list)
    ds_platform = defaultdict(set)
    ds_keyword = defaultdict(set)
    ds_project = defaultdict(set)
    platform_instr = defaultdict(set)
    keyword_subcat = defaultdict(set)
    paper_uses_ds = defaultdict(set)
    paper_area = defaultdict(set)

    for u, v, data in G.edges(data=True):
        e = _et(data)
        lu, lv = node_label.get(u), node_label.get(v)
        if e == "HAS_PLATFORM":
            if lu == "Dataset" and lv == "Platform": ds_platform[u].add(v)
            elif lv == "Dataset" and lu == "Platform": ds_platform[v].add(u)
        elif e == "HAS_SCIENCEKEYWORD":
            if lu == "Dataset" and lv == "ScienceKeyword": ds_keyword[u].add(v)
            elif lv == "Dataset" and lu == "ScienceKeyword": ds_keyword[v].add(u)
        elif e == "OF_PROJECT":
            if lu == "Dataset" and lv == "Project": ds_project[u].add(v)
            elif lv == "Dataset" and lu == "Project": ds_project[v].add(u)
        elif e == "HAS_INSTRUMENT":
            if lu == "Platform" and lv == "Instrument": platform_instr[u].add(v)
            elif lv == "Platform" and lu == "Instrument": platform_instr[v].add(u)
        elif e == "HAS_SUBCATEGORY":
            if lu == "ScienceKeyword" and lv == "ScienceKeyword":
                keyword_subcat[u].add(v)
        elif e == "USES_DATASET":
            if lu == "Publication" and lv == "Dataset" and v in kept_datasets:
                paper_uses_ds[u].add(v)
            elif lv == "Publication" and lu == "Dataset" and u in kept_datasets:
                paper_uses_ds[v].add(u)
        elif e == "HAS_APPLIEDRESEARCHAREA":
            if lu == "Publication" and lv == "ScienceKeyword":
                paper_area[u].add(v)
            elif lv == "Publication" and lu == "ScienceKeyword":
                paper_area[v].add(u)

    kept_papers = set(paper_uses_ds.keys())
    paper_area_f = {p: kws for p, kws in paper_area.items() if p in kept_papers}

    kept_platforms, kept_keywords, kept_projects = set(), set(), set()
    for d in kept_datasets:
        kept_platforms |= ds_platform.get(d, set())
        kept_keywords |= ds_keyword.get(d, set())
        kept_projects |= ds_project.get(d, set())
    for kws in paper_area_f.values():
        kept_keywords |= kws
    kept_instruments = set()
    for p in kept_platforms:
        kept_instruments |= platform_instr.get(p, set())

    sk_list = sorted(kept_keywords); inst_list = sorted(kept_instruments)
    plat_list = sorted(kept_platforms); proj_list = sorted(kept_projects)
    pub_list = sorted(kept_papers)
    sk_idx = {n: i for i, n in enumerate(sk_list)}
    inst_idx = {n: i for i, n in enumerate(inst_list)}
    plat_idx = {n: i for i, n in enumerate(plat_list)}
    proj_idx = {n: i for i, n in enumerate(proj_list)}
    pub_idx = {n: i for i, n in enumerate(pub_list)}

    def _tt(pairs):
        if not pairs:
            return torch.empty((2, 0), dtype=torch.long)
        a = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        b = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        return torch.stack([a, b], dim=0)

    ds_plat = [(ds_idx[d], plat_idx[p]) for d in kept_datasets
               for p in ds_platform.get(d, []) if p in plat_idx]
    ds_kw = [(ds_idx[d], sk_idx[k]) for d in kept_datasets
             for k in ds_keyword.get(d, []) if k in sk_idx]
    ds_proj = [(ds_idx[d], proj_idx[p]) for d in kept_datasets
               for p in ds_project.get(d, []) if p in proj_idx]
    plat_inst = [(plat_idx[p], inst_idx[i]) for p in kept_platforms
                 for i in platform_instr.get(p, []) if i in inst_idx]
    kw_sub = [(sk_idx[u], sk_idx[v]) for u in kept_keywords
              for v in keyword_subcat.get(u, []) if v in sk_idx]

    data = HeteroData()
    data["Dataset"].x = torch.from_numpy(specter)
    data["ScienceKeyword"].num_nodes = len(sk_list)
    data["Instrument"].num_nodes = len(inst_list)
    data["Platform"].num_nodes = len(plat_list)
    data["Project"].num_nodes = len(proj_list)
    data["Publication"].num_nodes = len(pub_list)

    cu_src, cu_dst = [], []
    for a, b, _ in train_pairs:
        if a in ds_idx and b in ds_idx:
            i, j = ds_idx[a], ds_idx[b]
            cu_src += [i, j]; cu_dst += [j, i]
    data["Dataset", "co_usage", "Dataset"].edge_index = \
        torch.tensor([cu_src, cu_dst], dtype=torch.long)

    def _add(ns, rel, nt, edges):
        data[ns, rel, nt].edge_index = _tt(edges)
        data[nt, "rev_" + rel, ns].edge_index = \
            data[ns, rel, nt].edge_index.flip(0)

    _add("Dataset", "has_platform", "Platform", ds_plat)
    _add("Dataset", "has_keyword", "ScienceKeyword", ds_kw)
    _add("Dataset", "of_project", "Project", ds_proj)
    _add("Platform", "has_instrument", "Instrument", plat_inst)
    _add("ScienceKeyword", "has_subcategory", "ScienceKeyword", kw_sub)

    paper_ds_edges = [(pub_idx[p], ds_idx[d]) for p, dset in paper_uses_ds.items()
                      for d in dset if d in ds_idx]
    _add("Publication", "uses_dataset", "Dataset", paper_ds_edges)

    area_edges = [(pub_idx[p], sk_idx[k]) for p, kws in paper_area_f.items()
                  for k in kws if k in sk_idx]
    _add("Publication", "has_area", "ScienceKeyword", area_edges)

    print(f"   nodes: DS={N_DS} KW={len(sk_list)} INST={len(inst_list)} "
          f"PLAT={len(plat_list)} PROJ={len(proj_list)} PUB={len(pub_list)}")
    print(f"   edges: DS-PLAT={len(ds_plat)} DS-KW={len(ds_kw)} "
          f"DS-PROJ={len(ds_proj)} PLAT-INST={len(plat_inst)} KW-SUB={len(kw_sub)} "
          f"PUB-USES={len(paper_ds_edges)} PUB-AREA={len(area_edges)}")
    return data


# ───── Models ────────────────────────────────────────────────
from torch_geometric.nn import SAGEConv, HeteroConv


class HeteroGNN(nn.Module):
    def __init__(self, data, hidden, n_layers=2):
        super().__init__()
        ds_in = data["Dataset"].x.size(1)
        self.ds_proj = nn.Linear(ds_in, hidden)
        embs = {}
        for t in ["ScienceKeyword", "Instrument", "Platform", "Project"]:
            embs[t] = nn.Embedding(data[t].num_nodes, hidden)
        if "Publication" in data.node_types:
            embs["Publication"] = nn.Embedding(data["Publication"].num_nodes, hidden)
        self.embs = nn.ModuleDict(embs)
        for e in self.embs.values():
            nn.init.xavier_uniform_(e.weight)
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(HeteroConv(
                {et: SAGEConv((hidden, hidden), hidden) for et in data.edge_types},
                aggr="sum"))
        self.dropout = 0.2

    def forward(self, data):
        x_dict = {"Dataset": self.ds_proj(data["Dataset"].x)}
        for t, emb in self.embs.items():
            x_dict[t] = emb.weight
        ei_dict = {k: v.edge_index for k, v in data.edge_items()}
        for i, conv in enumerate(self.convs):
            new_x = conv(x_dict, ei_dict)
            if i < len(self.convs) - 1:
                new_x = {k: F.relu(v) for k, v in new_x.items()}
                new_x = {k: F.dropout(v, p=self.dropout, training=self.training)
                         for k, v in new_x.items()}
            x_dict = new_x
        return x_dict["Dataset"]


class MLPScorer(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ha, hb):
        return self.mlp(torch.cat([ha, hb, ha * hb], dim=-1)).squeeze(-1)


# ───── Utils (same as Stage 2.5) ──────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sample_train_batch(rng_np):
    n_pos = len(train_pos_ij)
    perm = rng_np.permutation(n_pos)
    pos_ij = train_pos_ij[perm]
    src = pos_ij[:, 0]; pos = pos_ij[:, 1]
    neg = rng_np.choice(N_DS, size=n_pos, p=prob_train_neg)
    for _ in range(3):
        bad = np.array([
            tuple(sorted((ds_list[s], ds_list[neg[i]]))) in train_pair_set
            or s == neg[i]
            for i, s in enumerate(src)
        ])
        if not bad.any(): break
        new = rng_np.choice(N_DS, size=int(bad.sum()), p=prob_train_neg)
        neg[bad] = new
    return (torch.from_numpy(src).long(),
            torch.from_numpy(pos).long(),
            torch.from_numpy(neg).long())


def dot_scorer(ea, eb):
    return (ea * eb).sum(axis=-1)


def evaluate_pool(emb, pool, scorer_fn=dot_scorer):
    rng = np.random.default_rng(0)
    ranks = []; all_scores, all_labels = [], []
    for a, b_pos, b_negs in pool:
        if a not in ds_idx or b_pos not in ds_idx: continue
        cands = [b_pos] + [n for n in b_negs if n in ds_idx]
        if len(cands) < 2: continue
        a_i = ds_idx[a]
        cand_i = np.array([ds_idx[c] for c in cands], dtype=np.int64)
        ea = emb[np.full_like(cand_i, a_i)]; eb = emb[cand_i]
        scores = scorer_fn(ea, eb)
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


val_pool = []
rng_val = np.random.default_rng(1234)
for a, b, _ in val_pairs:
    if a not in ds_idx or b not in ds_idx: continue
    if tuple(sorted((a, b))) in train_pair_set: continue
    negs = []
    while len(negs) < 20:
        j = int(rng_val.choice(N_DS, p=prob_train_neg))
        cand = ds_list[j]
        if tuple(sorted((a, cand))) in train_pair_set or cand == a or cand == b: continue
        negs.append(cand)
    val_pool.append((a, b, negs))


# ───── Train one final model with seed 0 ────────────────────
def train_final(hetero_data, use_mlp, name):
    print(f"\n▶ Training final model: {name}")
    set_seed(SEED)
    rng_np = np.random.default_rng(SEED)
    data = hetero_data.to(DEVICE)
    model = HeteroGNN(data, HIDDEN, n_layers=2).to(DEVICE)
    scorer = MLPScorer(HIDDEN).to(DEVICE) if use_mlp else None
    params = list(model.parameters())
    if scorer is not None:
        params += list(scorer.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

    def fwd():
        return F.normalize(model(data), p=2, dim=-1)

    def pair_score(h, a_idx, b_idx):
        ea, eb = h[a_idx], h[b_idx]
        if scorer is not None:
            return scorer(ea, eb)
        return (ea * eb).sum(dim=-1)

    best_mrr = -1.0; best_state = None; best_scorer_state = None; stale = 0
    t0 = time.time()
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        if scorer is not None: scorer.train()
        opt.zero_grad()
        h = fwd()
        src, pos, neg = sample_train_batch(rng_np)
        src = src.to(DEVICE); pos = pos.to(DEVICE); neg = neg.to(DEVICE)
        pos_score = pair_score(h, src, pos)
        if scorer is not None:
            neg_score = pair_score(h, src, neg)
        else:
            neg_score = (h[src] * h[neg]).sum(dim=-1)
        logits = torch.cat([pos_score, neg_score])
        labels = torch.cat([torch.ones_like(pos_score),
                            torch.zeros_like(neg_score)])
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        opt.step()

        if epoch % 5 == 0:
            model.eval()
            if scorer is not None: scorer.eval()
            with torch.no_grad():
                h_eval = fwd().detach().cpu().numpy()
            if scorer is not None:
                W1 = scorer.mlp[0].weight.detach().cpu().numpy()
                b1 = scorer.mlp[0].bias.detach().cpu().numpy()
                W2 = scorer.mlp[2].weight.detach().cpu().numpy()
                b2 = scorer.mlp[2].bias.detach().cpu().numpy()
                def _score(ea, eb):
                    x_in = np.concatenate([ea, eb, ea * eb], axis=-1)
                    h1 = np.maximum(0, x_in @ W1.T + b1)
                    return (h1 @ W2.T + b2).squeeze(-1)
                m = evaluate_pool(h_eval, val_pool, scorer_fn=_score)
            else:
                m = evaluate_pool(h_eval, val_pool, scorer_fn=dot_scorer)

            if m["MRR"] > best_mrr + 1e-4:
                best_mrr = m["MRR"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if scorer is not None:
                    best_scorer_state = {k: v.detach().cpu().clone() for k, v in scorer.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale * 5 > PATIENCE:
                print(f"   early stop at epoch {epoch}  val MRR={best_mrr:.3f}")
                break

    model.load_state_dict(best_state)
    if scorer is not None and best_scorer_state is not None:
        scorer.load_state_dict(best_scorer_state)
    model.eval()
    if scorer is not None: scorer.eval()
    with torch.no_grad():
        h_final = fwd().detach().cpu().numpy()
    print(f"   trained in {time.time()-t0:.1f}s  val MRR={best_mrr:.3f}")

    mlp_weights = None
    if scorer is not None:
        mlp_weights = {
            "W1": scorer.mlp[0].weight.detach().cpu().numpy(),
            "b1": scorer.mlp[0].bias.detach().cpu().numpy(),
            "W2": scorer.mlp[2].weight.detach().cpu().numpy(),
            "b2": scorer.mlp[2].bias.detach().cpu().numpy(),
        }
    return h_final, mlp_weights, best_mrr


# ───── Run both configs ──────────────────────────────────────
print("\n" + "═" * 62)
print("Building hetero graph …")
print("═" * 62)
hetero_data = build_hetero()

emb_dot, _, val_dot = train_final(hetero_data, use_mlp=False, name="dot")
emb_mlp, mlp_w, val_mlp = train_final(hetero_data, use_mlp=True, name="mlp")


def apply_mlp(ea, eb, w):
    x_in = np.concatenate([ea, eb, ea * eb], axis=-1)
    h1 = np.maximum(0, x_in @ w["W1"].T + w["b1"])
    return (h1 @ w["W2"].T + w["b2"]).squeeze(-1)


# Test-pool sanity check
POOLS = [("all", eval_all), ("cold_start", eval_cold), ("cross_daac", eval_cross)]

print("\n" + "═" * 62)
print("Sanity check on test pools (seed 0 — compare to paper numbers)")
print("═" * 62)

print(f"\n{'Config':<18} {'Pool':<12} {'H@10':>8} {'MRR':>8}")
for name, emb, scorer_fn in [
    ("dot",  emb_dot,  dot_scorer),
    ("mlp",  emb_mlp,  lambda ea, eb: apply_mlp(ea, eb, mlp_w)),
]:
    for pool_name, pool in POOLS:
        m = evaluate_pool(emb, pool, scorer_fn=scorer_fn)
        print(f"{name:<18} {pool_name:<12} {m['Hits@10']:>8.3f} {m['MRR']:>8.3f}")


# ───── Save artifacts ────────────────────────────────────────
print("\n" + "═" * 62)
print("Saving final artifacts …")
print("═" * 62)

np.save(f"{FINAL_DIR}/final_dot.npy", emb_dot.astype(np.float32))
np.save(f"{FINAL_DIR}/final_mlp.npy", emb_mlp.astype(np.float32))
np.savez(f"{FINAL_DIR}/final_mlp_scorer.npz", **mlp_w)

# Copy the dataset-id order file over
with open(f"{FINAL_DIR}/dataset_ids.txt", "w") as f:
    for d in ds_list:
        f.write(d + "\n")

# Metadata with the 6-seed averages from Stage 2.5b
meta = {
    "configs": {
        "dot": {
            "description": ("Hetero + pubs_uses + pubs_area, dot scorer. "
                            "Best on 'all' and 'cold_start' pools."),
            "emb_file":    "final_dot.npy",
            "scorer":      "dot",
            "seed_used_for_saved_weights": SEED,
            "val_MRR_seed0": val_dot,
            "six_seed_mean_std": {
                "all":        {"Hits@10": 0.474, "std": 0.004,
                               "MRR": 0.226, "MRR_std": 0.003},
                "cold_start": {"Hits@10": 0.524, "std": 0.009,
                               "MRR": 0.262, "MRR_std": 0.009},
                "cross_daac": {"Hits@10": 0.170, "std": 0.010,
                               "MRR": 0.076, "MRR_std": 0.005},
            },
        },
        "mlp": {
            "description": ("Hetero + pubs_uses + pubs_area, MLP scorer. "
                            "Best on 'cross_daac' pool."),
            "emb_file":     "final_mlp.npy",
            "scorer_file":  "final_mlp_scorer.npz",
            "scorer":       "mlp",
            "seed_used_for_saved_weights": SEED,
            "val_MRR_seed0": val_mlp,
            "six_seed_mean_std": {
                "all":        {"Hits@10": 0.472, "std": 0.009,
                               "MRR": 0.229, "MRR_std": 0.007},
                "cold_start": {"Hits@10": 0.375, "std": 0.008,
                               "MRR": 0.182, "MRR_std": 0.005},
                "cross_daac": {"Hits@10": 0.282, "std": 0.012,
                               "MRR": 0.124, "MRR_std": 0.006},
            },
        },
    },
    "dataset_ids_file": "dataset_ids.txt",
    "N_datasets":       N_DS,
    "embedding_dim":    HIDDEN,
    "seed":             SEED,
}
with open(f"{FINAL_DIR}/final_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print(f"   ✅ {FINAL_DIR}/final_dot.npy           ({emb_dot.shape})")
print(f"   ✅ {FINAL_DIR}/final_mlp.npy           ({emb_mlp.shape})")
print(f"   ✅ {FINAL_DIR}/final_mlp_scorer.npz   (W1, b1, W2, b2)")
print(f"   ✅ {FINAL_DIR}/dataset_ids.txt         ({N_DS} ids)")
print(f"   ✅ {FINAL_DIR}/final_meta.json")
print("\nStage 4 can now load these without retraining. Ready for judge experiments.")