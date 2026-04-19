"""
STAGE 2.5b — Verification: 3 extra seeds for Hetero + pubs_area + mlp_scorer
============================================================================
Stage 2.5 reported cross-DAAC H@10 = 0.273 ± 0.008 with seeds [0, 1, 2] for
this configuration. We re-train with seeds [3, 4, 5] to confirm the result
isn't a fluke and to report 6-seed mean ± std for the paper.

This reuses helper code from stage2p5_ablation.py by subprocess of the key
logic — but for clarity we inline the minimal pieces we need.

Output:
    ./neurips_figs/stage2p5_verify.tsv
    Console: mean ± std over 6 seeds for the two top Hetero configs:
        - Hetero + pubs_area           (dot scorer, best on all + cold_start)
        - Hetero + pubs_area + mlp     (MLP scorer, best on cross_daac)
"""

from __future__ import annotations

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

HIDDEN = 128
N_EPOCHS = 300
LR = 1e-3
WD = 1e-5
PATIENCE = 30
NEG_ALPHA = 0.75
NEW_SEEDS = [3, 4, 5]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ───── Load artifacts ─────────────────────────────────────────
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


# ───── Hetero graph with pubs_uses + pubs_area ────────────────
def build_hetero_with_pubs_area():
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

    sk_list   = sorted(kept_keywords)
    inst_list = sorted(kept_instruments)
    plat_list = sorted(kept_platforms)
    proj_list = sorted(kept_projects)
    pub_list  = sorted(kept_papers)
    sk_idx   = {n: i for i, n in enumerate(sk_list)}
    inst_idx = {n: i for i, n in enumerate(inst_list)}
    plat_idx = {n: i for i, n in enumerate(plat_list)}
    proj_idx = {n: i for i, n in enumerate(proj_list)}
    pub_idx  = {n: i for i, n in enumerate(pub_list)}

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

    return data


# ───── Models (copied exactly from stage2p5_ablation.py) ──────
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


# ───── Utils ──────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def sample_train_batch(rng_np, neg_per_pos=1):
    n_pos = len(train_pos_ij)
    perm = rng_np.permutation(n_pos)
    pos_ij = train_pos_ij[perm]
    src = pos_ij[:, 0]; pos = pos_ij[:, 1]
    neg = rng_np.choice(N_DS, size=(n_pos, neg_per_pos), p=prob_train_neg)
    for k in range(neg_per_pos):
        for _ in range(3):
            bad = np.array([
                tuple(sorted((ds_list[s], ds_list[neg[i, k]]))) in train_pair_set
                or s == neg[i, k]
                for i, s in enumerate(src)
            ])
            if not bad.any():
                break
            new = rng_np.choice(N_DS, size=int(bad.sum()), p=prob_train_neg)
            neg[bad, k] = new
    return (torch.from_numpy(src).long(),
            torch.from_numpy(pos).long(),
            torch.from_numpy(neg).long())


def dot_scorer(ea, eb):
    return (ea * eb).sum(axis=-1)


def evaluate_pool(emb_ds_np, pool, scorer_fn=dot_scorer):
    rng = np.random.default_rng(0)
    ranks = []; all_scores, all_labels = [], []
    for a, b_pos, b_negs in pool:
        if a not in ds_idx or b_pos not in ds_idx:
            continue
        cands = [b_pos] + [n for n in b_negs if n in ds_idx]
        if len(cands) < 2:
            continue
        a_i = ds_idx[a]
        cand_i = np.array([ds_idx[c] for c in cands], dtype=np.int64)
        ea = emb_ds_np[np.full_like(cand_i, a_i)]
        eb = emb_ds_np[cand_i]
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
    if a not in ds_idx or b not in ds_idx:
        continue
    if tuple(sorted((a, b))) in train_pair_set:
        continue
    negs = []
    while len(negs) < 20:
        j = int(rng_val.choice(N_DS, p=prob_train_neg))
        cand = ds_list[j]
        if (tuple(sorted((a, cand))) in train_pair_set
                or cand == a or cand == b):
            continue
        negs.append(cand)
    val_pool.append((a, b, negs))


# ───── Training function for both configs ────────────────────
def train(seed, hetero_data, use_mlp):
    set_seed(seed)
    rng_np = np.random.default_rng(seed)
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

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        if scorer is not None: scorer.train()
        opt.zero_grad()
        h = fwd()
        src, pos, neg = sample_train_batch(rng_np, neg_per_pos=1)
        src = src.to(DEVICE); pos = pos.to(DEVICE); neg = neg.to(DEVICE)
        pos_score = pair_score(h, src, pos)
        if scorer is not None:
            neg_flat = neg.reshape(-1)
            src_rep = src  # neg_per_pos=1
            neg_score = pair_score(h, src_rep, neg_flat).view(-1, 1)
        else:
            ha = h[src].unsqueeze(1)
            hb = h[neg]
            neg_score = (ha * hb).sum(dim=-1)

        logits = torch.cat([pos_score, neg_score.reshape(-1)])
        labels = torch.cat([torch.ones_like(pos_score),
                            torch.zeros_like(neg_score.reshape(-1))])
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
                break

    model.load_state_dict(best_state)
    if scorer is not None and best_scorer_state is not None:
        scorer.load_state_dict(best_scorer_state)
    model.eval()
    if scorer is not None: scorer.eval()
    with torch.no_grad():
        h_final = fwd().detach().cpu().numpy()

    if scorer is not None:
        W1 = scorer.mlp[0].weight.detach().cpu().numpy()
        b1 = scorer.mlp[0].bias.detach().cpu().numpy()
        W2 = scorer.mlp[2].weight.detach().cpu().numpy()
        b2 = scorer.mlp[2].bias.detach().cpu().numpy()
        def final_scorer(ea, eb):
            x_in = np.concatenate([ea, eb, ea * eb], axis=-1)
            h1 = np.maximum(0, x_in @ W1.T + b1)
            return (h1 @ W2.T + b2).squeeze(-1)
    else:
        final_scorer = dot_scorer
    return h_final, final_scorer, best_mrr


# ───── Run ────────────────────────────────────────────────────
POOLS = [("all", eval_all), ("cold_start", eval_cold), ("cross_daac", eval_cross)]
print("\n" + "═" * 62)
print("Building hetero graph (pubs_area config) …")
print("═" * 62)
hetero_data = build_hetero_with_pubs_area()

results = []
# Existing Stage 2.5 seeds (0,1,2) already in stage2p5_ablation.tsv.
# We just add seeds 3,4,5 here for the two top configs.
for config_name, use_mlp in [
    ("+pubs_area",             False),
    ("+pubs_area_mlp_scorer",  True),
]:
    print(f"\n─── {config_name}  (seeds {NEW_SEEDS}) ───")
    for seed in NEW_SEEDS:
        t0 = time.time()
        emb, scorer, val_mrr = train(seed, hetero_data, use_mlp=use_mlp)
        dt = time.time() - t0
        print(f"   seed {seed}: {dt:5.1f}s  val MRR={val_mrr:.3f}", end="")
        for pool_name, pool in POOLS:
            m = evaluate_pool(emb, pool, scorer_fn=scorer)
            results.append({"config": config_name, "seed": seed,
                            "pool": pool_name, **m, "val_MRR": val_mrr})
        # Print just H@10s compactly
        h10s = {pn: [r for r in results if r["config"] == config_name
                     and r["seed"] == seed and r["pool"] == pn][0]["Hits@10"]
                for pn, _ in POOLS}
        print(f"   H@10  all={h10s['all']:.3f}  "
              f"cold={h10s['cold_start']:.3f}  cross={h10s['cross_daac']:.3f}")


# ───── Aggregate new seeds + combine with old ───────────────
print("\n" + "═" * 62)
print("SIX-SEED RESULTS  (seeds 0-5)")
print("═" * 62)

# Load the 3 original seeds from stage2p5_ablation.tsv
old_rows = {}
with open(f"{OUT_DIR}/stage2p5_ablation.tsv") as f:
    header = next(f).rstrip("\n").split("\t")
    for line in f:
        parts = line.rstrip("\n").split("\t")
        r = dict(zip(header, parts))
        if r["variant"] != "hetero":
            continue
        # Match config names
        if r["config"] in ("+pubs_area", "+mlp_scorer"):
            key = r["config"] if r["config"] == "+pubs_area" else "+pubs_area_mlp_scorer"
            old_rows.setdefault((key, int(r["seed"]), r["pool"]), r)


def agg(config_name, pool_name, metric):
    vals = []
    # old (seeds 0-2)
    for s in (0, 1, 2):
        key = (config_name, s, pool_name)
        if key in old_rows:
            vals.append(float(old_rows[key][metric]))
    # new (seeds 3-5)
    for r in results:
        if (r["config"] == config_name and r["pool"] == pool_name):
            vals.append(r[metric])
    return vals


print(f"\n{'Config':<28} {'Pool':<12} {'H@10':>18} {'MRR':>18}")
print("─" * 80)
for config_name in ["+pubs_area", "+pubs_area_mlp_scorer"]:
    for pool_name, _ in POOLS:
        h10 = agg(config_name, pool_name, "Hits@10")
        mrr = agg(config_name, pool_name, "MRR")
        if not h10:
            continue
        print(f"{config_name:<28} {pool_name:<12} "
              f"{np.mean(h10):>7.3f}±{np.std(h10):<5.3f}  (n={len(h10)})   "
              f"{np.mean(mrr):>7.3f}±{np.std(mrr):<5.3f}")


# Save new seeds separately
with open(f"{OUT_DIR}/stage2p5_verify.tsv", "w") as f:
    cols = ["config", "seed", "pool", "N",
            "Hits@10", "Hits@50", "MRR", "AUC", "AP", "val_MRR"]
    f.write("\t".join(cols) + "\n")
    for r in results:
        f.write("\t".join(str(r[c]) for c in cols) + "\n")
print(f"\n✅ saved new seeds: {OUT_DIR}/stage2p5_verify.tsv")

print("\n" + "═" * 62)
print("STAGE 2.5b COMPLETE")
print("═" * 62)