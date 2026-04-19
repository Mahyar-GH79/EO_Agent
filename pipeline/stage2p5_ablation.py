"""
STAGE 2.5 — Cumulative Ablation over 11 Levers
==============================================
Starting from the Stage 2 baseline (loaded from stage2_results.tsv), apply
11 levers one at a time. Each change is KEPT if it improves mean val MRR
over the current best configuration, otherwise DROPPED.

Levers (cumulative, in order):
    1.  +pubs_uses      Publication nodes + USES_DATASET           (hetero only)
    2.  +pubs_cites     Paper→CITES→Paper                          (hetero only)
    3.  +pubs_area      Paper→HAS_APPLIEDRESEARCHAREA              (hetero only)
    4.  +edge_weights   log1p-weighted co-usage edges (hetero only)
    5.  +3_layers       2 → 3 layers
    6.  +residual       Residual connections
    7.  +layer_concat   Concat outputs of all layers
    8.  +neg_5x         1:1 → 5:1 training negatives
    9.  +bpr_loss       BCE → BPR
    10. +mlp_scorer     Dot product → MLP pair scorer
    11. +ensemble       α · GNN + (1-α) · Adamic-Adar, α tuned on val

Outputs:
    ./neurips_figs/stage2p5_ablation.tsv         per-seed records
    ./neurips_figs/stage2p5_summary.tsv          aggregated mean±std
    ./neurips_figs/tab_stage2p5_ablation.tex     paper-ready LaTeX
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
PATIENCE = 30
NEG_ALPHA = 0.75
SEEDS = [0, 1, 2]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STAGE2_RESULTS_TSV = f"{OUT_DIR}/stage2_results.tsv"

print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# 1. Load artifacts
# ══════════════════════════════════════════════════════════════
def _load_pairs(path):
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            a, b, w = line.rstrip("\n").split("\t")
            out.append((a, b, int(w)))
    return out


def _load_eval_pool(path):
    out = []
    with open(path) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            a, b_pos = parts[0], parts[1]
            b_negs = parts[2].split(",") if len(parts) > 2 and parts[2] else []
            out.append((a, b_pos, b_negs))
    return out


print("\n" + "═" * 62)
print("Loading artifacts")
print("═" * 62)

train_pairs = _load_pairs(f"{OUT_DIR}/cousage_edges_train.tsv")
val_pairs   = _load_pairs(f"{OUT_DIR}/cousage_edges_val.tsv")
eval_all    = _load_eval_pool(f"{OUT_DIR}/eval_pool_all.tsv")
eval_cold   = _load_eval_pool(f"{OUT_DIR}/eval_pool_cold_start.tsv")
eval_cross  = _load_eval_pool(f"{OUT_DIR}/eval_pool_cross_daac.tsv")

ds_list = [l.strip() for l in open(f"{EMB_DIR}/dataset_ids.txt")]
ds_idx = {d: i for i, d in enumerate(ds_list)}
N_DS = len(ds_list)
specter = np.load(f"{EMB_DIR}/specter2.npy").astype(np.float32)

print(f"   train pairs: {len(train_pairs):,}  val: {len(val_pairs):,}")
print(f"   eval: all={len(eval_all):,}  cold={len(eval_cold):,}  cross={len(eval_cross):,}")
print(f"   SPECTER2: {specter.shape}")

# Degree / negative sampling distributions
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


# ══════════════════════════════════════════════════════════════
# 2. Graph builders
# ══════════════════════════════════════════════════════════════
def build_homo_edge_index():
    rows, cols = [], []
    for a, b, _ in train_pairs:
        if a in ds_idx and b in ds_idx:
            i, j = ds_idx[a], ds_idx[b]
            rows += [i, j]; cols += [j, i]
    return torch.tensor([rows, cols], dtype=torch.long)


HOMO_EI = build_homo_edge_index()
print(f"   homo edge_index: {HOMO_EI.shape}")


def build_hetero_data(include_pubs_uses=False,
                      include_pubs_cites=False,
                      include_pubs_area=False,
                      cousage_weighted=False):
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
    paper_cites_paper = []
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
        elif e == "USES_DATASET" and include_pubs_uses:
            if lu == "Publication" and lv == "Dataset" and v in kept_datasets:
                paper_uses_ds[u].add(v)
            elif lv == "Publication" and lu == "Dataset" and u in kept_datasets:
                paper_uses_ds[v].add(u)
        elif e == "CITES" and include_pubs_cites:
            if lu == "Publication" and lv == "Publication":
                paper_cites_paper.append((u, v))
        elif e == "HAS_APPLIEDRESEARCHAREA" and include_pubs_area:
            if lu == "Publication" and lv == "ScienceKeyword":
                paper_area[u].add(v)
            elif lv == "Publication" and lu == "ScienceKeyword":
                paper_area[v].add(u)

    kept_papers = set(paper_uses_ds.keys())
    paper_cites_filtered = [(u, v) for u, v in paper_cites_paper
                            if u in kept_papers and v in kept_papers]
    paper_area_filtered = {p: kws for p, kws in paper_area.items()
                           if p in kept_papers}

    kept_platforms, kept_keywords, kept_projects = set(), set(), set()
    for d in kept_datasets:
        kept_platforms |= ds_platform.get(d, set())
        kept_keywords |= ds_keyword.get(d, set())
        kept_projects |= ds_project.get(d, set())
    for kws in paper_area_filtered.values():
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
    if include_pubs_uses:
        data["Publication"].num_nodes = len(pub_list)

    cu_src, cu_dst, cu_w = [], [], []
    for a, b, w in train_pairs:
        if a in ds_idx and b in ds_idx:
            i, j = ds_idx[a], ds_idx[b]
            cu_src += [i, j]; cu_dst += [j, i]
            cu_w += [float(w), float(w)]
    data["Dataset", "co_usage", "Dataset"].edge_index = \
        torch.tensor([cu_src, cu_dst], dtype=torch.long)
    if cousage_weighted:
        data["Dataset", "co_usage", "Dataset"].edge_weight = \
            torch.log1p(torch.tensor(cu_w, dtype=torch.float32))

    def _add(ns, rel, nt, edges):
        data[ns, rel, nt].edge_index = _tt(edges)
        data[nt, "rev_" + rel, ns].edge_index = \
            data[ns, rel, nt].edge_index.flip(0)

    _add("Dataset", "has_platform",   "Platform",       ds_plat)
    _add("Dataset", "has_keyword",    "ScienceKeyword", ds_kw)
    _add("Dataset", "of_project",     "Project",        ds_proj)
    _add("Platform", "has_instrument", "Instrument",    plat_inst)
    _add("ScienceKeyword", "has_subcategory",
         "ScienceKeyword", kw_sub)

    n_pub_uses = n_pub_cites = n_pub_area = 0
    if include_pubs_uses:
        paper_ds_edges = [(pub_idx[p], ds_idx[d])
                          for p, dset in paper_uses_ds.items()
                          for d in dset if d in ds_idx]
        _add("Publication", "uses_dataset", "Dataset", paper_ds_edges)
        n_pub_uses = len(paper_ds_edges)
    if include_pubs_cites:
        cite_pairs = [(pub_idx[u], pub_idx[v]) for u, v in paper_cites_filtered
                      if u in pub_idx and v in pub_idx]
        _add("Publication", "cites", "Publication", cite_pairs)
        n_pub_cites = len(cite_pairs)
    if include_pubs_area:
        area_edges = [(pub_idx[p], sk_idx[k])
                      for p, kws in paper_area_filtered.items()
                      for k in kws if k in sk_idx]
        _add("Publication", "has_area", "ScienceKeyword", area_edges)
        n_pub_area = len(area_edges)

    print(f"      nodes: DS={N_DS} KW={len(sk_list)} INST={len(inst_list)} "
          f"PLAT={len(plat_list)} PROJ={len(proj_list)} PUB={len(pub_list)}")
    print(f"      edges: DS-PLAT={len(ds_plat)} DS-KW={len(ds_kw)} "
          f"DS-PROJ={len(ds_proj)} PLAT-INST={len(plat_inst)} KW-SUB={len(kw_sub)} "
          f"PUB-USES={n_pub_uses} PUB-CITES={n_pub_cites} PUB-AREA={n_pub_area}")
    return data


# ══════════════════════════════════════════════════════════════
# 3. Models
# ══════════════════════════════════════════════════════════════
from torch_geometric.nn import SAGEConv, HeteroConv


class HomoGNN(nn.Module):
    def __init__(self, in_dim, hidden, n_layers=2,
                 residual=False, layer_concat=False):
        super().__init__()
        self.layer_concat = layer_concat
        self.residual = residual
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(n_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.dropout = 0.2

    def forward(self, x, edge_index):
        h = x
        outs = []
        for i, conv in enumerate(self.convs):
            new_h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                new_h = F.relu(new_h)
                new_h = F.dropout(new_h, p=self.dropout, training=self.training)
            if self.residual and h.shape[-1] == new_h.shape[-1]:
                new_h = new_h + h
            h = new_h
            outs.append(h)
        if self.layer_concat:
            return torch.cat(outs, dim=-1)
        return h


class HeteroGNN(nn.Module):
    def __init__(self, data, hidden, n_layers=2,
                 residual=False, layer_concat=False):
        super().__init__()
        self.layer_concat = layer_concat
        self.residual = residual
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
            conv_dict = {et: SAGEConv((hidden, hidden), hidden)
                         for et in data.edge_types}
            self.convs.append(HeteroConv(conv_dict, aggr="sum"))
        self.dropout = 0.2

    def forward(self, data):
        x_dict = {"Dataset": self.ds_proj(data["Dataset"].x)}
        for t, emb in self.embs.items():
            x_dict[t] = emb.weight
        ei_dict = {k: v.edge_index for k, v in data.edge_items()}
        outs_ds = []
        for i, conv in enumerate(self.convs):
            new_x = conv(x_dict, ei_dict)
            if i < len(self.convs) - 1:
                new_x = {k: F.relu(v) for k, v in new_x.items()}
                new_x = {k: F.dropout(v, p=self.dropout, training=self.training)
                         for k, v in new_x.items()}
            if self.residual:
                for k in new_x:
                    if k in x_dict and x_dict[k].shape == new_x[k].shape:
                        new_x[k] = new_x[k] + x_dict[k]
            x_dict = new_x
            outs_ds.append(x_dict["Dataset"])
        if self.layer_concat:
            return torch.cat(outs_ds, dim=-1)
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


# ══════════════════════════════════════════════════════════════
# 4. Training utilities
# ══════════════════════════════════════════════════════════════
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


# Val pool for early stopping (20 negatives, Stage 2 protocol)
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
print(f"   val pool: {len(val_pool):,} pairs")


# Adamic-Adar for ensembling
import networkx as nx
_G_train = nx.Graph()
_G_train.add_nodes_from(ds_list)
for a, b, w in train_pairs:
    if a in ds_idx and b in ds_idx:
        _G_train.add_edge(a, b, weight=w)


def aa_score_pair(a, b):
    if a not in _G_train or b not in _G_train:
        return 0.0
    na = set(_G_train.neighbors(a))
    nb = set(_G_train.neighbors(b))
    s = 0.0
    for c in (na & nb):
        deg_c = _G_train.degree(c)
        if deg_c > 1:
            s += 1.0 / np.log(deg_c)
    return s


# ══════════════════════════════════════════════════════════════
# 5. Training loop
# ══════════════════════════════════════════════════════════════
def train_one_run(variant, seed, cfg, hetero_data=None):
    """Train one (variant, seed, cfg) run. Returns:
        {'emb': np.ndarray, 'scorer': callable, 'val_mrr': float}"""
    set_seed(seed)
    rng_np = np.random.default_rng(seed)

    use_mlp  = cfg.get("mlp_scorer",   False)
    use_bpr  = cfg.get("bpr_loss",     False)
    neg_k    = 5 if cfg.get("neg_5x",  False) else 1
    n_layers = 3 if cfg.get("3_layers", False) else 2
    residual = cfg.get("residual",     False)
    lconcat  = cfg.get("layer_concat", False)

    if variant == "homo":
        ei = HOMO_EI.to(DEVICE)
        x  = torch.from_numpy(specter).to(DEVICE)
        model = HomoGNN(specter.shape[1], HIDDEN, n_layers=n_layers,
                        residual=residual, layer_concat=lconcat).to(DEVICE)
    else:
        data = hetero_data.to(DEVICE)
        model = HeteroGNN(data, HIDDEN, n_layers=n_layers,
                          residual=residual, layer_concat=lconcat).to(DEVICE)

    final_dim = HIDDEN * n_layers if lconcat else HIDDEN
    scorer = MLPScorer(final_dim).to(DEVICE) if use_mlp else None
    params = list(model.parameters())
    if scorer is not None:
        params += list(scorer.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)

    def fwd():
        if variant == "homo":
            h = model(x, ei)
        else:
            h = model(data)
        return F.normalize(h, p=2, dim=-1)

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
        src, pos, neg = sample_train_batch(rng_np, neg_per_pos=neg_k)
        src = src.to(DEVICE); pos = pos.to(DEVICE); neg = neg.to(DEVICE)
        pos_score = pair_score(h, src, pos)

        if scorer is not None:
            N_pos = src.size(0)
            src_rep = src.unsqueeze(1).expand(-1, neg_k).reshape(-1)
            neg_flat = neg.reshape(-1)
            neg_score = pair_score(h, src_rep, neg_flat).view(N_pos, neg_k)
        else:
            ha = h[src].unsqueeze(1).expand(-1, neg_k, -1)
            hb = h[neg]
            neg_score = (ha * hb).sum(dim=-1)

        if use_bpr:
            diff = pos_score.unsqueeze(-1) - neg_score
            loss = -F.logsigmoid(diff).mean()
        else:
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

    return {"emb": h_final, "scorer": final_scorer, "val_mrr": best_mrr}


# ══════════════════════════════════════════════════════════════
# 6. Ensembling
# ══════════════════════════════════════════════════════════════
def evaluate_pool_ensemble(emb_np, gnn_scorer, pool, alpha):
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
        ea = emb_np[np.full_like(cand_i, a_i)]
        eb = emb_np[cand_i]
        g = gnn_scorer(ea, eb)
        aa = np.array([aa_score_pair(a, c) for c in cands])
        def z(x):
            s = x.std()
            return (x - x.mean()) / (s if s > 1e-8 else 1.0)
        score = alpha * z(g) + (1 - alpha) * z(aa)
        order = np.argsort(-score + rng.uniform(-1e-12, 1e-12, size=score.shape))
        rank = int(np.where(order == 0)[0][0]) + 1
        ranks.append(rank)
        all_scores.extend(score.tolist())
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


def tune_alpha_on_val(emb_np, gnn_scorer):
    best_a, best_m = 1.0, -1.0
    for alpha in np.linspace(0, 1, 11):
        m = evaluate_pool_ensemble(emb_np, gnn_scorer, val_pool, alpha)
        if m["MRR"] > best_m:
            best_m = m["MRR"]; best_a = float(alpha)
    return best_a, best_m


# ══════════════════════════════════════════════════════════════
# 7. Records / state
# ══════════════════════════════════════════════════════════════
def load_stage2_baseline_records():
    rows = []
    with open(STAGE2_RESULTS_TSV) as f:
        header = next(f).rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, parts)))
    recs = []
    for r in rows:
        v = "homo" if "Homo" in r["variant"] else "hetero"
        recs.append({
            "config":  "baseline",
            "variant": v,
            "seed":    int(r["seed"]),
            "pool":    r["pool"],
            "Hits@10": float(r["Hits@10"]),
            "Hits@50": float(r["Hits@50"]),
            "MRR":     float(r["MRR"]),
            "AUC":     float(r["AUC"]),
            "AP":      float(r["AP"]),
            "N":       int(r["N"]),
            "val_MRR": None,
            "kept":    True,
            "alpha":   None,
        })
    return recs


records = load_stage2_baseline_records()
print(f"   loaded {len(records)} baseline records from Stage 2")

POOLS = [("all", eval_all), ("cold_start", eval_cold), ("cross_daac", eval_cross)]
METRIC_KEYS = ["Hits@10", "Hits@50", "MRR", "AUC", "AP"]


def mean_val_mrr(variant, config_name):
    vs = [r["val_MRR"] for r in records
          if r["variant"] == variant and r["config"] == config_name
          and r["pool"] == "all" and r["val_MRR"] is not None]
    return float(np.mean(vs)) if len(vs) == 3 else None


def mean_metric(variant, config_name, pool, metric):
    vs = [r[metric] for r in records
          if r["variant"] == variant and r["config"] == config_name
          and r["pool"] == pool]
    return float(np.mean(vs)) if vs else None


def std_metric(variant, config_name, pool, metric):
    vs = [r[metric] for r in records
          if r["variant"] == variant and r["config"] == config_name
          and r["pool"] == pool]
    return float(np.std(vs)) if vs else None


def last_kept_config(variant):
    kept_configs = []
    seen = set()
    for r in records:
        if r["variant"] == variant and r["kept"] and r["config"] not in seen:
            kept_configs.append(r["config"])
            seen.add(r["config"])
    return kept_configs[-1] if kept_configs else "baseline"


# ══════════════════════════════════════════════════════════════
# 8. Lever list + helpers that depend on the functions above
# ══════════════════════════════════════════════════════════════
# (display_name, flag_name, hetero_only)
LEVERS = [
    ("+pubs_uses",    "pubs_uses",    True),
    ("+pubs_cites",   "pubs_cites",   True),
    ("+pubs_area",    "pubs_area",    True),
    ("+edge_weights", "edge_weights", True),  # only affects hetero co_usage edges
    ("+3_layers",     "3_layers",     False),
    ("+residual",     "residual",     False),
    ("+layer_concat", "layer_concat", False),
    ("+neg_5x",       "neg_5x",       False),
    ("+bpr_loss",     "bpr_loss",     False),
    ("+mlp_scorer",   "mlp_scorer",   False),
    ("+ensemble",     "ensemble",     False),
]

kept_levers = {"homo": set(), "hetero": set()}

# emb_cache[variant][config_name][seed] = {'emb','scorer','val_mrr'}
emb_cache = {"homo": {}, "hetero": {}}


def cfg_from_flags(flag_set):
    return {name: (name in flag_set) for _, name, _ in LEVERS}


def rebuild_hetero(flag_set):
    return build_hetero_data(
        include_pubs_uses=("pubs_uses"    in flag_set),
        include_pubs_cites=("pubs_cites"  in flag_set),
        include_pubs_area=("pubs_area"    in flag_set),
        cousage_weighted=("edge_weights"  in flag_set),
    )


def run_config(variant, config_name, flag_set):
    print(f"   [{variant}] config={config_name}  flags={sorted(flag_set)}")
    hdata = rebuild_hetero(flag_set) if variant == "hetero" else None
    cfg = cfg_from_flags(flag_set)
    emb_cache[variant][config_name] = {}
    for seed in SEEDS:
        t0 = time.time()
        out = train_one_run(variant, seed, cfg, hetero_data=hdata)
        dt = time.time() - t0
        emb_cache[variant][config_name][seed] = out
        for pool_name, pool in POOLS:
            m = evaluate_pool(out["emb"], pool, scorer_fn=out["scorer"])
            records.append({
                "config":  config_name,
                "variant": variant,
                "seed":    seed,
                "pool":    pool_name,
                **m,
                "val_MRR": out["val_mrr"],
                "kept":    None,
                "alpha":   None,
            })
        print(f"      seed {seed}: {dt:5.1f}s  val MRR={out['val_mrr']:.3f}")


def run_ensemble(variant, config_name):
    base = last_kept_config(variant)
    # We need cached embeddings from the base config. If not cached (e.g.
    # base is 'baseline' loaded from Stage 2), retrain the current-best
    # flag-set under a fresh name so we have emb/scorer in memory.
    if base not in emb_cache[variant]:
        print(f"   [{variant}] ensemble: base '{base}' not cached; retraining …")
        fs = set(kept_levers[variant])
        run_config(variant, base + "_retrain", fs)
        base = base + "_retrain"

    for seed in SEEDS:
        cached = emb_cache[variant][base][seed]
        emb_np = cached["emb"]; gnn_scorer = cached["scorer"]
        alpha, val_mrr = tune_alpha_on_val(emb_np, gnn_scorer)
        for pool_name, pool in POOLS:
            m = evaluate_pool_ensemble(emb_np, gnn_scorer, pool, alpha)
            records.append({
                "config":  config_name,
                "variant": variant,
                "seed":    seed,
                "pool":    pool_name,
                **m,
                "val_MRR": val_mrr,
                "kept":    True,   # ensemble is post-hoc, always included
                "alpha":   alpha,
            })
        print(f"      seed {seed}: val MRR={val_mrr:.3f}  α={alpha:.2f}")


# ══════════════════════════════════════════════════════════════
# 9. Main sweep
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("CUMULATIVE ABLATION SWEEP")
print("═" * 62)

for config_idx, (display_name, flag_name, hetero_only) in enumerate(LEVERS, start=1):
    print(f"\n─── Config {config_idx}/{len(LEVERS)}: {display_name} "
          f"{'(hetero only)' if hetero_only else ''} ───")

    for variant in ["homo", "hetero"]:
        if hetero_only and variant == "homo":
            print(f"   [skip] {variant} unaffected by {display_name}")
            continue

        if flag_name == "ensemble":
            run_ensemble(variant, display_name)
            for pool_name, _ in POOLS:
                h = mean_metric(variant, display_name, pool_name, "Hits@10")
                print(f"      {variant:<6} {pool_name:<11} H@10={h:.3f}")
            continue

        trial_flags = set(kept_levers[variant]) | {flag_name}
        run_config(variant, display_name, trial_flags)

        prev_best = last_kept_config(variant)
        prev_val = mean_val_mrr(variant, prev_best)
        cur_val = mean_val_mrr(variant, display_name)

        if prev_val is None:
            # Stage 2 baseline has no val_MRR; auto-accept first lever
            kept = True
            delta_str = "(first)"
        else:
            kept = cur_val > prev_val + 1e-3
            delta_str = f"Δ={cur_val - prev_val:+.4f}"

        for r in records:
            if r["variant"] == variant and r["config"] == display_name:
                r["kept"] = kept

        if kept:
            kept_levers[variant].add(flag_name)
        else:
            emb_cache[variant].pop(display_name, None)

        tag = "✓ KEEP" if kept else "✗ DROP"
        print(f"   [{variant}] val MRR={cur_val:.3f}  {delta_str}  {tag}")
        for pool_name, _ in POOLS:
            h = mean_metric(variant, display_name, pool_name, "Hits@10")
            print(f"       {pool_name:<11} H@10={h:.3f}")


# ══════════════════════════════════════════════════════════════
# 10. Save outputs
# ══════════════════════════════════════════════════════════════
print("\n" + "═" * 62)
print("Saving outputs")
print("═" * 62)

with open(f"{OUT_DIR}/stage2p5_ablation.tsv", "w") as f:
    cols = ["config", "variant", "seed", "pool", "N",
            "Hits@10", "Hits@50", "MRR", "AUC", "AP",
            "val_MRR", "kept", "alpha"]
    f.write("\t".join(cols) + "\n")
    for r in records:
        f.write("\t".join("" if r.get(c) is None else str(r[c])
                          for c in cols) + "\n")
print(f"   ✅ {OUT_DIR}/stage2p5_ablation.tsv")

summary = []
seen = set()
configs_seen_order = []
for r in records:
    key = (r["variant"], r["config"])
    if key not in seen:
        seen.add(key); configs_seen_order.append(key)
for variant, config in configs_seen_order:
    for pool_name, _ in POOLS:
        row = {"variant": variant, "config": config, "pool": pool_name,
               "kept": any(r["kept"] for r in records
                           if r["variant"] == variant and r["config"] == config)}
        for k in METRIC_KEYS:
            row[f"{k}_mean"] = mean_metric(variant, config, pool_name, k)
            row[f"{k}_std"] = std_metric(variant, config, pool_name, k)
        summary.append(row)

with open(f"{OUT_DIR}/stage2p5_summary.tsv", "w") as f:
    cols = ["variant", "config", "pool", "kept"] + \
           [f"{k}_mean" for k in METRIC_KEYS] + \
           [f"{k}_std" for k in METRIC_KEYS]
    f.write("\t".join(cols) + "\n")
    for r in summary:
        f.write("\t".join("" if r.get(c) is None else str(r[c])
                          for c in cols) + "\n")
print(f"   ✅ {OUT_DIR}/stage2p5_summary.tsv")


print("\n▶ Writing LaTeX ablation table …")
tex = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{Cumulative ablation on GNN-Homo and GNN-Hetero. Each row "
    r"adds the listed lever on top of previously-kept levers. A lever is "
    r"kept (\checkmark) iff mean val MRR improves by $>10^{-3}$. "
    r"Hits@10, mean $\pm$ std over 3 seeds.}",
    r"\label{tab:stage2p5_ablation}",
    r"\small",
    r"\begin{tabular}{llccc}",
    r"\toprule",
    r"\textbf{Variant} & \textbf{Config} & \textbf{all} & "
    r"\textbf{cold\_start} & \textbf{cross\_daac} \\",
    r"\midrule",
]
config_order = ["baseline"] + [name for name, _, _ in LEVERS]
for variant in ["homo", "hetero"]:
    first = True
    for config in config_order:
        if not any(r["variant"] == variant and r["config"] == config for r in records):
            continue
        kept_flag = any(r["kept"] for r in records
                        if r["variant"] == variant and r["config"] == config)
        mark = r"$\checkmark$" if kept_flag else ""
        var_cell = (r"\multirow{*}{\texttt{" + variant + r"}}" if first else "")
        first = False
        cells = []
        for pool_name, _ in POOLS:
            m = mean_metric(variant, config, pool_name, "Hits@10")
            s = std_metric(variant, config, pool_name, "Hits@10")
            cells.append(f"{m:.3f}{{\\scriptsize$\\pm${s:.3f}}}"
                         if m is not None else "--")
        tex.append(f"{var_cell} & \\texttt{{{config}}}~{mark} & "
                   + " & ".join(cells) + r" \\")
    tex.append(r"\midrule")
tex = tex[:-1]
tex += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\end{table}",
    "",
]
with open(f"{OUT_DIR}/tab_stage2p5_ablation.tex", "w") as f:
    f.write("\n".join(tex))
print(f"   ✅ {OUT_DIR}/tab_stage2p5_ablation.tex")


# Final pretty summary
print("\n" + "═" * 62)
print("FINAL ABLATION SUMMARY (Hits@10)")
print("═" * 62)
print(f"\n{'Variant':<8} {'Config':<20} "
      f"{'all':>16} {'cold_start':>16} {'cross_daac':>16}")
for variant in ["homo", "hetero"]:
    for config in config_order:
        if not any(r["variant"] == variant and r["config"] == config for r in records):
            continue
        kept_flag = any(r["kept"] for r in records
                        if r["variant"] == variant and r["config"] == config)
        cells = []
        for pool_name, _ in POOLS:
            m = mean_metric(variant, config, pool_name, "Hits@10")
            s = std_metric(variant, config, pool_name, "Hits@10")
            cells.append(f"{m:>7.3f}±{s:<5.3f}" if m is not None else "     --     ")
        tag = "✓" if kept_flag else " "
        print(f"{variant:<8} {config:<20}{tag} " + "  ".join(cells))

print("\nFinal kept levers per variant:")
for v in ["homo", "hetero"]:
    lst = sorted(kept_levers[v])
    print(f"   {v:<8}: {lst if lst else '(none — baseline wins)'}")

print("\n" + "═" * 62)
print("STAGE 2.5 COMPLETE")
print("═" * 62)