"""
STAGE 5 FACTORIAL — 2×2×2 Multi-Agent Hypothesis Pipeline
=========================================================
Three agents, each either GPT-5.2 or Claude Sonnet 4.6:

    Agent 1 (pair scorer):     rates 800 pairs on plausibility + novelty
    Agent 2 (hypothesis gen):  writes a hypothesis for a pair
    Agent 3 (hypothesis judge): rates hypotheses on imp/tract/novel

Produces 8 experiments (2³) covering every (a1, a2, a3) combo.

USAGE
-----
    # Entire pipeline end-to-end (each task will skip if already done)
    python3 stage5_factorial.py --live

    # Just one task at a time, for debugging / cost control
    python3 stage5_factorial.py --task agent1_claude --live
    python3 stage5_factorial.py --task select_top40 --live
    python3 stage5_factorial.py --task generate --live
    python3 stage5_factorial.py --task validate --live
    python3 stage5_factorial.py --task analyze

    # Dry run (always inspects + estimates cost; no API calls)
    python3 stage5_factorial.py                 # dry-run all
    python3 stage5_factorial.py --task generate # dry-run one task

PERSISTENCE
-----------
Every API call writes its result to disk immediately. Re-running after a
crash resumes from the last saved state. Cost of a crash = cost of at
most one incomplete call.

OUTPUTS
-------
./neurips_figs/stage5_factorial/
    agent1_gpt_scores.tsv          (from Stage 4 — already exists)
    agent1_claude_scores.tsv       (NEW, ~$8.80)
    selected_top40_gpt.tsv
    selected_top40_claude.tsv
    experiments/{a1}_{a2}_{a3}/hypotheses.tsv
    experiments/{a1}_{a2}_{a3}/validations.tsv
    final_factorial_results.tsv
    inter_rater_agreement.tsv

ENV VARS
--------
    OPENAI_API_KEY
    ANTHROPIC_API_KEY
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from typing import Any

# ══════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════
GPT_MODEL = "gpt-5.2"
CLAUDE_MODEL = "claude-sonnet-4-6"

OUT_DIR = "./neurips_figs"
STAGE4_DIR = f"{OUT_DIR}/stage4"
FACT_DIR = f"{OUT_DIR}/stage5_factorial"
PROMPTS_DIR = f"{FACT_DIR}/prompts"
RESPONSES_DIR = f"{FACT_DIR}/responses"
EXP_DIR = f"{FACT_DIR}/experiments"
os.makedirs(PROMPTS_DIR, exist_ok=True)
os.makedirs(RESPONSES_DIR, exist_ok=True)
os.makedirs(EXP_DIR, exist_ok=True)

TOP_K = 40
MAX_PER_SN = 2
SEED = 42

# Pricing (per 1M tokens)
PRICE = {
    "gpt":    {"in": 0.875,  "out": 7.00,  "batch_discount": 0.5},   # batch = 50% off
    "claude": {"in": 3.00,   "out": 15.00, "batch_discount": 1.0},   # no batch
}

# Token estimates (tuned conservatively; actual usage may be lower)
EST = {
    "score":    {"in": 2000, "out": 300},  # Agent 1
    "generate": {"in": 2500, "out": 700},  # Agent 2
    "judge":    {"in": 1200, "out": 400},  # Agent 3
}

EXPERIMENTS = [(a1, a2, a3) for a1 in ("gpt", "claude")
                             for a2 in ("gpt", "claude")
                             for a3 in ("gpt", "claude")]


# ══════════════════════════════════════════════════════════════
# Prompts — identical across judges/generators for fairness
# ══════════════════════════════════════════════════════════════
AGENT1_SYSTEM = (
    "You are an expert Earth-science research scientist. Your job is to "
    "evaluate whether two Earth-observation datasets could plausibly be "
    "combined in a single scientific study. Base your judgment on the "
    "datasets' scientific content as described in their abstracts, not on "
    "whether you have seen them combined before in published work. "
    "Respond with JSON only, in the exact schema requested."
)

AGENT1_USER = """Please evaluate the following pair of Earth-observation datasets.

Dataset A:
  Short name: {sn_a}
  Long name:  {ln_a}
  Abstract:   {abs_a}

Dataset B:
  Short name: {sn_b}
  Long name:  {ln_b}
  Abstract:   {abs_b}

Rate this pairing on two axes:

1. SCIENTIFIC PLAUSIBILITY (1-5): Could a research team reasonably combine
   these datasets in a single scientific study?
     1 = implausible   5 = highly plausible

2. NOVELTY (1-5): How novel or non-obvious is this combination?
     1 = obvious/trivial   5 = highly novel

Respond with JSON only:
{{
  "plausibility": <1-5 integer>,
  "novelty":      <1-5 integer>,
  "rationale":    "<2-3 sentence explanation>"
}}"""


AGENT2_SYSTEM = (
    "You are an expert Earth-observation research scientist. You will be "
    "given two NASA datasets. Generate ONE concrete, publishable research "
    "hypothesis that could be tested by combining them. Focus on scientific "
    "content, not data-availability caveats. Respond with JSON only."
)

AGENT2_USER = """Dataset A:
  Short name: {sn_a}
  Long name:  {ln_a}
  Abstract:   {abs_a}

Dataset B:
  Short name: {sn_b}
  Long name:  {ln_b}
  Abstract:   {abs_b}

Generate one specific, testable research hypothesis that combines these
two datasets. The hypothesis must be concrete enough that a team could
plan an actual study, not a vague "one could investigate X" statement.

Respond with JSON only:
{{
  "research_question":     "<one clear sentence stating the scientific question>",
  "hypothesis":            "<specific, testable hypothesis in 1-2 sentences>",
  "analysis_method":       "<how the datasets test the hypothesis, 1-2 sentences>",
  "expected_finding":      "<what would support the hypothesis, 1 sentence>",
  "scientific_importance": "<why answering this matters, 1-2 sentences>",
  "domain":                "<primary domain, 1-3 words>"
}}"""


AGENT3_SYSTEM_BLIND = (
    "You are a senior Earth-observation research scientist reviewing a "
    "proposed research hypothesis. Judge the hypothesis on its scientific "
    "merit alone; you do not need to verify data availability. "
    "Respond with JSON only."
)

AGENT3_USER_BLIND = """Here is a proposed Earth-science research hypothesis:

Research question:    {research_question}
Hypothesis:           {hypothesis}
Proposed method:      {analysis_method}
Expected finding:     {expected_finding}
Scientific importance: {scientific_importance}
Domain:               {domain}

Rate 1-5:
  IMPORTANCE:   1 = trivial, 5 = meaningfully advances the field
  TRACTABILITY: 1 = needs breakthroughs, 5 = ready to execute
  NOVELTY:      1 = already well-studied, 5 = opens new line of inquiry

Respond with JSON only:
{{
  "importance":   <1-5>,
  "tractability": <1-5>,
  "novelty":      <1-5>,
  "rationale":    "<2-3 sentences>"
}}"""


AGENT3_SYSTEM_CTX = (
    "You are a senior Earth-observation research scientist reviewing a "
    "proposed research hypothesis along with the datasets it uses. Judge "
    "the hypothesis on scientific merit and the appropriateness of the "
    "dataset combination. Respond with JSON only."
)

AGENT3_USER_CTX = """Dataset A:
  Short name: {sn_a}
  Long name:  {ln_a}
  Abstract:   {abs_a}

Dataset B:
  Short name: {sn_b}
  Long name:  {ln_b}
  Abstract:   {abs_b}

Proposed research hypothesis:

Research question:    {research_question}
Hypothesis:           {hypothesis}
Proposed method:      {analysis_method}
Expected finding:     {expected_finding}
Scientific importance: {scientific_importance}
Domain:               {domain}

Rate 1-5:
  IMPORTANCE:   1 = trivial, 5 = meaningfully advances the field
  TRACTABILITY: 1 = needs breakthroughs, 5 = ready to execute
  NOVELTY:      1 = already well-studied, 5 = opens new line of inquiry

Respond with JSON only:
{{
  "importance":   <1-5>,
  "tractability": <1-5>,
  "novelty":      <1-5>,
  "rationale":    "<2-3 sentences>"
}}"""


def truncate(text, max_chars=1200):
    if not text: return "(no abstract provided)"
    text = text.strip().replace("\n", " ").replace("\r", " ")
    return text if len(text) <= max_chars else text[:max_chars] + " […truncated]"


def clean_claude_json(text: str) -> str:
    """Claude occasionally wraps JSON in code fences. Strip defensively."""
    t = text.strip()
    if t.startswith("```"):
        t = t.lstrip("`").lstrip()
        if t.lower().startswith("json"):
            t = t[4:].lstrip()
        if t.endswith("```"):
            t = t[:-3].rstrip()
    return t


# ══════════════════════════════════════════════════════════════
# API client wrappers (lazy import, keeps dry-run dependency-free)
# ══════════════════════════════════════════════════════════════
_clients = {"gpt": None, "claude": None}


def call_gpt(system: str, user: str, max_tokens: int = 500) -> dict:
    """Single real-time GPT call; returns parsed JSON dict."""
    if _clients["gpt"] is None:
        from openai import OpenAI
        _clients["gpt"] = OpenAI()
    client = _clients["gpt"]
    resp = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        response_format={"type": "json_object"},
        max_completion_tokens=max_tokens,
        temperature=0,
    )
    return json.loads(resp.choices[0].message.content)


def call_claude(system: str, user: str, max_tokens: int = 500) -> dict:
    """Single real-time Claude call; returns parsed JSON dict."""
    if _clients["claude"] is None:
        import anthropic
        _clients["claude"] = anthropic.Anthropic()
    client = _clients["claude"]
    resp = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return json.loads(clean_claude_json(resp.content[0].text))


def call_judge(judge_model: str, system: str, user: str, max_tokens: int = 500):
    if judge_model == "gpt":
        return call_gpt(system, user, max_tokens)
    return call_claude(system, user, max_tokens)


# ══════════════════════════════════════════════════════════════
# TASK 1 — Agent 1 = Claude on all 800 pairs
# ══════════════════════════════════════════════════════════════
AGENT1_GPT_PATH    = f"{FACT_DIR}/agent1_gpt_scores.tsv"
AGENT1_CLAUDE_PATH = f"{FACT_DIR}/agent1_claude_scores.tsv"


def task_agent1_gpt(live):
    """GPT scores already exist in stage4_judgments.tsv — just copy relevant cols."""
    if os.path.exists(AGENT1_GPT_PATH):
        print(f"  [skip] {AGENT1_GPT_PATH} already exists")
        return
    src = f"{STAGE4_DIR}/stage4_judgments.tsv"
    if not os.path.exists(src):
        print(f"  [error] {src} not found"); return
    rows = []
    with open(src) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    cols_keep = ["stratum", "pair_idx", "dataset_i", "dataset_j",
                 "shortName_i", "shortName_j", "longName_i", "longName_j",
                 "abstract_i", "abstract_j", "daac_i", "daac_j",
                 "score_dot", "score_mlp", "rank_dot", "rank_mlp",
                 "source_model", "plausibility", "novelty", "rationale"]
    with open(AGENT1_GPT_PATH, "w") as f:
        f.write("\t".join(cols_keep) + "\n")
        for r in rows:
            f.write("\t".join(str(r.get(c, "")).replace("\t", " ").replace("\n", " ")
                              for c in cols_keep) + "\n")
    print(f"  ✅ {AGENT1_GPT_PATH}  ({len(rows)} rows — from Stage 4)")


def task_agent1_claude(live):
    """Run Claude on every Stage-4 pair. Resumable, persists after each call."""
    if os.path.exists(AGENT1_CLAUDE_PATH):
        # Check for completeness
        with open(AGENT1_CLAUDE_PATH) as f:
            done = sum(1 for _ in f) - 1  # minus header
        # Expected 800
        expected = _count_stage4_pairs()
        if done >= expected:
            print(f"  [skip] {AGENT1_CLAUDE_PATH} complete ({done}/{expected})")
            return
        print(f"  resuming: {done}/{expected} already done")
    else:
        print(f"  starting fresh")

    # Load Stage 4 pairs as source of truth
    src_rows = []
    with open(f"{STAGE4_DIR}/stage4_judgments.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            src_rows.append(r)
    total = len(src_rows)

    cost_est = total * (EST["score"]["in"]  * PRICE["claude"]["in"]  +
                        EST["score"]["out"] * PRICE["claude"]["out"]) / 1e6
    print(f"  pairs: {total}  est cost: ${cost_est:.2f}")

    if not live:
        print("  [DRY RUN] skipping API calls")
        return

    # Load existing rows so we can skip
    already = {}
    if os.path.exists(AGENT1_CLAUDE_PATH):
        with open(AGENT1_CLAUDE_PATH) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                already[(r["stratum"], r["pair_idx"])] = r

    cols_out = ["stratum", "pair_idx", "dataset_i", "dataset_j",
                "shortName_i", "shortName_j", "longName_i", "longName_j",
                "abstract_i", "abstract_j", "daac_i", "daac_j",
                "score_dot", "score_mlp", "rank_dot", "rank_mlp",
                "source_model", "plausibility", "novelty", "rationale"]

    # Open in append-or-create mode; write header if fresh
    if not os.path.exists(AGENT1_CLAUDE_PATH):
        with open(AGENT1_CLAUDE_PATH, "w") as f:
            f.write("\t".join(cols_out) + "\n")

    with open(AGENT1_CLAUDE_PATH, "a", buffering=1) as outf:  # line-buffered
        t_start = time.time()
        for i, r in enumerate(src_rows, 1):
            key = (r["stratum"], r["pair_idx"])
            if key in already:
                continue
            user = AGENT1_USER.format(
                sn_a=r["shortName_i"] or "(unknown)",
                ln_a=r["longName_i"]  or "(unknown)",
                abs_a=truncate(r["abstract_i"]),
                sn_b=r["shortName_j"] or "(unknown)",
                ln_b=r["longName_j"]  or "(unknown)",
                abs_b=truncate(r["abstract_j"]),
            )
            try:
                v = call_claude(AGENT1_SYSTEM, user, max_tokens=400)
                plaus = int(v["plausibility"])
                novel = int(v["novelty"])
                rat   = str(v.get("rationale", ""))
                if not (1 <= plaus <= 5) or not (1 <= novel <= 5):
                    raise ValueError(f"out of range: {plaus}, {novel}")
                row_out = {**r, "plausibility": plaus, "novelty": novel,
                           "rationale": rat}
                outf.write("\t".join(str(row_out.get(c, "")).replace("\t", " ").replace("\n", " ")
                                     for c in cols_out) + "\n")
                if i % 20 == 0:
                    elapsed = time.time() - t_start
                    rate = i / elapsed
                    eta = (total - i) / rate / 60
                    print(f"    [{i}/{total}] rate={rate:.2f}/s  eta={eta:.1f}m")
            except Exception as e:
                # Persist error; don't halt
                err_path = f"{RESPONSES_DIR}/agent1_claude_{key[0]}_{key[1]}_ERROR.txt"
                with open(err_path, "w") as fe:
                    fe.write(f"ERROR: {e}\n")
                print(f"    [{i}/{total}] ERROR on {key}: {e}")

    # Report any gaps
    with open(AGENT1_CLAUDE_PATH) as f:
        done = sum(1 for _ in f) - 1
    print(f"  ✅ {AGENT1_CLAUDE_PATH}  ({done}/{total} complete)")


def _count_stage4_pairs():
    with open(f"{STAGE4_DIR}/stage4_judgments.tsv") as f:
        return sum(1 for _ in f) - 1


# ══════════════════════════════════════════════════════════════
# TASK 2 — Select top-40 pairs for each Agent 1
# ══════════════════════════════════════════════════════════════
def _select_top40(scored_path, out_path):
    """Apply the same tier logic as Stage 5a: top 7 tier1 + top 10 tier2 + top 23 tier3."""
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} exists")
        return
    rows = []
    with open(scored_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["stratum"] != "A":  # Only from predicted-novel
                continue
            r["plausibility"] = int(r["plausibility"])
            r["novelty"] = int(r["novelty"])
            rows.append(r)

    def score(r):
        return 0.55 * r["plausibility"] + 0.45 * r["novelty"]

    tier1 = sorted([r for r in rows if r["plausibility"] == 5 and r["novelty"] >= 3],
                   key=score, reverse=True)
    tier2 = sorted([r for r in rows if r["plausibility"] == 4 and r["novelty"] >= 4],
                   key=score, reverse=True)
    tier3 = sorted([r for r in rows if r["plausibility"] == 5 and r["novelty"] == 2],
                   key=score, reverse=True)

    selected = (tier1 + tier2 + tier3)[:TOP_K]
    # If not enough pairs in those tiers, fall back to plaus>=4 top-by-score
    if len(selected) < TOP_K:
        extra = sorted([r for r in rows
                        if r["plausibility"] >= 4
                        and r not in tier1 and r not in tier2 and r not in tier3],
                       key=score, reverse=True)
        selected += extra[:TOP_K - len(selected)]
    selected = selected[:TOP_K]

    print(f"  selected {len(selected)} pairs from {scored_path}")
    print(f"    tier1(p=5,n≥3): {len(tier1)}  "
          f"tier2(p=4,n≥4): {len(tier2)}  tier3(p=5,n=2): {len(tier3)}")

    cols = list(selected[0].keys())
    with open(out_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in selected:
            f.write("\t".join(str(r.get(c, "")).replace("\t", " ").replace("\n", " ")
                              for c in cols) + "\n")
    print(f"  ✅ {out_path}")


SELECT_GPT_PATH    = f"{FACT_DIR}/selected_top40_gpt.tsv"
SELECT_CLAUDE_PATH = f"{FACT_DIR}/selected_top40_claude.tsv"


def task_select_top40(live):
    _select_top40(AGENT1_GPT_PATH,    SELECT_GPT_PATH)
    if os.path.exists(AGENT1_CLAUDE_PATH):
        _select_top40(AGENT1_CLAUDE_PATH, SELECT_CLAUDE_PATH)
    else:
        print(f"  [warn] {AGENT1_CLAUDE_PATH} missing — run agent1_claude first")


# ══════════════════════════════════════════════════════════════
# TASK 3 — Hypothesis generation, once per (a1, a2) pair
# ══════════════════════════════════════════════════════════════
def task_generate(live):
    """Generate hypotheses for each (a1, a2) combination.
    a1 determines which 40 pairs; a2 determines who writes the hypothesis.
    Produces 4 hypothesis files, each reused by 2 of the 8 experiments."""
    combos = [("gpt", "gpt"), ("gpt", "claude"),
              ("claude", "gpt"), ("claude", "claude")]

    for a1, a2 in combos:
        src = SELECT_GPT_PATH if a1 == "gpt" else SELECT_CLAUDE_PATH
        out_path = f"{FACT_DIR}/hypotheses_{a1}_{a2}.tsv"
        if not os.path.exists(src):
            print(f"  [skip {a1}_{a2}] {src} missing")
            continue
        _generate_hypotheses(src, out_path, a2, live)


def _generate_hypotheses(src, out_path, agent: str, live):
    # Load source pairs
    pairs = []
    with open(src) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            pairs.append(r)

    # Already-done
    done_keys = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                done_keys.add(r.get("orig_pair_key", ""))

    todo = [(i, r) for i, r in enumerate(pairs)
            if f"{r['stratum']}_{r['pair_idx']}" not in done_keys]
    cost_est = len(todo) * (EST["generate"]["in"]  * PRICE[agent]["in"] +
                            EST["generate"]["out"] * PRICE[agent]["out"]) / 1e6
    if agent == "gpt":
        cost_est *= PRICE["gpt"]["batch_discount"]  # we'd use batch; but real-time for simplicity
    print(f"  [{os.path.basename(out_path)}] agent={agent}  "
          f"todo={len(todo)}/{len(pairs)}  est cost=${cost_est:.2f}")

    if not live:
        return

    # Output columns
    cols = ["orig_pair_key", "source_stratum", "source_model",
            "orig_pair_idx", "dataset_i", "dataset_j",
            "shortName_i", "shortName_j", "longName_i", "longName_j",
            "abstract_i", "abstract_j", "daac_i", "daac_j",
            "agent1_plausibility", "agent1_novelty",
            "generator",
            "research_question", "hypothesis", "analysis_method",
            "expected_finding", "scientific_importance", "domain"]

    if not os.path.exists(out_path):
        with open(out_path, "w") as f:
            f.write("\t".join(cols) + "\n")

    with open(out_path, "a", buffering=1) as outf:
        for idx, r in todo:
            user = AGENT2_USER.format(
                sn_a=r["shortName_i"] or "(unknown)",
                ln_a=r["longName_i"]  or "(unknown)",
                abs_a=truncate(r["abstract_i"]),
                sn_b=r["shortName_j"] or "(unknown)",
                ln_b=r["longName_j"]  or "(unknown)",
                abs_b=truncate(r["abstract_j"]),
            )
            try:
                v = call_judge(agent, AGENT2_SYSTEM, user, max_tokens=900)
                row = {
                    "orig_pair_key":       f"{r['stratum']}_{r['pair_idx']}",
                    "source_stratum":      r["stratum"],
                    "source_model":        r.get("source_model", ""),
                    "orig_pair_idx":       r["pair_idx"],
                    "dataset_i":           r["dataset_i"],
                    "dataset_j":           r["dataset_j"],
                    "shortName_i":         r["shortName_i"],
                    "shortName_j":         r["shortName_j"],
                    "longName_i":          r["longName_i"],
                    "longName_j":          r["longName_j"],
                    "abstract_i":          r["abstract_i"],
                    "abstract_j":          r["abstract_j"],
                    "daac_i":              r["daac_i"],
                    "daac_j":              r["daac_j"],
                    "agent1_plausibility": r.get("plausibility", ""),
                    "agent1_novelty":      r.get("novelty", ""),
                    "generator":           agent,
                    "research_question":     v.get("research_question", ""),
                    "hypothesis":            v.get("hypothesis", ""),
                    "analysis_method":       v.get("analysis_method", ""),
                    "expected_finding":      v.get("expected_finding", ""),
                    "scientific_importance": v.get("scientific_importance", ""),
                    "domain":                v.get("domain", ""),
                }
                outf.write("\t".join(str(row.get(c, "")).replace("\t", " ").replace("\n", " ")
                                     for c in cols) + "\n")
            except Exception as e:
                with open(f"{RESPONSES_DIR}/gen_{agent}_{r['stratum']}_{r['pair_idx']}_ERROR.txt", "w") as fe:
                    fe.write(f"ERROR: {e}\n")
                print(f"    ERROR on {r['stratum']}_{r['pair_idx']}: {e}")

    with open(out_path) as f:
        n = sum(1 for _ in f) - 1
    print(f"  ✅ {out_path}  ({n} hypotheses)")


# ══════════════════════════════════════════════════════════════
# TASK 4 — Validate each hypothesis set with each judge, both conditions
# ══════════════════════════════════════════════════════════════
def task_validate(live):
    """For each (a1, a2) hypothesis file, run both judges × both conditions.
    8 experiments total. Result: 8 validations TSVs."""
    gen_combos = [("gpt", "gpt"), ("gpt", "claude"),
                  ("claude", "gpt"), ("claude", "claude")]
    judges = ["gpt", "claude"]

    for a1, a2 in gen_combos:
        hyp_path = f"{FACT_DIR}/hypotheses_{a1}_{a2}.tsv"
        if not os.path.exists(hyp_path):
            print(f"  [skip] {hyp_path} missing")
            continue
        for a3 in judges:
            exp_name = f"{a1}_{a2}_{a3}"
            out_path = f"{EXP_DIR}/{exp_name}_validations.tsv"
            _validate_hypotheses(hyp_path, out_path, a3, exp_name, live)


def _validate_hypotheses(hyp_path, out_path, judge: str, exp_name: str, live):
    hyps = []
    with open(hyp_path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            hyps.append(r)

    # Done keys: (hyp_key, condition)
    done_keys = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                done_keys.add((r["hyp_key"], r["condition"]))

    conditions = ["blind", "ctx"]
    todo = []
    for h in hyps:
        for cond in conditions:
            if (h["orig_pair_key"], cond) in done_keys:
                continue
            todo.append((h, cond))

    cost_est = len(todo) * (EST["judge"]["in"]  * PRICE[judge]["in"] +
                            EST["judge"]["out"] * PRICE[judge]["out"]) / 1e6
    print(f"  [{exp_name}] judge={judge}  todo={len(todo)}/{len(hyps)*2}  "
          f"est cost=${cost_est:.2f}")

    if not live:
        return

    cols = ["hyp_key", "experiment", "condition",
            "source_stratum", "generator", "judge",
            "importance", "tractability", "novelty", "rationale"]

    if not os.path.exists(out_path):
        with open(out_path, "w") as f:
            f.write("\t".join(cols) + "\n")

    with open(out_path, "a", buffering=1) as outf:
        for h, cond in todo:
            if cond == "blind":
                sys_p = AGENT3_SYSTEM_BLIND
                user_p = AGENT3_USER_BLIND.format(
                    research_question=h["research_question"],
                    hypothesis=h["hypothesis"],
                    analysis_method=h["analysis_method"],
                    expected_finding=h["expected_finding"],
                    scientific_importance=h["scientific_importance"],
                    domain=h["domain"],
                )
            else:
                sys_p = AGENT3_SYSTEM_CTX
                user_p = AGENT3_USER_CTX.format(
                    sn_a=h["shortName_i"] or "(unknown)",
                    ln_a=h["longName_i"]  or "(unknown)",
                    abs_a=truncate(h["abstract_i"]),
                    sn_b=h["shortName_j"] or "(unknown)",
                    ln_b=h["longName_j"]  or "(unknown)",
                    abs_b=truncate(h["abstract_j"]),
                    research_question=h["research_question"],
                    hypothesis=h["hypothesis"],
                    analysis_method=h["analysis_method"],
                    expected_finding=h["expected_finding"],
                    scientific_importance=h["scientific_importance"],
                    domain=h["domain"],
                )
            try:
                v = call_judge(judge, sys_p, user_p, max_tokens=500)
                row = {
                    "hyp_key":        h["orig_pair_key"],
                    "experiment":     exp_name,
                    "condition":      cond,
                    "source_stratum": h["source_stratum"],
                    "generator":      h["generator"],
                    "judge":          judge,
                    "importance":     int(v["importance"]),
                    "tractability":   int(v["tractability"]),
                    "novelty":        int(v["novelty"]),
                    "rationale":      str(v.get("rationale", "")),
                }
                outf.write("\t".join(str(row.get(c, "")).replace("\t", " ").replace("\n", " ")
                                     for c in cols) + "\n")
            except Exception as e:
                with open(f"{RESPONSES_DIR}/val_{exp_name}_{h['orig_pair_key']}_{cond}_ERROR.txt", "w") as fe:
                    fe.write(f"ERROR: {e}\n")
                print(f"    ERROR on {h['orig_pair_key']} {cond}: {e}")

    with open(out_path) as f:
        n = sum(1 for _ in f) - 1
    print(f"  ✅ {out_path}  ({n} validations)")


# ══════════════════════════════════════════════════════════════
# TASK 5 — Analyze the factorial
# ══════════════════════════════════════════════════════════════
def task_analyze(live):
    """Build final table, compute per-cell means and self-preference bias."""
    import statistics as st

    all_val = []
    for a1, a2, a3 in EXPERIMENTS:
        exp_name = f"{a1}_{a2}_{a3}"
        path = f"{EXP_DIR}/{exp_name}_validations.tsv"
        if not os.path.exists(path):
            print(f"  [skip] {path} missing")
            continue
        with open(path) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                r["agent1"] = a1
                r["agent2"] = a2
                r["agent3"] = a3
                r["importance"]   = int(r["importance"])
                r["tractability"] = int(r["tractability"])
                r["novelty"]      = int(r["novelty"])
                all_val.append(r)

    if not all_val:
        print("  no validations found"); return

    print(f"\nLoaded {len(all_val)} validator judgments across experiments\n")

    # ─── Per-experiment summary ───────────────────────────────
    print("═" * 78)
    print("PER-EXPERIMENT MEAN SCORES  (condition = both blind+ctx averaged)")
    print("═" * 78)
    print(f"{'a1':<8} {'a2':<8} {'a3':<8} {'cond':<6} {'n':<4}  "
          f"{'imp':<6} {'tract':<6} {'novel':<6}")
    rows_summary = []
    for a1, a2, a3 in EXPERIMENTS:
        for cond in ["blind", "ctx"]:
            vs = [r for r in all_val
                  if r["agent1"] == a1 and r["agent2"] == a2 and r["agent3"] == a3
                  and r["condition"] == cond]
            if not vs: continue
            n = len(vs)
            m_imp = st.mean([r["importance"]   for r in vs])
            m_tra = st.mean([r["tractability"] for r in vs])
            m_nov = st.mean([r["novelty"]      for r in vs])
            print(f"{a1:<8} {a2:<8} {a3:<8} {cond:<6} {n:<4}  "
                  f"{m_imp:.2f}   {m_tra:.2f}   {m_nov:.2f}")
            rows_summary.append({
                "agent1": a1, "agent2": a2, "agent3": a3,
                "condition": cond, "n": n,
                "importance_mean":   m_imp,
                "tractability_mean": m_tra,
                "novelty_mean":      m_nov,
            })

    # ─── Self-preference bias (A3 rating A2) ──────────────────
    print("\n" + "═" * 78)
    print("SELF-PREFERENCE EFFECT  (when Agent 3 = Agent 2, vs cross-model)")
    print("═" * 78)
    for cond in ["blind", "ctx"]:
        self_scores  = []
        cross_scores = []
        for r in all_val:
            if r["condition"] != cond: continue
            avg = (r["importance"] + r["tractability"] + r["novelty"]) / 3
            if r["agent2"] == r["agent3"]:
                self_scores.append(avg)
            else:
                cross_scores.append(avg)
        if self_scores and cross_scores:
            print(f"  [{cond}] self (same model): {st.mean(self_scores):.3f} (n={len(self_scores)})")
            print(f"  [{cond}] cross (diff model): {st.mean(cross_scores):.3f} (n={len(cross_scores)})")
            print(f"  [{cond}] self-preference bias: {st.mean(self_scores) - st.mean(cross_scores):+.3f}")

    # ─── Main effect of each agent ────────────────────────────
    print("\n" + "═" * 78)
    print("MARGINAL EFFECT  (mean score when agent_X = gpt vs claude)")
    print("═" * 78)
    for axis_num, axis in enumerate(["importance", "tractability", "novelty"]):
        print(f"\n  [{axis}]")
        for which in ["agent1", "agent2", "agent3"]:
            for m in ["gpt", "claude"]:
                vals = [r[axis] for r in all_val if r[which] == m]
                if not vals: continue
                print(f"    {which}={m:<6}  n={len(vals):<4}  "
                      f"mean={st.mean(vals):.3f}")

    # ─── Save final table ─────────────────────────────────────
    summary_path = f"{FACT_DIR}/final_factorial_results.tsv"
    cols = list(rows_summary[0].keys())
    with open(summary_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows_summary:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"\n✅ {summary_path}")

    # Save full raw for paper
    raw_path = f"{FACT_DIR}/all_validations_raw.tsv"
    cols = ["hyp_key", "agent1", "agent2", "agent3", "condition",
            "source_stratum", "generator", "judge",
            "importance", "tractability", "novelty", "rationale"]
    with open(raw_path, "w") as f:
        f.write("\t".join(cols) + "\n")
        for r in all_val:
            f.write("\t".join(str(r.get(c, "")).replace("\t", " ").replace("\n", " ")
                              for c in cols) + "\n")
    print(f"✅ {raw_path}")


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
TASKS = {
    "agent1_gpt":    task_agent1_gpt,
    "agent1_claude": task_agent1_claude,
    "select_top40":  task_select_top40,
    "generate":      task_generate,
    "validate":      task_validate,
    "analyze":       task_analyze,
    "all":           None,  # special
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default=None,
                    choices=list(TASKS.keys()),
                    help="Which task to run (default: dry-run all)")
    ap.add_argument("--live", action="store_true",
                    help="Actually make API calls. Default: dry-run.")
    args = ap.parse_args()

    print("═" * 72)
    print(f"STAGE 5 FACTORIAL  (live={args.live}, task={args.task or 'all'})")
    print("═" * 72)

    if args.task is None or args.task == "all":
        order = ["agent1_gpt", "agent1_claude", "select_top40",
                 "generate", "validate", "analyze"]
        for t in order:
            print(f"\n── {t.upper()} " + "─" * (66 - len(t)))
            TASKS[t](args.live)
    else:
        TASKS[args.task](args.live)


if __name__ == "__main__":
    main()