"""
STAGE 4b — GPT-5.2 Judge submission
====================================
Builds per-stratum JSONL batches and submits them to OpenAI's Batch API.

USAGE:
    # Dry run first — inspects prompts, estimates cost, no API calls
    python3 stage4_judge_submit.py

    # After confirming dry run looks good, submit ONE stratum:
    python3 stage4_judge_submit.py --live --stratum B

    # Then the others, one at a time, after inspecting each batch's outputs:
    python3 stage4_judge_submit.py --live --stratum C
    python3 stage4_judge_submit.py --live --stratum D
    python3 stage4_judge_submit.py --live --stratum A

WHY STRATUM B FIRST:
    Stratum B is held-out real co-usage pairs (positive control). If the
    judge doesn't rate these high, the judge is broken and we stop before
    wasting more money.

OUTPUTS:
    ./neurips_figs/stage4/batch_{stratum}_requests.jsonl   (prompts)
    ./neurips_figs/stage4/batch_{stratum}_submission.json  (submission metadata)
    ./neurips_figs/stage4/prompts/{stratum}_{idx}.txt      (audit copies)

Requires:
    export OPENAI_API_KEY=sk-...

    pip install openai
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

SEED = 42
MODEL = "gpt-5.2"   # exact model slug may differ; adjust if OpenAI uses a different name
OUT_DIR = "./neurips_figs"
STAGE4_DIR = f"{OUT_DIR}/stage4"
PROMPTS_DIR = f"{STAGE4_DIR}/prompts"
os.makedirs(PROMPTS_DIR, exist_ok=True)

# Pricing (per 1M tokens, Batch API = 50% off standard)
# As of 2026-01 assume: $0.875 input / $7.00 output per 1M tokens
# Update these if pricing has changed.
PRICE_INPUT_PER_1M  = 0.875
PRICE_OUTPUT_PER_1M = 7.00

# Token estimates per request (we'll refine with real counts)
EST_INPUT_TOKENS  = 2000
EST_OUTPUT_TOKENS = 500


# ══════════════════════════════════════════════════════════════
# Prompt template
# ══════════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    "You are an expert Earth-science research scientist. Your job is to "
    "evaluate whether two Earth-observation datasets could plausibly be "
    "combined in a single scientific study. Base your judgment on the "
    "datasets' scientific content as described in their abstracts, not on "
    "whether you have seen them combined before in published work. "
    "Respond with JSON only, in the exact schema requested."
)

USER_PROMPT_TEMPLATE = """Please evaluate the following pair of Earth-observation datasets.

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
   these datasets in a single scientific study? Consider measurement
   compatibility, spatial/temporal overlap, and whether their observed
   phenomena are scientifically connected.
     1 = implausible (no scientific reason to combine)
     2 = unlikely (weak or forced connection)
     3 = possible (some research context could combine them)
     4 = plausible (clear scientific motivation exists)
     5 = highly plausible (strong, well-motivated combination)

2. NOVELTY (1-5): How novel or non-obvious is this combination?
     1 = obvious / trivial (e.g., two versions of the same product,
         two channels of the same instrument)
     2 = expected (same instrument family or same narrow sub-domain)
     3 = standard (common complementary pairing)
     4 = non-obvious (combines distinct domains in a coherent way)
     5 = highly novel (unexpected combination across domains with
         coherent scientific motivation)

Respond with JSON only, this exact schema:
{{
  "plausibility": <1-5 integer>,
  "novelty": <1-5 integer>,
  "rationale": "<2-3 sentence explanation covering both axes>"
}}"""


# ══════════════════════════════════════════════════════════════
# Load strata
# ══════════════════════════════════════════════════════════════
def load_strata():
    rows = []
    with open(f"{STAGE4_DIR}/strata.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    return rows


def truncate_abstract(text, max_chars=1200):
    """Keep abstracts short. Truncate at sentence boundary if possible."""
    if not text:
        return "(no abstract provided)"
    text = text.strip().replace("\n", " ").replace("\r", " ")
    if len(text) <= max_chars:
        return text
    cut = text[:max_chars]
    # Back off to last period
    last_period = cut.rfind(". ")
    if last_period > max_chars * 0.6:
        cut = cut[:last_period + 1]
    return cut + " […truncated]"


def build_prompt(row, rng):
    """Build (system, user) prompts for one pair. Order is randomized."""
    sn_i, sn_j = row["shortName_i"], row["shortName_j"]
    ln_i, ln_j = row["longName_i"], row["longName_j"]
    ab_i, ab_j = truncate_abstract(row["abstract_i"]), truncate_abstract(row["abstract_j"])
    # Random order: present i first or j first (for position-bias control)
    if rng.random() < 0.5:
        sn_a, ln_a, abs_a = sn_i, ln_i, ab_i
        sn_b, ln_b, abs_b = sn_j, ln_j, ab_j
        order = "ij"
    else:
        sn_a, ln_a, abs_a = sn_j, ln_j, ab_j
        sn_b, ln_b, abs_b = sn_i, ln_i, ab_i
        order = "ji"
    user = USER_PROMPT_TEMPLATE.format(
        sn_a=sn_a or "(unknown)", ln_a=ln_a or "(unknown)", abs_a=abs_a,
        sn_b=sn_b or "(unknown)", ln_b=ln_b or "(unknown)", abs_b=abs_b,
    )
    return SYSTEM_PROMPT, user, order


def custom_id(stratum, pair_idx):
    return f"{stratum}_{pair_idx:04d}"


# ══════════════════════════════════════════════════════════════
# Build a JSONL batch file for one stratum
# ══════════════════════════════════════════════════════════════
def build_batch_jsonl(stratum_rows, stratum_name, out_path):
    rng = random.Random(SEED + ord(stratum_name))  # stable per-stratum
    order_log = []
    lines = []
    for r in stratum_rows:
        system, user, order = build_prompt(r, rng)
        cid = custom_id(stratum_name, int(r["pair_idx"]))
        # Also save a human-readable copy of every prompt
        with open(f"{PROMPTS_DIR}/{cid}.txt", "w") as f:
            f.write(f"=== SYSTEM ===\n{system}\n\n=== USER ===\n{user}\n")
        order_log.append({"custom_id": cid, "order": order})
        req = {
            "custom_id": cid,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                "response_format": {"type": "json_object"},
                "max_completion_tokens": 600,
                "temperature": 0,  # deterministic per run
            },
        }
        lines.append(json.dumps(req))

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    with open(out_path.replace(".jsonl", "_order.json"), "w") as f:
        json.dump(order_log, f, indent=2)
    return len(lines)


# ══════════════════════════════════════════════════════════════
# Cost estimation
# ══════════════════════════════════════════════════════════════
def estimate_cost(n_requests,
                  tokens_in=EST_INPUT_TOKENS,
                  tokens_out=EST_OUTPUT_TOKENS):
    cost_in  = n_requests * tokens_in  / 1e6 * PRICE_INPUT_PER_1M
    cost_out = n_requests * tokens_out / 1e6 * PRICE_OUTPUT_PER_1M
    return cost_in + cost_out, cost_in, cost_out


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="Actually submit to the OpenAI API. Default is dry-run.")
    ap.add_argument("--stratum", choices=["A", "B", "C", "D"],
                    help="Which stratum to build/submit. If omitted, dry-run "
                         "inspects all four.")
    ap.add_argument("--model", default=MODEL,
                    help=f"Model name (default: {MODEL})")
    args = ap.parse_args()

    all_rows = load_strata()
    by_stratum = defaultdict(list)
    for r in all_rows:
        by_stratum[r["stratum"]].append(r)

    strata_to_process = [args.stratum] if args.stratum else ["A", "B", "C", "D"]

    print("═" * 62)
    print(f"STAGE 4b — Judge submission   (live={args.live}, model={args.model})")
    print("═" * 62)

    total_cost = 0
    for s in strata_to_process:
        rows = by_stratum[s]
        jsonl_path = f"{STAGE4_DIR}/batch_{s}_requests.jsonl"
        n = build_batch_jsonl(rows, s, jsonl_path)
        cost, ci, co = estimate_cost(n)
        total_cost += cost
        print(f"\n── Stratum {s} ──")
        print(f"   pairs: {n}")
        print(f"   JSONL: {jsonl_path}")
        print(f"   Prompts saved to: {PROMPTS_DIR}/{s}_*.txt")
        print(f"   Est cost (batch pricing): ${cost:.2f}  "
              f"(input ${ci:.2f} + output ${co:.2f})")

    print(f"\nTotal estimated cost across {len(strata_to_process)} strata: "
          f"${total_cost:.2f}")

    if not args.live:
        print("\n[DRY RUN] No API calls made.")
        print("Inspect the JSONL files and a few prompt .txt files. When you're "
              "happy, rerun with --live --stratum B (positive control first).")
        # Show first prompt of the first stratum for sanity
        first_stratum = strata_to_process[0]
        first_cid = custom_id(first_stratum, 0)
        sample_path = f"{PROMPTS_DIR}/{first_cid}.txt"
        if os.path.exists(sample_path):
            print(f"\n────── Sample prompt from {sample_path} ──────")
            print(Path(sample_path).read_text())
        return

    # ───── Live submission ─────
    if args.stratum is None:
        print("\nERROR: --live requires --stratum (one at a time).")
        sys.exit(1)

    try:
        from openai import OpenAI
    except ImportError:
        print("\nERROR: pip install openai")
        sys.exit(1)

    client = OpenAI()
    s = args.stratum
    jsonl_path = f"{STAGE4_DIR}/batch_{s}_requests.jsonl"

    print(f"\n▶ Uploading {jsonl_path} to OpenAI Files API …")
    with open(jsonl_path, "rb") as f:
        file_obj = client.files.create(file=f, purpose="batch")
    print(f"   file_id = {file_obj.id}")

    print(f"\n▶ Creating batch job …")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"project": "nasa-kg-cousage", "stratum": s},
    )
    print(f"   batch_id = {batch.id}")
    print(f"   status   = {batch.status}")

    # Persist submission metadata
    submission = {
        "stratum":         s,
        "model":           args.model,
        "batch_id":        batch.id,
        "input_file_id":   file_obj.id,
        "created_at":      batch.created_at,
        "n_requests":      len(rows),
        "est_cost":        estimate_cost(len(rows))[0],
        "status":          batch.status,
    }
    sub_path = f"{STAGE4_DIR}/batch_{s}_submission.json"
    with open(sub_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"\n✅ Submission metadata: {sub_path}")

    print("\nNext:")
    print(f"   Check status:   python3 stage4_judge_parse.py --status --stratum {s}")
    print(f"   When complete:  python3 stage4_judge_parse.py --stratum {s}")


if __name__ == "__main__":
    main()