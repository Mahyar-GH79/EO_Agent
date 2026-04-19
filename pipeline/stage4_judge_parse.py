"""
STAGE 4c — GPT-5.2 Judge parsing
================================
Checks batch status, downloads completed batches, parses JSON responses,
and merges with strata.tsv into a single judgments.tsv.

USAGE:
    # Check status of one or all submitted batches
    python3 stage4_judge_parse.py --status

    # Check one stratum
    python3 stage4_judge_parse.py --status --stratum B

    # Parse a completed batch
    python3 stage4_judge_parse.py --stratum B

    # Parse all four strata (call after all are complete)
    python3 stage4_judge_parse.py

OUTPUTS:
    ./neurips_figs/stage4/responses/{stratum}_{idx}.json     (raw responses)
    ./neurips_figs/stage4/parse_failures.tsv                 (problem responses)
    ./neurips_figs/stage4/stage4_judgments.tsv               (final merged)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

OUT_DIR = "./neurips_figs"
STAGE4_DIR = f"{OUT_DIR}/stage4"
RESPONSES_DIR = f"{STAGE4_DIR}/responses"
os.makedirs(RESPONSES_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════
def load_submission(stratum):
    path = f"{STAGE4_DIR}/batch_{stratum}_submission.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_order_log(stratum):
    path = f"{STAGE4_DIR}/batch_{stratum}_requests_order.json"
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return {r["custom_id"]: r["order"] for r in json.load(f)}


def load_strata_rows():
    rows = []
    with open(f"{STAGE4_DIR}/strata.tsv") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rows.append(r)
    return rows


def parse_custom_id(cid):
    # e.g. "A_0017" -> ("A", 17)
    m = re.match(r"([ABCD])_(\d+)", cid)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


# ══════════════════════════════════════════════════════════════
# Status check
# ══════════════════════════════════════════════════════════════
def cmd_status(strata):
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai"); sys.exit(1)
    client = OpenAI()
    for s in strata:
        sub = load_submission(s)
        if sub is None:
            print(f"── Stratum {s}: not submitted")
            continue
        batch = client.batches.retrieve(sub["batch_id"])
        print(f"── Stratum {s} ──")
        print(f"   batch_id       = {batch.id}")
        print(f"   status         = {batch.status}")
        print(f"   request_counts = {batch.request_counts}")
        if batch.errors:
            print(f"   errors         = {batch.errors}")
        if batch.output_file_id:
            print(f"   output_file_id = {batch.output_file_id}")
        if batch.error_file_id:
            print(f"   error_file_id  = {batch.error_file_id}")


# ══════════════════════════════════════════════════════════════
# Parse one stratum's completed batch
# ══════════════════════════════════════════════════════════════
def cmd_parse(stratum, rows_by_stratum, order_log):
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: pip install openai"); sys.exit(1)
    client = OpenAI()

    sub = load_submission(stratum)
    if sub is None:
        print(f"No submission for stratum {stratum}. Did you run "
              f"stage4_judge_submit.py --live --stratum {stratum}?")
        return []

    batch = client.batches.retrieve(sub["batch_id"])
    if batch.status != "completed":
        print(f"Stratum {stratum} batch status: {batch.status}. "
              f"Check again when completed.")
        return []

    if not batch.output_file_id:
        print(f"No output_file_id for stratum {stratum}.")
        return []

    # Download and parse
    print(f"▶ Downloading output for stratum {stratum} …")
    content = client.files.content(batch.output_file_id).read().decode("utf-8")

    rows_by_cid = {}
    strata_rows = rows_by_stratum.get(stratum, [])
    for r in strata_rows:
        cid = f"{stratum}_{int(r['pair_idx']):04d}"
        rows_by_cid[cid] = r

    parsed = []
    failures = []
    for line in content.splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        cid = item.get("custom_id", "")
        row = rows_by_cid.get(cid)
        if row is None:
            failures.append({"custom_id": cid, "reason": "no matching strata row"})
            continue

        # Save raw response to audit dir
        resp_path = f"{RESPONSES_DIR}/{cid}.json"
        with open(resp_path, "w") as f:
            json.dump(item, f, indent=2)

        # Extract content
        try:
            resp = item["response"]
            if resp["status_code"] != 200:
                failures.append({"custom_id": cid,
                                 "reason": f"status {resp['status_code']}"})
                continue
            body = resp["body"]
            text = body["choices"][0]["message"]["content"]
            parsed_json = json.loads(text)
            plaus = int(parsed_json["plausibility"])
            novel = int(parsed_json["novelty"])
            rationale = parsed_json.get("rationale", "")
            if not (1 <= plaus <= 5) or not (1 <= novel <= 5):
                failures.append({"custom_id": cid, "reason": f"out-of-range scores {plaus},{novel}"})
                continue
            parsed.append({
                **row,
                "plausibility": plaus,
                "novelty":      novel,
                "rationale":    rationale.replace("\t", " ").replace("\n", " "),
                "presentation_order": order_log.get(cid, ""),
                "judge":        "gpt-5.2",
            })
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            failures.append({"custom_id": cid, "reason": f"parse error: {type(e).__name__}: {e}"})

    print(f"   parsed ok: {len(parsed)} / {len(strata_rows)}")
    print(f"   failures:  {len(failures)}")
    return parsed, failures


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratum", choices=["A", "B", "C", "D"])
    ap.add_argument("--status", action="store_true",
                    help="Just check status; don't download or parse.")
    args = ap.parse_args()

    strata = [args.stratum] if args.stratum else ["A", "B", "C", "D"]

    print("═" * 62)
    print(f"STAGE 4c — Judge parsing   (strata: {strata})")
    print("═" * 62)

    if args.status:
        cmd_status(strata)
        return

    rows = load_strata_rows()
    rows_by_stratum = {}
    for r in rows:
        rows_by_stratum.setdefault(r["stratum"], []).append(r)

    all_parsed = []
    all_failures = []
    for s in strata:
        order_log = load_order_log(s)
        parsed, failures = cmd_parse(s, rows_by_stratum, order_log)
        all_parsed.extend(parsed)
        all_failures.extend(failures)

    # Load any existing judgments so we can merge/refresh
    out_tsv = f"{STAGE4_DIR}/stage4_judgments.tsv"
    existing = {}
    if os.path.exists(out_tsv):
        with open(out_tsv) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                key = (r["stratum"], int(r["pair_idx"]))
                existing[key] = r

    # Overwrite with new parsed results for the strata we processed
    for r in all_parsed:
        key = (r["stratum"], int(r["pair_idx"]))
        existing[key] = r

    cols = [
        "stratum", "pair_idx", "dataset_i", "dataset_j",
        "shortName_i", "shortName_j", "longName_i", "longName_j",
        "daac_i", "daac_j", "abstract_i", "abstract_j",
        "score_dot", "score_mlp", "rank_dot", "rank_mlp",
        "source_model",
        "plausibility", "novelty", "rationale",
        "presentation_order", "judge",
    ]

    def safe(x):
        if x is None: return ""
        return str(x).replace("\t", " ").replace("\n", " ").replace("\r", "")

    with open(out_tsv, "w") as f:
        f.write("\t".join(cols) + "\n")
        for key in sorted(existing.keys()):
            row = existing[key]
            f.write("\t".join(safe(row.get(c)) for c in cols) + "\n")

    print(f"\n✅ {out_tsv}  ({len(existing)} rows)")

    if all_failures:
        fail_path = f"{STAGE4_DIR}/parse_failures.tsv"
        with open(fail_path, "w") as f:
            f.write("custom_id\treason\n")
            for r in all_failures:
                f.write(f"{r['custom_id']}\t{r['reason']}\n")
        print(f"⚠  {fail_path}  ({len(all_failures)} failures)")

    # Quick summary per stratum
    print("\n" + "═" * 62)
    print("QUICK SUMMARY")
    print("═" * 62)
    from collections import defaultdict
    per_stratum = defaultdict(list)
    for r in all_parsed:
        per_stratum[r["stratum"]].append(r)
    for s in ["A", "B", "C", "D"]:
        rs = per_stratum.get(s, [])
        if not rs: continue
        plaus = [r["plausibility"] for r in rs]
        novel = [r["novelty"] for r in rs]
        print(f"   Stratum {s} (n={len(rs)}): "
              f"plausibility mean={sum(plaus)/len(plaus):.2f}, "
              f"novelty mean={sum(novel)/len(novel):.2f}")


if __name__ == "__main__":
    main()