#!/usr/bin/env python3
"""Eval loop: detect weak topics and steer training (Track 5).

Closes Damru's self-improvement loop:
    feedback / eval scores  ->  weak_topics.json  ->  training focuses them

Sources (whatever is available):
  1. Supabase `damru_feedback` (question, answer, rating, intent)  [preferred]
  2. Fallback: HF dataset `Damaru-ai/damru-scorecards` if you wire it in.

Output: weak_topics.json  (sorted weakest-first) e.g.
    {"generated": "...", "threshold": 3.5,
     "weak": [{"topic": "organic_chemistry", "avg": 2.8, "n": 40}, ...]}

Then in phase5/config.py the engine reads weak_topics.json and raises
FOCUS_RATIO on those topics. Optionally this script pushes the file to a repo.

Run:
    pip install -q requests huggingface_hub
    SUPABASE_URL=... SUPABASE_KEY=... python3 build_weak_topics.py
    # optional: PUSH_REPO=Damaru-ai/damru-train HF_TOKEN=hf_xxx to upload
"""
import os
import json
import time
import collections

import requests

SB_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SB_KEY = os.environ.get("SUPABASE_KEY", "")
TABLE = os.environ.get("FEEDBACK_TABLE", "damru_feedback")
THRESHOLD = float(os.environ.get("WEAK_THRESHOLD", "3.5"))
MIN_SAMPLES = int(os.environ.get("MIN_SAMPLES", "5"))
OUT = os.environ.get("OUT_FILE", "weak_topics.json")
PUSH_REPO = os.environ.get("PUSH_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def fetch_feedback():
    """Pull rating rows from Supabase, paged."""
    if not (SB_URL and SB_KEY):
        print("No SUPABASE_URL/KEY set -- cannot read feedback.")
        return []
    rows, offset, page = [], 0, 1000
    headers = {"apikey": SB_KEY, "Authorization": "Bearer " + SB_KEY}
    while True:
        url = ("%s/rest/v1/%s?select=intent,rating&rating=not.is.null"
               "&order=id.desc" % (SB_URL, TABLE))
        h = dict(headers)
        h["Range"] = "%d-%d" % (offset, offset + page - 1)
        r = requests.get(url, headers=h, timeout=30)
        if not r.ok:
            print("fetch error", r.status_code, r.text[:200])
            break
        batch = r.json() or []
        rows.extend(batch)
        if len(batch) < page:
            break
        offset += page
    print("fetched %d rated rows" % len(rows))
    return rows


def compute(rows):
    agg = collections.defaultdict(list)
    for row in rows:
        topic = (row.get("intent") or "general").strip().lower().replace(" ", "_")
        try:
            agg[topic].append(float(row.get("rating")))
        except (TypeError, ValueError):
            continue
    stats = []
    for topic, vals in agg.items():
        if len(vals) < MIN_SAMPLES:
            continue
        stats.append({"topic": topic, "avg": round(sum(vals) / len(vals), 3),
                      "n": len(vals)})
    stats.sort(key=lambda x: x["avg"])  # weakest first
    weak = [s for s in stats if s["avg"] < THRESHOLD]
    return stats, weak


def maybe_push(path):
    if not (PUSH_REPO and HF_TOKEN):
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        api.upload_file(path_or_fileobj=path, path_in_repo=OUT,
                        repo_id=PUSH_REPO, repo_type="dataset")
        print("pushed %s -> %s" % (OUT, PUSH_REPO))
    except Exception as e:
        print("push failed:", str(e)[:200])


def main():
    rows = fetch_feedback()
    stats, weak = compute(rows)
    result = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": THRESHOLD,
        "min_samples": MIN_SAMPLES,
        "all": stats,
        "weak": weak,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n=== weakest topics (focus these in training) ===")
    for s in weak[:15]:
        print("  %-28s avg=%.2f  n=%d" % (s["topic"], s["avg"], s["n"]))
    if not weak:
        print("  (none below threshold -- Damru is balanced, raise the bar!)")
    print("\nwrote", OUT)
    maybe_push(OUT)


if __name__ == "__main__":
    main()
