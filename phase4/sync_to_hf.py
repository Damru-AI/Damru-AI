#!/usr/bin/env python3
"""
Damru AI - HOURLY INCREMENTAL Sync: Supabase -> HuggingFace (sharded)

Built to scale to 5M+ rows at ~90K/day with an hourly cron.

Why sharded (not full-mirror):
- Full-mirror re-uploads the WHOLE dataset every run -> at millions of rows
  that means 10-15GB up+down EVERY HOUR -> times out / dies.
- This script transfers ONLY the NEW rows (id > last_id) each run and writes
  them as a NEW parquet shard into the HF repo's data/ folder. Old shards are
  never touched. So hourly cost stays tiny no matter how big the dataset gets.

Flow each run:
  1. Read high-water-mark id from _sync_state.json on HF.
     (First run: baseline to current Supabase max id, because the existing
      rows are already on HF from the previous full-mirror sync -> no dupes.)
  2. Fetch ONLY Supabase rows with id > last_id.
  3. Clean + write them as data/train-<id>-<ts>.parquet on HF.
  4. Update _sync_state.json.
  5. Auto-delete OLD Supabase rows that are already safe on HF
     (keep newest CLEANUP_KEEP rows for live RAG).

load_dataset("Damaru-ai/damru-knowledge") reads ALL shards together.

Env vars:
  SUPABASE_URL, SUPABASE_KEY, HF_TOKEN          (required)
  HF_REPO           (default Damaru-ai/damru-knowledge)
  SUPABASE_CLEANUP  ("true"/"false", default true)
  CLEANUP_KEEP      (int, default 30000)
"""
import os
import io
import json
import time
import requests
from datasets import Dataset
from huggingface_hub import HfApi, hf_hub_download, login

SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ["SUPABASE_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = os.environ.get("HF_REPO", "Damaru-ai/damru-knowledge")
CLEANUP = os.environ.get("SUPABASE_CLEANUP", "true").lower() == "true"
CLEANUP_KEEP = int(os.environ.get("CLEANUP_KEEP", "30000"))

HEADERS = {"apikey": SB_KEY, "Authorization": "Bearer " + SB_KEY}
SELECT = "select=id,question,answer,intent,lang,upvotes,created_at"
STATE_FILE = "_sync_state.json"

api = HfApi(token=HF_TOKEN)


def valid(q, a):
    return len(q) > 3 and len(a) > 20


def row_obj(x):
    return {
        "question": (x.get("question") or "").strip(),
        "answer": (x.get("answer") or "").strip(),
        "intent": x.get("intent") or "general",
        "lang": x.get("lang") or "en",
        "upvotes": x.get("upvotes") or 0,
        "created_at": x.get("created_at") or "",
    }


def sb_max_id():
    url = SB_URL + "/rest/v1/damru_knowledge?select=id&order=id.desc&limit=1"
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    arr = r.json()
    return int(arr[0]["id"]) if arr else 0


def read_state():
    """Return last_id int, or None if no state file exists yet."""
    try:
        p = hf_hub_download(HF_REPO, STATE_FILE, repo_type="dataset", token=HF_TOKEN)
        with open(p) as f:
            return int(json.load(f).get("last_id", 0))
    except Exception as e:
        print("No state file yet ->", str(e)[:100])
        return None


def write_state(last_id):
    buf = json.dumps({"last_id": int(last_id), "updated": int(time.time())}).encode()
    api.upload_file(path_or_fileobj=io.BytesIO(buf), path_in_repo=STATE_FILE,
                    repo_id=HF_REPO, repo_type="dataset")


def fetch_new(last_id):
    rows, step, cur = [], 1000, last_id
    while True:
        url = (SB_URL + "/rest/v1/damru_knowledge?" + SELECT +
               "&id=gt." + str(cur) + "&order=id.asc&limit=" + str(step))
        r = requests.get(url, headers=HEADERS, timeout=90)
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        rows += batch
        cur = batch[-1]["id"]
        print("  fetched new", len(rows), "rows (up to id", cur, ")")
        if len(batch) < step:
            break
    return rows


def main():
    print("== Damru HOURLY incremental sync ==")
    login(HF_TOKEN)

    state = read_state()
    if state is None:
        # Transition: existing rows are already on HF (previous full-mirror).
        # Baseline to current Supabase max id so we DON'T re-upload them.
        base = sb_max_id()
        write_state(base)
        print("Baseline initialised -> last_id", base,
              "(existing rows already on HF, no re-upload)")
        last_id = base
    else:
        last_id = state
    print("Last synced id:", last_id)

    new_rows = fetch_new(last_id)
    print("New Supabase rows:", len(new_rows))

    clean, max_id = [], last_id
    for x in new_rows:
        max_id = max(max_id, int(x.get("id", last_id)))
        o = row_obj(x)
        if valid(o["question"], o["answer"]):
            clean.append(o)
    print("Clean new rows:", len(clean))

    if clean:
        local = "/tmp/shard.parquet"
        Dataset.from_list(clean).to_parquet(local)
        fname = "data/train-%013d-%d.parquet" % (max_id, int(time.time()))
        api.upload_file(path_or_fileobj=local, path_in_repo=fname,
                        repo_id=HF_REPO, repo_type="dataset")
        print("Uploaded shard:", fname, "(", len(clean), "rows )")
        write_state(max_id)
        print("State updated -> last_id", max_id)
    else:
        print("No new valid rows to upload this run.")

    if CLEANUP:
        cleanup_supabase(max_id)


def cleanup_supabase(safe_upto_id):
    """Delete Supabase rows already on HF (id <= safe_upto_id), keep newest CLEANUP_KEEP."""
    url = (SB_URL + "/rest/v1/damru_knowledge?select=id&order=id.desc&limit=1&offset="
           + str(CLEANUP_KEEP))
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    arr = r.json()
    if not arr:
        print("Cleanup: fewer than", CLEANUP_KEEP, "rows; nothing to delete.")
        return
    keep_cutoff = int(arr[0]["id"])           # rows with id >= this are the newest, kept
    del_cutoff = min(keep_cutoff, int(safe_upto_id))  # only delete what's safe on HF
    if del_cutoff <= 0:
        print("Cleanup: nothing safe to delete yet.")
        return
    del_url = SB_URL + "/rest/v1/damru_knowledge?id=lt." + str(del_cutoff)
    dr = requests.delete(del_url, headers={**HEADERS, "Prefer": "return=minimal"}, timeout=120)
    dr.raise_for_status()
    print("Cleanup: deleted Supabase rows with id <", del_cutoff,
          "(kept newest ~" + str(CLEANUP_KEEP) + " for RAG)")


if __name__ == "__main__":
    main()
