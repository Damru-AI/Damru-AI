#!/usr/bin/env python3
"""
Damru AI - ONE-TIME RESET re-sync: Supabase -> HuggingFace (clean, no dupes)

Use this ONCE to make HF exactly match Supabase when they drift apart.
It:
  1. Fetches ALL rows from Supabase (id asc).
  2. Deletes existing data parquet shards + _sync_state.json on HF.
  3. Uploads ALL clean rows as one fresh shard.
  4. Writes _sync_state.json (last_id = max id) so the HOURLY incremental
     sync (sync_to_hf.py) continues correctly from here.

Safe because Supabase still holds the full history (cleanup only triggers
above 30k rows). README.md / .gitattributes are left untouched.

Env: SUPABASE_URL, SUPABASE_KEY, HF_TOKEN, HF_REPO
"""
import os
import io
import json
import time
import requests
from datasets import Dataset
from huggingface_hub import HfApi, login

SB = os.environ["SUPABASE_URL"].rstrip("/")
K = os.environ["SUPABASE_KEY"]
T = os.environ["HF_TOKEN"]
REPO = os.environ.get("HF_REPO", "Damaru-ai/damru-knowledge")
H = {"apikey": K, "Authorization": "Bearer " + K}
SELECT = "select=id,question,answer,intent,lang,upvotes,created_at"


def main():
    print("== Damru ONE-TIME RESET re-sync ==")
    login(T)
    api = HfApi(token=T)

    # 1) Fetch ALL Supabase rows
    rows, cur, step = [], 0, 1000
    while True:
        u = (SB + "/rest/v1/damru_knowledge?" + SELECT +
             "&id=gt." + str(cur) + "&order=id.asc&limit=" + str(step))
        r = requests.get(u, headers=H, timeout=90)
        r.raise_for_status()
        b = r.json()
        if not b:
            break
        rows += b
        cur = b[-1]["id"]
        print("  fetched", len(rows), "rows (up to id", cur, ")")
        if len(b) < step:
            break

    clean, mx = [], 0
    for x in rows:
        mx = max(mx, int(x.get("id", 0)))
        q = (x.get("question") or "").strip()
        a = (x.get("answer") or "").strip()
        if len(q) > 3 and len(a) > 20:
            clean.append({
                "question": q,
                "answer": a,
                "intent": x.get("intent") or "general",
                "lang": x.get("lang") or "en",
                "upvotes": x.get("upvotes") or 0,
                "created_at": x.get("created_at") or "",
            })
    print("Total fetched:", len(rows), "| clean:", len(clean), "| max id:", mx)
    if not clean:
        print("Nothing to upload. Aborting (HF left unchanged).")
        return

    # 2) Delete existing data shards + state on HF (keep README/.gitattributes)
    try:
        files = api.list_repo_files(repo_id=REPO, repo_type="dataset")
    except Exception as e:
        files = []
        print("list_repo_files error:", str(e)[:120])
    for f in files:
        if f.startswith("data/") or f.endswith(".parquet") or f == "_sync_state.json":
            try:
                api.delete_file(path_in_repo=f, repo_id=REPO, repo_type="dataset")
                print("  deleted old", f)
            except Exception as e:
                print("  skip delete", f, str(e)[:80])

    # 3) Upload ALL clean rows as one fresh shard
    local = "/tmp/all.parquet"
    Dataset.from_list(clean).to_parquet(local)
    fname = "data/train-%013d-%d.parquet" % (mx, int(time.time()))
    api.upload_file(path_or_fileobj=local, path_in_repo=fname,
                    repo_id=REPO, repo_type="dataset")
    print("Uploaded fresh shard:", fname, "(", len(clean), "rows )")

    # 4) Write state so hourly incremental continues from here
    buf = json.dumps({"last_id": int(mx), "updated": int(time.time())}).encode()
    api.upload_file(path_or_fileobj=io.BytesIO(buf), path_in_repo="_sync_state.json",
                    repo_id=REPO, repo_type="dataset")
    print("State set -> last_id", mx)
    print("DONE. HF now has", len(clean), "rows. Hourly sync will continue from here.")


if __name__ == "__main__":
    main()
