#!/usr/bin/env python3
"""
Damru AI - Sync Supabase knowledge -> HuggingFace Dataset

Reads ALL rows from Supabase `damru_knowledge` and pushes them to a public
HuggingFace dataset repo. This is Damru's TRAINING data store (unlimited, free).
Supabase stays the LIVE RAG brain; HuggingFace holds the full history for
fine-tuning via load_dataset("Damaru-ai/damru-knowledge").

Env vars required:
  SUPABASE_URL, SUPABASE_KEY, HF_TOKEN
  HF_REPO (optional, defaults to Damaru-ai/damru-knowledge)
"""
import os
import requests
from datasets import Dataset
from huggingface_hub import login

SB_URL = os.environ["SUPABASE_URL"].rstrip("/")
SB_KEY = os.environ["SUPABASE_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
HF_REPO = os.environ.get("HF_REPO", "Damaru-ai/damru-knowledge")

HEADERS = {"apikey": SB_KEY, "Authorization": "Bearer " + SB_KEY}
SELECT = "select=question,answer,intent,lang,upvotes,created_at"


def fetch_all():
    rows, step, off = [], 1000, 0
    while True:
        url = SB_URL + "/rest/v1/damru_knowledge?" + SELECT + "&order=id.asc&limit=" + str(step) + "&offset=" + str(off)
        resp = requests.get(url, headers=HEADERS, timeout=90)
        resp.raise_for_status()
        batch = resp.json()
        if not isinstance(batch, list) or not batch:
            break
        rows += batch
        print("  fetched", len(rows), "rows so far...")
        if len(batch) < step:
            break
        off += step
    return rows


def clean(rows):
    seen, out = set(), []
    for x in rows:
        q = (x.get("question") or "").strip()
        a = (x.get("answer") or "").strip()
        if len(q) <= 3 or len(a) <= 20:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "question": q,
            "answer": a,
            "intent": x.get("intent") or "general",
            "lang": x.get("lang") or "en",
            "upvotes": x.get("upvotes") or 0,
            "created_at": x.get("created_at") or "",
        })
    return out


def main():
    print("Pulling rows from Supabase...")
    rows = fetch_all()
    print("Total raw rows:", len(rows))
    data = clean(rows)
    print("Clean, de-duped rows:", len(data))
    if not data:
        print("Nothing to push. Exiting.")
        return
    ds = Dataset.from_list(data)
    login(HF_TOKEN)
    print("Pushing to HuggingFace:", HF_REPO)
    ds.push_to_hub(HF_REPO, private=False)
    print("DONE. Training data live at https://huggingface.co/datasets/" + HF_REPO)


if __name__ == "__main__":
    main()
