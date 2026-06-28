#!/usr/bin/env python3
"""
Damru dataset DEDUP + QUALITY AUDIT (operates on the HuggingFace dataset).

Dry-run by default: prints a full report (duplicates, low-quality, by-lang, by-intent).
Set APPLY=1 to rewrite the dataset as ONE cleaned shard (data/train-*.parquet) and
delete the old shards. Safe: keeps the highest-upvoted copy of each duplicate.

Env:
  HF_TOKEN   (required to read/write the dataset)
  HF_REPO    default 'Damaru-ai/damru-knowledge'
  APPLY      '1' to actually rewrite; anything else = report only
  MIN_Q_LEN  default 4    (drop questions shorter than this)
  MIN_A_LEN  default 21   (drop answers shorter than this)
"""
import os
import time

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = os.environ.get("HF_REPO", "Damaru-ai/damru-knowledge")
TOKEN = os.environ.get("HF_TOKEN", "") or None
APPLY = os.environ.get("APPLY", "0") == "1"
MIN_Q = int(os.environ.get("MIN_Q_LEN", "4"))
MIN_A = int(os.environ.get("MIN_A_LEN", "21"))

BAD_STARTS = (
    "i don't", "i cannot", "i can't", "sorry", "as an ai", "i'm not able",
    "i am unable", "unknown", "n/a",
)


def _norm(q):
    return " ".join(str(q or "").lower().split())


def _bad_start(s):
    s = str(s or "").lower().strip()
    return any(s.startswith(b) for b in BAD_STARTS)


def main():
    api = HfApi(token=TOKEN)
    files = [
        f for f in api.list_repo_files(repo_id=REPO, repo_type="dataset")
        if f.endswith(".parquet")
    ]
    if not files:
        print("No parquet files found in", REPO)
        return
    print("Found %d parquet shard(s). Downloading..." % len(files))
    frames = []
    for f in files:
        try:
            p = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=f, token=TOKEN)
            frames.append(pd.read_parquet(p))
        except Exception as e:
            print("  skip", f, "->", str(e)[:100])
    if not frames:
        print("Nothing readable.")
        return
    df = pd.concat(frames, ignore_index=True)
    before = len(df)

    for col in ("question", "answer", "intent", "lang", "upvotes"):
        if col not in df.columns:
            df[col] = "" if col != "upvotes" else 0

    qs = df["question"].astype(str)
    a = df["answer"].astype(str)
    quality_mask = (qs.str.len() >= MIN_Q) & (a.str.len() >= MIN_A) & (~a.map(_bad_start))
    lowq = int((~quality_mask).sum())
    dfq = df[quality_mask].copy()

    dfq["_n"] = dfq["question"].map(_norm)
    try:
        dfq["upvotes"] = pd.to_numeric(dfq["upvotes"], errors="coerce").fillna(0)
        dfq = dfq.sort_values("upvotes", ascending=False)
    except Exception:
        pass
    deduped = dfq.drop_duplicates("_n", keep="first")
    dups = len(dfq) - len(deduped)
    after = len(deduped)

    print("\n================ DAMRU DATASET AUDIT ================")
    print("repo                : %s" % REPO)
    print("rows (before)       : %d" % before)
    print("low-quality dropped : %d" % lowq)
    print("duplicates dropped  : %d" % dups)
    print("rows (after clean)  : %d" % after)
    print("reduction           : %.1f%%" % (100.0 * (before - after) / max(1, before)))
    print("\nby language (after):")
    for lang, c in deduped["lang"].astype(str).value_counts().head(12).items():
        print("   %-10s %d" % (lang, c))
    print("\ntop intents (after):")
    for it, c in deduped["intent"].astype(str).value_counts().head(15).items():
        print("   %-40s %d" % (it[:40], c))
    print("====================================================\n")

    if not APPLY:
        print("DRY RUN (set APPLY=1 to rewrite the cleaned dataset). No changes made.")
        return

    out_df = deduped.drop(columns=["_n"], errors="ignore")
    maxid = 0
    if "id" in out_df.columns:
        try:
            maxid = int(pd.to_numeric(out_df["id"], errors="coerce").fillna(0).max())
        except Exception:
            maxid = 0
    ts = int(time.time())
    path_in_repo = "data/train-%013d-%d.parquet" % (maxid, ts)
    local = "/tmp/%s" % os.path.basename(path_in_repo)
    out_df.to_parquet(local, index=False)
    print("Uploading cleaned shard %s (%d rows)..." % (path_in_repo, after))
    api.upload_file(
        path_or_fileobj=local, path_in_repo=path_in_repo,
        repo_id=REPO, repo_type="dataset",
    )
    for f in files:
        try:
            api.delete_file(path_in_repo=f, repo_id=REPO, repo_type="dataset")
            print("  deleted old shard", f)
        except Exception as e:
            print("  could not delete", f, "->", str(e)[:80])
    print("APPLIED. Dataset is now deduped + quality-filtered.")


if __name__ == "__main__":
    main()
