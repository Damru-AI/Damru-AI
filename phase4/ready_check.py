#!/usr/bin/env python3
"""
Fine-tune READINESS check.

Counts current HuggingFace rows and reports progress toward READY_TARGET (default
2,000,000). When the target is reached it writes FINE_TUNE_READY.md (a big banner)
and opens ONE GitHub issue so you know the exact moment to start fine-tuning.
Always writes TRAINING_PROGRESS.md so you can watch the count grow.

Env:
  HF_TOKEN, HF_REPO (default 'Damaru-ai/damru-knowledge')
  READY_TARGET (default 2000000)
  GITHUB_TOKEN, GITHUB_REPOSITORY (provided automatically by GitHub Actions)
"""
import os
import json
import urllib.request
import urllib.error
from datetime import datetime, timezone

from huggingface_hub import HfApi, hf_hub_download

REPO = os.environ.get("HF_REPO", "Damaru-ai/damru-knowledge")
TOKEN = os.environ.get("HF_TOKEN", "") or None
TARGET = int(os.environ.get("READY_TARGET", "2000000"))
GH_TOKEN = os.environ.get("GITHUB_TOKEN", "")
GH_REPO = os.environ.get("GITHUB_REPOSITORY", "")
READY_FILE = "FINE_TUNE_READY.md"
PROGRESS_FILE = "TRAINING_PROGRESS.md"


def hf_row_count():
    import pandas as pd
    api = HfApi(token=TOKEN)
    files = [f for f in api.list_repo_files(repo_id=REPO, repo_type="dataset") if f.endswith(".parquet")]
    total = 0
    for f in files:
        try:
            p = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=f, token=TOKEN)
            total += len(pd.read_parquet(p, columns=["question"]))
        except Exception as e:
            print("skip", f, str(e)[:80])
    return total, len(files)


def open_issue(title, body):
    if not GH_TOKEN or not GH_REPO:
        print("No GITHUB_TOKEN/REPOSITORY -> skip issue")
        return
    url = "https://api.github.com/repos/%s/issues" % GH_REPO
    data = json.dumps({"title": title, "body": body}).encode()
    req = urllib.request.Request(url, data=data, headers={
        "Authorization": "Bearer " + GH_TOKEN,
        "Accept": "application/vnd.github+json",
        "User-Agent": "damru-ready-check",
        "Content-Type": "application/json",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print("Opened issue:", json.loads(r.read().decode()).get("html_url"))
    except Exception as e:
        print("issue creation failed:", str(e)[:120])


def main():
    total, shards = hf_row_count()
    pct = 100.0 * total / max(1, TARGET)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    bar = "#" * int(min(40, pct / 2.5)) + "-" * (40 - int(min(40, pct / 2.5)))
    print("HF rows: %d / %d  (%.2f%%)  shards=%d" % (total, TARGET, pct, shards))

    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        f.write("# \U0001F415 Damru Training Progress\n\n")
        f.write("Updated: %s\n\n" % now)
        f.write("**Rows:** %d / %d  (**%.2f%%**)\n\n" % (total, TARGET, pct))
        f.write("```\n[%s] %.1f%%\n```\n\n" % (bar, pct))
        f.write("Shards: %d\n\n" % shards)
        f.write("Fine-tuning starts automatically-ready at %d rows.\n" % TARGET)

    if total >= TARGET and not os.path.exists(READY_FILE):
        with open(READY_FILE, "w", encoding="utf-8") as f:
            f.write("# \u2705 FINE-TUNE READY!\n\n")
            f.write("Damru has reached **%d rows** (target %d) on %s.\n\n" % (total, TARGET, now))
            f.write("Open `phase4/damru_finetune.ipynb` in Colab and start training.\n")
        open_issue(
            "\u2705 Damru is FINE-TUNE READY (%d rows)" % total,
            "Damru's dataset reached **%d rows** (target %d) on %s.\n\n"
            "Time to start fine-tuning: open `phase4/damru_finetune.ipynb` in Google Colab."
            % (total, TARGET, now),
        )
        print("*** TARGET REACHED -> wrote FINE_TUNE_READY.md + opened issue ***")
    else:
        print("Not yet at target (or already alerted).")


if __name__ == "__main__":
    main()
