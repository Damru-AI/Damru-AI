#!/usr/bin/env python3
"""
Damru EMERGENCY data restore
============================
The main branch lost rows (6.5M -> 269k) because a DESTRUCTIVE writer
(the legacy Sync / reset-sync workflow doing push_to_hub) deleted the
harvested data/ parquet shards. HuggingFace keeps the FULL git history, so
every deleted shard still exists in an older commit.

This scans recent commits, finds the revision that had the MOST data/*.parquet
shards (the pre-clobber state), and restores every MISSING shard back onto
main in a SINGLE commit (so it never trips the 128-commits/hour limit).
It is purely ADDITIVE -- it never deletes anything currently on main.

Env:
  HF_TOKEN            (required)
  HF_REPO             default Damaru-ai/damru-knowledge
  MAX_COMMITS_SCAN    how many recent commits to inspect (default 150)
  DRY_RUN             "1" = only report what WOULD be restored (default 0)
"""
import os
import time

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_REPO", "Damaru-ai/damru-knowledge")
MAX_COMMITS_SCAN = int(os.environ.get("MAX_COMMITS_SCAN", "150"))
DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"


def _data_shards(api, revision):
    try:
        files = api.list_repo_files(HF_REPO, repo_type="dataset",
                                    revision=revision, token=HF_TOKEN)
    except Exception as e:
        print("  list failed @%s: %s" % (revision[:8], str(e)[:80]), flush=True)
        return set()
    return set(f for f in files
              if f.startswith("data/") and f.endswith(".parquet"))


def main():
    assert HF_TOKEN, "HF_TOKEN required"
    from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
    api = HfApi(token=HF_TOKEN)

    current = _data_shards(api, "main")
    print("main currently has %d data shards" % len(current), flush=True)

    commits = api.list_repo_commits(HF_REPO, repo_type="dataset")
    print("repo has %d commits in history" % len(commits), flush=True)

    best_rev = None
    best_files = set()
    for c in commits[:MAX_COMMITS_SCAN]:
        shards = _data_shards(api, c.commit_id)
        if len(shards) > len(best_files):
            best_files = shards
            best_rev = c.commit_id
            print("  candidate %s -> %d shards (%s)"
                  % (c.commit_id[:8], len(shards),
                     (c.title or "")[:50]), flush=True)

    if not best_rev:
        print("No richer revision found. Nothing to restore.", flush=True)
        return

    missing = sorted(best_files - current)
    print("\nBest revision %s had %d shards; main has %d; MISSING %d"
          % (best_rev[:8], len(best_files), len(current), len(missing)),
          flush=True)
    if not missing:
        print("main already has all shards. Nothing to restore.", flush=True)
        return
    if DRY_RUN:
        for m in missing[:20]:
            print("  would restore:", m, flush=True)
        print("DRY_RUN -> stopping (set DRY_RUN=0 to actually restore).",
              flush=True)
        return

    print("Downloading %d shards from history..." % len(missing), flush=True)
    ops = []
    for i, fn in enumerate(missing):
        for attempt in range(6):
            try:
                local = hf_hub_download(HF_REPO, fn, repo_type="dataset",
                                        revision=best_rev, token=HF_TOKEN)
                break
            except Exception as e:
                if attempt == 5:
                    raise
                time.sleep(min(120, 8 * (2 ** attempt)))
        ops.append(CommitOperationAdd(path_in_repo=fn, path_or_fileobj=local))
        if (i + 1) % 25 == 0:
            print("  fetched %d/%d" % (i + 1, len(missing)), flush=True)

    print("Committing %d restored shards in ONE commit..." % len(ops),
          flush=True)
    for attempt in range(6):
        try:
            api.create_commit(
                HF_REPO, repo_type="dataset", operations=ops,
                commit_message="EMERGENCY restore %d shards from %s"
                               % (len(ops), best_rev[:8]))
            break
        except Exception as e:
            s = str(e)
            if attempt == 5:
                raise
            wait = 1900 if ("per hour" in s or "repository commits" in s) \
                else min(180, 15 * (2 ** attempt))
            print("  commit retry in %ds (%s)" % (wait, s[:70]), flush=True)
            time.sleep(wait)
    print("RESTORE COMPLETE -> %d shards back on main." % len(ops), flush=True)


if __name__ == "__main__":
    main()
