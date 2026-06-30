#!/usr/bin/env python3
"""
Damru Dataset Backup / Versioning  (gap #7)
===========================================
The HF knowledge base is Damru's single source of truth. This creates an
immutable, timestamped GIT TAG on the dataset repo so any run can be rolled
back if a future harvest ever corrupts data. Also snapshots _bulk_state.json
into a versioned snapshots/ path for quick inspection.

Env: HF_TOKEN (required), HF_REPO (default Damaru-ai/damru-knowledge),
     KEEP_TAGS (default 30 -- prune older auto tags beyond this many).
"""
import os, io, json, time
from datetime import datetime, timezone

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_REPO", "Damaru-ai/damru-knowledge")
KEEP_TAGS = int(os.environ.get("KEEP_TAGS", "30"))
STATE_FILE = "_bulk_state.json"


def main():
    assert HF_TOKEN, "HF_TOKEN required"
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=HF_TOKEN)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    tag = "backup-" + stamp

    # 1) snapshot the state file into a versioned path
    try:
        p = hf_hub_download(HF_REPO, STATE_FILE, repo_type="dataset",
                            token=HF_TOKEN)
        with open(p) as f:
            st = json.load(f)
        total = st.get("total", 0)
        for attempt in range(6):
            try:
                api.upload_file(
                    path_or_fileobj=io.BytesIO(
                        json.dumps(st, indent=2).encode()),
                    path_in_repo="snapshots/state-%s.json" % stamp,
                    repo_id=HF_REPO, repo_type="dataset")
                break
            except Exception as e:
                s = str(e)
                transient = any(x in s for x in
                                ("504", "503", "502", "500", "429",
                                 "Time-out", "Timeout", "Gateway"))
                if not transient or attempt == 5:
                    raise
                time.sleep(min(120, 8 * (2 ** attempt)))
        print("Snapshotted state (total=%s rows)" % total, flush=True)
    except Exception as e:
        print("state snapshot skipped:", str(e)[:140], flush=True)

    # 2) immutable tag of the whole repo at this commit
    try:
        api.create_tag(HF_REPO, tag=tag, repo_type="dataset",
                       tag_message="Auto backup %s" % stamp)
        print("Created tag:", tag, flush=True)
    except Exception as e:
        print("tag failed:", str(e)[:140], flush=True)

    # 3) prune old auto tags beyond KEEP_TAGS
    try:
        refs = api.list_repo_refs(HF_REPO, repo_type="dataset")
        auto = sorted([t.name for t in refs.tags
                       if t.name.startswith("backup-")])
        for old in auto[:-KEEP_TAGS] if len(auto) > KEEP_TAGS else []:
            try:
                api.delete_tag(HF_REPO, tag=old, repo_type="dataset")
                print("pruned tag", old, flush=True)
            except Exception:
                pass
    except Exception as e:
        print("prune skipped:", str(e)[:140], flush=True)
    print("BACKUP COMPLETE", flush=True)


if __name__ == "__main__":
    main()
