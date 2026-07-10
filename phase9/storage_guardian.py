#!/usr/bin/env python3
"""Damru Storage Guardian v1.

When public.damru_knowledge reaches ARCHIVE_THRESHOLD_MB (default 400 MB):
1. Read trusted HF _sync_state.json.
2. Start a short Supabase maintenance lock (writers receive retryable errors).
3. Upload only rows newer than HF last_id as a new Parquet shard.
4. Verify the shard exists, update HF state, and write an archive manifest.
5. Ask a guarded Supabase RPC to TRUNCATE damru_knowledge (CONTINUE IDENTITY).

Existing HF shards are never deleted or rewritten. Embeddings are intentionally
not archived; they can be rebuilt in the HF FAISS/RAG index.

Destructive mode is OFF unless ARCHIVE_ENABLED=1. With it unset, the script is
a read-only dry run and prints what it would do.

Required environment:
  SUPABASE_URL
  SUPABASE_SERVICE_KEY
  HF_TOKEN
Optional:
  HF_REPO=Damaru-ai/damru-knowledge
  ARCHIVE_THRESHOLD_MB=400
  ARCHIVE_ENABLED=0
  LOCK_MINUTES=30
  STATE_FILE=_sync_state.json
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

SB_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SB_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_REPO = os.getenv("HF_REPO", "Damaru-ai/damru-knowledge")
STATE_FILE = os.getenv("STATE_FILE", "_sync_state.json")
THRESHOLD_MB = int(os.getenv("ARCHIVE_THRESHOLD_MB", "400"))
ENABLED = os.getenv("ARCHIVE_ENABLED", "0") == "1"
LOCK_MINUTES = max(5, min(int(os.getenv("LOCK_MINUTES", "30")), 120))
PAGE = 1000

HEADERS = {
    "apikey": SB_KEY,
    "Authorization": "Bearer " + SB_KEY,
    "Content-Type": "application/json",
}
_API = None


def hf_api():
    global _API
    if _API is None:
        from huggingface_hub import HfApi
        _API = HfApi(token=HF_TOKEN)
    return _API


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def rpc(name: str, payload: dict[str, Any] | None = None) -> Any:
    r = requests.post(
        f"{SB_URL}/rest/v1/rpc/{name}", headers=HEADERS,
        json=payload or {}, timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"RPC {name} failed {r.status_code}: {r.text[:400]}")
    if not r.text:
        return None
    data = r.json()
    if isinstance(data, list) and len(data) == 1:
        return data[0]
    return data


def status() -> dict[str, Any]:
    data = rpc("damru_storage_status")
    if not isinstance(data, dict):
        raise RuntimeError(f"Unexpected storage status: {data!r}")
    return data


def read_hf_state() -> int:
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(
            HF_REPO, STATE_FILE, repo_type="dataset", token=HF_TOKEN,
        )
        with open(p, encoding="utf-8") as f:
            return int(json.load(f).get("last_id", -1))
    except Exception as exc:
        raise RuntimeError(
            "HF sync state is missing/unreadable; refusing to archive: " + str(exc)[:200]
        )


def fetch_unsynced(last_id: int, max_id: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cur = last_id
    select = "id,question,answer,intent,lang,upvotes,created_at"
    while cur < max_id:
        url = f"{SB_URL}/rest/v1/damru_knowledge"
        params = {
            "select": select,
            "id": f"gt.{cur}",
            "order": "id.asc",
            "limit": str(PAGE),
        }
        r = requests.get(url, headers=HEADERS, params=params, timeout=90)
        if not r.ok:
            raise RuntimeError(f"Supabase fetch failed {r.status_code}: {r.text[:300]}")
        batch = [x for x in (r.json() or []) if int(x.get("id", 0)) <= max_id]
        if not batch:
            break
        rows.extend(batch)
        cur = int(batch[-1]["id"])
        print(f"fetched unsynced: {len(rows)} rows through id {cur}", flush=True)
        if len(batch) < PAGE:
            break
    return rows


def clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for x in rows:
        q = (x.get("question") or "").strip()
        a = (x.get("answer") or "").strip()
        if len(q) <= 3 or len(a) <= 20:
            continue
        out.append({
            "id": int(x["id"]),
            "question": q,
            "answer": a,
            "intent": x.get("intent") or "general",
            "lang": x.get("lang") or "en",
            "upvotes": x.get("upvotes") or 0,
            "created_at": x.get("created_at") or "",
        })
    return out


def upload_new_rows(rows: list[dict[str, Any]], max_id: int, archive_id: str) -> tuple[str, str]:
    if not rows:
        return f"already-synced-through-{max_id}", ""
    from datasets import Dataset
    local = Path("/tmp/damru-storage-guardian.parquet")
    Dataset.from_list(rows).to_parquet(str(local))
    digest = hashlib.sha256(local.read_bytes()).hexdigest()
    path = f"data/storage-archive-{max_id:013d}-{archive_id}.parquet"
    api = hf_api()
    api.upload_file(
        path_or_fileobj=str(local), path_in_repo=path,
        repo_id=HF_REPO, repo_type="dataset",
    )
    files = set(api.list_repo_files(HF_REPO, repo_type="dataset"))
    if path not in files:
        raise RuntimeError("HF upload verification failed: shard not listed")
    return path, digest


def write_hf_state(last_id: int) -> None:
    buf = io.BytesIO(json.dumps({
        "last_id": int(last_id), "updated": int(time.time()),
        "writer": "storage_guardian_v1",
    }).encode())
    hf_api().upload_file(
        path_or_fileobj=buf, path_in_repo=STATE_FILE,
        repo_id=HF_REPO, repo_type="dataset",
    )


def create_backup_tag(archive_id: str) -> str:
    tag = "pre-archive-" + archive_id
    hf_api().create_tag(
        HF_REPO, tag=tag, repo_type="dataset",
        tag_message="Damru Storage Guardian pre-archive checkpoint",
    )
    return tag


def upload_manifest(manifest: dict[str, Any], archive_id: str) -> str:
    path = f"archives/storage-guardian-{archive_id}.json"
    hf_api().upload_file(
        path_or_fileobj=io.BytesIO(json.dumps(manifest, indent=2).encode()),
        path_in_repo=path, repo_id=HF_REPO, repo_type="dataset",
    )
    return path


def require_env() -> None:
    missing = [name for name, val in [
        ("SUPABASE_URL", SB_URL),
        ("SUPABASE_SERVICE_KEY", SB_KEY),
        ("HF_TOKEN", HF_TOKEN),
    ] if not val]
    if missing:
        raise RuntimeError("Missing required secrets: " + ", ".join(missing))


def main() -> int:
    require_env()
    before = status()
    size_mb = float(before.get("total_bytes", 0)) / 1024 / 1024
    row_count = int(before.get("row_count", 0))
    max_id = int(before.get("max_id", 0))
    print(json.dumps({
        "mode": "enabled" if ENABLED else "dry-run",
        "threshold_mb": THRESHOLD_MB,
        "current_mb": round(size_mb, 2),
        "rows": row_count,
        "max_id": max_id,
    }, indent=2), flush=True)

    if size_mb < THRESHOLD_MB:
        print("Below threshold; nothing to archive.", flush=True)
        return 0
    if not ENABLED:
        print("WOULD ARCHIVE, but ARCHIVE_ENABLED is not 1. No data changed.", flush=True)
        return 0
    if row_count <= 0 or max_id <= 0:
        print("No knowledge rows to archive.", flush=True)
        return 0

    archive_id = utc_stamp()
    locked = False
    try:
        snap = rpc("damru_archive_begin", {
            "p_archive_id": archive_id,
            "p_lock_minutes": LOCK_MINUTES,
        })
        locked = True
        expected_count = int(snap["row_count"])
        expected_max = int(snap["max_id"])
        if expected_count != row_count or expected_max != max_id:
            raise RuntimeError("Snapshot changed before maintenance lock; retry later")

        hf_last = read_hf_state()
        if hf_last < 0:
            raise RuntimeError("Invalid HF last_id")
        print(f"HF state last_id={hf_last}; Supabase max_id={max_id}", flush=True)

        rows = clean_rows(fetch_unsynced(hf_last, max_id)) if hf_last < max_id else []
        if hf_last < max_id and (not rows or max(x["id"] for x in rows) != max_id):
            raise RuntimeError("Could not fetch every unsynced row through snapshot max_id")

        tag = create_backup_tag(archive_id)
        hf_path, sha256 = upload_new_rows(rows, max_id, archive_id)
        if hf_last < max_id:
            write_hf_state(max_id)
        verified_last = read_hf_state()
        if verified_last < max_id:
            raise RuntimeError("HF state verification failed; refusing Supabase truncate")

        manifest = {
            "archive_id": archive_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "supabase_rows": row_count,
            "supabase_max_id": max_id,
            "previous_hf_last_id": hf_last,
            "uploaded_new_rows": len(rows),
            "hf_path": hf_path,
            "sha256": sha256,
            "pre_archive_tag": tag,
        }
        manifest_path = upload_manifest(manifest, archive_id)

        result = rpc("damru_archive_finalize", {
            "p_archive_id": archive_id,
            "p_expected_count": row_count,
            "p_expected_max_id": max_id,
            "p_hf_path": hf_path,
            "p_sha256": sha256,
            "p_manifest_path": manifest_path,
        })
        locked = False
        print("ARCHIVE COMPLETE", json.dumps(result, indent=2), flush=True)
        print("Supabase usage dashboard may take up to one hour to refresh.", flush=True)
        return 0
    except Exception as exc:
        print("ARCHIVE ABORTED:", str(exc), flush=True)
        if locked:
            try:
                rpc("damru_archive_abort", {"p_archive_id": archive_id})
                print("Maintenance lock released.", flush=True)
            except Exception as unlock_exc:
                print("Unlock warning:", str(unlock_exc)[:200], flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
