#!/usr/bin/env python3
"""Embed new Damru knowledge with BAAI/bge-m3 via HF InferenceClient.

Designed as a small post-step after Supabase->HF sync. It processes a bounded
batch, upserts halfvec(1024) rows, then prunes the hot cache to HOT_KEEP_ROWS.
Failure never deletes knowledge and should not block the core HF sync workflow.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests

SB_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SB_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")
PROVIDER = os.getenv("EMBED_PROVIDER", "hf-inference")
MAX_ROWS = max(1, min(int(os.getenv("HOT_EMBED_MAX_ROWS", "32")), 256))
BATCH = max(1, min(int(os.getenv("HOT_EMBED_BATCH", "4")), 32))
KEEP_ROWS = max(100, min(int(os.getenv("HOT_KEEP_ROWS", "10000")), 50000))
REPORT = os.getenv("HOT_EMBED_REPORT", "hot_embed_report.json")

HEADERS = {"apikey": SB_KEY, "Authorization": "Bearer " + SB_KEY, "Content-Type": "application/json"}


def last_hot_id() -> int:
    r = requests.get(
        f"{SB_URL}/rest/v1/damru_hot_vectors", headers=HEADERS,
        params={"select": "knowledge_id", "order": "knowledge_id.desc", "limit": "1"}, timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"hot table read failed {r.status_code}: {r.text[:250]}")
    rows = r.json() or []
    return int(rows[0]["knowledge_id"]) if rows else 0


def fetch_new(after_id: int) -> list[dict[str, Any]]:
    r = requests.get(
        f"{SB_URL}/rest/v1/damru_knowledge", headers=HEADERS,
        params={
            "select": "id,question,answer,intent,lang,created_at",
            "id": f"gt.{after_id}", "order": "id.asc", "limit": str(MAX_ROWS),
        }, timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"knowledge read failed {r.status_code}: {r.text[:250]}")
    return r.json() or []


def pool_and_normalize(value: Any, expected_batch: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 3:  # batched token embeddings -> mean pool
        arr = arr.mean(axis=1)
    elif arr.ndim == 2 and expected_batch == 1 and arr.shape[0] != 1 and arr.shape[1] == 1024:
        arr = arr.mean(axis=0, keepdims=True)  # single-text token embeddings
    if arr.ndim != 2 or arr.shape[0] != expected_batch:
        raise RuntimeError(f"Unexpected embedding shape {arr.shape}; batch={expected_batch}")
    if arr.shape[1] != 1024:
        raise RuntimeError(f"BGE-M3 dimension must be 1024, got {arr.shape[1]}")
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.maximum(norm, 1e-12)
    return arr


def embed(client: InferenceClient, texts: list[str]) -> np.ndarray:
    last = None
    for attempt in range(4):
        try:
            value = client.feature_extraction(texts, normalize=True)
            return pool_and_normalize(value, len(texts))
        except Exception as exc:
            last = exc
            time.sleep(2 ** attempt)
    raise RuntimeError(f"HF feature_extraction failed: {str(last)[:300]}")


def halfvec(v: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in v) + "]"


def upsert(rows: list[dict[str, Any]], vecs: np.ndarray) -> None:
    payload = []
    for row, vec in zip(rows, vecs):
        payload.append({
            "knowledge_id": int(row["id"]),
            "question": (row.get("question") or "")[:2000],
            "answer": (row.get("answer") or "")[:6000],
            "intent": (row.get("intent") or "general")[:80],
            "lang": row.get("lang") or "en",
            "source": "damru_knowledge",
            "created_at": row.get("created_at"),
            "embedding": halfvec(vec),
        })
    r = requests.post(
        f"{SB_URL}/rest/v1/damru_hot_vectors", headers={**HEADERS, "Prefer": "return=minimal,resolution=merge-duplicates"},
        params={"on_conflict": "knowledge_id"}, json=payload, timeout=90,
    )
    if not r.ok:
        raise RuntimeError(f"hot vector upsert failed {r.status_code}: {r.text[:300]}")


def prune() -> int:
    r = requests.post(
        f"{SB_URL}/rest/v1/rpc/prune_damru_hot", headers=HEADERS,
        json={"keep_rows": KEEP_ROWS}, timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"hot prune failed {r.status_code}: {r.text[:250]}")
    return int(r.json() or 0)


def self_test() -> None:
    x = np.ones((2, 1024), dtype=np.float32)
    y = pool_and_normalize(x, 2)
    assert y.shape == (2, 1024)
    assert abs(float(np.linalg.norm(y[0])) - 1.0) < 1e-5
    assert halfvec(y[0]).startswith("[")
    print("Hot Embed Sync self-test PASS")


def main() -> int:
    if os.getenv("SELF_TEST") == "1":
        self_test(); return 0
    if not (SB_URL and SB_KEY and HF_TOKEN):
        raise RuntimeError("SUPABASE_URL, SUPABASE_SERVICE_KEY and HF_TOKEN required")
    from huggingface_hub import InferenceClient
    client = InferenceClient(model=MODEL, token=HF_TOKEN, provider=PROVIDER)
    start = last_hot_id()
    rows = fetch_new(start)
    done = 0
    for pos in range(0, len(rows), BATCH):
        part = rows[pos:pos + BATCH]
        texts = [((x.get("question") or "") + "\n" + (x.get("answer") or "")[:1200]).strip() for x in part]
        vecs = embed(client, texts)
        upsert(part, vecs)
        done += len(part)
        print(f"embedded {done}/{len(rows)} through id {part[-1]['id']}", flush=True)
    removed = prune()
    report = {"model": MODEL, "provider": PROVIDER, "start_id": start,
              "new_rows_found": len(rows), "embedded": done, "pruned": removed,
              "hot_keep_rows": KEEP_ROWS}
    Path(REPORT).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
