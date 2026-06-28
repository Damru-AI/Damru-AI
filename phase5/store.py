"""
Storage layer: dedup (local sqlite) + batched insert into Supabase.
Schema written matches the HF sync (question, answer, intent, lang, upvotes, created_at).
intent encodes the subject/type; upvotes encodes quality (higher = better) for training weighting.
Crash-proof: retries with backoff; marks dedup only AFTER a successful insert.
"""
import json
import time
import hashlib
import sqlite3
import os
import urllib.request
import urllib.error
from datetime import datetime, timezone

import config

_DB = os.path.join(config.DATA_DIR, "seen.db")


def _conn():
    c = sqlite3.connect(_DB, timeout=30)
    c.execute("CREATE TABLE IF NOT EXISTS seen (h TEXT PRIMARY KEY)")
    return c


def _hash(q):
    norm = " ".join((q or "").lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def is_new(q):
    if not q:
        return False
    c = _conn()
    try:
        return c.execute("SELECT 1 FROM seen WHERE h=?", (_hash(q),)).fetchone() is None
    finally:
        c.close()


def _mark(qs):
    c = _conn()
    try:
        c.executemany("INSERT OR IGNORE INTO seen(h) VALUES(?)", [(_hash(q),) for q in qs])
        c.commit()
    finally:
        c.close()


def make_row(question, answer, intent, lang="en", quality=0.6, upvotes=None):
    q = (question or "").strip()
    a = (answer or "").strip()
    uv = int(upvotes) if upvotes is not None else int(round(max(0.0, min(1.0, quality)) * 10))
    return {
        "question": q,
        "answer": a,
        "intent": (intent or "general")[:80],
        "lang": lang or "en",
        "upvotes": uv,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def insert_batch(rows):
    """Insert valid + de-duped rows. Returns number actually inserted."""
    clean = []
    seen_local = set()
    for r in rows:
        q = (r.get("question") or "").strip()
        a = (r.get("answer") or "").strip()
        if len(q) <= 3 or len(a) <= 20:
            continue
        h = _hash(q)
        if h in seen_local:
            continue
        if not is_new(q):
            continue
        seen_local.add(h)
        clean.append(r)
    if not clean:
        return 0
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        # No DB configured (e.g. dry run) -> still mark to avoid repeats in-process
        _mark([r["question"] for r in clean])
        return len(clean)
    url = config.SUPABASE_URL + "/rest/v1/" + config.TABLE
    data = json.dumps(clean).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "apikey": config.SUPABASE_KEY,
            "Authorization": "Bearer " + config.SUPABASE_KEY,
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60):
                _mark([r["question"] for r in clean])
                return len(clean)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            # Bad request etc -> don't loop forever
            return 0
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return 0
