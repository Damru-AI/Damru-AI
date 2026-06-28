"""
Storage layer: dedup (local sqlite) + batched insert into Supabase.
Schema written matches the HF sync (question, answer, intent, lang, upvotes, created_at).
intent encodes the subject/type; upvotes encodes quality (higher = better) for training weighting.
Crash-proof: retries with backoff; marks dedup only AFTER a successful insert.

SOURCE-LEVEL DEDUP (permanent): if you add a UNIQUE index on the normalized
question in Supabase (see phase4/SUPABASE_DEDUP.sql), duplicate rows are rejected
by the database with HTTP 409. A whole batch would normally fail on one conflict,
so insert_batch() falls back to row-by-row insert on 409 and silently skips the
conflicting (already-present) rows while still saving the new ones.
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


def _headers():
    return {
        "apikey": config.SUPABASE_KEY,
        "Authorization": "Bearer " + config.SUPABASE_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }


def _post(url, payload):
    """POST json payload. Returns ('ok', None) or ('http', code) or ('err', msg)."""
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=_headers())
    try:
        with urllib.request.urlopen(req, timeout=60):
            return ("ok", None)
    except urllib.error.HTTPError as e:
        return ("http", e.code)
    except Exception as e:
        return ("err", str(e)[:120])


def _insert_one_by_one(url, rows):
    """Insert rows individually; skip duplicates (409) and bad rows. Returns inserted count."""
    inserted, ok_qs = 0, []
    for r in rows:
        status, info = _post(url, [r])
        if status == "ok":
            inserted += 1
            ok_qs.append(r["question"])
        elif status == "http" and info == 409:
            # already present in DB (unique-index conflict) -> mark seen, skip
            ok_qs.append(r["question"])
        elif status == "http" and info in (429, 500, 502, 503):
            time.sleep(0.8)
            status2, _ = _post(url, [r])
            if status2 == "ok":
                inserted += 1
                ok_qs.append(r["question"])
        # other errors -> drop silently
    if ok_qs:
        _mark(ok_qs)
    return inserted


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
    for attempt in range(4):
        status, info = _post(url, clean)
        if status == "ok":
            _mark([r["question"] for r in clean])
            return len(clean)
        if status == "http":
            if info == 409:
                # A duplicate exists in the batch vs the DB unique index. PostgREST
                # fails the whole batch -> retry row-by-row, skipping duplicates.
                return _insert_one_by_one(url, clean)
            if info in (429, 500, 502, 503):
                time.sleep(2 ** attempt)
                continue
            return 0  # bad request etc -> don't loop forever
        # network/other error -> backoff and retry
        time.sleep(1.5 * (attempt + 1))
    return 0
