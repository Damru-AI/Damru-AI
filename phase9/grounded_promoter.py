#!/usr/bin/env python3
"""Damru Grounded Promoter v1.

Converts only VERIFIED rows from damru_observations into deterministic,
source-cited Q&A rows in damru_knowledge. No LLM is used, so the promotion
stage cannot invent facts. Pending/rejected observations are never touched.

After a successful insert (or duplicate-safe conflict), the observation is
marked promoted. Writing is OFF unless PROMOTE_ENABLED=1.

Required env:
  SUPABASE_URL
  SUPABASE_SERVICE_KEY
Optional:
  OBS_TABLE=damru_observations
  KNOWLEDGE_TABLE=damru_knowledge
  PROMOTE_ENABLED=0
  PROMOTE_LIMIT=500
  MIN_VERIFICATION_SCORE=0.70
  MAX_AGE_DAYS=30
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

SB_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SB_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
OBS_TABLE = os.getenv("OBS_TABLE", "damru_observations")
KNOWLEDGE_TABLE = os.getenv("KNOWLEDGE_TABLE", "damru_knowledge")
ENABLED = os.getenv("PROMOTE_ENABLED", "0") == "1"
LIMIT = max(1, min(int(os.getenv("PROMOTE_LIMIT", "500")), 5000))
MIN_SCORE = float(os.getenv("MIN_VERIFICATION_SCORE", "0.70"))
MAX_AGE_DAYS = int(os.getenv("MAX_AGE_DAYS", "30"))
REPORT_FILE = os.getenv("PROMOTER_REPORT", "grounded_promoter_report.json")

HEADERS = {
    "apikey": SB_KEY,
    "Authorization": "Bearer " + SB_KEY,
    "Content-Type": "application/json",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(value: Any, limit: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:limit]


def parse_time(value: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(value or "").replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def language_of(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    return "en"


def verification_reason(row: dict[str, Any]) -> str:
    raw = row.get("verification_notes")
    if not raw:
        return "verified_source"
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return clean((data or {}).get("reason"), 120) or "verified_source"
    except Exception:
        return "verified_source"


def build_knowledge(row: dict[str, Any]) -> dict[str, Any] | None:
    if str(row.get("verification_status")) != "verified":
        return None
    score = float(row.get("verification_score") or 0)
    if score < MIN_SCORE:
        return None
    observed = parse_time(row.get("observed_at"))
    if observed < datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS):
        return None

    source = clean(row.get("source"), 80)
    source_type = clean(row.get("source_type"), 50).lower().replace(" ", "_") or "world"
    title = clean(row.get("title"), 500)
    summary = clean(row.get("summary"), 5000)
    url = clean(row.get("url"), 2000)
    if not source or not title or not url.startswith(("https://", "http://")):
        return None

    date_label = observed.strftime("%Y-%m-%d %H:%M UTC")
    question = f'What verified live update did {source} report about "{title}"?'
    body = summary if summary and summary.lower() != title.lower() else title
    answer = (
        f"As of {date_label}, {source} reported: {title}. {body}\n\n"
        f"Source: {url}\n"
        f"Verification: {verification_reason(row)} (score {score:.2f}).\n"
        "This is a time-sensitive observation; consult the cited source for later changes."
    )
    return {
        "question": question[:2000],
        "answer": answer[:12000],
        "intent": ("live_" + source_type)[:80],
        "lang": language_of(title + " " + summary),
        "upvotes": max(7, min(10, int(round(score * 10)))),
        "created_at": now_iso(),
    }


def fetch_verified() -> list[dict[str, Any]]:
    select = (
        "id,source,source_type,title,summary,url,observed_at,trust_tier,"
        "verification_status,verification_score,verification_notes"
    )
    r = requests.get(
        f"{SB_URL}/rest/v1/{OBS_TABLE}", headers=HEADERS,
        params={
            "select": select,
            "verification_status": "eq.verified",
            "order": "observed_at.asc",
            "limit": str(LIMIT),
        }, timeout=60,
    )
    if not r.ok:
        raise RuntimeError(f"Verified fetch failed {r.status_code}: {r.text[:300]}")
    return r.json() or []


def insert_knowledge(item: dict[str, Any]) -> None:
    r = requests.post(
        f"{SB_URL}/rest/v1/{KNOWLEDGE_TABLE}",
        headers={**HEADERS, "Prefer": "return=minimal,resolution=ignore-duplicates"},
        params={"on_conflict": "qnorm"}, json=[item], timeout=45,
    )
    if not r.ok:
        raise RuntimeError(f"Knowledge insert failed {r.status_code}: {r.text[:300]}")


def mark_promoted(observation_id: int) -> None:
    r = requests.patch(
        f"{SB_URL}/rest/v1/{OBS_TABLE}",
        headers={**HEADERS, "Prefer": "return=minimal"},
        params={"id": f"eq.{observation_id}", "verification_status": "eq.verified"},
        json={"verification_status": "promoted", "promoted_at": now_iso()}, timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"Promotion mark failed id={observation_id} {r.status_code}: {r.text[:250]}")


def proposals(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        item = build_knowledge(row)
        if item:
            out.append({"observation_id": int(row["id"]), "knowledge": item})
    return out


def self_test() -> int:
    row = {
        "id": 1,
        "source": "USGS",
        "source_type": "earthquake",
        "title": "M 5.2 earthquake near Test Region",
        "summary": "Magnitude 5.2; tsunami 0",
        "url": "https://earthquake.usgs.gov/example",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "verification_status": "verified",
        "verification_score": 0.98,
        "verification_notes": json.dumps({"reason": "official_authority_feed"}),
    }
    item = build_knowledge(row)
    assert item and item["intent"] == "live_earthquake"
    assert "https://earthquake.usgs.gov/example" in item["answer"]
    assert "official_authority_feed" in item["answer"]
    bad = dict(row, verification_status="pending")
    assert build_knowledge(bad) is None
    print("Grounded Promoter self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if not (SB_URL and SB_KEY):
        raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY")

    rows = fetch_verified()
    items = proposals(rows)
    report = {
        "generated": now_iso(),
        "mode": "enabled" if ENABLED else "dry-run",
        "verified_rows_found": len(rows),
        "promotion_candidates": len(items),
        "minimum_score": MIN_SCORE,
        "max_age_days": MAX_AGE_DAYS,
        "sample": items[:10],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)

    promoted = 0
    errors = []
    if not ENABLED:
        print("DRY RUN ONLY: PROMOTE_ENABLED is not 1. No rows changed.", flush=True)
    else:
        for entry in items:
            try:
                insert_knowledge(entry["knowledge"])
                mark_promoted(entry["observation_id"])
                promoted += 1
            except Exception as exc:
                errors.append({"id": entry["observation_id"], "error": str(exc)[:250]})
        print(f"PROMOTED {promoted} verified observations", flush=True)
        if errors:
            print("PROMOTION ERRORS", json.dumps(errors[:20], indent=2), flush=True)

    report["promoted"] = promoted
    report["errors"] = errors
    Path(REPORT_FILE).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
