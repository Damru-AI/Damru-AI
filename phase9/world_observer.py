#!/usr/bin/env python3
"""Damru World Observer v1.

Collects lawful, public, near-real-time signals into a QUARANTINE table.
It never writes directly to damru_knowledge and never trains on raw feeds.

Keyless sources:
- GDELT: global news index
- NASA EONET: open natural-event feed
- USGS: live earthquake feed
- Hacker News: public tech signal feed
- Configurable RSS/Atom feeds
- Bluesky public search (optional queries)

Optional official API:
- YouTube Data API (YOUTUBE_KEY + YOUTUBE_QUERIES)

Environment:
  SUPABASE_URL
  SUPABASE_SERVICE_KEY   service-side secret; never put in frontend
  OBS_TABLE              default damru_observations
  WORLD_QUERIES          comma-separated news queries
  RSS_URLS               comma-separated public RSS/Atom URLs
  BLUESKY_QUERIES        comma-separated public search queries
  YOUTUBE_KEY
  YOUTUBE_QUERIES
  MAX_PER_SOURCE         default 20
  OUT_FILE               default world_observations.jsonl

Usage:
  python phase9/world_observer.py
  python phase9/world_observer.py --self-test
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote, urlencode
import xml.etree.ElementTree as ET

import requests

SB_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SB_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
OBS_TABLE = os.getenv("OBS_TABLE", "damru_observations")
MAX_PER_SOURCE = max(1, min(int(os.getenv("MAX_PER_SOURCE", "20")), 100))
OUT_FILE = os.getenv("OUT_FILE", "world_observations.jsonl")
TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "15"))
USER_AGENT = "Damru-World-Observer/1.0 (+https://damru-ai.vercel.app)"

WORLD_QUERIES = [x.strip() for x in os.getenv(
    "WORLD_QUERIES",
    "artificial intelligence,India science,space mission,climate disaster,public health,cybersecurity",
).split(",") if x.strip()]
RSS_URLS = [x.strip() for x in os.getenv("RSS_URLS", "").split(",") if x.strip()]
BLUESKY_QUERIES = [x.strip() for x in os.getenv("BLUESKY_QUERIES", "").split(",") if x.strip()]
YT_QUERIES = [x.strip() for x in os.getenv("YOUTUBE_QUERIES", "").split(",") if x.strip()]
YOUTUBE_KEY = os.getenv("YOUTUBE_KEY", "")

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json,text/xml,application/xml,*/*"})


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(value: Any, limit: int = 4000) -> str:
    text = html.unescape(str(value or ""))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def parse_time(value: Any) -> str:
    if value is None or value == "":
        return now_iso()
    if isinstance(value, (int, float)):
        # seconds or milliseconds
        v = float(value)
        if v > 10_000_000_000:
            v /= 1000.0
        try:
            return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()
        except Exception:
            return now_iso()
    raw = str(value).strip()
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).isoformat()
    except Exception:
        pass
    try:
        return parsedate_to_datetime(raw).astimezone(timezone.utc).isoformat()
    except Exception:
        return now_iso()


def fingerprint(source: str, url: str, title: str) -> str:
    stable = "|".join([source.lower().strip(), url.strip(), title.lower().strip()])
    return hashlib.sha256(stable.encode("utf-8", "ignore")).hexdigest()


def observation(*, source: str, source_type: str, title: Any, summary: Any,
                url: Any, observed_at: Any, trust_tier: str,
                metadata: dict[str, Any] | None = None) -> dict[str, Any] | None:
    title_s = clean_text(title, 500)
    url_s = str(url or "").strip()[:2000]
    summary_s = clean_text(summary, 6000)
    if not title_s or not url_s:
        return None
    return {
        "fingerprint": fingerprint(source, url_s, title_s),
        "source": source[:80],
        "source_type": source_type[:50],
        "title": title_s,
        "summary": summary_s,
        "url": url_s,
        "observed_at": parse_time(observed_at),
        "fetched_at": now_iso(),
        "trust_tier": trust_tier,
        "verification_status": "pending",
        "metadata": metadata or {},
    }


def get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    last = None
    for attempt in range(3):
        try:
            r = SESSION.get(url, params=params, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as exc:
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET failed: {url}: {last}")


def fetch_gdelt() -> list[dict[str, Any]]:
    out = []
    per_query = max(3, min(MAX_PER_SOURCE, 10))
    for query in WORLD_QUERIES[:10]:
        data = get_json("https://api.gdeltproject.org/api/v2/doc/doc", {
            "query": query,
            "mode": "ArtList",
            "maxrecords": per_query,
            "format": "json",
            "sort": "HybridRel",
        })
        for item in (data or {}).get("articles", [])[:per_query]:
            row = observation(
                source="GDELT", source_type="news", title=item.get("title"),
                summary=item.get("seendate") or item.get("domain"),
                url=item.get("url"), observed_at=item.get("seendate"),
                trust_tier="B",
                metadata={"query": query, "domain": item.get("domain"), "language": item.get("language")},
            )
            if row:
                out.append(row)
    return out


def fetch_nasa_eonet() -> list[dict[str, Any]]:
    data = get_json("https://eonet.gsfc.nasa.gov/api/v3/events", {"status": "open", "limit": MAX_PER_SOURCE})
    out = []
    for item in (data or {}).get("events", [])[:MAX_PER_SOURCE]:
        geom = (item.get("geometry") or [{}])[-1]
        categories = [x.get("title") for x in item.get("categories", []) if x.get("title")]
        sources = [x.get("url") for x in item.get("sources", []) if x.get("url")]
        row = observation(
            source="NASA EONET", source_type="earth_event", title=item.get("title"),
            summary=item.get("description") or ", ".join(categories),
            url=item.get("link") or (sources[0] if sources else "https://eonet.gsfc.nasa.gov/"),
            observed_at=geom.get("date"), trust_tier="A",
            metadata={"event_id": item.get("id"), "categories": categories, "geometry": geom},
        )
        if row:
            out.append(row)
    return out


def fetch_usgs() -> list[dict[str, Any]]:
    data = get_json("https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson")
    out = []
    for feature in (data or {}).get("features", [])[:MAX_PER_SOURCE]:
        prop = feature.get("properties") or {}
        row = observation(
            source="USGS", source_type="earthquake", title=prop.get("title"),
            summary=f"Magnitude {prop.get('mag')}; alert {prop.get('alert')}; tsunami {prop.get('tsunami')}",
            url=prop.get("url"), observed_at=prop.get("time"), trust_tier="A",
            metadata={"magnitude": prop.get("mag"), "place": prop.get("place"), "coordinates": (feature.get("geometry") or {}).get("coordinates")},
        )
        if row:
            out.append(row)
    return out


def fetch_hacker_news() -> list[dict[str, Any]]:
    ids = get_json("https://hacker-news.firebaseio.com/v0/newstories.json") or []
    out = []
    for item_id in ids[:MAX_PER_SOURCE]:
        try:
            item = get_json("https://hacker-news.firebaseio.com/v0/item/%s.json" % item_id) or {}
        except Exception:
            continue
        url = item.get("url") or ("https://news.ycombinator.com/item?id=%s" % item_id)
        row = observation(
            source="Hacker News", source_type="technology_signal", title=item.get("title"),
            summary=item.get("text") or f"score={item.get('score')}; comments={item.get('descendants')}",
            url=url, observed_at=item.get("time"), trust_tier="C",
            metadata={"item_id": item_id, "score": item.get("score"), "author": item.get("by")},
        )
        if row:
            out.append(row)
    return out


def _xml_text(node: ET.Element, names: Iterable[str]) -> str:
    for child in node.iter():
        tag = child.tag.split("}")[-1].lower()
        if tag in names and child.text:
            return child.text.strip()
    return ""


def fetch_rss() -> list[dict[str, Any]]:
    out = []
    for feed_url in RSS_URLS[:30]:
        try:
            r = SESSION.get(feed_url, timeout=TIMEOUT)
            r.raise_for_status()
            root = ET.fromstring(r.content)
        except Exception as exc:
            print("RSS error", feed_url, str(exc)[:100], flush=True)
            continue
        entries = [x for x in root.iter() if x.tag.split("}")[-1].lower() in {"item", "entry"}]
        for item in entries[:MAX_PER_SOURCE]:
            link = _xml_text(item, {"link"})
            if not link:
                for c in item.iter():
                    if c.tag.split("}")[-1].lower() == "link" and c.attrib.get("href"):
                        link = c.attrib["href"]
                        break
            row = observation(
                source="RSS", source_type="rss", title=_xml_text(item, {"title"}),
                summary=_xml_text(item, {"description", "summary", "content"}),
                url=link, observed_at=_xml_text(item, {"pubdate", "published", "updated"}),
                trust_tier="B", metadata={"feed_url": feed_url},
            )
            if row:
                out.append(row)
    return out


def fetch_bluesky() -> list[dict[str, Any]]:
    out = []
    for query in BLUESKY_QUERIES[:10]:
        data = get_json("https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts", {"q": query, "limit": MAX_PER_SOURCE})
        for item in (data or {}).get("posts", [])[:MAX_PER_SOURCE]:
            rec = item.get("record") or {}
            author = item.get("author") or {}
            uri = item.get("uri") or ""
            rkey = uri.rsplit("/", 1)[-1] if uri else ""
            handle = author.get("handle") or ""
            post_url = ("https://bsky.app/profile/%s/post/%s" % (handle, rkey)) if handle and rkey else "https://bsky.app/"
            text = rec.get("text") or ""
            row = observation(
                source="Bluesky", source_type="social_signal", title=text[:180] or query,
                summary=text, url=post_url, observed_at=rec.get("createdAt"), trust_tier="C",
                metadata={"query": query, "author": handle, "likes": item.get("likeCount"), "reposts": item.get("repostCount")},
            )
            if row:
                out.append(row)
    return out


def fetch_youtube() -> list[dict[str, Any]]:
    if not (YOUTUBE_KEY and YT_QUERIES):
        return []
    out = []
    for query in YT_QUERIES[:10]:
        data = get_json("https://www.googleapis.com/youtube/v3/search", {
            "part": "snippet", "type": "video", "order": "date",
            "maxResults": min(MAX_PER_SOURCE, 25), "q": query, "key": YOUTUBE_KEY,
        })
        for item in (data or {}).get("items", []):
            vid = (item.get("id") or {}).get("videoId")
            sn = item.get("snippet") or {}
            if not vid:
                continue
            row = observation(
                source="YouTube", source_type="video_signal", title=sn.get("title"),
                summary=sn.get("description"), url=("https://www.youtube.com/watch?v=%s" % vid),
                observed_at=sn.get("publishedAt"), trust_tier="C",
                metadata={"query": query, "channel": sn.get("channelTitle"), "video_id": vid},
            )
            if row:
                out.append(row)
    return out


SOURCES = [
    ("usgs", fetch_usgs),
    ("nasa_eonet", fetch_nasa_eonet),
    ("gdelt", fetch_gdelt),
    ("hacker_news", fetch_hacker_news),
    ("rss", fetch_rss),
    ("bluesky", fetch_bluesky),
    ("youtube", fetch_youtube),
]


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    found = {}
    for row in rows:
        fp = row.get("fingerprint")
        if fp and fp not in found:
            found[fp] = row
    return list(found.values())


def write_jsonl(rows: list[dict[str, Any]], path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def insert_supabase(rows: list[dict[str, Any]]) -> int:
    if not (SB_URL and SB_KEY and rows):
        print("Supabase insert skipped (missing service credentials or no rows).", flush=True)
        return 0
    endpoint = f"{SB_URL}/rest/v1/{OBS_TABLE}?on_conflict=fingerprint"
    headers = {
        "apikey": SB_KEY,
        "Authorization": "Bearer " + SB_KEY,
        "Content-Type": "application/json",
        "Prefer": "return=minimal,resolution=ignore-duplicates",
    }
    inserted = 0
    for pos in range(0, len(rows), 100):
        batch = rows[pos:pos + 100]
        for attempt in range(4):
            try:
                r = SESSION.post(endpoint, headers=headers, json=batch, timeout=30)
                if r.ok:
                    inserted += len(batch)
                    break
                if r.status_code in {429, 500, 502, 503, 504}:
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Supabase {r.status_code}: {r.text[:300]}")
            except Exception as exc:
                if attempt == 3:
                    print("Supabase batch failed:", str(exc)[:200], flush=True)
                else:
                    time.sleep(2 ** attempt)
    return inserted


def self_test() -> int:
    a = observation(source="USGS", source_type="earthquake", title="M 5 event", summary="test", url="https://example.org/a", observed_at=0, trust_tier="A")
    b = observation(source="USGS", source_type="earthquake", title="M 5 event", summary="duplicate", url="https://example.org/a", observed_at=0, trust_tier="A")
    c = observation(source="RSS", source_type="rss", title="Science update", summary="ok", url="https://example.org/b", observed_at="2026-07-10T00:00:00Z", trust_tier="B")
    assert a and b and c
    rows = dedupe([a, b, c])
    assert len(rows) == 2
    assert rows[0]["verification_status"] == "pending"
    assert rows[0]["trust_tier"] == "A"
    assert len(rows[0]["fingerprint"]) == 64
    print("World Observer self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    rows = []
    stats = {}
    for name, fn in SOURCES:
        try:
            part = fn()
            rows.extend(part)
            stats[name] = {"ok": True, "rows": len(part)}
            print(name, "+", len(part), flush=True)
        except Exception as exc:
            stats[name] = {"ok": False, "error": str(exc)[:180]}
            print(name, "ERROR", str(exc)[:180], flush=True)

    rows = dedupe(rows)
    write_jsonl(rows, OUT_FILE)
    inserted = insert_supabase(rows)
    report = {
        "generated": now_iso(), "observations": len(rows),
        "insert_attempt_rows": inserted, "sources": stats,
        "out_file": OUT_FILE, "table": OBS_TABLE,
    }
    Path("world_observer_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
