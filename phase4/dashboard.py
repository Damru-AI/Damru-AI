#!/usr/bin/env python3
"""
Damru GROWTH DASHBOARD -> writes 'damru_dashboard.html' and prints a summary.

Reads the HuggingFace dataset (full breakdown) + the live Supabase row count.

Env:
  HF_TOKEN, HF_REPO (default 'Damaru-ai/damru-knowledge')
  SUPABASE_URL, SUPABASE_KEY, DAMRU_TABLE (default 'damru_knowledge')
"""
import os
import html
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

REPO = os.environ.get("HF_REPO", "Damaru-ai/damru-knowledge")
TOKEN = os.environ.get("HF_TOKEN", "") or None
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
TABLE = os.environ.get("DAMRU_TABLE", "damru_knowledge")
OUT = os.environ.get("DASHBOARD_OUT", "damru_dashboard.html")


def supabase_count():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    url = SUPABASE_URL + "/rest/v1/" + TABLE + "?select=id&limit=1"
    req = urllib.request.Request(url, headers={
        "apikey": SUPABASE_KEY,
        "Authorization": "Bearer " + SUPABASE_KEY,
        "Prefer": "count=exact",
        "Range-Unit": "items",
        "Range": "0-0",
    })
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            return int(cr.split("/")[-1])
    except Exception as e:
        print("supabase count failed:", str(e)[:100])
    return None


def load_hf():
    api = HfApi(token=TOKEN)
    files = [f for f in api.list_repo_files(repo_id=REPO, repo_type="dataset") if f.endswith(".parquet")]
    frames = []
    for f in files:
        try:
            p = hf_hub_download(repo_id=REPO, repo_type="dataset", filename=f, token=TOKEN)
            frames.append(pd.read_parquet(p))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bars(pairs, total):
    rows = []
    for label, c in pairs:
        pct = (100.0 * c / total) if total else 0
        rows.append(
            "<tr><td>%s</td><td style='text-align:right'>%d</td>"
            "<td><div style='background:#6c5ce7;height:12px;width:%.1f%%;border-radius:6px'></div></td></tr>"
            % (html.escape(str(label)[:48]), c, max(1.0, pct))
        )
    return "\n".join(rows)


def main():
    df = load_hf()
    hf_total = len(df)
    supa = supabase_count()

    by_lang, by_intent, by_day = [], [], []
    avg_up = 0
    if hf_total:
        for col in ("lang", "intent", "upvotes", "created_at"):
            if col not in df.columns:
                df[col] = "" if col != "upvotes" else 0
        by_lang = list(df["lang"].astype(str).value_counts().head(10).items())
        by_intent = list(df["intent"].astype(str).value_counts().head(15).items())
        try:
            avg_up = round(float(pd.to_numeric(df["upvotes"], errors="coerce").fillna(0).mean()), 2)
        except Exception:
            avg_up = 0
        try:
            d = pd.to_datetime(df["created_at"], errors="coerce", utc=True).dt.date
            by_day = list(d.value_counts().sort_index().tail(14).items())
        except Exception:
            by_day = []

    print("==== DAMRU DASHBOARD ====")
    print("HF rows       :", hf_total)
    print("Supabase rows :", supa)
    print("Avg upvotes   :", avg_up)
    print("Languages     :", by_lang)
    print("Top intents   :", by_intent[:5])

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>Damru Dashboard</title>"
        "<style>body{font-family:system-ui,Segoe UI,Roboto,Arial;background:#0f0f17;color:#eee;margin:0;padding:18px}"
        "h1{font-size:22px}h2{font-size:16px;color:#a29bfe;margin-top:26px}"
        ".cards{display:flex;flex-wrap:wrap;gap:12px}"
        ".card{background:#1b1b27;border:1px solid #2a2a3a;border-radius:14px;padding:16px 20px;min-width:130px}"
        ".big{font-size:30px;font-weight:700}"
        ".muted{color:#888;font-size:12px}"
        "table{width:100%;border-collapse:collapse;font-size:13px}td{padding:4px 8px;border-bottom:1px solid #23232f}"
        "</style></head><body>"
    )
    doc += "<h1>\U0001F415 Damru AI \u2014 Growth Dashboard</h1>"
    doc += "<div class='muted'>Updated: %s</div>" % now
    doc += "<div class='cards'>"
    doc += "<div class='card'><div class='muted'>HuggingFace rows</div><div class='big'>%s</div></div>" % f"{hf_total:,}"
    doc += "<div class='card'><div class='muted'>Supabase (live)</div><div class='big'>%s</div></div>" % (f"{supa:,}" if supa is not None else "-")
    doc += "<div class='card'><div class='muted'>Avg quality (upvotes)</div><div class='big'>%s</div></div>" % avg_up
    doc += "<div class='card'><div class='muted'>Target (training floor)</div><div class='big'>2M</div></div>"
    doc += "</div>"

    if by_day:
        doc += "<h2>Rows per day (HF, last 14 days)</h2><table>" + _bars(by_day, max(c for _, c in by_day)) + "</table>"
    if by_lang:
        doc += "<h2>By language</h2><table>" + _bars(by_lang, hf_total) + "</table>"
    if by_intent:
        doc += "<h2>Top intents / subjects</h2><table>" + _bars(by_intent, hf_total) + "</table>"
    doc += "</body></html>"

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(doc)
    print("Wrote", OUT)


if __name__ == "__main__":
    main()
