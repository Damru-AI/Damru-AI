"""
Stack Exchange harvester: REAL human Q&A with accepted answers.
Routes a subject to the right site (Stack Overflow for coding, Math.SE for maths, etc.).
Handles gzip, strips HTML. Optional STACKEXCHANGE_KEY raises the quota.
"""
import json
import gzip
import re
import html
import time
import urllib.request
import urllib.parse

import config

_TAG = re.compile(r"<[^>]+>")
_SITE_MAP = [
    ("coding", "stackoverflow"), ("programming", "stackoverflow"),
    ("algorithm", "stackoverflow"), ("data structure", "stackoverflow"),
    ("mathemat", "math"), ("algebra", "math"), ("calculus", "math"),
    ("geometry", "math"), ("number theory", "math"), ("probability", "math"),
    ("statistics", "stats"), ("physics", "physics"), ("chemistry", "chemistry"),
    ("biology", "biology"), ("computer science", "cs"), ("economics", "economics"),
    ("philosophy", "philosophy"), ("law", "law"), ("electronic", "electronics"),
    ("electrical", "electronics"), ("medicine", "medicalsciences"),
]


def site_for(subject):
    s = (subject or "").lower()
    for key, site in _SITE_MAP:
        if key in s:
            return site
    return "stackoverflow"


def _clean(t):
    return " ".join(_TAG.sub(" ", html.unescape(t or "")).split())


def _get(url, timeout=40):
    req = urllib.request.Request(
        url, headers={"User-Agent": "DamruAI/1.0", "Accept-Encoding": "gzip"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    try:
        raw = gzip.decompress(raw)
    except Exception:
        pass
    return json.loads(raw.decode("utf-8", "ignore"))


def harvest(subject, pagesize=100):
    site = site_for(subject)
    base = "https://api.stackexchange.com/2.3"
    keyp = ("&key=" + config.STACKEXCHANGE_KEY) if config.STACKEXCHANGE_KEY else ""
    qurl = (
        "%s/search/advanced?order=desc&sort=votes&accepted=True&site=%s"
        "&pagesize=%d&filter=withbody&intitle=%s%s"
        % (base, site, pagesize, urllib.parse.quote(subject), keyp)
    )
    try:
        dq = _get(qurl)
    except Exception:
        return []
    items = dq.get("items", [])
    aids = [str(it["accepted_answer_id"]) for it in items if it.get("accepted_answer_id")]
    amap = {}
    if aids:
        ids = ";".join(aids[:100])
        aurl = (
            "%s/answers/%s?order=desc&sort=votes&site=%s&pagesize=100&filter=withbody%s"
            % (base, ids, site, keyp)
        )
        try:
            da = _get(aurl)
            for a in da.get("items", []):
                amap[a["answer_id"]] = _clean(a.get("body", ""))
        except Exception:
            pass
    out = []
    for it in items:
        ans = amap.get(it.get("accepted_answer_id"), "")
        title = _clean(it.get("title", ""))
        body = _clean(it.get("body", ""))
        if title and ans and len(ans) > 40:
            q = title if len(body) < 20 else (title + " — " + body[:600])
            out.append({
                "question": q,
                "answer": ans,
                "intent": subject.lower().replace(" ", "_"),
                "lang": "en",
                "upvotes": int(it.get("score", 0)),
            })
        time.sleep(0.01)
    return out
