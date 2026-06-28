"""Wikipedia harvester (bulk). Returns knowledge Q&A. No API key, very high yield."""
import json
import time
import urllib.request
import urllib.parse

_UA = {"User-Agent": "DamruAI/1.0 (educational self-learning bot)"}


def _get(url, timeout=30):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def search_titles(subject, limit=40):
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "list": "search", "srsearch": subject,
         "srlimit": limit, "format": "json", "srnamespace": 0}
    )
    try:
        d = _get(url)
        return [x["title"] for x in d.get("query", {}).get("search", [])]
    except Exception:
        return []


def extracts(titles):
    """Get intro extracts for up to 20 titles in ONE call."""
    if not titles:
        return {}
    url = "https://en.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
        {"action": "query", "prop": "extracts", "exintro": 1, "explaintext": 1,
         "format": "json", "redirects": 1, "titles": "|".join(titles[:20])}
    )
    out = {}
    try:
        d = _get(url)
        for _, page in d.get("query", {}).get("pages", {}).items():
            t = page.get("title")
            ex = page.get("extract", "")
            if t and ex:
                out[t] = ex
    except Exception:
        pass
    return out


def harvest(subject, limit=40):
    titles = search_titles(subject, limit)
    out = []
    for i in range(0, len(titles), 20):
        chunk = titles[i:i + 20]
        for title, ex in extracts(chunk).items():
            if ex and len(ex) > 120:
                out.append({
                    "question": "Explain in depth: %s (in the context of %s)." % (title, subject),
                    "answer": ex,
                    "intent": subject.lower().replace(" ", "_"),
                    "lang": "en",
                })
        time.sleep(0.05)
    return out
