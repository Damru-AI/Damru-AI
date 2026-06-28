"""arXiv harvester. Returns research-grade Q&A from paper abstracts. No key."""
import re
import html
import urllib.request
import urllib.parse

_UA = {"User-Agent": "DamruAI/1.0 (educational self-learning bot)"}


def _get(url, timeout=40):
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "ignore")


def harvest(subject, limit=50):
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": "all:" + subject, "start": 0,
         "max_results": limit, "sortBy": "relevance"}
    )
    try:
        xml = _get(url)
    except Exception:
        return []
    out = []
    for e in re.findall(r"<entry>(.*?)</entry>", xml, re.S):
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        s = re.search(r"<summary>(.*?)</summary>", e, re.S)
        if not t or not s:
            continue
        title = html.unescape(" ".join(t.group(1).split()))
        summ = html.unescape(" ".join(s.group(1).split()))
        if len(summ) > 120:
            out.append({
                "question": "Summarize and critically explain the key idea of the "
                            "research '%s' (%s)." % (title, subject),
                "answer": summ,
                "intent": subject.lower().replace(" ", "_"),
                "lang": "en",
            })
    return out
