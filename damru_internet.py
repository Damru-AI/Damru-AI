"""
damru_internet.py
=================
Step 2 / 5 -- Live Internet A->E  (Damru ko insaan-jaisa internet access)

Layers (each optional, graceful fallback, env-gated):
  A  SEARCH   SearXNG -> Tavily -> Brave -> Wikipedia -> DuckDuckGo(lite)
  B  READ     Crawl4AI -> Jina Reader(r.jina.ai) -> trafilatura -> BeautifulSoup
  C  BROWSE   Playwright(sync) -> plain read      [JS-heavy / multi-step nav]
  D  WATCH    yt-dlp captions -> faster-whisper    [video -> text]
  E  RESEARCH search -> read top-K in parallel -> synthesize w/ [n] citations

Plugs into Step 1 (damru_tools):
    from damru_internet import internet_providers
    from damru_tools import get_belt, Providers
    belt = get_belt(Providers(**internet_providers(llm_complete=my_llm),
                              knowledge=..., code_exec=...))

FastAPI router for app.py:
    from damru_internet import build_internet_router
    api.include_router(build_internet_router(llm_complete=_llm_complete))

ENV: SEARXNG_URL, TAVILY_KEY, BRAVE_KEY, JINA_API_KEY(optional),
     INTERNET_SEARCH_ORDER, INTERNET_READ_ORDER, INTERNET_MAX_RESULTS,
     INTERNET_RESEARCH_PAGES, INTERNET_TIMEOUT, WHISPER_MODEL(base)
Optional deps (see requirements-internet.txt): requests, beautifulsoup4, lxml,
     trafilatura, crawl4ai, playwright, yt-dlp, faster-whisper.

ZERO hard deps: every backend is lazy-imported; `import damru_internet` never breaks.
Built by Shiva AI for Damru.
"""

from __future__ import annotations

import os
import re
import json
import html
import logging
import threading
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional

__version__ = "1.0.0"
__all__ = ["Internet", "get_internet", "internet_providers",
           "build_internet_router", "SearchResult"]

log = logging.getLogger("damru.internet")
if not logging.getLogger().handlers:
    logging.basicConfig(level=os.getenv("DAMRU_LOG_LEVEL", "INFO"))

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
UA = os.getenv("INTERNET_UA",
               "Mozilla/5.0 (compatible; DamruBot/1.0; +https://github.com/Damru-AI)")
TIMEOUT        = int(os.getenv("INTERNET_TIMEOUT", "15"))
MAX_RESULTS    = int(os.getenv("INTERNET_MAX_RESULTS", "6"))
RESEARCH_PAGES = int(os.getenv("INTERNET_RESEARCH_PAGES", "4"))

SEARXNG_URL = os.getenv("SEARXNG_URL", "").rstrip("/")
TAVILY_KEY  = os.getenv("TAVILY_KEY") or os.getenv("TAVILY_API_KEY")
BRAVE_KEY   = os.getenv("BRAVE_KEY") or os.getenv("BRAVE_API_KEY")
JINA_READER = os.getenv("JINA_READER_URL", "https://r.jina.ai/").rstrip("/") + "/"
JINA_KEY    = os.getenv("JINA_API_KEY")

SEARCH_ORDER = [s.strip() for s in os.getenv(
    "INTERNET_SEARCH_ORDER", "searxng,tavily,brave,wikipedia,ddg").split(",") if s.strip()]
READ_ORDER = [s.strip() for s in os.getenv(
    "INTERNET_READ_ORDER", "crawl4ai,jina,trafilatura,bs4").split(",") if s.strip()]


# --------------------------------------------------------------------------- #
# HTTP + HTML helpers (requests preferred, urllib fallback -> works anywhere)
# --------------------------------------------------------------------------- #
def _http_get(url: str, params: dict = None, headers: dict = None,
              timeout: int = TIMEOUT) -> str:
    if params:
        url = url + ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
    h = {"User-Agent": UA}
    if headers:
        h.update(headers)
    try:
        import requests  # type: ignore
        r = requests.get(url, headers=h, timeout=timeout)
        r.raise_for_status()
        return r.text
    except ImportError:
        req = urllib.request.Request(url, headers=h)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", "replace")


def _http_get_json(url: str, params: dict = None, headers: dict = None,
                   timeout: int = TIMEOUT) -> Any:
    return json.loads(_http_get(url, params, headers, timeout) or "null")


def _http_post_json(url: str, payload: dict, headers: dict = None,
                    timeout: int = TIMEOUT) -> Any:
    h = {"User-Agent": UA, "Content-Type": "application/json"}
    if headers:
        h.update(headers)
    try:
        import requests  # type: ignore
        r = requests.post(url, json=payload, headers=h, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except ImportError:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=h, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))


_TAG_RE = re.compile(r"<[^>]+>")


def _clean_html(raw: str, max_chars: int = 6000) -> str:
    if not raw:
        return ""
    try:
        from bs4 import BeautifulSoup  # type: ignore
        soup = BeautifulSoup(raw, "html.parser")
        for t in soup(["script", "style", "noscript", "header", "footer",
                       "nav", "svg", "form", "aside"]):
            t.decompose()
        text = soup.get_text("\n")
    except Exception:
        text = html.unescape(_TAG_RE.sub(" ", raw))
    text = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", text)).strip()
    return text[:max_chars]


@dataclass
class SearchResult:
    title: str = ""
    url: str = ""
    snippet: str = ""
    source: str = ""

    def to_dict(self) -> Dict[str, str]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Internet engine
# --------------------------------------------------------------------------- #
class Internet:
    def __init__(self, llm_complete: Optional[Callable] = None) -> None:
        self.llm_complete = llm_complete

    # ------------------------- A: SEARCH ------------------------- #
    def _s_searxng(self, q: str, n: int) -> List[SearchResult]:
        if not SEARXNG_URL:
            return []
        data = _http_get_json(f"{SEARXNG_URL}/search",
                              {"q": q, "format": "json", "safesearch": 1})
        return [SearchResult(r.get("title", ""), r.get("url", ""),
                             r.get("content", ""), "searxng")
                for r in (data or {}).get("results", [])[:n]]

    def _s_tavily(self, q: str, n: int) -> List[SearchResult]:
        if not TAVILY_KEY:
            return []
        data = _http_post_json("https://api.tavily.com/search",
                               {"api_key": TAVILY_KEY, "query": q,
                                "max_results": n, "search_depth": "basic"})
        out = [SearchResult(r.get("title", ""), r.get("url", ""),
                            r.get("content", ""), "tavily")
               for r in (data or {}).get("results", [])[:n]]
        if data and data.get("answer"):
            out.insert(0, SearchResult("Tavily direct answer", "", data["answer"], "tavily"))
        return out

    def _s_brave(self, q: str, n: int) -> List[SearchResult]:
        if not BRAVE_KEY:
            return []
        data = _http_get_json("https://api.search.brave.com/res/v1/web/search",
                              {"q": q, "count": n},
                              headers={"X-Subscription-Token": BRAVE_KEY,
                                       "Accept": "application/json"})
        web = ((data or {}).get("web", {}) or {}).get("results", [])
        return [SearchResult(r.get("title", ""), r.get("url", ""),
                             r.get("description", ""), "brave") for r in web[:n]]

    def _s_wikipedia(self, q: str, n: int) -> List[SearchResult]:
        data = _http_get_json("https://en.wikipedia.org/w/api.php",
                              {"action": "query", "list": "search", "srsearch": q,
                               "format": "json", "srlimit": n})
        hits = (((data or {}).get("query", {}) or {}).get("search", []))
        out = []
        for r in hits[:n]:
            title = r.get("title", "")
            slug = urllib.parse.quote(title.replace(" ", "_"))
            out.append(SearchResult(title, f"https://en.wikipedia.org/wiki/{slug}",
                                    _clean_html(r.get("snippet", "")), "wikipedia"))
        return out

    def _s_ddg(self, q: str, n: int) -> List[SearchResult]:
        raw = _http_get("https://lite.duckduckgo.com/lite/", {"q": q})
        out: List[SearchResult] = []
        try:
            from bs4 import BeautifulSoup  # type: ignore
            soup = BeautifulSoup(raw, "html.parser")
            for a in soup.select("a.result-link")[:n]:
                out.append(SearchResult(a.get_text(" ", strip=True),
                                        a.get("href", ""), "", "ddg"))
        except Exception as e:
            log.info("ddg parse failed: %s", e)
        return out

    def search(self, query: str, n: int = MAX_RESULTS,
               order: List[str] = None) -> List[SearchResult]:
        fns = {"searxng": self._s_searxng, "tavily": self._s_tavily,
               "brave": self._s_brave, "wikipedia": self._s_wikipedia,
               "ddg": self._s_ddg}
        for name in (order or SEARCH_ORDER):
            fn = fns.get(name)
            if not fn:
                continue
            try:
                res = fn(query, n)
                if res:
                    log.info("search via %s -> %d hits", name, len(res))
                    return res
            except Exception as e:
                log.info("search %s failed: %s", name, e)
        return []

    # ------------------------- B: READ ------------------------- #
    def _r_crawl4ai(self, url: str) -> str:
        import asyncio
        from crawl4ai import AsyncWebCrawler  # type: ignore

        async def _go():
            async with AsyncWebCrawler(verbose=False) as c:
                r = await c.arun(url=url)
                return r.markdown or r.cleaned_html or ""
        return asyncio.run(_go())

    def _r_jina(self, url: str) -> str:
        headers = {"Accept": "text/plain"}
        if JINA_KEY:
            headers["Authorization"] = f"Bearer {JINA_KEY}"
        return _http_get(JINA_READER + url, headers=headers)

    def _r_trafilatura(self, url: str) -> str:
        import trafilatura  # type: ignore
        return trafilatura.extract(trafilatura.fetch_url(url)) or ""

    def _r_bs4(self, url: str) -> str:
        return _clean_html(_http_get(url))

    def read(self, url: str, order: List[str] = None, max_chars: int = 6000) -> str:
        if not url:
            return ""
        fns = {"crawl4ai": self._r_crawl4ai, "jina": self._r_jina,
               "trafilatura": self._r_trafilatura, "bs4": self._r_bs4}
        for name in (order or READ_ORDER):
            fn = fns.get(name)
            if not fn:
                continue
            try:
                txt = fn(url)
                if txt and txt.strip():
                    log.info("read via %s -> %d chars", name, len(txt))
                    return txt.strip()[:max_chars]
            except Exception as e:
                log.info("read %s failed: %s", name, e)
        return ""

    # ------------------------- C: BROWSE ------------------------- #
    def browse(self, target: str, max_chars: int = 6000) -> str:
        url = target if re.match(r"^https?://", target or "") else None
        if url is None:
            hits = self.search(target, n=1)
            if not hits:
                return ""
            url = hits[0].url
        try:
            from playwright.sync_api import sync_playwright  # type: ignore
            with sync_playwright() as p:
                b = p.chromium.launch(headless=True)
                pg = b.new_page(user_agent=UA)
                pg.goto(url, timeout=TIMEOUT * 1000, wait_until="domcontentloaded")
                pg.wait_for_timeout(1500)
                text = pg.inner_text("body")
                b.close()
                return (text or "").strip()[:max_chars]
        except Exception as e:
            log.info("playwright browse failed (%s); falling back to read", e)
        return self.read(url, max_chars=max_chars)

    # ------------------------- D: WATCH ------------------------- #
    @staticmethod
    def _vtt_text(vtt: str) -> str:
        lines: List[str] = []
        for ln in vtt.splitlines():
            ln = ln.strip()
            if (not ln or "-->" in ln or ln.isdigit()
                    or ln.upper().startswith(("WEBVTT", "KIND", "LANGUAGE"))):
                continue
            ln = re.sub(r"<[^>]+>", "", ln)
            if ln and (not lines or lines[-1] != ln):
                lines.append(ln)
        return " ".join(lines)

    def _transcript(self, url: str) -> str:
        # 1) captions via yt-dlp (fast, no GPU/Whisper)
        try:
            import glob
            import tempfile
            import yt_dlp  # type: ignore
            tmp = tempfile.mkdtemp(prefix="damru_yt_")
            opts = {"skip_download": True, "writesubtitles": True,
                    "writeautomaticsub": True,
                    "subtitleslangs": ["en", "hi", "en-US"],
                    "subtitlesformat": "vtt",
                    "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
                    "quiet": True, "no_warnings": True}
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
            subs = glob.glob(os.path.join(tmp, "*.vtt"))
            if subs:
                with open(subs[0], encoding="utf-8", errors="replace") as fh:
                    return self._vtt_text(fh.read())
        except Exception as e:
            log.info("captions path failed: %s", e)
        # 2) audio -> faster-whisper
        try:
            import glob
            import tempfile
            import yt_dlp  # type: ignore
            from faster_whisper import WhisperModel  # type: ignore
            tmp = tempfile.mkdtemp(prefix="damru_aud_")
            opts = {"format": "bestaudio/best",
                    "outtmpl": os.path.join(tmp, "%(id)s.%(ext)s"),
                    "quiet": True, "no_warnings": True,
                    "postprocessors": [{"key": "FFmpegExtractAudio",
                                        "preferredcodec": "mp3"}]}
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.extract_info(url, download=True)
            files = (glob.glob(os.path.join(tmp, "*.mp3"))
                     or glob.glob(os.path.join(tmp, "*")))
            if files:
                model = WhisperModel(os.getenv("WHISPER_MODEL", "base"),
                                     device="cpu", compute_type="int8")
                segs, _ = model.transcribe(files[0])
                return " ".join(s.text.strip() for s in segs)
        except Exception as e:
            log.info("whisper path failed: %s", e)
        return ""

    def watch(self, url: str, max_chars: int = 6000, summarize: bool = True) -> str:
        transcript = (self._transcript(url) or "").strip()
        if not transcript:
            return ""
        transcript = transcript[:max_chars]
        if summarize and self.llm_complete:
            try:
                s = self.llm_complete([
                    {"role": "system", "content":
                        "Summarise this video transcript in 6-8 crisp bullets; "
                        "keep key facts, names and numbers."},
                    {"role": "user", "content": transcript}])
                return f"SUMMARY:\n{s}\n\n--- TRANSCRIPT (excerpt) ---\n{transcript[:1500]}"
            except Exception as e:
                log.info("watch summarize failed: %s", e)
        return transcript

    # ------------------------- E: RESEARCH ------------------------- #
    def deep_research(self, query: str, pages: int = RESEARCH_PAGES,
                      synth: bool = True) -> Dict[str, Any]:
        hits = self.search(query, n=max(pages, MAX_RESULTS))
        picked = [h for h in hits if h.url][:pages]
        docs: List[tuple] = []
        if picked:
            with ThreadPoolExecutor(max_workers=min(4, len(picked))) as ex:
                for h, txt in ex.map(lambda x: (x, self.read(x.url, max_chars=3000)), picked):
                    if txt:
                        docs.append((h, txt))
        notes = "\n\n".join(f"[{i + 1}] {h.title} -- {h.url}\n{txt[:1200]}"
                            for i, (h, txt) in enumerate(docs))
        answer = None
        if synth and self.llm_complete and notes:
            try:
                answer = self.llm_complete([
                    {"role": "system", "content":
                        "You are a research assistant. Using ONLY the numbered "
                        "sources, write a concise well-structured answer with inline "
                        "[n] citations. If sources conflict, say so."},
                    {"role": "user", "content": f"Question: {query}\n\nSources:\n{notes}"}])
            except Exception as e:
                log.info("research synth failed: %s", e)
        return {"query": query, "answer": answer,
                "sources": [h.to_dict() for h, _ in docs], "notes": notes[:4000]}

    # ------------------------- health ------------------------- #
    def health(self) -> Dict[str, Any]:
        def _imp(m: str) -> bool:
            try:
                __import__(m)
                return True
            except Exception:
                return False
        return {
            "search": {"searxng": bool(SEARXNG_URL), "tavily": bool(TAVILY_KEY),
                       "brave": bool(BRAVE_KEY), "wikipedia": True,
                       "ddg": _imp("bs4"), "order": SEARCH_ORDER},
            "read": {"crawl4ai": _imp("crawl4ai"), "jina": True,
                     "trafilatura": _imp("trafilatura"), "bs4": _imp("bs4"),
                     "order": READ_ORDER},
            "browse": {"playwright": _imp("playwright")},
            "watch": {"yt_dlp": _imp("yt_dlp"),
                      "faster_whisper": _imp("faster_whisper")},
            "research": {"llm": bool(self.llm_complete)},
            "http": {"requests": _imp("requests")},
        }


# --------------------------------------------------------------------------- #
# Singleton + Step-1 providers + FastAPI router
# --------------------------------------------------------------------------- #
_INET: Optional[Internet] = None
_INET_LOCK = threading.Lock()


def get_internet(llm_complete: Optional[Callable] = None) -> Internet:
    global _INET
    if _INET is None or llm_complete is not None:
        with _INET_LOCK:
            if _INET is None or llm_complete is not None:
                _INET = Internet(llm_complete=llm_complete)
    return _INET


def internet_providers(llm_complete: Optional[Callable] = None,
                       internet: Optional[Internet] = None) -> Dict[str, Callable]:
    """Return callables keyed to match damru_tools.Providers (web_search, deep_read,
    browse, watch_video). Feed straight into Providers(**internet_providers())."""
    net = internet or get_internet(llm_complete)
    return {
        "web_search": lambda query, **_: [r.to_dict() for r in net.search(query)],
        "deep_read": lambda url=None, query=None, **_: net.read(url or query or ""),
        "browse": lambda url=None, query=None, **_: net.browse(url or query or ""),
        "watch_video": lambda url=None, query=None, **_: net.watch(url or query or ""),
    }


def build_internet_router(llm_complete: Optional[Callable] = None,
                          internet: Optional[Internet] = None):
    """Return a FastAPI APIRouter exposing /internet/*. fastapi imported lazily."""
    from fastapi import APIRouter, Body  # type: ignore
    net = internet or get_internet(llm_complete)
    r = APIRouter(prefix="/internet", tags=["internet"])

    @r.get("/health")
    def _health():
        return net.health()

    @r.get("/search")
    def _search(q: str, n: int = MAX_RESULTS):
        return {"query": q, "results": [x.to_dict() for x in net.search(q, n)]}

    @r.get("/read")
    def _read(url: str):
        return {"url": url, "text": net.read(url)}

    @r.get("/watch")
    def _watch(url: str):
        return {"url": url, "transcript": net.watch(url)}

    @r.get("/browse")
    def _browse(q: str):
        return {"target": q, "text": net.browse(q)}

    @r.get("/research")
    def _research(q: str, pages: int = RESEARCH_PAGES):
        return net.deep_research(q, pages)

    @r.post("/research")
    def _research_post(payload: dict = Body(...)):
        return net.deep_research(payload.get("q", ""),
                                 int(payload.get("pages", RESEARCH_PAGES)))
    return r


# --------------------------------------------------------------------------- #
# Self-test (offline-safe: no network calls are executed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    net = get_internet()
    print("=== HEALTH ===")
    print(json.dumps(net.health(), indent=2))

    print("\n=== PROVIDERS ===")
    prov = internet_providers()
    print("keys:", list(prov.keys()))

    print("\n=== WIRE INTO STEP-1 BELT ===")
    try:
        from damru_tools import get_belt, Providers
        belt = get_belt(Providers(**prov, knowledge=lambda q, **_: f"(kb) {q}"))
        print("belt tools:", belt.names())
        for q in ["latest ISRO news today",
                  "read https://example.com/article",
                  "watch this https://youtu.be/abc",
                  "what is PRAYAS"]:
            print("  route", repr(q), "->", [p["name"] for p in belt.plan(q)])
    except Exception as e:
        print("tools wiring skipped:", e)

    print("\n=== ROUTER ===")
    try:
        rt = build_internet_router()
        print("routes:", sorted({getattr(x, "path", "?") for x in rt.routes}))
    except Exception as e:
        print("router build skipped (fastapi not in sandbox):", e)

    print("\n=== VTT PARSER UNIT TEST (offline) ===")
    vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nHello <c>world</c>\n\n"
           "00:00:03.000 --> 00:00:05.000\nHello world\n\n"
           "00:00:05.000 --> 00:00:07.000\nDamru is live\n")
    print("parsed:", Internet._vtt_text(vtt))

    print("\nOK damru_internet v" + __version__)
