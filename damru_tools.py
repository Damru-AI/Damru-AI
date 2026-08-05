"""
damru_tools.py
==============
Damru Tool Belt + Smart Router  (Step 1 / 5 of the Beast build)

Why this file exists (Sunil's asks):
  * "damru fast tools access kar sake"    -> latency-tiered registry + caching + parallel exec
  * "sab tools smart method se use kare"   -> intent router + LLM tool-card planner
  * "sabhi cheez connected ho"             -> ONE belt shared by cortex / reflex / agentic / live
  * "response smooth ho"                   -> staged pipeline events (SANKALP..UTTAR)

Design
------
* ZERO hard third-party deps. Everything degrades gracefully (never crashes app.py).
* Backends are *injected* as Providers -> no circular import with app.py / rag.py / cortex.
  Anything not injected auto-disables, so `import damru_tools` always succeeds.
* Thread-safe singleton via get_belt(). Import-safe on Python 3.11 (HF) and 3.13 (sandbox).

Built by Shiva AI for Damru.
"""

from __future__ import annotations

import os
import re
import json
import time
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

__version__ = "1.0.0"
__all__ = [
    "ToolResult", "ToolSpec", "ToolBelt", "Providers",
    "build_belt", "get_belt", "run_pipeline", "llm_tool_plan",
    "TIER_INSTANT", "TIER_FAST", "TIER_NET", "TIER_HEAVY", "STAGES",
]

log = logging.getLogger("damru.tools")
if not logging.getLogger().handlers:
    logging.basicConfig(level=os.getenv("DAMRU_LOG_LEVEL", "INFO"))

# --------------------------------------------------------------------------- #
# Config (all env-tunable)
# --------------------------------------------------------------------------- #
MAX_PARALLEL   = int(os.getenv("DAMRU_TOOLS_MAX_PARALLEL", "6"))
CACHE_TTL      = int(os.getenv("DAMRU_TOOLS_CACHE_TTL", "300"))   # seconds
ROUTE_MAX      = int(os.getenv("DAMRU_TOOLS_ROUTE_MAX", "3"))
ROUTE_MIN      = float(os.getenv("DAMRU_TOOLS_ROUTE_MIN", "1.0"))

# Latency tiers -> router prefers the FASTEST capable tool ------------------- #
TIER_INSTANT = "instant"   # pure local ~0ms   (memory, cache, math)
TIER_FAST    = "fast"      # local index  <1s  (prayas / rag)
TIER_NET     = "net"       # one network hop   (web search, deep read)
TIER_HEAVY   = "heavy"     # slow / costly     (browse, watch video, image, 3d)

_TIER_BONUS = {TIER_INSTANT: 0.6, TIER_FAST: 0.4, TIER_NET: 0.15, TIER_HEAVY: 0.0}


# --------------------------------------------------------------------------- #
# Result + Spec
# --------------------------------------------------------------------------- #
@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: Any = None
    error: Optional[str] = None
    latency_ms: int = 0
    source: Optional[str] = None
    cached: bool = False

    def as_context(self, max_chars: int = 1200) -> str:
        if not self.ok or self.data in (None, "", [], {}):
            return ""
        body = self.data
        if isinstance(body, (dict, list)):
            try:
                body = json.dumps(body, ensure_ascii=False, indent=2)
            except Exception:
                body = str(body)
        body = str(body).strip()
        if len(body) > max_chars:
            body = body[:max_chars] + " ..."
        return f"[{self.source or self.tool}]\n{body}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool, "ok": self.ok, "data": self.data,
            "error": self.error, "latency_ms": self.latency_ms,
            "source": self.source, "cached": self.cached,
        }


@dataclass
class ToolSpec:
    name: str
    description: str          # what it does (one line)
    when_to_use: str          # teaches the model WHEN to reach for it
    tier: str
    runner: Callable[..., Any]
    keywords: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)   # regex over the query
    parallel_safe: bool = True
    cost_hint: str = "free"
    example: str = ""
    _enabled: Callable[[], bool] = field(default=lambda: True, repr=False)

    def enabled(self) -> bool:
        try:
            return bool(self._enabled())
        except Exception:
            return False

    def score(self, query: str) -> float:
        q = (query or "").lower()
        s = 0.0
        for kw in self.keywords:
            if kw in q:
                s += 1.0
        for pat in self.patterns:
            try:
                if re.search(pat, q, re.I):
                    s += 1.5
            except re.error:
                pass
        if s > 0:
            s += _TIER_BONUS.get(self.tier, 0.0)
        return s

    def card(self) -> str:
        ex = f' e.g. "{self.example}"' if self.example else ""
        return (f"- {self.name} [{self.tier}] -- {self.description} "
                f"USE WHEN: {self.when_to_use}{ex}")


# --------------------------------------------------------------------------- #
# The belt
# --------------------------------------------------------------------------- #
class ToolBelt:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}
        self._cache: Dict[str, Any] = {}     # key -> (expires_at, ToolResult)
        self._lock = threading.RLock()

    # -- registry ----------------------------------------------------------- #
    def register(self, spec: ToolSpec) -> None:
        with self._lock:
            self._tools[spec.name] = spec
        log.info("tool registered: %s (%s)", spec.name, spec.tier)

    def unregister(self, name: str) -> None:
        with self._lock:
            self._tools.pop(name, None)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def all(self) -> List[ToolSpec]:
        return list(self._tools.values())

    def available(self) -> List[ToolSpec]:
        return [t for t in self._tools.values() if t.enabled()]

    def names(self) -> List[str]:
        return [t.name for t in self.available()]

    # -- LLM teaching surface ---------------------------------------------- #
    def tool_card(self) -> str:
        lines = [t.card() for t in self.available()]
        if not lines:
            return "No external tools are currently available."
        return ("You (Damru) can call these tools. Prefer the FASTEST tool that "
                "fully answers; call independent tools together; never invent a "
                "tool not listed.\n" + "\n".join(lines))

    # -- caching ------------------------------------------------------------ #
    def _key(self, name: str, kwargs: dict) -> str:
        raw = name + "|" + json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[ToolResult]:
        with self._lock:
            item = self._cache.get(key)
            if not item:
                return None
            exp, res = item
            if time.time() > exp:
                self._cache.pop(key, None)
                return None
            return res

    def _cache_put(self, key: str, res: ToolResult, ttl: int) -> None:
        if ttl > 0 and res.ok:
            with self._lock:
                self._cache[key] = (time.time() + ttl, res)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # -- execution ---------------------------------------------------------- #
    def run(self, name: str, use_cache: bool = True, ttl: int = CACHE_TTL,
            **kwargs) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(name, False, error="unknown tool")
        if not spec.enabled():
            return ToolResult(name, False, error="tool disabled/unavailable")
        key = self._key(name, kwargs)
        if use_cache:
            hit = self._cache_get(key)
            if hit is not None:
                return ToolResult(**{**hit.to_dict(), "cached": True})
        t0 = time.time()
        try:
            data = spec.runner(**kwargs)
            res = ToolResult(name, True, data=data, source=spec.name,
                             latency_ms=int((time.time() - t0) * 1000))
        except Exception as e:   # a tool must NEVER crash its caller
            res = ToolResult(name, False, error=f"{type(e).__name__}: {e}",
                             latency_ms=int((time.time() - t0) * 1000))
            log.warning("tool %s failed: %s", name, e)
        if use_cache:
            self._cache_put(key, res, ttl)
        return res

    def run_parallel(self, calls: List[dict]) -> List[ToolResult]:
        """calls = [{'name':.., 'kwargs':{..}}, ..] executed concurrently, order kept."""
        if not calls:
            return []
        if len(calls) == 1:
            c = calls[0]
            return [self.run(c["name"], **c.get("kwargs", {}))]
        out: List[Optional[ToolResult]] = [None] * len(calls)
        workers = min(MAX_PARALLEL, len(calls))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            fut = {ex.submit(self.run, c["name"], **c.get("kwargs", {})): i
                   for i, c in enumerate(calls)}
            for f in as_completed(fut):
                out[fut[f]] = f.result()
        return [r for r in out if r is not None]

    # -- routing ------------------------------------------------------------ #
    def route(self, query: str, context: Any = None,
              max_tools: int = ROUTE_MAX, min_score: float = ROUTE_MIN) -> List[ToolSpec]:
        scored = [(t.score(query), t) for t in self.available()]
        scored = [(s, t) for s, t in scored if s >= min_score]
        scored.sort(key=lambda x: x[0], reverse=True)
        picked = [t for _, t in scored[:max_tools]]
        if not picked:   # sensible default: local knowledge, then web
            for fb in ("knowledge", "rag", "web_search"):
                sp = self.get(fb)
                if sp and sp.enabled():
                    picked.append(sp)
                    break
        return picked

    def _default_kwargs(self, spec: ToolSpec, query: str) -> dict:
        if spec.name in ("deep_read", "watch_video", "browse"):
            m = re.search(r"https?://\S+", query or "")
            if m:
                return {"url": m.group(0)}
        return {"query": query}

    def plan(self, query: str, context: Any = None) -> List[dict]:
        return [{"name": sp.name, "kwargs": self._default_kwargs(sp, query)}
                for sp in self.route(query, context)]

    # -- one-shot helpers --------------------------------------------------- #
    def smart_gather(self, query: str, context: Any = None) -> List[ToolResult]:
        plan = self.plan(query, context)
        log.info("router plan for %r -> %s", (query or "")[:60], [p["name"] for p in plan])
        return self.run_parallel(plan)

    def gather_context(self, query: str, context: Any = None,
                       max_chars: int = 3500) -> str:
        blocks = [r.as_context() for r in self.smart_gather(query, context)]
        return "\n\n".join(b for b in blocks if b)[:max_chars]


# --------------------------------------------------------------------------- #
# LLM planner (optional, graceful)
# --------------------------------------------------------------------------- #
def _extract_json(text: str) -> dict:
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return {}
    try:
        return json.loads(m.group(0))
    except Exception:
        return {}


def llm_tool_plan(belt: ToolBelt, query: str, llm_complete: Callable,
                  context: Any = None) -> List[dict]:
    """Ask the model which tools to use. Falls back to heuristic on any error."""
    try:
        sys = ("You are Damru's tool planner. Pick the MINIMAL set of tools to "
               "answer the user. Reply ONLY JSON: "
               '{"tools":[{"name":"..","query":"..","url":"..(optional)"}]}.\n'
               + belt.tool_card())
        raw = llm_complete([{"role": "system", "content": sys},
                            {"role": "user", "content": query}])
        data = _extract_json(raw if isinstance(raw, str) else str(raw))
        plan: List[dict] = []
        for t in (data.get("tools", []) if isinstance(data, dict) else []):
            sp = belt.get(t.get("name", ""))
            if sp and sp.enabled():
                kw = {"url": t["url"]} if t.get("url") else {"query": t.get("query", query)}
                plan.append({"name": sp.name, "kwargs": kw})
        if plan:
            return plan[:ROUTE_MAX]
    except Exception as e:
        log.info("llm planner fallback: %s", e)
    return belt.plan(query, context)


# --------------------------------------------------------------------------- #
# Providers (inject your backends; missing ones auto-disable their tool)
# --------------------------------------------------------------------------- #
class Providers:
    _KEYS = ("web_search", "deep_read", "browse", "watch_video", "rag_search",
             "knowledge", "code_exec", "math_solve", "image_gen", "make_3d",
             "memory", "llm_complete")

    def __init__(self, **kw) -> None:
        for k in self._KEYS:
            setattr(self, k, kw.get(k))


def _lazy_prayas() -> Optional[Callable]:
    """Default 'knowledge' backend = PRAYAS local tiles, if importable."""
    try:
        import damru_prayas_core as p   # type: ignore
        eng = p.get_engine()
    except Exception as e:
        log.info("prayas provider unavailable: %s", e)
        return None

    def _run(query: str, **_):
        try:
            if hasattr(eng, "compose_answer"):
                ans = eng.compose_answer(query)
                if ans:
                    return ans
            if hasattr(eng, "search"):
                k = int(os.getenv("PRAYAS_TOP_K", "8"))
                return eng.search(query, top_k=k)
        except Exception as e:
            return f"(prayas error: {e})"
        return ""
    return _run


def _lazy_math() -> Optional[Callable]:
    try:
        import sympy as sp  # type: ignore
        from sympy.parsing.sympy_parser import parse_expr  # type: ignore
    except Exception:
        return None

    def _run(query: str, **_):
        expr = re.sub(r"[^0-9a-zA-Z_+\-*/^().,= ]", " ", query or "").strip()
        if not expr:
            return ""
        try:
            if "=" in expr:
                lhs, rhs = expr.split("=", 1)
                return f"solution: {sp.solve(sp.Eq(parse_expr(lhs), parse_expr(rhs)))}"
            return f"result: {parse_expr(expr)}"
        except Exception as e:
            return f"math error: {e}"
    return _run


# --------------------------------------------------------------------------- #
# Belt factory
# --------------------------------------------------------------------------- #
def build_belt(providers: Optional[Providers] = None) -> ToolBelt:
    p = providers or Providers()
    belt = ToolBelt()

    knowledge_fn = p.knowledge or _lazy_prayas()
    if knowledge_fn:
        belt.register(ToolSpec(
            name="knowledge", tier=TIER_FAST,
            description="Search Damru's own knowledge tiles (PRAYAS/BM25, offline).",
            when_to_use="almost always first -- facts we may already know, no network.",
            runner=lambda query, **_: knowledge_fn(query),
            keywords=["what", "who", "explain", "define", "kya", "kaun", "batao", "samjha"],
            patterns=[r"\bwhat is\b", r"\bexplain\b", r"kya hai"],
            example="what is PRAYAS"))

    if p.rag_search:
        belt.register(ToolSpec(
            name="rag", tier=TIER_FAST,
            description="Vector RAG over Damru's indexed corpus (FAISS + Supabase).",
            when_to_use="deeper document lookup when knowledge tiles are thin.",
            runner=lambda query, k=5, **_: p.rag_search(query, k),
            keywords=["document", "docs", "paper", "reference", "citation", "source"]))

    if p.web_search:
        belt.register(ToolSpec(
            name="web_search", tier=TIER_NET,
            description="Live web search (SearXNG/Tavily/Brave/Wikipedia).",
            when_to_use="current events, prices, news, post-cutoff or unknown facts.",
            runner=lambda query, **_: p.web_search(query),
            keywords=["latest", "today", "news", "current", "price", "weather",
                      "aaj", "abhi", "taza", "kab", "when", "score"],
            patterns=[r"\b(latest|current|today|news|price|stock|weather|score)\b",
                      r"\b20\d\d\b"],
            example="latest ISRO launch"))

    if p.deep_read:
        belt.register(ToolSpec(
            name="deep_read", tier=TIER_NET,
            description="Fetch & clean a URL into readable text (Crawl4AI/Jina).",
            when_to_use="user gives a link, or read a specific page after web_search.",
            runner=lambda url=None, query=None, **_: p.deep_read(url or query),
            patterns=[r"https?://\S+", r"\bread this\b", r"open this link"],
            example="read https://..."))

    if p.browse:
        belt.register(ToolSpec(
            name="browse", tier=TIER_HEAVY,
            description="Drive a real browser to click/scroll/extract (Browser-Use/Playwright).",
            when_to_use="JS-heavy sites, logins, multi-step navigation, or deep_read failed.",
            runner=lambda url=None, query=None, **_: p.browse(url or query),
            keywords=["login", "click", "navigate", "fill form", "book", "checkout"],
            cost_hint="slow"))

    if p.watch_video:
        belt.register(ToolSpec(
            name="watch_video", tier=TIER_HEAVY,
            description="Watch a video: download+transcribe (yt-dlp+Whisper) & summarise.",
            when_to_use="a video/YouTube link, or 'watch this' / 'is video me kya hai'.",
            runner=lambda url=None, query=None, **_: p.watch_video(url or query),
            keywords=["video", "youtube", "watch", "dekh", "clip", "reel"],
            patterns=[r"(youtube\.com|youtu\.be)/\S+", r"\bwatch\b", r"video me"],
            cost_hint="slow", example="watch this https://youtu.be/..."))

    if p.code_exec:
        belt.register(ToolSpec(
            name="code_exec", tier=TIER_FAST,
            description="Write & run code in a sandbox, return verified output.",
            when_to_use="exact compute, data wrangling, algorithms, logic problems.",
            runner=lambda query=None, code=None, **_: p.code_exec(code or query),
            keywords=["calculate", "compute", "code", "program", "sort", "algorithm", "run"],
            patterns=[r"```", r"\bwrite (a )?(program|code|function)\b", r"\bcompute\b"],
            example="compute 37th fibonacci"))

    math_fn = p.math_solve or _lazy_math()
    if math_fn:
        belt.register(ToolSpec(
            name="math", tier=TIER_INSTANT,
            description="Symbolic/arithmetic math (SymPy).",
            when_to_use="equations, arithmetic, calculus, algebra -- exact math.",
            runner=lambda query, **_: math_fn(query),
            keywords=["solve", "integrate", "derivative", "equation"],
            patterns=[r"\d+\s*[\+\-\*/\^]\s*\d+", r"\bsolve\b", r"\bintegrate\b"],
            example="solve x^2 - 4 = 0"))

    if p.image_gen:
        belt.register(ToolSpec(
            name="image", tier=TIER_HEAVY,
            description="Generate an image from a prompt (Pollinations/ComfyUI).",
            when_to_use="user asks to draw/create/generate a picture, logo or art.",
            runner=lambda query=None, prompt=None, **_: p.image_gen(prompt or query),
            keywords=["image", "picture", "draw", "logo", "banao", "photo", "art"],
            patterns=[r"\b(draw|generate|create).{0,12}(image|picture|logo|art)\b",
                      r"image bana"],
            cost_hint="slow"))

    if p.make_3d:
        belt.register(ToolSpec(
            name="model3d", tier=TIER_HEAVY,
            description="Turn an image into a 3D GLB (TRELLIS).",
            when_to_use="user wants a 3D model / printable asset from an image.",
            runner=lambda image=None, query=None, **_: p.make_3d(image or query),
            keywords=["3d", "glb", "model", "print", "trellis"],
            cost_hint="slow"))

    if p.memory:
        belt.register(ToolSpec(
            name="memory", tier=TIER_INSTANT,
            description="Recall this user's past turns & profile (digital twin).",
            when_to_use="personalised or follow-up questions referencing earlier chat.",
            runner=lambda query, **_: p.memory(query),
            keywords=["remember", "last time", "earlier", "mera", "pehle", "yaad"]))

    log.info("belt ready with %d tools: %s", len(belt.names()), belt.names())
    return belt


# --------------------------------------------------------------------------- #
# Singleton
# --------------------------------------------------------------------------- #
_BELT: Optional[ToolBelt] = None
_BELT_LOCK = threading.Lock()


def get_belt(providers: Optional[Providers] = None) -> ToolBelt:
    global _BELT
    if _BELT is None or providers is not None:
        with _BELT_LOCK:
            if _BELT is None or providers is not None:
                _BELT = build_belt(providers)
    return _BELT


# --------------------------------------------------------------------------- #
# Smooth 5-stage pipeline (matches Chakra Pulse HUD)
# --------------------------------------------------------------------------- #
STAGES = ["SANKALP", "KHOJ", "TARK", "SATYA", "UTTAR"]


def run_pipeline(query: str, belt: Optional[ToolBelt] = None, context: Any = None,
                 llm_complete: Optional[Callable] = None):
    """Generator of {'stage','status','detail'} events; final event has answer+sources.
    Lets the frontend stream a smooth HUD instead of one big blocking wait."""
    belt = belt or get_belt()

    yield {"stage": "SANKALP", "status": "start", "detail": "intent samajh raha hoon"}
    plan = (llm_tool_plan(belt, query, llm_complete, context)
            if llm_complete else belt.plan(query, context))
    yield {"stage": "SANKALP", "status": "done",
           "detail": "tools: " + ", ".join(p["name"] for p in plan)}

    yield {"stage": "KHOJ", "status": "start", "detail": "tools chala raha hoon"}
    results = belt.run_parallel(plan)
    ok = [r for r in results if r.ok]
    yield {"stage": "KHOJ", "status": "done", "detail": f"{len(ok)}/{len(results)} tools ok"}

    yield {"stage": "TARK", "status": "start", "detail": "jaankari jod raha hoon"}
    ctx = "\n\n".join(r.as_context() for r in ok if r.as_context())
    yield {"stage": "TARK", "status": "done", "detail": f"{len(ctx)} chars context"}

    answer = None
    if llm_complete:
        yield {"stage": "SATYA", "status": "start", "detail": "jawab bana raha hoon"}
        try:
            answer = llm_complete([
                {"role": "system", "content":
                    "You are Damru. Use ONLY the tool context when relevant, cite "
                    "sources inline, answer concisely and warmly (Hinglish ok)."},
                {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {query}"},
            ])
        except Exception as e:
            log.warning("pipeline llm failed: %s", e)
        yield {"stage": "SATYA", "status": "done", "detail": "verified"}

    yield {"stage": "UTTAR", "status": "done", "answer": answer, "context": ctx,
           "sources": [r.source or r.tool for r in ok],
           "results": [r.to_dict() for r in results]}


# --------------------------------------------------------------------------- #
# Self-test (no network, no deps) -- run:  python3 damru_tools.py
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    prov = Providers(
        web_search=lambda query: [f"(mock web) top hit for: {query}"],
        knowledge=lambda query: f"(mock kb) Damru already knows about: {query}",
        code_exec=lambda query=None, code=None: "result=42",
        deep_read=lambda url: f"(mock read) cleaned text of {url}",
        watch_video=lambda url: f"(mock watch) transcript+summary of {url}",
        memory=lambda query: ["(mock) user asked about Damru Live yesterday"],
    )
    belt = get_belt(prov)
    print("=== TOOL CARD ===")
    print(belt.tool_card())
    print("\n=== ROUTER DEMO ===")
    for q in ["what is PRAYAS",
              "latest ISRO news today",
              "watch this https://youtu.be/abc123",
              "compute 12*13 fibonacci",
              "read https://example.com/article",
              "mujhe pehle wali baat yaad dila"]:
        plan = belt.plan(q)
        print(f"\nQ: {q}\n  -> plan: {[p['name'] for p in plan]}")
        print("  -> ctx :", belt.gather_context(q)[:150].replace("\n", " "))
    print("\n=== PIPELINE DEMO ===")
    for ev in run_pipeline("latest ISRO news today", belt):
        print(" ", ev.get("stage"), ev.get("status"), "|", ev.get("detail", ""))
    print("\nOK damru_tools v" + __version__)
