#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 DAMRU RAG SERVE  --  STRONG BUILD v1
================================================================================
Fast hybrid answer system:
  1. Query aata hai
  2. FAISS index se top-K chunks retrieve karo (milliseconds)
  3. Base LLM ko context + query dedo
  4. Fast verified answer milta hai -- NO TRAINING NEEDED!

STRONG BUILD PRINCIPLES:
  - Self-healing: koi bhi component fail ho -> graceful fallback
  - Atomic state: kabhi corrupt nahi hoga
  - Resume: FAISS index HF se load, local cache mein save
  - Multiple providers: ek down -> dusra try
  - Logging: har step transparent
  - No silent failures: sab kuch log hota hai

RUN (local ya HF Space):
  pip install sentence-transformers faiss-cpu huggingface_hub requests flask
  export HF_TOKEN=hf_xxx
  python serve/damru_rag_serve.py

API:
  POST /ask  {"query": "Bharat ki capital kya hai?"}
  GET  /health
  GET  /stats

ENV:
  HF_TOKEN           HuggingFace token
  INDEX_REPO         Damaru-ai/damru-rag-index
  EMBED_MODEL        BAAI/bge-small-en-v1.5
  TOP_K              top results, default 5
  MAX_CTX_CHARS      max context chars, default 3000
  PORT               server port, default 7860
  # LLM providers (at least one needed):
  CEREBRAS_API_KEY   fastest free inference
  OPENROUTER_API_KEY backup
  GITHUB_MODELS_TOKEN backup
  HF_TOKEN           also used for HF router fallback
================================================================================
"""

import os, json, time, threading, traceback
from typing import List, Dict, Optional

_START = time.time()

def log(*a):
    print(f"[rag +{int(time.time()-_START)}s]", *a, flush=True)

def env(k, d=None):
    v = os.environ.get(k)
    return v if (v is not None and str(v).strip()) else d

# ================================================================ CONFIG
HF_TOKEN    = env("HF_TOKEN", "")
INDEX_REPO  = env("INDEX_REPO",   "Damaru-ai/damru-rag-index")
EMBED_MODEL = env("EMBED_MODEL",  "BAAI/bge-small-en-v1.5")
TOP_K       = int(env("TOP_K",    "5"))
MAX_CTX     = int(env("MAX_CTX_CHARS", "3000"))
PORT        = int(env("PORT",     "7860"))
CACHE_DIR   = env("CACHE_DIR",    "/tmp/damru_rag_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ================================================================ STATS
_stats = {
    "queries": 0, "rag_hits": 0, "llm_calls": 0, "errors": 0,
    "start_time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "index_loaded": False, "index_size": 0,
}
_stats_lock = threading.Lock()

def bump(k, n=1):
    with _stats_lock:
        _stats[k] = _stats.get(k, 0) + n


# ================================================================ RAG INDEX
class RAGIndex:
    """
    Strong build:
      - Loads FAISS + metadata parquet from HF, caches locally.
      - Background load -- server answers immediately (with fallback) while index loads.
      - Survives: network fail, partial download, corrupt cache.
    """
    def __init__(self):
        self.index    = None
        self.meta     = {}      # {question: [...], answer: [...], domain: [...]}
        self.embedder = None
        self.ready    = False
        self._lock    = threading.Lock()

    def _local(self, name):
        return os.path.join(CACHE_DIR, name)

    def _dl(self, fname):
        """Download from HF if not cached. Returns local path or None."""
        local = self._local(fname)
        if os.path.exists(local) and os.path.getsize(local) > 100:
            log(f"  cache-hit: {fname}")
            return local
        if not HF_TOKEN:
            log(f"  no HF_TOKEN -- cannot dl {fname}")
            return None
        try:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(INDEX_REPO, fname, repo_type="dataset",
                                token=HF_TOKEN, local_dir=CACHE_DIR)
            log(f"  dl ok: {fname} ({os.path.getsize(p)//1024}KB)")
            return p
        except Exception as e:
            log(f"  dl-fail: {fname}: {str(e)[:120]}")
            return None

    def load(self):
        threading.Thread(target=self._load_bg, daemon=True).start()

    def _load_bg(self):
        log("[index] Loading FAISS index...")
        try:
            import faiss
            import pyarrow.parquet as pq
            from sentence_transformers import SentenceTransformer

            log("[index] embedder load...")
            self.embedder = SentenceTransformer(EMBED_MODEL)
            log(f"[index] embedder ready dim={self.embedder.get_sentence_embedding_dimension()}")

            idx_p = self._dl("index.faiss")
            if idx_p:
                self.index = faiss.read_index(idx_p)
                log(f"[index] FAISS ok: {self.index.ntotal} vectors")
            else:
                log("[index] WARN: no FAISS index -- RAG disabled (build it with build_index.py)")
                return

            meta_p = self._dl("meta.parquet")
            if meta_p:
                self.meta = pq.read_table(meta_p).to_pydict()
                log(f"[index] meta ok: {len(self.meta.get('question',[]))} rows")

            with self._lock:
                self.ready = True
            with _stats_lock:
                _stats["index_loaded"] = True
                _stats["index_size"] = self.index.ntotal
            log("[index] RAG READY!")
        except Exception:
            log("[index] ERROR:", traceback.format_exc())

    def search(self, query: str, k: int = TOP_K) -> List[Dict]:
        if not self.ready or self.index is None:
            return []
        try:
            import numpy as np
            vec = self.embedder.encode([query], normalize_embeddings=True,
                                       show_progress_bar=False).astype("float32")
            scores, ids = self.index.search(vec, k)
            qs = self.meta.get("question", [])
            ans = self.meta.get("answer", [])
            ds = self.meta.get("domain", [])
            out = []
            for sc, idx in zip(scores[0], ids[0]):
                if idx < 0 or idx >= len(qs): continue
                out.append({"question": qs[idx], "answer": ans[idx],
                            "domain": ds[idx] if idx < len(ds) else "general",
                            "score": float(sc)})
            return out
        except Exception as e:
            log(f"[search] err: {e}")
            return []


# ================================================================ LLM
class Provider:
    def __init__(self, name, url, model, key, extra_headers=None):
        self.name = name; self.url = url; self.model = model
        self.key = key; self.extra_headers = extra_headers or {}
        self.fails = 0; self.cool_until = 0.0

    def ok(self): return time.time() >= self.cool_until

    def ask(self, messages, max_tokens=800, temperature=0.3, timeout=25):
        try:
            import requests
            h = {"Content-Type": "application/json",
                 "Authorization": f"Bearer {self.key}"}
            h.update(self.extra_headers)
            r = requests.post(self.url, headers=h, timeout=timeout,
                json={"model": self.model, "messages": messages,
                      "max_tokens": max_tokens, "temperature": temperature})
            if r.status_code == 200:
                t = r.json()["choices"][0]["message"]["content"]
                if t: self.fails = 0; return t
            if r.status_code in (429, 500, 502, 503): self._cool(r.status_code)
        except Exception as e:
            self._cool(type(e).__name__)
        return None

    def _cool(self, why):
        self.fails += 1
        cd = min(600, 30 * self.fails)
        self.cool_until = time.time() + cd
        log(f"  [{self.name}] cooling {cd}s ({why})")


class LLMBrain:
    def __init__(self):
        self.providers = []
        for key in (env("CEREBRAS_API_KEY") or "").split(","):
            if k := key.strip():
                self.providers.append(Provider("cerebras",
                    "https://api.cerebras.ai/v1/chat/completions",
                    "llama-3.3-70b", k))
        for key in (env("OPENROUTER_API_KEY") or "").split(","):
            if k := key.strip():
                self.providers.append(Provider("openrouter",
                    "https://openrouter.ai/api/v1/chat/completions",
                    "meta-llama/llama-3.3-70b-instruct:free", k,
                    {"HTTP-Referer": "https://damru-ai.vercel.app",
                     "X-Title": "Damru"}))
        for key in (env("GITHUB_MODELS_TOKEN") or "").split(","):
            if k := key.strip():
                self.providers.append(Provider("github-models",
                    "https://models.github.ai/inference/chat/completions",
                    "meta-llama/Llama-3.3-70B-Instruct", k))
        if HF_TOKEN:
            self.providers.append(Provider("hf-router",
                "https://router.huggingface.co/v1/chat/completions",
                "meta-llama/Llama-3.3-70B-Instruct", HF_TOKEN))
        self._rr = 0
        log(f"[llm] {len(self.providers)} providers: " +
            ", ".join(p.name for p in self.providers))

    def ask(self, messages, **kw):
        n = len(self.providers)
        if not n: return None
        for _ in range(n):
            p = self.providers[self._rr % n]; self._rr += 1
            if not p.ok(): continue
            out = p.ask(messages, **kw)
            if out: bump("llm_calls"); return out
        return None


# ================================================================ PIPELINE
SYS = """Tu Damru hai -- Bharat ka apna AI. Short, clear, helpful answers de.
Agar context mein answer hai to context use kar.
Agar nahi hai to apni knowledge se de.
Hindi/English dono samajhta hai, user ki language mein jawab de."""


def build_ctx(chunks):
    if not chunks: return ""
    parts = ["=== Retrieved Context ==="]
    for i, c in enumerate(chunks, 1):
        parts.append(f"[{i}] Q: {c['question'][:200]}\n    A: {c['answer'][:600]}")
    return "\n\n".join(parts)


def answer_query(query: str, rag: RAGIndex, llm: LLMBrain) -> Dict:
    t0 = time.time(); bump("queries")

    chunks = []
    if rag.ready:
        try:
            chunks = rag.search(query)
            if chunks: bump("rag_hits")
        except Exception as e:
            log(f"[answer] rag err: {e}")

    ctx = build_ctx(chunks)
    user_content = f"{ctx}\n\n---\nSawal: {query}" if ctx else query
    messages = [{"role": "system", "content": SYS},
                {"role": "user",   "content": user_content}]

    resp = llm.ask(messages)
    if not resp:
        if chunks:
            resp = chunks[0]["answer"][:800]; mode = "rag_direct"
        else:
            resp = "Abhi system load ho raha hai. 15 second mein dubara try karo!"
            mode = "fallback"
        bump("errors")
    else:
        mode = "rag_llm" if chunks else "llm_only"

    return {
        "answer": resp,
        "sources": [{"q": c["question"][:100], "domain": c["domain"],
                     "score": round(c["score"], 3)} for c in chunks[:3]],
        "latency_ms": int((time.time() - t0) * 1000),
        "mode": mode,
        "rag_ready": rag.ready,
    }


# ================================================================ FLASK
def make_app(rag, llm):
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        log("pip install flask"); return None

    app = Flask("damru-rag")

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "rag_ready": rag.ready,
                        "index_size": _stats["index_size"],
                        "providers": len(llm.providers)})

    @app.route("/stats")
    def stats():
        with _stats_lock: return jsonify(dict(_stats))

    @app.route("/ask", methods=["POST"])
    def ask():
        try:
            data = request.get_json(force=True, silent=True) or {}
            q = str(data.get("query", "")).strip()
            if not q: return jsonify({"error": "query required"}), 400
            return jsonify(answer_query(q, rag, llm))
        except Exception as e:
            log("[/ask] err:", e); bump("errors")
            return jsonify({"error": str(e)[:200]}), 500

    @app.route("/")
    def root():
        return jsonify({"name": "Damru RAG Server v1",
                        "endpoints": ["/ask", "/health", "/stats"],
                        "rag_ready": rag.ready})
    return app


# ================================================================ MAIN
def main():
    log("=" * 56)
    log("DAMRU RAG SERVER v1 -- STRONG BUILD")
    log(f"  INDEX_REPO  = {INDEX_REPO}")
    log(f"  EMBED_MODEL = {EMBED_MODEL}")
    log(f"  TOP_K       = {TOP_K}")
    log(f"  PORT        = {PORT}")
    log("=" * 56)

    rag = RAGIndex(); rag.load()
    llm = LLMBrain()
    if not llm.providers:
        log("[WARN] No LLM providers! Set CEREBRAS_API_KEY / OPENROUTER_API_KEY")

    app = make_app(rag, llm)
    if not app: return
    log(f"[server] Listening on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Stopped.")
    except Exception:
        log("[FATAL]", traceback.format_exc())
