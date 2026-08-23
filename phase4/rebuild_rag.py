#!/usr/bin/env python3
"""
Damru RAG INDEX REBUILDER  --  FULL KNOWLEDGE BASE (v4)
=======================================================
v4 FIX: streams every file (parquet iter_batches / jsonl line-by-line / csv
chunks) so big shards never blow up RAM -- that OOM is what crashed the v3 run.
Also: dep auto-install, explicit GPU device, IVFPQ->flat fallback, upload retry.

Rebuilds Damaru-ai/damru-rag-index over the ENTIRE Damru knowledge base
(11.46M+ rows) PLUS any HF "bucket" dataset repos, into ONE searchable FAISS
index that the live Space (rag.py) loads at answer time.

WHY A NEW BUILDER (vs serve/build_index.py):
  - old one CAPPED at MAX_INDEX=30000  -> that is why the index is tiny/stale.
  - old one only read data/*.parquet    -> it MISSED the curious/, world_tiles/,
    live_chats/, daily/ and buckets/ shards.
  This one reads EVERY data file across ALL prefixes + ALL formats, is memory
  safe (streams meta to disk, buffers only a training sample), and auto-picks a
  compressed FAISS index so 11.46M vectors actually ship.

WHAT IT DOES
  1. Lists + reads every data file in SRC_REPO across all prefixes
     (data/ curious/ world_tiles/ live_chats/ daily/ buckets/ ...) and formats
     (.parquet/.jsonl/.ndjson/.json/.csv/.tsv).  = the full 11.46M corpus.
  2. Optionally folds in extra bucket repos (EXTRA_REPOS or auto-discovered
     under BUCKETS_AUTHOR). If you already merged buckets into damru-knowledge
     with merge_buckets.py, they are picked up in step 1 -> leave these empty.
  3. Normalises every row -> {question, answer, intent, domain, lang, source}.
  4. Embeds with BAAI/bge-small-en-v1.5 (384d) on GPU, cosine via L2-norm.
  5. Builds a memory-smart FAISS index:
        < IVF_THRESHOLD rows  -> IndexFlatIP  (exact)
        >= IVF_THRESHOLD rows -> IndexIVFPQ   (compressed, ships small);
     nprobe is baked into the index so rag.py needs NO change.
  6. Writes index.faiss + meta.parquet + config.json (columns EXACTLY what
     rag.py reads) and uploads them to INDEX_REPO.

RUN ON KAGGLE/COLAB WITH GPU + INTERNET ON (not runnable in the sandbox):
  pip install -U huggingface_hub sentence-transformers faiss-cpu pandas pyarrow
  import os; os.environ["HF_TOKEN"] = "hf_xxx"   # WRITE token
  # os.environ["DRY_RUN"] = "1"                   # inventory only, no embed
  # os.environ["MAX_INDEX"] = "200000"           # fast test run first
  # then paste + run this whole file

ENV (all optional except HF_TOKEN)
  HF_TOKEN        required (WRITE)
  SRC_REPO        Damaru-ai/damru-knowledge
  INDEX_REPO      Damaru-ai/damru-rag-index   (created if missing)
  EMBED_MODEL     BAAI/bge-small-en-v1.5
  EXTRA_REPOS     ""     comma/space list of extra bucket repos to fold in
  BUCKETS_AUTHOR  ""     HF user/org to auto-discover bucket datasets from
  DRY_RUN         0      1 = list files only (no embed/upload)
  MAX_INDEX       0      cap indexed rows (0 = ALL). Use 200000 for a test run.
  MAX_PER_DOMAIN  0      cap dominant domains (0 = no cap); PRIORITY uncapped
  DEDUP           1      drop duplicate q+a (md5)
  BATCH           512    embed batch size
  ANS_STORE       1200   chars of answer kept in meta for context injection
  MIN_Q           8      min question chars
  MIN_A           12     min answer chars
  INDEX_TYPE      auto   auto | flat | ivfpq
  IVF_THRESHOLD   200000 switch to IVFPQ above this many rows
  NLIST           4096   IVF lists
  PQ_M            48     PQ subquantizers (must divide dim; 384 %% 48 == 0)
  PQ_NBITS        8
  NPROBE          16     search lists (baked into the saved index)
  TRAIN_SAMPLE    300000 vectors used to train IVFPQ
  LOG_EVERY       20000  heartbeat rows
"""
import os
import re
import json
import time
import hashlib
from collections import Counter

import numpy as np


def _env(k, d):
    v = os.environ.get(k)
    return v if v not in (None, "") else d


def _resolve_hf_token():
    """WRITE HF token env se ya Kaggle Secrets se auto-nikaalo. Kaggle 'Save
    Version' (papermill) run me interactive env nahi hota, isliye secret bhi padho."""
    for _k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
               "HF_API_TOKEN", "HUGGINGFACE_TOKEN"):
        _v = (os.environ.get(_k) or "").strip()
        if _v:
            return _v
    try:
        from kaggle_secrets import UserSecretsClient
        _c = UserSecretsClient()
        for _k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            try:
                _v = (_c.get_secret(_k) or "").strip()
                if _v:
                    return _v
            except Exception:
                pass
    except Exception:
        pass
    return ""


HF_TOKEN       = _resolve_hf_token()
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", HF_TOKEN)
SRC_REPO       = _env("SRC_REPO", "Damaru-ai/damru-knowledge")
INDEX_REPO     = _env("INDEX_REPO", "Damaru-ai/damru-rag-index")
EMBED_MODEL    = _env("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EXTRA_REPOS    = _env("EXTRA_REPOS", "")
BUCKETS_AUTHOR = _env("BUCKETS_AUTHOR", "")
DRY_RUN        = _env("DRY_RUN", "0") == "1"
MAX_INDEX      = int(_env("MAX_INDEX", "0"))
MAX_PER_DOMAIN = int(_env("MAX_PER_DOMAIN", "0"))
DEDUP          = _env("DEDUP", "1") == "1"
BATCH          = int(_env("BATCH", "512"))
ANS_STORE      = int(_env("ANS_STORE", "1200"))
MIN_Q          = int(_env("MIN_Q", "8"))
MIN_A          = int(_env("MIN_A", "12"))
INDEX_TYPE     = _env("INDEX_TYPE", "auto").lower()
IVF_THRESHOLD  = int(_env("IVF_THRESHOLD", "200000"))
NLIST          = int(_env("NLIST", "4096"))
PQ_M           = int(_env("PQ_M", "48"))
PQ_NBITS       = int(_env("PQ_NBITS", "8"))
NPROBE         = int(_env("NPROBE", "16"))
TRAIN_SAMPLE   = int(_env("TRAIN_SAMPLE", "300000"))
LOG_EVERY      = int(_env("LOG_EVERY", "20000"))

# infra repos that must never be treated as knowledge "buckets"
EXCLUDE = {
    "damru-knowledge", "damru-rag-index", "damru-14b-lora", "damru-14b-gguf",
    "damru-gguf", "damru-train", "damru-gurukul", "damru-oracle",
}
PRIORITY = {"medical", "holy"}
META_COLS = ("question", "answer", "intent", "domain", "lang", "source")

DATA_EXT = (".parquet", ".jsonl", ".ndjson", ".json", ".csv", ".tsv")
SKIP_TOKENS = ("manifest", "_state", "stats", "_meta", "metadata", "readme",
               "dataset_infos", "gitattributes", "checkpoint", "progress",
               "_index", "config", "scorecard", "license", ".gitignore")

Q_KEYS = ("question", "prompt", "instruction", "problem", "query", "task",
          "input", "title", "text", "content", "body")
A_KEYS = ("answer", "output", "response", "completion", "solution", "target",
          "label", "assistant", "chosen", "value")


def _log(*a):
    print("[rebuild-rag]", *a, flush=True)


def _is_data_file(path):
    p = path.lower()
    if not p.endswith(DATA_EXT):
        return False
    base = p.rsplit("/", 1)[-1]
    return not any(tok in base for tok in SKIP_TOKENS)


def _s(x):
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    if isinstance(x, (list, dict)):
        try:
            return json.dumps(x, ensure_ascii=False)
        except Exception:
            return str(x)
    return str(x)


def _from_messages(val):
    """Extract (q, a) from a chat messages / conversations list."""
    if not isinstance(val, list):
        return "", ""
    q, a = "", ""
    for m in val:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role") or m.get("from") or "").lower()
        content = m.get("content")
        if content is None:
            content = m.get("value")
        content = _s(content)
        if role in ("user", "human", "question") and not q:
            q = content
        elif role in ("assistant", "gpt", "bot", "ai", "answer") and not a:
            a = content
    return q, a


def normalize(row):
    """row(dict) -> (question, answer, intent, lang). Blank fields if unusable."""
    if not isinstance(row, dict):
        return "", "", "", "en"
    lang = _s(row.get("lang") or row.get("language") or "en") or "en"
    intent = _s(row.get("intent") or row.get("domain") or row.get("topic")
                or row.get("subject") or row.get("category"))
    for mk in ("messages", "conversations", "conversation", "dialog"):
        v = row.get(mk)
        if isinstance(v, list):
            q, a = _from_messages(v)
            if q or a:
                return q.strip(), a.strip(), intent.strip(), lang.strip() or "en"
    q, qk = "", None
    for k in Q_KEYS:
        if row.get(k):
            q = _s(row[k]); qk = k; break
    a = ""
    for k in A_KEYS:
        if k != qk and row.get(k):
            a = _s(row[k]); break
    if qk == "instruction" and row.get("input"):
        q = (q + "\n" + _s(row.get("input"))).strip()
    return q.strip(), a.strip(), intent.strip(), lang.strip() or "en"


def domain_of(intent):
    s = (intent or "").lower()

    def has(*ks):
        return any(k in s for k in ks)

    if has("nurs", "med", "clinic", "disease", "anatom", "physio", "pharma",
           "patho", "health", "surg", "nutri"):
        return "medical"
    if has("veda", "gita", "bible", "quran", "holy", "itihasa", "mahabharat",
           "ramayan", "upanishad", "verse", "scripture"):
        return "holy"
    if has("cod", "program", "python", "algorithm", "competitive", "devops"):
        return "coding"
    if has("physic", "chem", "math", "reason", "logic", "calcul", "science"):
        return "stem"
    if has("agent", "tool", "plan"):
        return "agentic"
    return "general"


def read_rows(repo, fname):
    """Download one data file and STREAM dict rows -- never loads a whole big
    shard into RAM (that RAM blow-up is what crashed the v3 run)."""
    import json as _json
    from huggingface_hub import hf_hub_download
    _rc = int(_env("READ_CHUNK", "50000"))
    try:
        local = hf_hub_download(repo, fname, repo_type="dataset",
                                token=HF_TOKEN or None)
    except Exception as e:
        _log("dl-fail", repo, fname, str(e)[:100]); return
    ext = fname.lower().rsplit(".", 1)[-1]
    try:
        if ext == "parquet":
            import pyarrow.parquet as _pq
            pf = _pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=_rc):
                for r in batch.to_pylist():
                    yield r
        elif ext in ("jsonl", "ndjson"):
            with open(local, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = _json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        yield obj
        elif ext == "json":
            with open(local, "r", encoding="utf-8", errors="ignore") as fh:
                try:
                    obj = _json.load(fh)
                except Exception:
                    obj = None
            rows = []
            if isinstance(obj, dict):
                for key in ("data", "rows", "items", "examples", "problems"):
                    if isinstance(obj.get(key), list):
                        rows = obj[key]; break
            elif isinstance(obj, list):
                rows = obj
            for r in rows:
                if isinstance(r, dict):
                    yield r
        elif ext in ("csv", "tsv"):
            import pandas as pd
            _sep = "\t" if ext == "tsv" else ","
            for chunk in pd.read_csv(local, sep=_sep, chunksize=_rc,
                                     dtype=str, keep_default_na=False):
                for r in chunk.to_dict("records"):
                    yield r
    except Exception as e:
        _log("read-fail", fname, str(e)[:100]); return


def gather_files(api):
    """Return (files, repos): every data file across SRC_REPO + extra buckets."""
    extra = [r.strip() for r in re.split(r"[,\s]+", EXTRA_REPOS) if r.strip()]
    if BUCKETS_AUTHOR:
        try:
            for d in api.list_datasets(author=BUCKETS_AUTHOR):
                rid = getattr(d, "id", None) or str(d)
                nm = rid.split("/")[-1].lower()
                if nm in EXCLUDE or rid in (SRC_REPO, INDEX_REPO):
                    continue
                extra.append(rid)
        except Exception as e:
            _log("list_datasets fail", str(e)[:120])
    seen_r, repos = set(), []
    for r in [SRC_REPO] + extra:
        if r and r not in seen_r:
            seen_r.add(r); repos.append(r)
    files = []
    for repo in repos:
        try:
            for f in api.list_repo_files(repo, repo_type="dataset"):
                if _is_data_file(f):
                    files.append((repo, f))
        except Exception as e:
            _log("list_repo_files fail", repo, str(e)[:120])
    return files, repos


class Builder:
    """Memory-smart incremental FAISS builder (auto Flat vs IVFPQ)."""

    def __init__(self, dim):
        self.dim = dim
        self.index = None
        self.mode = None
        self.trained = False
        self.buf_vecs = []
        self.buf_meta = []
        self.buffered = 0
        self.meta_writer = None
        self._mbatch = []
        self.total_added = 0

    def _build_flat(self):
        import faiss
        self.index = faiss.IndexFlatIP(self.dim)
        self.mode = "flat"

    def _build_ivfpq(self, train_arr):
        import faiss
        n = int(train_arr.shape[0])
        nlist = min(NLIST, max(256, n // 40))
        nlist = max(1, min(nlist, n))   # never more lists than training vectors
        quant = faiss.IndexFlatIP(self.dim)
        idx = faiss.IndexIVFPQ(quant, self.dim, nlist, PQ_M, PQ_NBITS,
                               faiss.METRIC_INNER_PRODUCT)
        _log("training IVFPQ nlist=%d m=%d nbits=%d on %d vecs ..."
             % (nlist, PQ_M, PQ_NBITS, n))
        idx.train(train_arr)
        idx.nprobe = NPROBE
        self.index = idx
        self.mode = "ivfpq"

    def add_batch(self, vecs, metas):
        if self.trained:
            self.index.add(vecs); self._write_meta(metas)
            self.total_added += int(vecs.shape[0]); return
        if INDEX_TYPE == "flat":
            self._build_flat(); self.trained = True
            self.index.add(vecs); self._write_meta(metas)
            self.total_added += int(vecs.shape[0]); return
        self.buf_vecs.append(vecs); self.buf_meta.extend(metas)
        self.buffered += int(vecs.shape[0])
        if self.buffered >= TRAIN_SAMPLE:
            self._finalize_training()

    def _finalize_training(self):
        if self.trained:
            return
        if not self.buf_vecs:
            self._build_flat(); self.trained = True; return
        arr = np.vstack(self.buf_vecs).astype("float32")
        n = int(arr.shape[0])
        use_ivfpq = (INDEX_TYPE == "ivfpq") or (INDEX_TYPE == "auto"
                                                and n >= IVF_THRESHOLD)
        if use_ivfpq and n >= 1000:
            try:
                self._build_ivfpq(arr)
            except Exception as e:
                _log("IVFPQ build failed -> flat fallback:", str(e)[:120])
                self.index = None
                self._build_flat()
        else:
            self._build_flat()
        self.index.add(arr); self._write_meta(self.buf_meta)
        self.total_added += n
        self.buf_vecs = []; self.buf_meta = []
        self.trained = True
        _log("index mode = %s, seeded with %d vectors" % (self.mode, n))

    def _write_meta(self, metas):
        self._mbatch.extend(metas)
        if len(self._mbatch) >= 50000:
            self._flush_meta()

    def _flush_meta(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        if not self._mbatch:
            return
        cols = {k: [m.get(k) for m in self._mbatch] for k in META_COLS}
        table = pa.table(cols)
        if self.meta_writer is None:
            self.meta_writer = pq.ParquetWriter("meta.parquet", table.schema,
                                                compression="zstd")
        self.meta_writer.write_table(table)
        self._mbatch = []

    def finish(self):
        import faiss
        self._finalize_training()   # small-dataset case (never hit TRAIN_SAMPLE)
        self._flush_meta()
        if self.meta_writer is not None:
            self.meta_writer.close()
        faiss.write_index(self.index, "index.faiss")
        return self.total_added


def _ensure_deps():
    """Auto-install the build stack if a fresh Kaggle/Colab kernel lacks it.
    Internet must be ON. No-op when everything is already present."""
    import importlib.util as _u
    need = []
    for mod, pip in (("sentence_transformers", "sentence-transformers"),
                     ("faiss", "faiss-cpu"), ("pyarrow", "pyarrow"),
                     ("pandas", "pandas"), ("huggingface_hub", "huggingface_hub")):
        if _u.find_spec(mod) is None:
            need.append(pip)
    if need:
        import subprocess, sys
        _log("installing missing deps:", " ".join(need))
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *need])


def main():
    _ensure_deps()
    from huggingface_hub import HfApi
    assert HF_TOKEN, (
        "HF_TOKEN (WRITE) nahi mila. Do me se ek karo:  "
        "(A) Kaggle: Add-ons -> Secrets -> 'HF_TOKEN' (WRITE) add + Attach karo "
        "(Save Version pe bhi chalega),  ya  "
        "(B) notebook ke SABSE PEHLE cell me: "
        "import os; os.environ['HF_TOKEN']='hf_xxx'"
    )
    ncpu = max(1, os.cpu_count() or 2)
    os.environ.setdefault("OMP_NUM_THREADS", str(ncpu))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    api = HfApi(token=HF_TOKEN)

    files, repos = gather_files(api)
    _log("resolved %d repo(s), %d data file(s)" % (len(repos), len(files)))
    repo_ct, prefix_ct = Counter(), Counter()
    src_has_buckets = False
    for (repo, f) in files:
        repo_ct[repo] += 1
        pre = f.split("/", 1)[0] if "/" in f else "(root)"
        prefix_ct["%s : %s" % (repo.split("/")[-1], pre)] += 1
        if repo == SRC_REPO and f.lower().startswith("buckets/"):
            src_has_buckets = True
    for r in repos:
        _log("  repo", r, "->", repo_ct[r], "files")
    for k, v in sorted(prefix_ct.items()):
        _log("    ", k, "->", v)
    _log("buckets/ already inside %s : %s" % (SRC_REPO, src_has_buckets))
    if not files:
        _log("no data files found -- check HF_TOKEN / repo names"); return
    if DRY_RUN:
        _log("DRY_RUN=1 -> inventory only. Set DRY_RUN=0 to build + upload.")
        return

    from sentence_transformers import SentenceTransformer
    try:
        import torch
        _dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        _dev = "cpu"
    _log("loading embedder", EMBED_MODEL, "on", _dev, "...")
    embedder = SentenceTransformer(EMBED_MODEL, device=_dev)
    dim = int(embedder.get_sentence_embedding_dimension())
    _log("embedder ready dim=%d device=%s" % (dim, _dev))
    if _dev == "cpu":
        _log("WARNING: no GPU -- embedding millions of rows on CPU is very slow; "
             "use a Kaggle/Colab GPU runtime.")
    builder = Builder(dim)
    cap, seen = Counter(), set()
    batch_txt, batch_meta = [], []
    scanned = kept = 0
    t0 = time.time()

    def flush():
        if not batch_txt:
            return
        vecs = embedder.encode(batch_txt, batch_size=BATCH,
                               convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False).astype("float32")
        builder.add_batch(vecs, list(batch_meta))
        _log("  +batch indexed=%d buffered=%d kept=%d scanned=%d %.0fs"
             % (builder.total_added, builder.buffered, kept, scanned,
                time.time() - t0))
        batch_txt.clear()
        batch_meta.clear()

    done = False
    for (repo, fname) in files:
        if done:
            break
        rid = repo.split("/")[-1]
        for row in read_rows(repo, fname):
            scanned += 1
            if scanned % LOG_EVERY == 0:
                _log("scanned=%d kept=%d indexed~=%d %.0fs"
                     % (scanned, kept, builder.total_added, time.time() - t0))
            q, a, intent, lang = normalize(row)
            if len(q) < MIN_Q or len(a) < MIN_A:
                continue
            if DEDUP:
                h = hashlib.blake2b((q[:400] + "\u241f" + a[:200])
                                    .encode("utf-8", "ignore"),
                                    digest_size=8).digest()
                if h in seen:
                    continue
                seen.add(h)
            dom = domain_of(intent or q[:200])
            if MAX_PER_DOMAIN and dom not in PRIORITY and cap[dom] >= MAX_PER_DOMAIN:
                continue
            cap[dom] += 1
            batch_meta.append({"question": q[:500], "answer": a[:ANS_STORE],
                               "intent": intent[:120], "domain": dom,
                               "lang": (lang or "en")[:8], "source": rid})
            batch_txt.append(q + "\n" + a[:400])
            kept += 1
            if len(batch_txt) >= BATCH:
                flush()
            if MAX_INDEX and kept >= MAX_INDEX:
                done = True
                break
    flush()
    total = builder.finish()
    _log("BUILD DONE: %d vectors | scanned %d | %.0fs | mode=%s"
         % (total, scanned, time.time() - t0, builder.mode))

    cfg = {"embed_model": EMBED_MODEL, "dim": dim, "count": total,
           "index_type": builder.mode,
           "nprobe": (NPROBE if builder.mode == "ivfpq" else None),
           "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "per_domain": dict(cap)}
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    try:
        api.create_repo(INDEX_REPO, repo_type="dataset", exist_ok=True,
                        private=False)
    except Exception as e:
        _log("create_repo", str(e)[:100])
    for path in ("index.faiss", "meta.parquet", "config.json"):
        sz = os.path.getsize(path) if os.path.exists(path) else 0
        _log("uploading", path, "(%d KB) ..." % (sz // 1024))
        for attempt in range(3):
            try:
                api.upload_file(path_or_fileobj=path, path_in_repo=path,
                                repo_id=INDEX_REPO, repo_type="dataset")
                break
            except Exception as e:
                _log("upload retry %d/3 for %s: %s"
                     % (attempt + 1, path, str(e)[:120]))
                time.sleep(5 * (attempt + 1))
        else:
            _log("UPLOAD FAILED for", path,
                 "-- check HF_TOKEN WRITE scope and repo access")
    _log("UPLOADED ->", INDEX_REPO)
    _log("config:", json.dumps(cfg, ensure_ascii=False))
    _log("Now restart the HF Space (or wait for cold start); rag.py will load "
         "the new index.")


if __name__ == "__main__":
    main()