#!/usr/bin/env python3
"""
Damru RAG BEAST rebuild -- full-corpus FAISS index for damru-rag-index
======================================================================
Rebuilds the RAG index that rag.py loads at serve time, scaling from the
current ~30k rows up to the FULL knowledge corpus (Damaru-ai/damru-knowledge,
~11.4M rows).

Produces the EXACT 3 files rag.py expects and pushes them to INDEX_REPO:
  * config.json   {embed_model, dim, count, index_type, metric, nprobe}
  * index.faiss   FAISS index (cosine via normalized vectors + inner product)
  * meta.parquet  per-vector rows: question, answer, domain, source, intent, lang

Why this is safe (non-destructive to the running Space):
  * Same embed_model + same file names + same schema -> rag.py loads it with
    ZERO code changes. Query path (BGE prefix) is unchanged.
  * Memory-bounded: embeddings + meta are streamed to shards on disk, so RAM
    stays flat even for 11.4M rows (naive flat index = ~17GB RAM; here IVFPQ
    keeps index.faiss < ~1GB).
  * Resumable: re-run after a crash -> shards already on disk are skipped.

Run on Kaggle/Colab T4 (Internet ON). NOT on GitHub Actions (no GPU there).
Deps auto-install on first run.

  HF_TOKEN=... python rebuild_rag.py            # full corpus
  MAX_ROWS=2000000 python rebuild_rag.py        # quick 2M first pass

Env (all optional):
  HF_TOKEN       push/pull private repos
  SRC_REPO       source dataset       (default Damaru-ai/damru-knowledge)
  INDEX_REPO     output dataset       (default Damaru-ai/damru-rag-index)
  EMBED_MODEL    embedder             (default BAAI/bge-small-en-v1.5 -> dim 384)
                 BAAI/bge-large-en-v1.5 -> dim 1024 (enables Supabase hot RAG)
  MAX_ROWS       cap rows, 0 = all    (default 0)
  CHUNK          rows per shard       (default 100000)
  BATCH          embed batch size     (default 256)
  INDEX_TYPE     auto|flat|ivfpq      (default auto: flat if <=FLAT_MAX else ivfpq)
  FLAT_MAX       flat cutoff          (default 150000)
  PQ_M           PQ subquantizers     (default 64; auto-reduced to divide dim)
  NPROBE         IVF probes (baked)   (default 32)
  META_MAX_ANS   answer chars in meta (default 1200)
  DEDUP          1 = drop dup Qs      (default 1)
  WORK_DIR       shard/work dir       (default rag_build)
  PUSH           1 = push to HF       (default 1)

Built by Shiva AI for Damru.
"""
import os
import json
import math
import time
import hashlib

SRC_REPO     = os.environ.get("SRC_REPO",     "Damaru-ai/damru-knowledge")
INDEX_REPO   = os.environ.get("INDEX_REPO",   "Damaru-ai/damru-rag-index")
EMBED_MODEL  = os.environ.get("EMBED_MODEL",  "BAAI/bge-small-en-v1.5")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
MAX_ROWS     = int(os.environ.get("MAX_ROWS", "0"))
CHUNK        = int(os.environ.get("CHUNK", "100000"))
BATCH        = int(os.environ.get("BATCH", "256"))
INDEX_TYPE   = os.environ.get("INDEX_TYPE", "auto").lower()
FLAT_MAX     = int(os.environ.get("FLAT_MAX", "150000"))
PQ_M         = int(os.environ.get("PQ_M", "64"))
NPROBE       = int(os.environ.get("NPROBE", "32"))
META_MAX_ANS = int(os.environ.get("META_MAX_ANS", "1200"))
DEDUP        = (os.environ.get("DEDUP", "1") == "1")
WORK_DIR     = os.environ.get("WORK_DIR", "rag_build")
PUSH         = (os.environ.get("PUSH", "1") == "1")

EMB_DIR  = os.path.join(WORK_DIR, "emb")
META_DIR = os.path.join(WORK_DIR, "meta")
OUT_DIR  = os.path.join(WORK_DIR, "out")


def _ensure_deps():
    """Auto-install the build stack if missing (Internet must be ON)."""
    import importlib.util
    need = [("sentence_transformers", "sentence-transformers"),
            ("faiss", "faiss-cpu"), ("pyarrow", "pyarrow"),
            ("datasets", "datasets"), ("huggingface_hub", "huggingface_hub"),
            ("numpy", "numpy")]
    missing = [pip for mod, pip in need if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    import subprocess, sys
    print(">> installing RAG build deps:", missing, "-- one-time, ~2-4 min", flush=True)
    rc = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *missing]).returncode
    if rc != 0:
        print(">> AUTO-INSTALL failed. Kaggle: Settings > Internet = ON, phir dobara run karo;", flush=True)
        print(">>   ya pehle: pip install -U sentence-transformers faiss-cpu pyarrow datasets huggingface_hub", flush=True)
        raise RuntimeError("dependency install failed -- enable Internet and re-run")
    print(">> RAG build deps installed OK", flush=True)


def _norm(s):
    return " ".join((s or "").lower().split())


def _row_fields(ex):
    q = (ex.get("question") or ex.get("q") or "").strip()
    a = (ex.get("answer") or ex.get("a") or "").strip()
    domain = (ex.get("domain") or ex.get("intent") or "general")
    source = (ex.get("source") or SRC_REPO.split("/")[-1])
    intent = (ex.get("intent") or "")
    lang = (ex.get("lang") or "en")
    return q, a, domain, source, intent, lang


def build_shards(model):
    """Stream SRC_REPO, embed questions, write emb_*.npy + meta_*.parquet shards.
    One CHUNK in RAM at a time. Existing shards are skipped (resume)."""
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from datasets import load_dataset
    os.makedirs(EMB_DIR, exist_ok=True)
    os.makedirs(META_DIR, exist_ok=True)

    ds = load_dataset(SRC_REPO, split="train", streaming=True, token=HF_TOKEN or None)
    seen = set()
    txt, meta = [], []
    shard = 0
    total = 0
    t0 = time.time()

    def flush(shard_idx, texts, metas):
        emb_p = os.path.join(EMB_DIR, "emb_%05d.npy" % shard_idx)
        meta_p = os.path.join(META_DIR, "meta_%05d.parquet" % shard_idx)
        if os.path.exists(emb_p) and os.path.exists(meta_p):
            print("   shard %d exists -> skip embed" % shard_idx, flush=True)
            return
        vecs = model.encode(texts, batch_size=BATCH, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False)
        np.save(emb_p, vecs.astype("float32"))
        pq.write_table(pa.Table.from_pylist(metas), meta_p)
        print("   shard %d written (%d rows)" % (shard_idx, len(texts)), flush=True)

    for ex in ds:
        q, a, domain, source, intent, lang = _row_fields(ex)
        text = q or a
        if len(text) < 6:
            continue
        if DEDUP and q:
            h = hashlib.md5(_norm(q).encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
        txt.append(text)
        meta.append({"question": q[:500], "answer": a[:META_MAX_ANS],
                     "domain": domain, "source": source,
                     "intent": intent, "lang": lang})
        total += 1
        if len(txt) >= CHUNK:
            flush(shard, txt, meta)
            shard += 1
            txt, meta = [], []
            print("   ... %d rows streamed (%.0fs)" % (total, time.time() - t0), flush=True)
        if MAX_ROWS and total >= MAX_ROWS:
            break
    if txt:
        flush(shard, txt, meta)
        shard += 1
    print(">> shards done: %d shards, %d rows (%.0fs)" % (shard, total, time.time() - t0), flush=True)
    return shard, total


def _load_npy(path):
    import numpy as np
    return np.load(path)


def build_index(shard_count, total, dim):
    import numpy as np
    import faiss
    itype = INDEX_TYPE
    if itype == "auto":
        itype = "flat" if total <= FLAT_MAX else "ivfpq"
    emb_files = [os.path.join(EMB_DIR, "emb_%05d.npy" % s) for s in range(shard_count)]

    if itype == "flat":
        index = faiss.IndexFlatIP(dim)
        for p in emb_files:
            index.add(_load_npy(p))
    else:
        nlist = int(4 * math.sqrt(max(total, 1)))
        nlist = max(256, min(65536, nlist))
        nlist = min(nlist, max(256, total // 40))
        m = PQ_M
        while m > 1 and dim % m != 0:
            m //= 2
        print(">> IVFPQ nlist=%d m=%d" % (nlist, m), flush=True)
        index = faiss.index_factory(dim, "IVF%d,PQ%d" % (nlist, m),
                                    faiss.METRIC_INNER_PRODUCT)
        need = min(max(nlist * 40, 100000), total)
        sample, got = [], 0
        for p in emb_files:
            v = _load_npy(p)
            sample.append(v)
            got += len(v)
            if got >= need:
                break
        train = np.vstack(sample)
        print(">> training IVFPQ on %d vectors ..." % len(train), flush=True)
        index.train(train)
        for p in emb_files:
            index.add(_load_npy(p))
        index.nprobe = NPROBE
    os.makedirs(OUT_DIR, exist_ok=True)
    idx_p = os.path.join(OUT_DIR, "index.faiss")
    faiss.write_index(index, idx_p)
    print(">> index.faiss written: %d vectors (%s)" % (index.ntotal, itype), flush=True)
    return itype, index.ntotal, idx_p


def merge_meta(shard_count):
    import pyarrow.parquet as pq
    os.makedirs(OUT_DIR, exist_ok=True)
    out_p = os.path.join(OUT_DIR, "meta.parquet")
    writer = None
    rows = 0
    for s in range(shard_count):
        p = os.path.join(META_DIR, "meta_%05d.parquet" % s)
        t = pq.read_table(p)
        if writer is None:
            writer = pq.ParquetWriter(out_p, t.schema)
        writer.write_table(t)
        rows += t.num_rows
    if writer is not None:
        writer.close()
    print(">> meta.parquet written: %d rows" % rows, flush=True)
    return out_p, rows


def write_config(dim, count, itype):
    os.makedirs(OUT_DIR, exist_ok=True)
    cfg = {"embed_model": EMBED_MODEL, "dim": int(dim), "count": int(count),
           "index_type": itype, "metric": "ip", "nprobe": NPROBE,
           "source": SRC_REPO,
           "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    p = os.path.join(OUT_DIR, "config.json")
    with open(p, "w") as f:
        json.dump(cfg, f, indent=2)
    print(">> config.json:", json.dumps(cfg), flush=True)
    return p, cfg


def push(files):
    from huggingface_hub import HfApi, create_repo
    if not HF_TOKEN:
        print(">> no HF_TOKEN -> skip push. Files ready in:", OUT_DIR, flush=True)
        return
    create_repo(INDEX_REPO, repo_type="dataset", token=HF_TOKEN, exist_ok=True)
    api = HfApi(token=HF_TOKEN)
    for p in files:
        name = os.path.basename(p)
        print(">> uploading %s ..." % name, flush=True)
        api.upload_file(path_or_fileobj=p, path_in_repo=name,
                        repo_id=INDEX_REPO, repo_type="dataset")
    print(">> pushed ->", INDEX_REPO, flush=True)


def main():
    print("== Damru RAG BEAST rebuild ==", flush=True)
    print("src=%s -> index=%s | embed=%s | max_rows=%s" %
          (SRC_REPO, INDEX_REPO, EMBED_MODEL, MAX_ROWS or "ALL"), flush=True)
    _ensure_deps()
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    from sentence_transformers import SentenceTransformer
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        dev = "cpu"
    print(">> loading embedder on", dev, flush=True)
    model = SentenceTransformer(EMBED_MODEL, device=dev)
    dim = model.get_sentence_embedding_dimension()
    print(">> embed dim =", dim, flush=True)

    shard_count, total = build_shards(model)
    if total == 0:
        raise RuntimeError("no rows embedded -- check SRC_REPO / HF_TOKEN")
    itype, ntotal, idx_p = build_index(shard_count, total, dim)
    meta_p, meta_rows = merge_meta(shard_count)
    cfg_p, _cfg = write_config(dim, ntotal, itype)
    if PUSH:
        push([cfg_p, idx_p, meta_p])
    print("== DONE ==  vectors=%d meta=%d type=%s" % (ntotal, meta_rows, itype), flush=True)
    print(">> NOTE: the Space caches the index in /tmp; after push, RESTART / Factory", flush=True)
    print(">>       reboot the Space so rag.py re-downloads the new index (else stale).", flush=True)
    if dim != 1024:
        print(">> NOTE: dim=%d -> Supabase HOT-RAG stays off (it needs dim 1024)." % dim, flush=True)
        print(">>       For hot RAG: EMBED_MODEL=BAAI/bge-large-en-v1.5 (heavier on the Space).", flush=True)


if __name__ == "__main__":
    main()
