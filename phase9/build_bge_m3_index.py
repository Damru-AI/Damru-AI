#!/usr/bin/env python3
"""
Damru RAG INDEX BUILDER
=======================
Builds a semantic-search index over the Damru knowledge base so the served
model can retrieve grounded, CITED context at answer time (RAG).

- Reads ALL shards (data/*.parquet) from damru-knowledge (bulk-* + train-*).
- Prioritises minority/high-value domains (medical/nursing, holy scripture)
  so exam + scripture citations stay strong; caps dominant domains.
- Embeds with multilingual BAAI/bge-m3 (1024-dim, 100+ languages).
- Builds a FAISS inner-product index over L2-normalised vectors (= cosine).
- Pushes index.faiss + meta.parquet + config.json to an HF dataset repo.

Env:
  HF_TOKEN         (required)
  SRC_REPO         Damaru-ai/damru-knowledge
  INDEX_REPO       Damaru-ai/damru-rag-index   (created if missing)
  EMBED_MODEL      BAAI/bge-m3
  MAX_INDEX        200000     (total rows to index; 0 = no cap)
  MAX_PER_DOMAIN   60000      (cap dominant domains; priority ones uncapped)
  BATCH            256
  ANS_STORE        1200       (chars of answer kept for context injection)
"""
import os
import json
import time
from collections import Counter

HF_TOKEN = os.environ.get("HF_TOKEN", "")
SRC_REPO = os.environ.get("SRC_REPO", "Damaru-ai/damru-knowledge")
INDEX_REPO = os.environ.get("INDEX_REPO", "Damaru-ai/damru-rag-index")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
MAX_INDEX = int(os.environ.get("MAX_INDEX") or "200000")
MAX_PER_DOMAIN = int(os.environ.get("MAX_PER_DOMAIN") or "60000")
BATCH = int(os.environ.get("BATCH") or "64")
ANS_STORE = int(os.environ.get("ANS_STORE") or "1200")
MIN_Q = int(os.environ.get("MIN_Q") or "8")
MIN_A = int(os.environ.get("MIN_A") or "20")

# domains we never cap (want full citation coverage for exam + scripture)
PRIORITY = {"medical", "holy"}


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


def main():
    assert HF_TOKEN, "HF_TOKEN required"
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    import faiss
    import pyarrow as pa
    import pyarrow.parquet as pq

    api = HfApi(token=HF_TOKEN)
    try:
        api.create_repo(INDEX_REPO, repo_type="dataset", exist_ok=True,
                        private=False)
    except Exception as e:
        print("create_repo:", str(e)[:80], flush=True)

    print("Loading embedder:", EMBED_MODEL, flush=True)
    embedder = SentenceTransformer(EMBED_MODEL)
    dim = embedder.get_sentence_embedding_dimension()
    index = faiss.IndexFlatIP(dim)

    # read EVERY shard (data_files overrides README config that hides bulk-*)
    ds = load_dataset(SRC_REPO, data_files="data/*.parquet",
                      split="train", streaming=True)

    cap = Counter()
    metas = []
    batch_txt = []
    kept = 0
    scanned = 0
    t0 = time.time()

    def flush_batch():
        if not batch_txt:
            return
        vecs = embedder.encode(batch_txt, batch_size=BATCH,
                               convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False)
        index.add(vecs.astype("float32"))
        batch_txt.clear()

    for ex in ds:
        scanned += 1
        q = (ex.get("question") or "").strip()
        a = (ex.get("answer") or "").strip()
        if len(q) < MIN_Q or len(a) < MIN_A:
            continue
        intent = (ex.get("intent") or "").strip()
        dom = domain_of(intent)
        if dom not in PRIORITY and MAX_PER_DOMAIN and cap[dom] >= MAX_PER_DOMAIN:
            continue
        cap[dom] += 1
        metas.append({"question": q[:500], "answer": a[:ANS_STORE],
                      "intent": intent, "domain": dom,
                      "lang": (ex.get("lang") or "en"), "source": SRC_REPO})
        batch_txt.append(q + "\n" + a[:400])
        kept += 1
        if len(batch_txt) >= BATCH:
            flush_batch()
        if MAX_INDEX and kept >= MAX_INDEX:
            break
        if scanned % 100000 == 0:
            print("scanned %d kept %d | %.0fs" %
                  (scanned, kept, time.time() - t0), flush=True)
    flush_batch()

    print("Indexed %d vectors (scanned %d) in %.0fs" %
          (index.ntotal, scanned, time.time() - t0), flush=True)

    faiss.write_index(index, "index.faiss")
    cols = {k: [m.get(k) for m in metas] for k in
            ("question", "answer", "intent", "domain", "lang", "source")}
    pq.write_table(pa.table(cols), "meta.parquet", compression="zstd")
    cfg = {"embed_model": EMBED_MODEL, "dim": dim, "count": kept,
           "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "per_domain": dict(cap)}
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    for path in ("index.faiss", "meta.parquet", "config.json"):
        api.upload_file(path_or_fileobj=path, path_in_repo=path,
                        repo_id=INDEX_REPO, repo_type="dataset")
        print("uploaded", path, flush=True)
    print("DONE.", json.dumps(cfg, indent=2), flush=True)


if __name__ == "__main__":
    main()
