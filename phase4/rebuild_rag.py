#!/usr/bin/env python3
"""
Damru RAG INDEX REBUILDER  --  FULL KNOWLEDGE BASE (v5)
=======================================================
v5 = v4 engine (streaming reader, auto Flat/IVFPQ, upload retry, HF_TOKEN
auto-resolve) PLUS the two things that burned the last 12-hour Kaggle run:

  1. TIME BUDGET -- the run STOPS ITSELF at TIME_BUDGET_SEC (default 41400 s
     = 11 h 30 m). It then finishes the index and uploads index.faiss +
     meta.parquet + config.json + state.json BEFORE Kaggle's hard 12 h kill
     (exit code 137) can throw 12 hours of embedding away.
  2. RESUME -- every run writes state.json into INDEX_REPO: which data files
     are 100% done, how many rows of the in-progress file were consumed, the
     dedup hash set (seen.u64.bin), plus the partial index.faiss/meta.parquet.
     The next run downloads all of that and KEEPS ADDING to the SAME index.
     Nothing is ever embedded twice.

WORKFLOW
  run 1 : rebuild_rag.py          -> prints COMPLETE, or PARTIAL + how much left
  run 2+: rebuild_rag_resume.py   -> repeat until it prints COMPLETE
  then  : restart the HF Space; rag.py loads index.faiss + meta.parquet as-is.

WHAT IT DOES (unchanged from v4)
  1. Lists + streams every data file in SRC_REPO across all prefixes
     (data/ curious/ world_tiles/ live_chats/ daily/ buckets/ ...) and formats
     (.parquet/.jsonl/.ndjson/.json/.csv/.tsv) = the full 11.46M corpus.
  2. Normalises every row -> {question, answer, intent, domain, lang, source}.
  3. Embeds with BAAI/bge-small-en-v1.5 (384d) on GPU, cosine via L2-norm.
  4. < IVF_THRESHOLD rows -> IndexFlatIP (exact); above -> IndexIVFPQ
     (compressed, nprobe baked in, so rag.py needs NO change).
  5. Uploads index.faiss + meta.parquet + config.json (+ state.json,
     seen.u64.bin for resume) to INDEX_REPO.

RUN ON KAGGLE WITH GPU + INTERNET ON (not runnable in the sandbox):
  Add-ons -> Secrets -> HF_TOKEN (WRITE)  [needed for Save Version runs]
  # os.environ["DRY_RUN"] = "1"          # inventory only, no embed
  # os.environ["MAX_INDEX"] = "200000"   # fast smoke test first
  # os.environ["TIME_BUDGET_SEC"] = "41400"   # 11h30m (default)
  then paste + run this whole file as ONE cell.

ENV (all optional except HF_TOKEN)
  HF_TOKEN          required (WRITE)
  SRC_REPO          Damaru-ai/damru-knowledge
  INDEX_REPO        Damaru-ai/damru-rag-index   (created if missing)
  EMBED_MODEL       BAAI/bge-small-en-v1.5
  TIME_BUDGET_SEC   41400   HARD self-stop budget (11h30m) for the whole run
  UPLOAD_RESERVE_SEC 1500   time kept aside to finish + upload (25m)
  CKPT_EVERY_SEC    21600   mid-run safety checkpoint upload (6h; 0 = off)
  RESUME            auto    auto | 1 (must resume) | 0 (ignore state)
  FRESH             0       1 = ignore state.json, rebuild from scratch
  SEEN_PERSIST      1       save/load dedup hash set across runs
  SEEN_MAX_MB       400     skip persisting the hash set above this size
  EXTRA_REPOS       ""      comma/space list of extra bucket repos
  BUCKETS_AUTHOR    ""      HF user/org to auto-discover bucket datasets from
  DRY_RUN           0       1 = list files only (no embed/upload)
  MAX_INDEX         0       cap total indexed rows (0 = ALL)
  MAX_PER_DOMAIN    0       cap dominant domains (PRIORITY stays uncapped)
  DEDUP             1       drop duplicate q+a (blake2b)
  BATCH             512     embed batch size
  ANS_STORE         1200    chars of answer kept in meta
  MIN_Q / MIN_A     8 / 12  min question / answer chars
  INDEX_TYPE        auto    auto | flat | ivfpq
  IVF_THRESHOLD     200000  switch to IVFPQ above this many rows
  NLIST / PQ_M      4096 / 48   (PQ_M must divide dim; 384 % 48 == 0)
  PQ_NBITS / NPROBE 8 / 16
  TRAIN_SAMPLE      300000  vectors used to train IVFPQ
  LOG_EVERY         20000   heartbeat rows
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

# ---- v5: time budget + resume knobs -------------------------------------
TIME_BUDGET    = int(_env("TIME_BUDGET_SEC", "41400"))      # 11h30m
UPLOAD_RESERVE = int(_env("UPLOAD_RESERVE_SEC", "1500"))    # 25m
CKPT_EVERY     = int(_env("CKPT_EVERY_SEC", "21600"))       # 6h, 0 = off
RESUME_MODE    = _env("RESUME", "auto").lower()
FRESH          = _env("FRESH", "0") == "1"
SEEN_PERSIST   = _env("SEEN_PERSIST", "1") == "1"
SEEN_MAX_MB    = int(_env("SEEN_MAX_MB", "400"))
EMBED_DEADLINE = max(300, TIME_BUDGET - UPLOAD_RESERVE)

STATE_FILE = "state.json"
SEEN_FILE  = "seen.u64.bin"
BASE_META  = "_base_meta.parquet"
SRC_COPY   = "builder_src.py"
STATE_VER  = 5

# infra repos that must never be treated as knowledge "buckets"
EXCLUDE = {
    "damru-knowledge", "damru-rag-index", "damru-14b-lora", "damru-14b-gguf",
    "damru-gguf", "damru-train", "damru-gurukul", "damru-oracle",
}
PRIORITY = {"medical", "holy"}
META_COLS = ("question", "answer", "intent", "domain", "lang", "source")

DATA_EXT = (".parquet", ".jsonl", ".ndjson", ".json", ".csv", ".tsv")
# v5: our own build artifacts must NEVER be ingested as knowledge rows
OWN_ARTIFACTS = {"state.json", "config.json", "meta.parquet", "index.faiss",
                 "builder_src.py", "seen.u64.bin", "_base_meta.parquet",
                 "_merged_meta.parquet", "_state_in.json"}
SKIP_TOKENS = ("manifest", "_state", "stats", "_meta", "metadata", "readme",
               "dataset_infos", "gitattributes", "checkpoint", "progress",
               "_index", "config", "scorecard", "license", ".gitignore")

Q_KEYS = ("question", "prompt", "instruction", "problem", "query", "task",
          "input", "title", "text", "content", "body")
A_KEYS = ("answer", "output", "response", "completion", "solution", "target",
          "label", "assistant", "chosen", "value")


def _log(*a):
    print("[rebuild-rag v5]", *a, flush=True)


def _hms(sec):
    sec = int(max(0, sec))
    return "%dh%02dm%02ds" % (sec // 3600, (sec % 3600) // 60, sec % 60)


def _is_data_file(path):
    p = path.lower()
    if not p.endswith(DATA_EXT):
        return False
    base = p.rsplit("/", 1)[-1]
    if base in OWN_ARTIFACTS:
        return False
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
    """Return (files, repos): every data file across SRC_REPO + extra buckets.
    v5: the list is SORTED so resume offsets stay valid across runs."""
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
    files.sort()
    return files, repos


class Builder:
    """Memory-smart incremental FAISS builder (auto Flat vs IVFPQ).
    v5: can be seeded with an EXISTING index + meta parquet (resume), and can
    snapshot itself mid-run without losing alignment (index add-order == meta
    row-order, so row i of meta.parquet describes vector i)."""

    def __init__(self, dim, index=None, mode=None, base_meta=None, base_count=0):
        self.dim = dim
        self.index = index
        self.mode = mode
        self.trained = index is not None
        self.buf_vecs = []
        self.buf_meta = []
        self.buffered = 0
        self.meta_writer = None
        self._cur_path = None
        self.part_paths = []
        self.base_meta_paths = [base_meta] if base_meta else []
        self._mbatch = []
        self._wseq = 0
        self.total_added = int(base_count)
        self.added_this_run = 0

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
            n = int(vecs.shape[0])
            self.total_added += n; self.added_this_run += n; return
        if INDEX_TYPE == "flat":
            self._build_flat(); self.trained = True
            self.index.add(vecs); self._write_meta(metas)
            n = int(vecs.shape[0])
            self.total_added += n; self.added_this_run += n; return
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
        self.total_added += n; self.added_this_run += n
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
            self._wseq += 1
            self._cur_path = "meta_part_%03d.parquet" % self._wseq
            self.meta_writer = pq.ParquetWriter(self._cur_path, table.schema,
                                               compression="zstd")
            self.part_paths.append(self._cur_path)
        self.meta_writer.write_table(table)
        self._mbatch = []

    def _close_writer(self):
        if self.meta_writer is not None:
            try:
                self.meta_writer.close()
            except Exception as e:
                _log("meta close warn", str(e)[:100])
            self.meta_writer = None

    def merge_meta(self, out_path="meta.parquet"):
        """Stream base meta (previous runs) + this run's parts into ONE
        meta.parquet, row-group by row-group (never loads it all in RAM).
        After the merge the output itself becomes the new base."""
        import pyarrow.parquet as pq
        self._flush_meta()
        self._close_writer()
        parts = [p for p in (self.base_meta_paths + self.part_paths)
                 if p and os.path.exists(p)]
        if not parts:
            return 0
        tmp = "_merged_meta.parquet"
        writer = None
        rows = 0
        for p in parts:
            try:
                pf = pq.ParquetFile(p)
            except Exception as e:
                _log("meta part unreadable", p, str(e)[:100]); continue
            for gi in range(pf.num_row_groups):
                t = pf.read_row_group(gi)
                if writer is None:
                    writer = pq.ParquetWriter(tmp, t.schema, compression="zstd")
                elif t.schema != writer.schema:
                    try:
                        t = t.select(list(META_COLS)).cast(writer.schema)
                    except Exception:
                        pass
                writer.write_table(t)
                rows += int(t.num_rows)
        if writer is None:
            return 0
        writer.close()
        os.replace(tmp, out_path)
        for p in self.part_paths:
            try:
                os.remove(p)
            except Exception:
                pass
        self.part_paths = []
        self.base_meta_paths = [out_path]
        self._wseq = 0
        return rows

    def write_index(self, path="index.faiss"):
        import faiss
        self._finalize_training()   # small-dataset case (never hit TRAIN_SAMPLE)
        faiss.write_index(self.index, path)
        return self.total_added

    def finish(self):
        total = self.write_index("index.faiss")
        rows = self.merge_meta("meta.parquet")
        return total, rows


# ------------------------------------------------------------------ resume I/O
def _dl_asset(name, local=None):
    """Download one file from INDEX_REPO into cwd. None if absent."""
    import shutil
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(INDEX_REPO, name, repo_type="dataset",
                            token=HF_TOKEN or None)
    except Exception as e:
        _log("resume: no", name, "(%s)" % str(e)[:70]); return None
    dst = local or name
    try:
        if os.path.abspath(p) != os.path.abspath(dst):
            shutil.copy(p, dst)
        sz = os.path.getsize(dst)
        _log("resume: got", name, "(%d KB)" % (sz // 1024))
        return dst
    except Exception as e:
        _log("resume: copy fail", name, str(e)[:90]); return None


def _load_state():
    """state.json from INDEX_REPO -> dict ({} if none/unusable)."""
    p = _dl_asset(STATE_FILE, "_state_in.json")
    if not p:
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {}
    except Exception as e:
        _log("state.json unreadable ->  fresh build", str(e)[:100]); return {}


def _load_seen():
    """Dedup hash set from the previous run (uint64 array on disk)."""
    if not (DEDUP and SEEN_PERSIST):
        return set()
    p = _dl_asset(SEEN_FILE, "_seen_in.bin")
    if not p:
        return set()
    try:
        arr = np.fromfile(p, dtype=np.uint64)
        s = set(arr.tolist())
        _log("resume: dedup hashes loaded = %d" % len(s))
        return s
    except Exception as e:
        _log("seen load fail", str(e)[:100]); return set()


def _save_seen(seen):
    """Persist dedup hashes; returns path or None."""
    if not (DEDUP and SEEN_PERSIST) or not seen:
        return None
    mb = (len(seen) * 8) / 1e6
    if mb > SEEN_MAX_MB:
        _log("seen set %.0f MB > SEEN_MAX_MB=%d -> not persisted"
             % (mb, SEEN_MAX_MB))
        return None
    try:
        np.fromiter(seen, dtype=np.uint64, count=len(seen)).tofile(SEEN_FILE)
        return SEEN_FILE
    except Exception as e:
        _log("seen save fail", str(e)[:100]); return None


def _self_source_path():
    """Copy this script next to the artifacts so rebuild_rag_resume.py can pull
    the EXACT engine version that produced the state. No-op in a notebook cell
    (no __file__) -- resume.py then falls back to GitHub raw."""
    import shutil
    try:
        me = os.path.abspath(__file__)
    except Exception:
        return None
    try:
        if os.path.exists(me) and os.path.abspath(SRC_COPY) != me:
            shutil.copy(me, SRC_COPY)
            return SRC_COPY
    except Exception:
        pass
    return None


def _upload_paths(api, paths):
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        sz = os.path.getsize(path)
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


def _publish(api, builder, cfg, state, seen, tag="final"):
    """Write index + merged meta + config + state (+ dedup hashes + engine src)
    and push everything to INDEX_REPO. Safe to call mid-run (checkpoint)."""
    t = time.time()
    total, mrows = builder.finish()
    cfg["count"] = total
    cfg["meta_rows"] = mrows
    cfg["index_type"] = builder.mode
    cfg["nprobe"] = (NPROBE if builder.mode == "ivfpq" else None)
    cfg["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    state["count"] = total
    state["meta_rows"] = mrows
    state["index_type"] = builder.mode
    state["updated_at"] = cfg["built_at"]
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)
    if total != mrows:
        _log("WARN: index vectors=%d but meta rows=%d (alignment!)"
             % (total, mrows))
    try:
        api.create_repo(INDEX_REPO, repo_type="dataset", exist_ok=True,
                        private=False)
    except Exception as e:
        _log("create_repo", str(e)[:100])
    paths = ["index.faiss", "meta.parquet", "config.json", STATE_FILE]
    sp = _save_seen(seen)
    if sp:
        paths.append(sp)
    src = _self_source_path()
    if src:
        paths.append(src)
    _upload_paths(api, paths)
    _log("%s publish done: %d vectors | meta %d rows | mode=%s | took %s"
         % (tag, total, mrows, builder.mode, _hms(time.time() - t)))
    return total


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
    t0 = time.time()
    _log("time budget = %s | embed till %s | %s reserved for finish+upload"
         % (_hms(TIME_BUDGET), _hms(EMBED_DEADLINE), _hms(UPLOAD_RESERVE)))
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

    # ---------------- resume state ----------------
    if FRESH:
        _log("FRESH=1 -> existing state.json ignored, rebuilding from scratch")
        state = {}
    elif RESUME_MODE == "0":
        state = {}
    else:
        state = _load_state()
    if RESUME_MODE == "1" and not state:
        _log("RESUME=1 but no usable state.json in", INDEX_REPO,
             "-> starting a FRESH build")

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

    base_index = base_mode = base_meta = None
    base_count = 0
    done_keys, partial = set(), {}
    prev_scanned = prev_kept = 0
    runs = []
    cap = Counter()
    seen = set()
    if state:
        if state.get("embed_model") and state["embed_model"] != EMBED_MODEL:
            _log("state embed_model=%s != %s -> state dropped (fresh build)"
                 % (state.get("embed_model"), EMBED_MODEL))
            state = {}
        elif int(state.get("dim") or dim) != dim:
            _log("state dim=%s != %d -> state dropped (fresh build)"
                 % (state.get("dim"), dim))
            state = {}
    if state:
        ip = _dl_asset("index.faiss", "index.faiss")
        mp = _dl_asset("meta.parquet", BASE_META)
        idx = None
        if ip and mp:
            import faiss
            try:
                idx = faiss.read_index(ip)
            except Exception as e:
                _log("index.faiss unreadable:", str(e)[:110]); idx = None
        if idx is not None and int(idx.d) == dim:
            base_index = idx
            base_mode = ("ivfpq" if "IVF" in type(idx).__name__.upper()
                         else "flat")
            if base_mode == "ivfpq":
                try:
                    idx.nprobe = NPROBE
                except Exception:
                    pass
            base_count = int(idx.ntotal)
            base_meta = mp
            done_keys = set(state.get("done_files") or [])
            partial = state.get("partial") or {}
            prev_scanned = int(state.get("scanned") or 0)
            prev_kept = int(state.get("kept") or 0)
            cap = Counter(state.get("per_domain") or {})
            runs = list(state.get("runs") or [])[-19:]
            seen = _load_seen()
            _log("RESUMING: %d vectors already in index | %d/%d files done | "
                 "mode=%s | partial=%s"
                 % (base_count, len(done_keys), len(files), base_mode,
                    (partial.get("file") or "-")))
            if state.get("complete"):
                _log("previous state was COMPLETE -> only NEW files will be added")
        else:
            _log("resume assets missing/incompatible -> FRESH build")
            base_index = base_mode = base_meta = None
            base_count = 0
            done_keys, partial = set(), {}
    else:
        _log("fresh build (no state to resume from)")

    builder = Builder(dim, index=base_index, mode=base_mode,
                      base_meta=base_meta, base_count=base_count)
    batch_txt, batch_meta = [], []
    scanned = kept = skipped_files = 0
    last_ckpt = time.time()
    stop_reason = "complete"
    new_partial = None

    def flush():
        if not batch_txt:
            return
        vecs = embedder.encode(batch_txt, batch_size=BATCH,
                               convert_to_numpy=True,
                               normalize_embeddings=True,
                               show_progress_bar=False).astype("float32")
        builder.add_batch(vecs, list(batch_meta))
        _log("  +batch indexed=%d buffered=%d kept=%d scanned=%d %s"
             % (builder.total_added, builder.buffered, kept, scanned,
                _hms(time.time() - t0)))
        batch_txt.clear()
        batch_meta.clear()

    def _mk_cfg():
        return {"embed_model": EMBED_MODEL, "dim": dim,
                "count": builder.total_added, "index_type": builder.mode,
                "nprobe": (NPROBE if builder.mode == "ivfpq" else None),
                "built_at": "", "per_domain": dict(cap)}

    def _mk_state(complete, reason, part):
        return {"version": STATE_VER, "complete": bool(complete),
                "embed_model": EMBED_MODEL, "dim": dim,
                "index_type": builder.mode, "count": builder.total_added,
                "files_total": len(files), "files_done": len(done_keys),
                "done_files": sorted(done_keys), "partial": part,
                "scanned": prev_scanned + scanned, "kept": prev_kept + kept,
                "per_domain": dict(cap), "stop_reason": reason,
                "time_budget_sec": TIME_BUDGET,
                "runs": runs + [{
                    "ended": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "added": builder.added_this_run,
                    "elapsed_sec": int(time.time() - t0),
                    "reason": reason}]}

    for (repo, fname) in files:
        key = repo + "::" + fname
        if key in done_keys:
            skipped_files += 1
            continue
        rid = repo.split("/")[-1]
        skip_rows = (int(partial.get("rows_done") or 0)
                     if partial.get("file") == key else 0)
        if skip_rows:
            _log("resuming inside", fname, "-> skipping first %d row(s)"
                 % skip_rows)
        i_row = 0
        hit = ""
        for row in read_rows(repo, fname):
            i_row += 1
            if i_row <= skip_rows:
                continue
            scanned += 1
            if scanned % LOG_EVERY == 0:
                _log("scanned=%d kept=%d indexed~=%d %s"
                     % (scanned, kept, builder.total_added,
                        _hms(time.time() - t0)))
            q, a, intent, lang = normalize(row)
            if len(q) < MIN_Q or len(a) < MIN_A:
                continue
            if DEDUP:
                h = int.from_bytes(
                    hashlib.blake2b((q[:400] + "\u241f" + a[:200])
                                    .encode("utf-8", "ignore"),
                                    digest_size=8).digest(), "big")
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
            if MAX_INDEX and (prev_kept + kept) >= MAX_INDEX:
                hit = "max_index"
                break
            if i_row % 2000 == 0 and (time.time() - t0) > EMBED_DEADLINE:
                hit = "budget"
                break
        if hit:
            new_partial = {"file": key, "rows_done": i_row}
            stop_reason = hit
            _log("STOP (%s) inside %s at row %d | elapsed %s"
                 % (hit, fname, i_row, _hms(time.time() - t0)))
            break
        done_keys.add(key)
        if (time.time() - t0) > EMBED_DEADLINE:
            stop_reason = "budget"
            _log("STOP (budget) at file boundary after %s | %d/%d files done"
                 % (_hms(time.time() - t0), len(done_keys), len(files)))
            break
        if CKPT_EVERY and (time.time() - last_ckpt) >= CKPT_EVERY:
            flush()
            _log("--- mid-run safety checkpoint (%s elapsed) ---"
                 % _hms(time.time() - t0))
            _publish(api, builder, _mk_cfg(), _mk_state(False, "checkpoint", None),
                     seen, tag="checkpoint")
            last_ckpt = time.time()

    flush()
    complete = (stop_reason == "complete")
    cfg = _mk_cfg()
    total = _publish(api, builder, cfg, _mk_state(complete, stop_reason,
                                                  new_partial),
                     seen, tag="final")
    _log("=" * 64)
    _log("RUN SUMMARY: reason=%s | added this run=%d | index total=%d vectors"
         % (stop_reason, builder.added_this_run, total))
    _log("files: %d done / %d total (already-done skipped=%d)"
         % (len(done_keys), len(files), skipped_files))
    _log("scanned=%d kept=%d | elapsed %s of budget %s"
         % (scanned, kept, _hms(time.time() - t0), _hms(TIME_BUDGET)))
    _log("per_domain:", json.dumps(dict(cap), ensure_ascii=False)[:600])
    if complete:
        _log("STATUS: COMPLETE -- pura knowledge base index ho gaya.")
        _log("Ab HF Space restart karo (ya cold start ka wait); rag.py naya "
             "index utha lega.")
    else:
        left = max(0, len(files) - len(done_keys))
        _log("STATUS: PARTIAL (%s) -- %d file(s) baaki hain." % (stop_reason, left))
        _log("Next: naya Kaggle notebook me phase4/rebuild_rag_resume.py chala do "
             "(same HF_TOKEN secret). Index + dedup hashes safe hain -- ek bhi "
             "row dobara embed nahi hogi.")
    _log("config:", json.dumps(cfg, ensure_ascii=False)[:800])


if __name__ == "__main__":
    main()
