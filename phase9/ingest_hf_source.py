#!/usr/bin/env python3
"""Generic HF dataset -> Damru knowledge ingester.

Pulls any HuggingFace dataset, auto-detects its question/answer columns,
normalizes it into the Damru knowledge schema, dedups, and uploads a single
parquet shard to the damru-knowledge repo as data/<SOURCE_NAME>-0000.parquet.

The cold index builder (build_bge_m3_index.py) reads data/*.parquet, so the new
shard is automatically included on the next cold-index rebuild. Reusable for
NCERT now and exambench / other sources later -- just change env vars.

Env:
  HF_TOKEN        (required, needs WRITE access to DST_REPO)
  SRC_DATASET     (required, e.g. KadamParth/Ncert_dataset)
  SOURCE_NAME     (required, short slug, e.g. ncert)   -> shard + source tag
  DST_REPO        default Damaru-ai/damru-knowledge
  SRC_CONFIG      optional dataset config name
  SRC_SPLIT       default train
  INTENT_BASE     default = SOURCE_NAME (domain hint for the index)
  LANG_DEFAULT    default en
  MAX_ROWS        default 0 (0 = all rows)
  MIN_Q           default 8
  MIN_A           default 20
  DRY_RUN         default 0 (1 = detect + preview only, no upload)
"""
from __future__ import annotations
import os, re, json
from datetime import datetime, timezone

HF_TOKEN    = os.getenv("HF_TOKEN", "")
SRC_DATASET = os.getenv("SRC_DATASET", "")
SOURCE_NAME = os.getenv("SOURCE_NAME", "").strip().lower().replace(" ", "_")
DST_REPO    = os.getenv("DST_REPO", "Damaru-ai/damru-knowledge")
SRC_CONFIG  = os.getenv("SRC_CONFIG", "") or None
SRC_SPLIT   = os.getenv("SRC_SPLIT", "train")
INTENT_BASE = os.getenv("INTENT_BASE", "").strip() or SOURCE_NAME
LANG_DEF    = os.getenv("LANG_DEFAULT", "en")
MAX_ROWS    = int(os.getenv("MAX_ROWS", "0") or "0")
MIN_Q       = int(os.getenv("MIN_Q", "8"))
MIN_A       = int(os.getenv("MIN_A", "20"))
DRY_RUN     = os.getenv("DRY_RUN", "0") == "1"

# Column name candidates (checked in priority order, case-insensitive).
Q_CANDS = ["question", "instruction", "prompt", "query", "input", "title", "topic"]
A_CANDS = ["answer", "response", "output", "completion", "solution",
           "explanation", "text", "content", "body", "passage"]
CTX_CANDS = ["input", "context", "passage"]  # extra context appended to question
META_CANDS = ["subject", "class", "standard", "grade", "chapter", "topic", "category"]

# Keyword -> domain hint so the index buckets NCERT science/math into STEM etc.
STEM_KW = ("scien", "physic", "chem", "bio", "math", "comput")


def norm(s):
    return " ".join(str(s or "").lower().split())


def pick(cols_lower, cands, exclude=()):
    for c in cands:
        if c in cols_lower and c not in exclude:
            return cols_lower[c]
    return None


def main():
    assert HF_TOKEN, "HF_TOKEN required (must have WRITE access to DST_REPO)"
    assert SRC_DATASET, "SRC_DATASET required"
    assert SOURCE_NAME, "SOURCE_NAME required"
    from datasets import load_dataset
    from huggingface_hub import HfApi
    import pyarrow as pa
    import pyarrow.parquet as pq

    api = HfApi(token=HF_TOKEN)

    # --- verify write access early (clear error instead of failing at upload) ---
    if not DRY_RUN:
        try:
            api.auth_check() if hasattr(api, "auth_check") else api.whoami()
            api.create_repo(DST_REPO, repo_type="dataset", exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                "HF_TOKEN cannot write to %s. Use a WRITE token. Detail: %s"
                % (DST_REPO, str(e)[:160]))

    print("Loading dataset:", SRC_DATASET, "config=", SRC_CONFIG, "split=", SRC_SPLIT, flush=True)
    ds = load_dataset(SRC_DATASET, SRC_CONFIG, split=SRC_SPLIT, streaming=True, token=HF_TOKEN)

    # --- detect schema from the first row ---
    first = next(iter(ds))
    cols_lower = {c.lower(): c for c in first.keys()}
    print("Columns found:", list(first.keys()), flush=True)

    qcol = pick(cols_lower, Q_CANDS)
    acol = pick(cols_lower, A_CANDS, exclude=({qcol.lower()} if qcol else set()))
    ctxcol = pick(cols_lower, CTX_CANDS, exclude={c.lower() for c in (qcol, acol) if c})
    metacol = pick(cols_lower, META_CANDS, exclude={c.lower() for c in (qcol, acol, ctxcol) if c})

    text_only = False
    if qcol and acol:
        mode = "qa"
    elif acol and not qcol:
        mode = "text_only"; text_only = True
    elif qcol and not acol:
        # only a question-like col -> treat it as the passage text
        acol = qcol; qcol = None; mode = "text_only"; text_only = True
    else:
        raise RuntimeError("Could not detect usable text columns in: %s" % list(first.keys()))

    print("Detected mode=%s  question=%s  answer=%s  context=%s  meta=%s"
          % (mode, qcol, acol, ctxcol, metacol), flush=True)
    print("Sample row (truncated):",
          json.dumps({k: (str(v)[:160]) for k, v in first.items()}, ensure_ascii=False)[:600],
          flush=True)

    # re-open stream so we include the first row too
    ds = load_dataset(SRC_DATASET, SRC_CONFIG, split=SRC_SPLIT, streaming=True, token=HF_TOKEN)

    now = datetime.now(timezone.utc).isoformat()
    seen = set()
    Q, A, I, L, C = [], [], [], [], []
    scanned = kept = skipped = 0

    for ex in ds:
        scanned += 1
        if text_only:
            body = str(ex.get(acol) or "").strip()
            # derive a short question/title from the first line or meta
            head = re.split(r"[.\n?]", body, 1)[0].strip()[:180]
            meta = str(ex.get(metacol) or "").strip() if metacol else ""
            q = (meta + ": " + head).strip(": ").strip() if meta else head
            a = body
        else:
            q = str(ex.get(qcol) or "").strip()
            if ctxcol:
                ctx = str(ex.get(ctxcol) or "").strip()
                if ctx:
                    q = (q + "\n" + ctx).strip()
            a = str(ex.get(acol) or "").strip()

        if len(q) < MIN_Q or len(a) < MIN_A:
            skipped += 1
            continue
        key = norm(q)[:400]
        if key in seen:
            skipped += 1
            continue
        seen.add(key)

        # intent / domain hint
        meta = str(ex.get(metacol) or "").strip().lower() if metacol else ""
        intent = INTENT_BASE + ((" " + meta) if meta else "")
        if not any(kw in intent for kw in STEM_KW) and any(kw in norm(q + " " + meta) for kw in STEM_KW):
            intent += " science"

        Q.append(q[:2000]); A.append(a[:6000]); I.append(intent[:80])
        L.append((str(ex.get(cols_lower.get("lang", "")) or "").strip() or LANG_DEF)[:12])
        C.append(now)
        kept += 1
        if MAX_ROWS and kept >= MAX_ROWS:
            break
        if scanned % 20000 == 0:
            print("scanned %d kept %d skipped %d" % (scanned, kept, skipped), flush=True)

    print("TOTAL scanned=%d kept=%d skipped=%d" % (scanned, kept, skipped), flush=True)
    if kept == 0:
        raise RuntimeError("No usable rows produced -- check detected columns above.")

    table = pa.table({"question": Q, "answer": A, "intent": I, "lang": L, "created_at": C})
    out_path = "%s-0000.parquet" % SOURCE_NAME
    pq.write_table(table, out_path, compression="zstd")
    size_mb = os.path.getsize(out_path) / 1e6
    print("Wrote %s  rows=%d  size=%.1f MB" % (out_path, kept, size_mb), flush=True)

    if DRY_RUN:
        print("DRY_RUN=1 -> not uploading. Review the detection above, then run for real.")
        return

    dest = "data/%s-0000.parquet" % SOURCE_NAME
    api.upload_file(path_or_fileobj=out_path, path_in_repo=dest,
                    repo_id=DST_REPO, repo_type="dataset",
                    commit_message="Add %s knowledge shard (%d rows)" % (SOURCE_NAME, kept))
    print(json.dumps({"ok": True, "source": SOURCE_NAME, "dst": DST_REPO,
                      "shard": dest, "rows": kept, "size_mb": round(size_mb, 1)}, indent=2))
    print("DONE. Next: rebuild the cold index (RUN_COLD_INDEX=True) to include", SOURCE_NAME)


if __name__ == "__main__":
    main()
