#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
damru_ingest_gold.py  --  STEP 3 of the Damru 14B Master Plan
=============================================================
Curate a HIGH-TRUST, license-clean "gold" corpus from permissively-licensed
open datasets (LIMA philosophy: quality > quantity). This is SEPARATE from the
bulk `damru-omni`; gold rows carry gold=True and are weighted higher at SFT.

Engine = the same battle-tested v3 core as the omni/oracle builders:
  * COMMIT BATCHING: one throttled upload_folder -> never hits HF 429.
  * 429-AWARE RETRY: parse "retry after N" and sleep, no crash.
  * checkpoint + resume mirrored to the repo, atomic writes, dedup.
  * self time-budget, per-row/per-source try/except, file-level resume.
  * schema log + drop-sample log for painless diagnosis.
GOLD-SPECIFIC:
  * per-source declared LICENSE + copyleft/non-commercial BLOCKLIST (refuse).
  * strict QUALITY GATES (length bounds, dedup, text/lang heuristic).
  * per-source cap (GOLD_MAX_PER_SRC) to keep the corpus balanced.
  * splits: sft {prompt,response} / text {text} / reasoning {problem,answer,steps}

SOURCE SPEC: "repo:hint:mode[:license[:extra]]"
  hint  = sft | text | reasoning
  mode  = stream | files
  extra = (files mode) path-substring filter e.g. 20231101.en
          (stream mode) dataset config name e.g. sample-10BT

RUN:
  pip install "datasets>=2.19" "huggingface_hub>=0.24" "pyarrow>=15"
  export HF_TOKEN=hf_xxx
  python damru_ingest_gold.py
"""

import gzip
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUT_REPO        = os.environ.get("GOLD_OUT_REPO", "Damaru-ai/damru-gold")
SHARD_ROWS      = int(os.environ.get("GOLD_SHARD_ROWS", "25000"))
MAX_PER_SRC     = int(os.environ.get("GOLD_MAX_PER_SRC", "150000"))
TIME_BUDGET_MIN = int(os.environ.get("GOLD_TIME_BUDGET_MIN", "300"))
COMMIT_EVERY    = int(os.environ.get("GOLD_COMMIT_EVERY_SEC", "600"))
WORKDIR         = os.environ.get("GOLD_WORKDIR", "./gold_work")
MIRROR_STATE    = os.environ.get("GOLD_MIRROR_STATE", "1") == "1"
HF_TOKEN        = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
RESET_SOURCES   = [r.strip() for r in os.environ.get("GOLD_RESET_SOURCES", "").split(",") if r.strip()]

MIN_PROMPT = int(os.environ.get("GOLD_MIN_PROMPT", "8"))
MIN_RESP   = int(os.environ.get("GOLD_MIN_RESP", "20"))
MAX_LEN    = int(os.environ.get("GOLD_MAX_LEN", "12000"))
MIN_TEXT   = int(os.environ.get("GOLD_MIN_TEXT", "200"))
MAX_TEXT   = int(os.environ.get("GOLD_MAX_TEXT", "20000"))

# "repo:hint:mode[:license[:extra]]"
DEFAULT_SOURCES = [
    "nvidia/Open-SWE-Traces:sft:stream:cc-by-4.0",
    "greghavens/kimi-k3-coding-and-debugging-traces:sft:stream:apache-2.0",
    "AletheiaResearch/GLM-5.2-Agent:sft:stream:apache-2.0",
    "ianncity/GLM-5.2-Science:sft:stream:apache-2.0",
    "SupraLabs/reasoning-corpus-4K-5M-v1:reasoning:stream:apache-2.0",
    "HuggingFaceFW/fineweb-edu:text:stream:odc-by",
    "roneneldan/TinyStories:text:stream:cdla-sharing-1.0",
    "wikimedia/wikipedia:text:files:cc-by-sa-3.0:20231101.en",
]
SOURCES = [s.strip() for s in os.environ.get(
    "GOLD_SOURCES", ",".join(DEFAULT_SOURCES)).split(",") if s.strip()]

SPLITS = ("sft", "text", "reasoning")
MAX_RETRIES = 6

# copyleft / non-commercial -> refuse (user constraint: avoid AGPL/GPL/NC)
LICENSE_BLOCKLIST = ("agpl", "gpl-3", "gpl-2", "gpl-v", "lgpl",
                     "noncommercial", "non-commercial", "cc-by-nc", "cc-nc",
                     "-nc-", "-nc.", "proprietary", "closed")
# share-alike / weak-copyleft -> allowed but tagged for audit
LICENSE_SHAREALIKE = ("by-sa", "sharing", "share-alike", "sharealike", "mpl", "epl")

CKPT_DIR   = os.path.join(WORKDIR, "_ckpt")
DATA_DIR   = os.path.join(WORKDIR, "data")
DL_DIR     = os.path.join(WORKDIR, "_dl")
STATE_PATH = os.path.join(CKPT_DIR, "state.json")
SEEN_PATH  = os.path.join(CKPT_DIR, "seen.txt.gz")
README_PATH = os.path.join(WORKDIR, "README.md")
ERR_PATH   = os.path.join(WORKDIR, "errors.log")

for _d in (WORKDIR, CKPT_DIR, DATA_DIR, DL_DIR):
    os.makedirs(_d, exist_ok=True)
for _s in SPLITS:
    os.makedirs(os.path.join(DATA_DIR, _s), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(WORKDIR, "run.log"))],
)
log = logging.getLogger("gold")


def log_error(where: str, exc: Exception) -> None:
    try:
        with open(ERR_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {where}: "
                    f"{type(exc).__name__}: {exc}\n")
            f.write(traceback.format_exc() + "\n")
    except Exception:
        pass


def _is_rate_limit(msg: str) -> bool:
    m = msg.lower()
    return "429" in m or "too many requests" in m or "rate limit" in m


def retry(fn, *args, what: str = "op", **kwargs):
    delay = 3.0
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            msg = str(exc)
            log_error(f"retry:{what}:attempt{attempt}", exc)
            if _is_rate_limit(msg):
                wait = 300.0
                m = re.search(r"retry after (\d+)", msg.lower())
                if m:
                    wait = float(m.group(1)) + 5
                log.warning("[retry] %s rate-limited (%d/%d); sleeping %.0fs",
                            what, attempt, MAX_RETRIES, wait)
                if attempt == MAX_RETRIES:
                    break
                time.sleep(wait)
                continue
            log.warning("[retry] %s failed (%d/%d): %s", what, attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                break
            time.sleep(delay + random.uniform(0, 2.0))
            delay = min(delay * 2, 120.0)
    raise last  # type: ignore[misc]


def atomic_write_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sha_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((p or "").encode("utf-8", "ignore"))
        h.update(b"\x1f")
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# State + seen
# ---------------------------------------------------------------------------
def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            log_error("load_state", exc)
    return {"version": 1, "sources": {},
            "shard_index": {s: 0 for s in SPLITS},
            "rows_out": {s: 0 for s in SPLITS}, "total_out": 0}


def save_state(state: Dict[str, Any]) -> None:
    atomic_write_text(STATE_PATH, json.dumps(state, ensure_ascii=False, indent=2))


def load_seen() -> set:
    seen: set = set()
    if os.path.exists(SEEN_PATH):
        try:
            with gzip.open(SEEN_PATH, "rt", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        seen.add(line)
        except Exception as exc:
            log_error("load_seen", exc)
    log.info("[seen] loaded %d known ids", len(seen))
    return seen


def save_seen(seen: set) -> None:
    tmp = SEEN_PATH + ".tmp"
    try:
        with gzip.open(tmp, "wt", encoding="utf-8") as f:
            for sid in seen:
                f.write(sid + "\n")
        os.replace(tmp, SEEN_PATH)
    except Exception as exc:
        log_error("save_seen", exc)


# ---------------------------------------------------------------------------
# HF client + THROTTLED folder commit (the 429 fix)
# ---------------------------------------------------------------------------
_api = None
_last_commit = [0.0]


def hf_api():
    global _api
    if _api is None:
        from huggingface_hub import HfApi
        _api = HfApi(token=HF_TOKEN)
    return _api


def ensure_repo() -> None:
    def _mk():
        hf_api().create_repo(OUT_REPO, repo_type="dataset", exist_ok=True, private=True)
    retry(_mk, what="create_repo")


def commit_all(force: bool = False) -> None:
    """ONE upload_folder commit, throttled to <= 1 per COMMIT_EVERY seconds."""
    if not force and (time.time() - _last_commit[0] < COMMIT_EVERY):
        return

    def _up():
        hf_api().upload_folder(
            folder_path=WORKDIR, repo_id=OUT_REPO, repo_type="dataset",
            allow_patterns=["data/**", "_ckpt/**", "README.md"],
            commit_message=f"gold update {time.strftime('%Y-%m-%d %H:%M:%S')}",
        )
    try:
        retry(_up, what="commit_folder")
        _last_commit[0] = time.time()
        log.info("[commit] uploaded folder snapshot")
    except Exception as exc:
        log_error("commit_all", exc)
        log.warning("[commit] failed -- progress kept locally, will retry later")


def hf_try_download(path_in_repo: str, dest: str) -> bool:
    from huggingface_hub import hf_hub_download
    try:
        from huggingface_hub.utils import EntryNotFoundError
    except Exception:
        class EntryNotFoundError(Exception):
            pass
    delay = 3.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            p = hf_hub_download(repo_id=OUT_REPO, repo_type="dataset",
                                filename=path_in_repo, token=HF_TOKEN)
            with open(p, "rb") as src, open(dest, "wb") as out:
                out.write(src.read())
            return True
        except EntryNotFoundError:
            return False
        except Exception as exc:
            msg = str(exc)
            if "404" in msg or "Entry Not Found" in msg or "not found" in msg.lower():
                return False
            log_error(f"download:{path_in_repo}:attempt{attempt}", exc)
            if attempt == MAX_RETRIES:
                return False
            time.sleep(delay + random.uniform(0, 2.0))
            delay = min(delay * 2, 120.0)
    return False


def restore_from_hf() -> None:
    if not MIRROR_STATE:
        return
    if hf_try_download("_ckpt/state.json", STATE_PATH):
        log.info("[resume] restored state.json from HF")
    hf_try_download("_ckpt/seen.txt.gz", SEEN_PATH)


# ---------------------------------------------------------------------------
# License gate
# ---------------------------------------------------------------------------
def license_verdict(lic: str) -> Tuple[str, str]:
    low = (lic or "").lower()
    for bad in LICENSE_BLOCKLIST:
        if bad in low:
            return ("block", low)
    for sa in LICENSE_SHAREALIKE:
        if sa in low:
            return ("share_alike", low)
    return ("ok", low)


# ---------------------------------------------------------------------------
# Normalization + quality gates
# ---------------------------------------------------------------------------
P_KEYS = ("prompt", "instruction", "question", "query", "input", "task",
          "text_input", "user", "human", "context")
R_KEYS = ("response", "output", "answer", "completion", "solution", "chosen",
          "target", "label", "text_output", "assistant", "gpt", "code")
T_KEYS = ("text", "content", "document", "page_content", "body", "chunk",
          "data", "raw", "passage", "markdown")
M_KEYS = ("messages", "conversations", "conversation", "chat", "dialogue", "turns")
RSN_ANS = ("answer", "final_answer", "solution", "label", "output", "result")
RSN_STEP = ("steps", "reasoning", "cot", "rationale", "thinking", "explanation", "chain_of_thought")

_DBG = [0]


def _first(row: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] not in (None, "", []):
            v = row[k]
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return None


def _from_messages(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    msgs = None
    for mk in M_KEYS:
        if isinstance(row.get(mk), list) and row[mk]:
            msgs = row[mk]
            break
    if not msgs:
        return None
    prompt, response = [], None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or m.get("from") or "").lower()
        content = m.get("content") or m.get("value") or m.get("text") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role in ("assistant", "gpt", "model", "bot"):
            response = content
        elif role in ("user", "human", "system", "prompter"):
            prompt.append(content)
    if response and prompt:
        return ("\n\n".join(prompt).strip(), response.strip())
    return None


def _longest_str(row: Dict[str, Any]) -> Optional[str]:
    best = None
    for v in row.values():
        if isinstance(v, str) and len(v) >= 60 and (best is None or len(v) > len(best)):
            best = v
    return best


def _looks_texty(s: str) -> bool:
    if not s:
        return False
    sample = s[:1000]
    good = sum(1 for c in sample if c.isalpha() or c.isspace())
    return good / max(1, len(sample)) > 0.6


def _quality_ok_sft(p: str, r: str) -> bool:
    if len(p) < MIN_PROMPT or len(r) < MIN_RESP:
        return False
    if len(p) > MAX_LEN or len(r) > MAX_LEN:
        return False
    return True


def _text_ok(t: str) -> bool:
    return bool(t) and MIN_TEXT <= len(t) <= MAX_TEXT and _looks_texty(t)


def _normalize_inner(row: Dict[str, Any], hint: str) -> Optional[Dict[str, Any]]:
    if hint == "reasoning":
        problem = _first(row, ("problem", "question", "task", "prompt", "instruction"))
        if problem and len(problem) <= MAX_LEN:
            answer = _first(row, RSN_ANS)
            steps = _first(row, RSN_STEP)
            if answer or steps:
                return {"split": "reasoning", "problem": problem.strip(),
                        "answer": answer, "steps": steps}
    if hint == "text":
        text = _first(row, T_KEYS) or _longest_str(row)
        if text:
            text = text.strip()
        if _text_ok(text or ""):
            return {"split": "text", "text": text}
        return None
    # default: sft (with text as last resort)
    pair = _from_messages(row)
    if pair is None:
        p = _first(row, P_KEYS)
        r = _first(row, R_KEYS)
        if p and r:
            pair = (p, r)
    if pair:
        p, r = pair[0].strip(), pair[1].strip()
        if _quality_ok_sft(p, r):
            return {"split": "sft", "prompt": p, "response": r}
    text = _first(row, T_KEYS)
    if text:
        text = text.strip()
    if _text_ok(text or ""):
        return {"split": "text", "text": text}
    return None


def normalize(row: Dict[str, Any], hint: str) -> Optional[Dict[str, Any]]:
    rec = _normalize_inner(row, hint)
    if rec is None and _DBG[0] < 12:
        _DBG[0] += 1
        prev = {k: (str(v)[:80] if v not in (None, "") else repr(v))
                for k, v in list(row.items())[:10]}
        log.info("[drop] hint=%s no-match sample=%s", hint, prev)
    return rec


# ---------------------------------------------------------------------------
# Shard writer (local-first; commits are throttled)
# ---------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, state: Dict[str, Any], seen: set):
        self.state = state
        self.seen = seen
        self.buffers: Dict[str, List[str]] = {s: [] for s in SPLITS}
        self.dupes = 0

    def add(self, rec: Dict[str, Any], source: str, lic: str, sa: bool) -> bool:
        split = rec.pop("split")
        rid = sha_id(split, json.dumps(rec, ensure_ascii=False, sort_keys=True))
        if rid in self.seen:
            self.dupes += 1
            return False
        self.seen.add(rid)
        out = {"id": rid, "source": source, "provenance": source,
               "license": lic, "gold": True, **rec}
        if sa:
            out["share_alike"] = True
            out["needs_audit"] = True
        self.buffers[split].append(json.dumps(out, ensure_ascii=False))
        if len(self.buffers[split]) >= SHARD_ROWS:
            self.flush(split)
        return True

    def flush(self, split: str) -> None:
        buf = self.buffers[split]
        if not buf:
            return
        idx = self.state["shard_index"][split]
        fname = f"{split}-{idx:05d}.jsonl"
        local = os.path.join(DATA_DIR, split, fname)
        atomic_write_text(local, "\n".join(buf) + "\n")
        self.state["shard_index"][split] = idx + 1
        self.state["rows_out"][split] = self.state["rows_out"].get(split, 0) + len(buf)
        self.state["total_out"] = self.state.get("total_out", 0) + len(buf)
        self.buffers[split] = []
        save_state(self.state)
        save_seen(self.seen)
        commit_all()  # throttled -> avoids 429
        log.info("[flush] %s shard %d (+%d) | total=%d", split, idx, len(buf), self.state["total_out"])

    def flush_all(self) -> None:
        for s in SPLITS:
            self.flush(s)


# ---------------------------------------------------------------------------
# Source processing
# ---------------------------------------------------------------------------
def process_source_stream(repo, hint, writer, state, spec, deadline, lic, sa, extra):
    src_state = state["sources"].get(spec, {"status": "pending", "rows_seen": 0, "rows_kept": 0})
    already = src_state.get("rows_seen", 0)
    log.info("[stream] %s (hint=%s cfg=%s) resume_from=%d", spec, hint, extra, already)
    from datasets import load_dataset

    def _open():
        return load_dataset(repo, name=extra, split="train", streaming=True, token=HF_TOKEN)
    try:
        ds = retry(_open, what=f"load:{repo}")
    except Exception as exc:
        log_error(f"open_source:{spec}", exc)
        log.error("[stream] cannot open %s -- skipping.", spec)
        src_state["status"] = "error"
        state["sources"][spec] = src_state
        save_state(state)
        return "error"

    seen_n, kept_n = 0, src_state.get("rows_kept", 0)
    logged_schema = False
    iterator = iter(ds)
    while True:
        try:
            row = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            log_error(f"iter:{spec}", exc)
            continue
        if not logged_schema and isinstance(row, dict):
            log.info("[schema] %s columns: %s", spec, list(row.keys()))
            logged_schema = True
        seen_n += 1
        if seen_n <= already:
            continue
        try:
            rec = normalize(row, hint)
            if rec and writer.add(rec, repo, lic, sa):
                kept_n += 1
        except Exception as exc:
            log_error(f"row:{spec}:{seen_n}", exc)
        if kept_n and MAX_PER_SRC and kept_n >= MAX_PER_SRC:
            src_state.update(status="done", rows_seen=seen_n, rows_kept=kept_n)
            state["sources"][spec] = src_state
            writer.flush_all()
            save_state(state)
            commit_all(force=True)
            log.info("[cap] %s reached kept cap %d (seen=%d)", spec, MAX_PER_SRC, seen_n)
            return "done"
        if seen_n % 2000 == 0:
            src_state.update(status="partial", rows_seen=seen_n, rows_kept=kept_n)
            state["sources"][spec] = src_state
            save_state(state)
            if time.time() > deadline:
                writer.flush_all()
                commit_all(force=True)
                log.info("[budget] %s paused seen=%d kept=%d", spec, seen_n, kept_n)
                return "budget"
            if seen_n % 10000 == 0:
                log.info("[progress] %s seen=%d kept=%d dupes=%d", spec, seen_n, kept_n, writer.dupes)

    writer.flush_all()
    src_state.update(status="done", rows_seen=seen_n, rows_kept=kept_n)
    state["sources"][spec] = src_state
    save_state(state)
    commit_all(force=True)
    log.info("[done] %s seen=%d kept=%d dupes=%d", spec, seen_n, kept_n, writer.dupes)
    return "done"


def process_source_files(repo, hint, writer, state, spec, deadline, lic, sa, extra):
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    src_state = state["sources"].get(spec, {"status": "pending", "files_done": [], "rows_kept": 0})
    done = set(src_state.get("files_done", []))
    try:
        files = retry(hf_api().list_repo_files, repo_id=repo, repo_type="dataset", what=f"list:{repo}")
    except Exception as exc:
        log_error(f"list:{spec}", exc)
        src_state["status"] = "error"
        state["sources"][spec] = src_state
        save_state(state)
        return "error"
    parquets = sorted(f for f in files
                      if f.endswith(".parquet") and (not extra or extra in f))
    log.info("[files] %s: %d parquet files (%d done, filter=%s)", spec, len(parquets), len(done), extra)
    logged_schema = False
    capped = False

    for fn in parquets:
        if capped:
            break
        if fn in done:
            continue
        if time.time() > deadline:
            src_state.update(status="partial", files_done=sorted(done))
            state["sources"][spec] = src_state
            save_state(state)
            commit_all(force=True)
            log.info("[budget] %s paused before %s", spec, fn)
            return "budget"
        try:
            local = retry(hf_hub_download, repo_id=repo, repo_type="dataset",
                          filename=fn, token=HF_TOKEN, local_dir=DL_DIR, what=f"dlfile:{fn}")
        except Exception as exc:
            log_error(f"dlfile:{spec}:{fn}", exc)
            continue
        try:
            pf = pq.ParquetFile(local)
            for batch in pf.iter_batches(batch_size=2000):
                if capped:
                    break
                for row in batch.to_pylist():
                    if not logged_schema and isinstance(row, dict):
                        log.info("[schema] %s columns: %s", spec, list(row.keys()))
                        logged_schema = True
                    try:
                        rec = normalize(row, hint)
                        if rec and writer.add(rec, repo, lic, sa):
                            src_state["rows_kept"] = src_state.get("rows_kept", 0) + 1
                    except Exception as exc:
                        log_error(f"row:{spec}:{fn}", exc)
                    if MAX_PER_SRC and src_state.get("rows_kept", 0) >= MAX_PER_SRC:
                        capped = True
                        break
        except Exception as exc:
            log_error(f"parse:{spec}:{fn}", exc)
        finally:
            try:
                os.remove(local)
            except Exception:
                pass
        done.add(fn)
        writer.flush_all()
        src_state.update(status="partial", files_done=sorted(done))
        state["sources"][spec] = src_state
        save_state(state)
        commit_all()  # throttled
        log.info("[file-done] %s %s kept=%d dupes=%d", spec, fn, src_state.get("rows_kept", 0), writer.dupes)

    writer.flush_all()
    src_state.update(status="done", files_done=sorted(done))
    state["sources"][spec] = src_state
    save_state(state)
    commit_all(force=True)
    log.info("[done] %s files=%d kept=%d dupes=%d", spec, len(done), src_state.get("rows_kept", 0), writer.dupes)
    return "done"


def process_source(spec, writer, state, deadline):
    parts = spec.split(":")
    repo = parts[0]
    hint = parts[1] if len(parts) > 1 else "sft"
    mode = parts[2] if len(parts) > 2 else "stream"
    lic  = parts[3] if len(parts) > 3 else "unknown"
    extra = parts[4] if len(parts) > 4 else None

    verdict, low = license_verdict(lic)
    if verdict == "block":
        log.warning("[license] BLOCKED %s (license=%s) -- refused, not ingested.", spec, lic)
        state["sources"][spec] = {"status": "blocked", "license": lic}
        save_state(state)
        return "blocked"
    sa = (verdict == "share_alike")
    if sa:
        log.info("[license] %s share-alike (%s) -- allowed, tagged needs_audit.", spec, lic)

    src_state = state["sources"].get(spec)
    if src_state and src_state.get("status") == "done":
        log.info("[skip] %s already done", spec)
        return "done"
    if mode == "files":
        return process_source_files(repo, hint, writer, state, spec, deadline, lic, sa, extra)
    return process_source_stream(repo, hint, writer, state, spec, deadline, lic, sa, extra)


def write_readme(state: Dict[str, Any]) -> None:
    rows = state.get("rows_out", {})
    md = [
        "---", "license: other", "tags: [damru, gold, curated, sft, reasoning]", "---", "",
        "# Damru Gold (curated high-trust corpus)", "",
        "Auto-built by `damru_ingest_gold.py` (Step 3). Quality > quantity (LIMA).",
        "Only permissive licenses; copyleft/non-commercial refused. Share-alike tagged.", "",
        "## Splits", "",
        f"- **sft**: {rows.get('sft', 0)} -- {{prompt, response}}",
        f"- **text**: {rows.get('text', 0)} -- {{text}} (clean pretrain-quality)",
        f"- **reasoning**: {rows.get('reasoning', 0)} -- {{problem, answer, steps}}",
        "", f"**Total kept:** {state.get('total_out', 0)}", "",
        "Rows carry id (sha1), source, provenance, license, gold=true, share_alike/needs_audit.",
    ]
    atomic_write_text(README_PATH, "\n".join(md) + "\n")


def main() -> int:
    if not HF_TOKEN:
        log.error("HF_TOKEN env var is required.")
        return 2
    log.info("=== Damru Gold Ingest (Step 3) | out=%s | sources=%d | budget=%dmin | cap=%d ===",
             OUT_REPO, len(SOURCES), TIME_BUDGET_MIN, MAX_PER_SRC)
    ensure_repo()
    restore_from_hf()
    state = load_state()
    if RESET_SOURCES:
        for _k in list(state.get("sources", {}).keys()):
            if any(r in _k for r in RESET_SOURCES):
                log.info("[reset] clearing %s state -> will reprocess from 0", _k)
                state["sources"].pop(_k, None)
        save_state(state)
    seen = load_seen()
    writer = ShardWriter(state, seen)
    deadline = time.time() + TIME_BUDGET_MIN * 60

    stopped = False
    for spec in SOURCES:
        try:
            res = process_source(spec, writer, state, deadline)
        except Exception as exc:
            log_error(f"process_source:{spec}", exc)
            log.error("[source] %s failed hard -- continuing.", spec)
            res = "error"
        if res == "budget":
            stopped = True
            break

    writer.flush_all()
    write_readme(state)
    commit_all(force=True)
    if stopped:
        log.info("=== PARTIAL saved. Re-run to continue. total=%d ===", state.get("total_out", 0))
    else:
        log.info("=== DONE. total kept=%d | splits=%s ===",
                 state.get("total_out", 0), state.get("rows_out", {}))
    log.info("Dataset: https://huggingface.co/datasets/%s", OUT_REPO)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warning("Interrupted -- state saved, safe to re-run to resume.")
        sys.exit(130)
