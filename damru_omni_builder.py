#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
damru_omni_builder.py  --  STEP 1 of the Damru 14B Master Plan  (v3, rate-safe)
=============================================================================
Merge ALL Hugging Face datasets into ONE `Damaru-ai/damru-omni` with clean
splits + sha1 dedup + provenance tags.

v3 FIXES (why the run hit 429 Too Many Requests):
  * HF limits repo commits to 128/hour. v2 committed on EVERY shard (shard +
    state + seen = 3 commits/flush) -> blew the limit on millions of rows.
  1. COMMIT BATCHING: write everything locally, then ONE `upload_folder`
     commit at most every OMNI_COMMIT_EVERY_SEC (default 600s = 10 min).
     Unchanged files are auto-skipped by the Hub, so re-commits are cheap.
  2. 429-AWARE RETRY: on a rate-limit, parse "retry after N" and sleep, no crash.
  3. SCHEMA LOG: print the first row's column names per source, so we can see
     exactly why a source yields 0 rows and patch field names precisely.
  4. WIDER FIELD MATCHING for prompt/response/text/messages.

Kept from v2: reorder (small high-value first, giant knowledge last), file-
level resume for huge sources, self time-budget, checkpoint+resume mirrored
to the repo, atomic writes, retry+backoff, dedup, per-row/per-source try/except.

RUN:
  pip install "datasets>=2.19" "huggingface_hub>=0.24" "pyarrow>=15"
  export HF_TOKEN=hf_xxx
  python damru_omni_builder.py
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

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
OUT_REPO        = os.environ.get("OMNI_OUT_REPO", "Damaru-ai/damru-omni")
SHARD_ROWS      = int(os.environ.get("OMNI_SHARD_ROWS", "50000"))
MAX_PER_SRC     = int(os.environ.get("OMNI_MAX_PER_SRC", "0"))
TIME_BUDGET_MIN = int(os.environ.get("OMNI_TIME_BUDGET_MIN", "300"))
COMMIT_EVERY    = int(os.environ.get("OMNI_COMMIT_EVERY_SEC", "600"))
WORKDIR         = os.environ.get("OMNI_WORKDIR", "./omni_work")
MIRROR_STATE    = os.environ.get("OMNI_MIRROR_STATE", "1") == "1"
HF_TOKEN        = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

DEFAULT_SOURCES = [
    "Damaru-ai/damru-dpo:pref:stream",
    "Damaru-ai/damru-gurukul:sft:stream",
    "Damaru-ai/damru-train:sft:stream",
    "Damaru-ai/damru-reasoning-traces:sft:stream",
    "Damaru-ai/damru-knowledge:knowledge:files",   # 10.8M -> file-level resume, LAST
]
SOURCES = [s.strip() for s in os.environ.get(
    "OMNI_SOURCES", ",".join(DEFAULT_SOURCES)).split(",") if s.strip()]

SPLITS = ("sft", "pref", "rl", "knowledge")
MAX_RETRIES = 6

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
log = logging.getLogger("omni")


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


# ----------------------------------------------------------------------------
# State + seen
# ----------------------------------------------------------------------------
def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            log_error("load_state", exc)
    return {"version": 3, "sources": {},
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


# ----------------------------------------------------------------------------
# HF client + THROTTLED folder commit (the 429 fix)
# ----------------------------------------------------------------------------
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
    """ONE upload_folder commit, throttled to <= 1 per COMMIT_EVERY seconds.
    Unchanged files are skipped by the Hub, so this is cheap to call often."""
    if not force and (time.time() - _last_commit[0] < COMMIT_EVERY):
        return

    def _up():
        hf_api().upload_folder(
            folder_path=WORKDIR, repo_id=OUT_REPO, repo_type="dataset",
            allow_patterns=["data/**", "_ckpt/**", "README.md"],
            commit_message=f"omni update {time.strftime('%Y-%m-%d %H:%M:%S')}",
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


# ----------------------------------------------------------------------------
# Normalization (wider field matching + schema logging)
# ----------------------------------------------------------------------------
P_KEYS = ("prompt", "instruction", "question", "query", "input", "task",
          "text_input", "user", "human", "context")
R_KEYS = ("response", "output", "answer", "completion", "solution", "chosen",
          "target", "label", "text_output", "assistant", "gpt", "code")
T_KEYS = ("text", "content", "document", "page_content", "body", "chunk",
          "data", "raw", "passage")
M_KEYS = ("messages", "conversations", "conversation", "chat", "dialogue", "turns")


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


def normalize(row: Dict[str, Any], hint: str) -> Optional[Dict[str, Any]]:
    chosen = _first(row, ("chosen", "chosen_response", "preferred"))
    rejected = _first(row, ("rejected", "rejected_response", "dispreferred"))
    if chosen and rejected:
        prompt = _first(row, P_KEYS) or ""
        return {"split": "pref", "prompt": prompt, "chosen": chosen, "rejected": rejected}

    if hint == "rl":
        problem = _first(row, ("problem", "task", "question"))
        if problem:
            answer = _first(row, ("answer", "final_answer", "solution", "label"))
            tests = row.get("tests") or row.get("test_cases") or row.get("unit_tests")
            return {"split": "rl", "problem": problem, "answer": answer, "tests": tests}

    pair = _from_messages(row)
    if pair is None:
        p = _first(row, P_KEYS)
        r = _first(row, R_KEYS)
        if p and r:
            pair = (p, r)
    if pair:
        return {"split": "sft", "prompt": pair[0], "response": pair[1]}

    text = _first(row, T_KEYS)
    if not text and hint == "knowledge":
        text = _longest_str(row)
    if text and len(text) >= 40:
        return {"split": "knowledge", "text": text}
    return None


# ----------------------------------------------------------------------------
# Shard writer (local-first; commits are throttled)
# ----------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, state: Dict[str, Any], seen: set):
        self.state = state
        self.seen = seen
        self.buffers: Dict[str, List[str]] = {s: [] for s in SPLITS}

    def add(self, rec: Dict[str, Any], source: str, hint: str) -> bool:
        split = rec.pop("split")
        rid = sha_id(split, json.dumps(rec, ensure_ascii=False, sort_keys=True))
        if rid in self.seen:
            return False
        self.seen.add(rid)
        out = {"id": rid, "source": source, "provenance": source, "license": "see-source", **rec}
        low = source.lower()
        if any(t in low for t in ("reasoning-traces", "knowledge", "codex", "bulk")):
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


# ----------------------------------------------------------------------------
# Source processing
# ----------------------------------------------------------------------------
def process_source_stream(repo, hint, writer, state, spec, deadline):
    src_state = state["sources"].get(spec, {"status": "pending", "rows_seen": 0, "rows_kept": 0})
    already = src_state.get("rows_seen", 0)
    log.info("[stream] %s (hint=%s) resume_from=%d", spec, hint, already)
    from datasets import load_dataset

    def _open():
        return load_dataset(repo, split="train", streaming=True, token=HF_TOKEN)
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
            if rec and writer.add(rec, repo, hint):
                kept_n += 1
        except Exception as exc:
            log_error(f"row:{spec}:{seen_n}", exc)
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
                log.info("[progress] %s seen=%d kept=%d", spec, seen_n, kept_n)
        if MAX_PER_SRC and (seen_n - already) >= MAX_PER_SRC:
            log.info("[cap] %s hit MAX_PER_SRC=%d", spec, MAX_PER_SRC)
            break

    writer.flush_all()
    src_state.update(status="done", rows_seen=seen_n, rows_kept=kept_n)
    state["sources"][spec] = src_state
    save_state(state)
    commit_all(force=True)
    log.info("[done] %s seen=%d kept=%d", spec, seen_n, kept_n)
    return "done"


def process_source_files(repo, hint, writer, state, spec, deadline):
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
    parquets = sorted(f for f in files if f.endswith(".parquet"))
    log.info("[files] %s: %d parquet files (%d done)", spec, len(parquets), len(done))
    logged_schema = False

    for fn in parquets:
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
                for row in batch.to_pylist():
                    if not logged_schema and isinstance(row, dict):
                        log.info("[schema] %s columns: %s", spec, list(row.keys()))
                        logged_schema = True
                    try:
                        rec = normalize(row, hint)
                        if rec and writer.add(rec, repo, hint):
                            src_state["rows_kept"] = src_state.get("rows_kept", 0) + 1
                    except Exception as exc:
                        log_error(f"row:{spec}:{fn}", exc)
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
        log.info("[file-done] %s %s kept=%d", spec, fn, src_state.get("rows_kept", 0))

    writer.flush_all()
    src_state.update(status="done", files_done=sorted(done))
    state["sources"][spec] = src_state
    save_state(state)
    commit_all(force=True)
    log.info("[done] %s files=%d kept=%d", spec, len(done), src_state.get("rows_kept", 0))
    return "done"


def process_source(spec, writer, state, deadline):
    parts = spec.split(":")
    repo = parts[0]
    hint = parts[1] if len(parts) > 1 else "sft"
    mode = parts[2] if len(parts) > 2 else "stream"
    src_state = state["sources"].get(spec)
    if src_state and src_state.get("status") == "done":
        log.info("[skip] %s already done", spec)
        return "done"
    if mode == "files":
        return process_source_files(repo, hint, writer, state, spec, deadline)
    return process_source_stream(repo, hint, writer, state, spec, deadline)


def write_readme(state: Dict[str, Any]) -> None:
    rows = state.get("rows_out", {})
    md = [
        "---", "license: other", "tags: [damru, unified, sft, dpo, rag]", "---", "",
        "# Damru Omni (unified corpus)", "",
        "Auto-built by `damru_omni_builder.py` (Step 1).", "", "## Splits", "",
        f"- **sft**: {rows.get('sft', 0)} -- {{prompt, response}}",
        f"- **pref**: {rows.get('pref', 0)} -- {{prompt, chosen, rejected}}",
        f"- **rl**: {rows.get('rl', 0)} -- {{problem, answer, tests}}",
        f"- **knowledge**: {rows.get('knowledge', 0)} -- {{text}} (RAG only)",
        "", f"**Total kept:** {state.get('total_out', 0)}", "",
        "Rows carry id (sha1), source, provenance, needs_audit.",
    ]
    atomic_write_text(README_PATH, "\n".join(md) + "\n")


def main() -> int:
    if not HF_TOKEN:
        log.error("HF_TOKEN env var is required.")
        return 2
    log.info("=== Damru Omni Builder v3 | out=%s | sources=%d | budget=%dmin | commit_every=%ds ===",
             OUT_REPO, len(SOURCES), TIME_BUDGET_MIN, COMMIT_EVERY)
    ensure_repo()
    restore_from_hf()
    state = load_state()
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
