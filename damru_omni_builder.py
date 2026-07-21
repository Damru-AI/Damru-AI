#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
damru_omni_builder.py  --  STEP 1 of the Damru 14B Master Plan
===============================================================
Merge ALL of your Hugging Face datasets into ONE unified dataset
`Damaru-ai/damru-omni` with clean splits + sha1 dedup + provenance tags.

Splits produced:
  - sft        : {prompt, response}          (instruction tuning)
  - pref       : {prompt, chosen, rejected}   (DPO / preference)
  - rl         : {problem, answer, tests}     (verifiable RL, if present)
  - knowledge  : {text}                        (for RAG, NOT weight-training)

----------------------------------------------------------------
WHY THIS IS "SOLID" (survives crashes / ephemeral runners):
  1. Checkpoint + resume via state.json  (also mirrored to the OUTPUT HF repo,
     so GitHub Actions / Kaggle runners resume even after a full restart).
  2. Atomic writes (temp file -> os.replace) so a killed process never leaves
     a half-written shard.
  3. Retry + exponential backoff on EVERY network call.
  4. Idempotent: sha1 dedup + completed-source skip => re-run never doubles data.
  5. Per-row AND per-source try/except => one bad row/source never crashes the run;
     everything problematic is logged to errors.log and processing continues.
  6. Incremental push to HF after every shard (progress is never lost).
----------------------------------------------------------------

RUN:
  pip install "datasets>=2.19" "huggingface_hub>=0.24"
  export HF_TOKEN=hf_xxx           # write token for the Damaru-ai org
  python damru_omni_builder.py

ENV KNOBS (all optional except HF_TOKEN):
  HF_TOKEN          (required) HF write token
  OMNI_OUT_REPO     default: Damaru-ai/damru-omni
  OMNI_SOURCES      comma list "repo[:split][:config]"; default = your 5 core sets
  OMNI_SHARD_ROWS   rows per shard/push          (default 50000)
  OMNI_MAX_PER_SRC  cap rows per source, 0=all   (default 0; set small to test)
  OMNI_WORKDIR      local scratch dir            (default ./omni_work)
  OMNI_MIRROR_STATE 1=mirror state to HF repo    (default 1; set 0 for local-only)
"""

import gzip
import hashlib
import json
import logging
import os
import random
import sys
import time
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
OUT_REPO      = os.environ.get("OMNI_OUT_REPO", "Damaru-ai/damru-omni")
SHARD_ROWS    = int(os.environ.get("OMNI_SHARD_ROWS", "50000"))
MAX_PER_SRC   = int(os.environ.get("OMNI_MAX_PER_SRC", "0"))
WORKDIR       = os.environ.get("OMNI_WORKDIR", "./omni_work")
MIRROR_STATE  = os.environ.get("OMNI_MIRROR_STATE", "1") == "1"
HF_TOKEN      = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# "repo:split" -- split is a HINT; the normalizer still auto-detects per row.
# knowledge -> RAG, sft -> instruction, pref -> DPO.
DEFAULT_SOURCES = [
    "Damaru-ai/damru-knowledge:knowledge",
    "Damaru-ai/damru-train:sft",
    "Damaru-ai/damru-dpo:pref",
    "Damaru-ai/damru-gurukul:sft",
    "Damaru-ai/damru-reasoning-traces:sft",
]
SOURCES = [s.strip() for s in os.environ.get(
    "OMNI_SOURCES", ",".join(DEFAULT_SOURCES)).split(",") if s.strip()]

SPLITS = ("sft", "pref", "rl", "knowledge")
STATE_PATH   = os.path.join(WORKDIR, "state.json")
SEEN_PATH    = os.path.join(WORKDIR, "seen.txt.gz")
ERR_PATH     = os.path.join(WORKDIR, "errors.log")
MAX_RETRIES  = 6

os.makedirs(WORKDIR, exist_ok=True)
for _s in SPLITS:
    os.makedirs(os.path.join(WORKDIR, _s), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(WORKDIR, "run.log"))],
)
log = logging.getLogger("omni")


def log_error(where: str, exc: Exception) -> None:
    """Never crash: record the problem and keep going."""
    try:
        with open(ERR_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {where}: "
                    f"{type(exc).__name__}: {exc}\n")
            f.write(traceback.format_exc() + "\n")
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Retry helper (exponential backoff + jitter)
# ----------------------------------------------------------------------------
def retry(fn, *args, what: str = "op", **kwargs):
    delay = 3.0
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberate broad catch
            last = exc
            log_error(f"retry:{what}:attempt{attempt}", exc)
            log.warning("[retry] %s failed (attempt %d/%d): %s",
                        what, attempt, MAX_RETRIES, exc)
            if attempt == MAX_RETRIES:
                break
            time.sleep(delay + random.uniform(0, 2.0))
            delay = min(delay * 2, 120.0)
    raise last  # type: ignore[misc]


# ----------------------------------------------------------------------------
# Atomic IO
# ----------------------------------------------------------------------------
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
# State + seen-set (persisted, so we resume after any crash)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# HF client (lazy import so --help / syntax works without deps)
# ----------------------------------------------------------------------------
_api = None


def hf_api():
    global _api
    if _api is None:
        from huggingface_hub import HfApi
        _api = HfApi(token=HF_TOKEN)
    return _api


def ensure_repo() -> None:
    def _mk():
        hf_api().create_repo(OUT_REPO, repo_type="dataset",
                             exist_ok=True, private=True)
    retry(_mk, what="create_repo")


def hf_upload(local_path: str, path_in_repo: str) -> None:
    def _up():
        hf_api().upload_file(path_or_fileobj=local_path,
                             path_in_repo=path_in_repo,
                             repo_id=OUT_REPO, repo_type="dataset")
    retry(_up, what=f"upload:{path_in_repo}")


def hf_try_download(path_in_repo: str, dest: str) -> bool:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    try:
        p = retry(hf_hub_download, repo_id=OUT_REPO, repo_type="dataset",
                  filename=path_in_repo, token=HF_TOKEN, what=f"dl:{path_in_repo}")
        with open(p, "rb") as src, open(dest, "wb") as out:
            out.write(src.read())
        return True
    except EntryNotFoundError:
        return False
    except Exception as exc:
        log_error(f"download:{path_in_repo}", exc)
        return False


def restore_from_hf() -> None:
    """Pull state.json + seen set from the repo so ephemeral runners resume."""
    if not MIRROR_STATE:
        return
    if hf_try_download("_ckpt/state.json", STATE_PATH):
        log.info("[resume] restored state.json from HF")
    hf_try_download("_ckpt/seen.txt.gz", SEEN_PATH)


def mirror_to_hf() -> None:
    if not MIRROR_STATE:
        return
    try:
        if os.path.exists(STATE_PATH):
            hf_upload(STATE_PATH, "_ckpt/state.json")
        if os.path.exists(SEEN_PATH):
            hf_upload(SEEN_PATH, "_ckpt/seen.txt.gz")
    except Exception as exc:
        log_error("mirror_to_hf", exc)


# ----------------------------------------------------------------------------
# Normalization: turn ANY row into a unified record (schema-agnostic)
# ----------------------------------------------------------------------------
def _first(row: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        if k in row and row[k] not in (None, "", []):
            v = row[k]
            return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    return None


def _from_messages(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    msgs = row.get("messages") or row.get("conversations") or row.get("conversation")
    if not isinstance(msgs, list) or not msgs:
        return None
    prompt, response = [], None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = (m.get("role") or m.get("from") or "").lower()
        content = m.get("content") or m.get("value") or ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False)
        if role in ("assistant", "gpt", "model", "bot"):
            response = content
        elif role in ("user", "human", "system", "prompter"):
            prompt.append(content)
    if response and prompt:
        return ("\n\n".join(prompt).strip(), response.strip())
    return None


def normalize(row: Dict[str, Any], hint: str) -> Optional[Dict[str, Any]]:
    """Return a unified record dict with a 'split' key, or None to skip."""
    # 1) preference / DPO
    chosen = _first(row, ("chosen", "chosen_response", "preferred"))
    rejected = _first(row, ("rejected", "rejected_response", "dispreferred"))
    if chosen and rejected:
        prompt = _first(row, ("prompt", "question", "instruction", "query", "input")) or ""
        return {"split": "pref", "prompt": prompt, "chosen": chosen, "rejected": rejected}

    # 2) verifiable RL (problem + answer/tests)
    problem = _first(row, ("problem", "task", "question")) if hint == "rl" else None
    if problem:
        answer = _first(row, ("answer", "final_answer", "solution", "label"))
        tests = row.get("tests") or row.get("test_cases") or row.get("unit_tests")
        return {"split": "rl", "problem": problem, "answer": answer, "tests": tests}

    # 3) instruction / SFT (explicit pair or chat messages)
    pair = _from_messages(row)
    if pair is None:
        p = _first(row, ("prompt", "instruction", "question", "input", "query"))
        r = _first(row, ("response", "output", "answer", "completion", "solution", "chosen"))
        if p and r:
            pair = (p, r)
    if pair:
        return {"split": "sft", "prompt": pair[0], "response": pair[1]}

    # 4) knowledge (free text) -> RAG
    text = _first(row, ("text", "content", "document", "page_content", "body", "chunk"))
    if text and len(text) >= 40:
        return {"split": "knowledge", "text": text}

    return None


# ----------------------------------------------------------------------------
# Shard writer
# ----------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, state: Dict[str, Any], seen: set):
        self.state = state
        self.seen = seen
        self.buffers: Dict[str, List[str]] = {s: [] for s in SPLITS}

    def add(self, rec: Dict[str, Any], source: str, hint: str) -> bool:
        split = rec.pop("split")
        key_material = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        rid = sha_id(split, key_material)
        if rid in self.seen:
            return False
        self.seen.add(rid)
        out = {"id": rid, "source": source, "provenance": source,
               "license": "see-source", **rec}
        # provenance audit flag for harvested/uncertain corpora
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
        local = os.path.join(WORKDIR, split, fname)
        atomic_write_text(local, "\n".join(buf) + "\n")
        hf_upload(local, f"data/{split}/{fname}")
        self.state["shard_index"][split] = idx + 1
        self.state["rows_out"][split] = self.state["rows_out"].get(split, 0) + len(buf)
        self.state["total_out"] = self.state.get("total_out", 0) + len(buf)
        self.buffers[split] = []
        save_state(self.state)
        save_seen(self.seen)
        mirror_to_hf()
        log.info("[flush] %s shard %d (+%d rows) | total=%d",
                 split, idx, len(buf), self.state["total_out"])

    def flush_all(self) -> None:
        for s in SPLITS:
            self.flush(s)


# ----------------------------------------------------------------------------
# Per-source processing (streaming, resumable)
# ----------------------------------------------------------------------------
def process_source(spec: str, writer: ShardWriter, state: Dict[str, Any]) -> None:
    parts = spec.split(":")
    repo = parts[0]
    hint = parts[1] if len(parts) > 1 else "sft"
    config = parts[2] if len(parts) > 2 else None

    src_state = state["sources"].get(spec, {"status": "pending", "rows_seen": 0,
                                            "rows_kept": 0})
    if src_state.get("status") == "done":
        log.info("[skip] %s already done (%d kept)", spec, src_state.get("rows_kept", 0))
        return
    already = src_state.get("rows_seen", 0)
    log.info("[source] %s (hint=%s) resume_from=%d", spec, hint, already)

    from datasets import load_dataset

    def _open():
        return load_dataset(repo, config, split="train", streaming=True,
                            token=HF_TOKEN)
    try:
        ds = retry(_open, what=f"load:{repo}")
    except Exception as exc:
        log_error(f"open_source:{spec}", exc)
        log.error("[source] cannot open %s -- skipping. See errors.log", spec)
        src_state["status"] = "error"
        state["sources"][spec] = src_state
        save_state(state)
        return

    seen_n = 0
    kept_n = src_state.get("rows_kept", 0)
    iterator = iter(ds)
    while True:
        try:
            row = next(iterator)
        except StopIteration:
            break
        except Exception as exc:  # transient stream hiccup -> log + continue
            log_error(f"iter:{spec}", exc)
            continue
        seen_n += 1
        # resume: fast-forward past rows already processed in a prior run
        if seen_n <= already:
            continue
        try:
            rec = normalize(row, hint)
            if rec and writer.add(rec, repo, hint):
                kept_n += 1
        except Exception as exc:
            log_error(f"row:{spec}:{seen_n}", exc)
        # periodic progress checkpoint (cheap, no upload)
        if seen_n % 5000 == 0:
            src_state.update(status="partial", rows_seen=seen_n, rows_kept=kept_n)
            state["sources"][spec] = src_state
            save_state(state)
            log.info("[progress] %s seen=%d kept=%d", spec, seen_n, kept_n)
        if MAX_PER_SRC and (seen_n - already) >= MAX_PER_SRC:
            log.info("[cap] %s hit MAX_PER_SRC=%d", spec, MAX_PER_SRC)
            break

    writer.flush_all()
    src_state.update(status="done", rows_seen=seen_n, rows_kept=kept_n)
    state["sources"][spec] = src_state
    save_state(state)
    mirror_to_hf()
    log.info("[done] %s seen=%d kept=%d", spec, seen_n, kept_n)


def write_readme(state: Dict[str, Any]) -> None:
    rows = state.get("rows_out", {})
    md = [
        "---", "license: other", "tags: [damru, unified, sft, dpo, rag]", "---", "",
        "# Damru Omni (unified corpus)", "",
        "Auto-built by `damru_omni_builder.py` (Step 1 of the Damru 14B plan).",
        "", "## Splits", "",
        f"- **sft**: {rows.get('sft', 0)} rows -- {{prompt, response}}",
        f"- **pref**: {rows.get('pref', 0)} rows -- {{prompt, chosen, rejected}}",
        f"- **rl**: {rows.get('rl', 0)} rows -- {{problem, answer, tests}}",
        f"- **knowledge**: {rows.get('knowledge', 0)} rows -- {{text}} (RAG only)",
        "", f"**Total kept:** {state.get('total_out', 0)} rows", "",
        "Each row carries `id` (sha1), `source`, `provenance`, and `needs_audit`.",
        "`knowledge` is for RAG, NOT weight-training. Curate `sft` before SFT.",
    ]
    path = os.path.join(WORKDIR, "README.md")
    atomic_write_text(path, "\n".join(md) + "\n")
    try:
        hf_upload(path, "README.md")
    except Exception as exc:
        log_error("write_readme", exc)


def main() -> int:
    if not HF_TOKEN:
        log.error("HF_TOKEN env var is required (write token for Damaru-ai).")
        return 2
    log.info("=== Damru Omni Builder | out=%s | sources=%d ===",
             OUT_REPO, len(SOURCES))
    ensure_repo()
    restore_from_hf()
    state = load_state()
    seen = load_seen()
    writer = ShardWriter(state, seen)

    for spec in SOURCES:
        try:
            process_source(spec, writer, state)
        except Exception as exc:  # a whole source blew up -> log, keep others
            log_error(f"process_source:{spec}", exc)
            log.error("[source] %s failed hard -- continuing. See errors.log", spec)

    writer.flush_all()
    write_readme(state)
    mirror_to_hf()
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
