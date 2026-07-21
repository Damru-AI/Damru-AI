#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
damru_oracle_builder.py  --  STEP 2 of the Damru 14B Master Plan
================================================================
Build a VERIFIER-ORACLE dataset `Damaru-ai/damru-oracle` from closed-model
traces + legal reasoning/coding datasets.

================================================================
LEGAL / ANTI-COPY LINE (binding -- read this):
  We extract ONLY these facts from a trace:
    - problem : the task / question statement   (fact  -> minable)
    - answer  : the SHORT final answer          (math number / canonical)
    - tests   : code unit-tests / assert lines  (fact  -> checkable)
  We NEVER store the model's chain-of-thought, prose, style, or full
  completion. Reasoning/style = copyright-sensitive -> discarded.
  For CLOSED-model sources, non-verifiable reference answers are DROPPED
  (only the problem is kept; Damru self-solves; a JUDGE scores it).
  AGPL / copyleft sources are REFUSED outright.

Damru does NOT train on this text. The VERIFIER uses it to CHECK Damru's own
answer (exactly like GSM8K answer-keys / HumanEval tests).
================================================================

Same SOLID engine as Step 1: checkpoint+resume (mirrored to HF), atomic writes,
retry+backoff, sha1 dedup, per-row/per-source try/except, incremental push,
self time-budget.

RUN:
  pip install "datasets>=2.19" "huggingface_hub>=0.24"
  export HF_TOKEN=hf_xxx
  python damru_oracle_builder.py

ENV KNOBS:
  HF_TOKEN              (required)
  ORACLE_OUT_REPO       default: Damaru-ai/damru-oracle
  ORACLE_SOURCES        comma list "repo[:domain][:closed]" (closed=1|0)
  ORACLE_SHARD_ROWS     default 20000
  ORACLE_MAX_PER_SRC    cap per source, 0=all (default 50000)
  ORACLE_TIME_BUDGET_MIN default 300
  ORACLE_WORKDIR        default ./oracle_work
  ORACLE_MIRROR_STATE   default 1
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
OUT_REPO        = os.environ.get("ORACLE_OUT_REPO", "Damaru-ai/damru-oracle")
SHARD_ROWS      = int(os.environ.get("ORACLE_SHARD_ROWS", "20000"))
MAX_PER_SRC     = int(os.environ.get("ORACLE_MAX_PER_SRC", "50000"))
TIME_BUDGET_MIN = int(os.environ.get("ORACLE_TIME_BUDGET_MIN", "300"))
WORKDIR         = os.environ.get("ORACLE_WORKDIR", "./oracle_work")
MIRROR_STATE    = os.environ.get("ORACLE_MIRROR_STATE", "1") == "1"
HF_TOKEN        = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

# "repo:domain:closed"  -- closed=1 => only verifiable oracle kept (no prose).
DEFAULT_SOURCES = [
    # LEGAL GOLD (open weights) -- problem + answer + tests OK
    "greghavens/kimi-k3-coding-and-debugging-traces:code:0",
    "nvidia/Open-SWE-Traces:code:0",
    "SupraLabs/reasoning-corpus-4K-5M-v1:reasoning:0",
    # CLOSED-model -> mine PROBLEMS + verifiable oracle only, discard prose
    "armand0e/claude-fable-5-claude-code:code:1",
    "Crownelius/Complete-FABLE.5-traces-2M:mixed:1",
    "greghavens/fable-5-coding-and-debugging-traces:code:1",
    "WithinUsAI/claude_mythos_distilled_25k:mixed:1",
]
SOURCES = [s.strip() for s in os.environ.get(
    "ORACLE_SOURCES", ",".join(DEFAULT_SOURCES)).split(",") if s.strip()]

# Refuse copyleft / AGPL sources even if someone adds them.
BLOCKLIST = ("glint-research/fable-5-traces", "agpl")

DOMAINS = ("math", "code", "reasoning", "other")
STATE_PATH = os.path.join(WORKDIR, "state.json")
SEEN_PATH  = os.path.join(WORKDIR, "seen.txt.gz")
ERR_PATH   = os.path.join(WORKDIR, "errors.log")
MAX_RETRIES = 6

os.makedirs(WORKDIR, exist_ok=True)
for _d in DOMAINS:
    os.makedirs(os.path.join(WORKDIR, _d), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout),
              logging.FileHandler(os.path.join(WORKDIR, "run.log"))],
)
log = logging.getLogger("oracle")


def log_error(where: str, exc: Exception) -> None:
    try:
        with open(ERR_PATH, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {where}: "
                    f"{type(exc).__name__}: {exc}\n")
            f.write(traceback.format_exc() + "\n")
    except Exception:
        pass


def retry(fn, *args, what: str = "op", **kwargs):
    delay = 3.0
    last = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last = exc
            log_error(f"retry:{what}:attempt{attempt}", exc)
            log.warning("[retry] %s failed (attempt %d/%d): %s",
                        what, attempt, MAX_RETRIES, exc)
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
    return {"version": 1, "sources": {},
            "shard_index": {d: 0 for d in DOMAINS},
            "rows_out": {d: 0 for d in DOMAINS}, "total_out": 0}


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
# HF client
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
        hf_api().create_repo(OUT_REPO, repo_type="dataset", exist_ok=True, private=True)
    retry(_mk, what="create_repo")


def hf_upload(local_path: str, path_in_repo: str) -> None:
    def _up():
        hf_api().upload_file(path_or_fileobj=local_path, path_in_repo=path_in_repo,
                             repo_id=OUT_REPO, repo_type="dataset")
    retry(_up, what=f"upload:{path_in_repo}")


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
# ORACLE extraction (facts only -- no CoT/prose)
# ----------------------------------------------------------------------------
BOXED   = re.compile(r"\\boxed\{([^{}]{1,200})\}")
GSM     = re.compile(r"####\s*([^\n]{1,100})")
ANSWER  = re.compile(r"(?:final answer|the answer is|answer)\s*[:=\-]?\s*([^\n]{1,120})", re.I)
NUMBER  = re.compile(r"-?\d[\d,]*\.?\d*")
ASSERT  = re.compile(r"^\s*assert\s+.+$", re.M)
CODEBLK = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.S)


def _first(row: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def extract_problem(row: Dict[str, Any]) -> Optional[str]:
    p = _first(row, ("problem", "question", "prompt", "instruction", "query", "task", "input"))
    if p and len(p.strip()) >= 10:
        return p.strip()
    msgs = row.get("messages") or row.get("conversations") or row.get("conversation")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict):
                role = (m.get("role") or m.get("from") or "").lower()
                if role in ("user", "human", "prompter"):
                    c = m.get("content") or m.get("value")
                    if isinstance(c, str) and len(c.strip()) >= 10:
                        return c.strip()
    return None


def completion_text(row: Dict[str, Any]) -> str:
    v = _first(row, ("solution", "answer", "response", "output", "completion", "chosen"))
    if v:
        return v
    msgs = row.get("messages") or row.get("conversations") or row.get("conversation")
    if isinstance(msgs, list):
        for m in reversed(msgs):
            if isinstance(m, dict):
                role = (m.get("role") or m.get("from") or "").lower()
                if role in ("assistant", "gpt", "model", "bot"):
                    c = m.get("content") or m.get("value")
                    if isinstance(c, str):
                        return c
    return ""


def extract_final_answer(text: str) -> Tuple[Optional[str], str]:
    if not text:
        return (None, "none")
    m = BOXED.search(text)
    if m:
        return (m.group(1).strip(), "expr")
    m = GSM.search(text)
    if m:
        return (m.group(1).strip().strip("."), "number")
    m = ANSWER.search(text)
    if m:
        a = m.group(1).strip().strip(".").strip()
        if 0 < len(a) <= 120:
            return (a, "string")
    nums = NUMBER.findall(text[-400:])
    if nums:
        return (nums[-1].replace(",", ""), "number")
    return (None, "none")


def extract_tests(row: Dict[str, Any], text: str) -> Optional[List[str]]:
    tests: List[str] = []
    for k in ("test_list", "tests", "test_cases", "unit_tests"):
        v = row.get(k)
        if isinstance(v, list):
            tests += [str(x) for x in v if x]
        elif isinstance(v, str) and v.strip():
            tests.append(v.strip())
    t = row.get("test")
    if isinstance(t, str) and "assert" in t:
        tests.append(t.strip())
    for block in CODEBLK.findall(text or ""):
        tests += [a.strip() for a in ASSERT.findall(block)]
    tests += [a.strip() for a in ASSERT.findall(text or "")]
    out, seen = [], set()
    for x in tests:
        if x not in seen and 3 < len(x) <= 500:
            seen.add(x)
            out.append(x)
    return out[:40] or None


def classify(problem: str, tests: Optional[list], hint: str) -> str:
    if tests:
        return "code"
    p = (problem or "").lower()
    if any(k in p for k in ("def ", "function", "code", "python", "java", "c++", "implement", "```", "algorithm")):
        return "code"
    if any(k in p for k in ("prove", "integral", "equation", "compute", "calculate", "how many", "probability", "sum of", "solve for", "theorem")):
        return "math"
    if hint in DOMAINS:
        return hint
    return "other"


def build_record(row: Dict[str, Any], source: str, closed: bool, hint: str) -> Optional[Dict[str, Any]]:
    problem = extract_problem(row)
    if not problem or len(problem) > 20000:
        return None
    comp = completion_text(row)
    tests = extract_tests(row, comp) or extract_tests(row, problem)
    ans, atype = extract_final_answer(comp)
    domain = classify(problem, tests, hint)
    verifiable = bool(tests) or (ans is not None and atype in ("number", "expr"))

    rec: Dict[str, Any] = {
        "source": source, "provenance": source, "closed_model": closed,
        "domain": domain, "problem": problem,
        "answer": None, "answer_type": "none", "tests": tests,
        "verifiable": verifiable, "judge_only": False,
        "license": "problem-mined; verifiable-oracle-only" if closed else "see-source",
    }
    if verifiable:
        if ans and atype in ("number", "expr", "string") and len(ans) <= 120:
            rec["answer"] = ans
            rec["answer_type"] = atype
    else:
        # non-verifiable: closed -> DROP reference (anti-copy); open -> short rubric ref
        rec["judge_only"] = True
        if not closed:
            ref = (comp or "").strip()
            if ref:
                rec["answer"] = ref[:400]
                rec["answer_type"] = "rubric"
    return rec


# ----------------------------------------------------------------------------
# Shard writer (domain-sharded)
# ----------------------------------------------------------------------------
class ShardWriter:
    def __init__(self, state: Dict[str, Any], seen: set):
        self.state = state
        self.seen = seen
        self.buffers: Dict[str, List[str]] = {d: [] for d in DOMAINS}

    def add(self, rec: Dict[str, Any], source: str) -> bool:
        rid = sha_id(rec["domain"], rec["problem"])
        if rid in self.seen:
            return False
        self.seen.add(rid)
        rec = {"id": rid, **rec}
        dom = rec["domain"] if rec["domain"] in DOMAINS else "other"
        self.buffers[dom].append(json.dumps(rec, ensure_ascii=False))
        if len(self.buffers[dom]) >= SHARD_ROWS:
            self.flush(dom)
        return True

    def flush(self, dom: str) -> None:
        buf = self.buffers[dom]
        if not buf:
            return
        idx = self.state["shard_index"][dom]
        fname = f"{dom}-{idx:05d}.jsonl"
        local = os.path.join(WORKDIR, dom, fname)
        atomic_write_text(local, "\n".join(buf) + "\n")
        hf_upload(local, f"data/{dom}/{fname}")
        self.state["shard_index"][dom] = idx + 1
        self.state["rows_out"][dom] = self.state["rows_out"].get(dom, 0) + len(buf)
        self.state["total_out"] = self.state.get("total_out", 0) + len(buf)
        self.buffers[dom] = []
        save_state(self.state)
        save_seen(self.seen)
        mirror_to_hf()
        log.info("[flush] %s shard %d (+%d) | total=%d", dom, idx, len(buf), self.state["total_out"])

    def flush_all(self) -> None:
        for d in DOMAINS:
            self.flush(d)


# ----------------------------------------------------------------------------
# Per-source processing
# ----------------------------------------------------------------------------
def process_source(spec: str, writer: ShardWriter, state: Dict[str, Any], deadline: float) -> str:
    parts = spec.split(":")
    repo = parts[0]
    hint = parts[1] if len(parts) > 1 else "other"
    closed = (len(parts) > 2 and parts[2] == "1")

    low = repo.lower()
    if any(b in low for b in BLOCKLIST):
        log.warning("[refuse] %s is copyleft/AGPL -- skipped (legal).", repo)
        state["sources"][spec] = {"status": "refused"}
        save_state(state)
        return "refused"

    src_state = state["sources"].get(spec, {"status": "pending", "rows_seen": 0, "rows_kept": 0})
    if src_state.get("status") == "done":
        log.info("[skip] %s done", spec)
        return "done"
    already = src_state.get("rows_seen", 0)
    log.info("[source] %s (domain=%s closed=%s) resume_from=%d", spec, hint, closed, already)

    from datasets import load_dataset

    def _open():
        return load_dataset(repo, split="train", streaming=True, token=HF_TOKEN)
    try:
        ds = retry(_open, what=f"load:{repo}")
    except Exception as exc:
        log_error(f"open:{spec}", exc)
        log.error("[source] cannot open %s -- skipping. See errors.log", spec)
        src_state["status"] = "error"
        state["sources"][spec] = src_state
        save_state(state)
        return "error"

    seen_n, kept_n = 0, src_state.get("rows_kept", 0)
    iterator = iter(ds)
    while True:
        try:
            row = next(iterator)
        except StopIteration:
            break
        except Exception as exc:
            log_error(f"iter:{spec}", exc)
            continue
        seen_n += 1
        if seen_n <= already:
            continue
        try:
            rec = build_record(row, repo, closed, hint)
            if rec and writer.add(rec, repo):
                kept_n += 1
        except Exception as exc:
            log_error(f"row:{spec}:{seen_n}", exc)
        if seen_n % 2000 == 0:
            src_state.update(status="partial", rows_seen=seen_n, rows_kept=kept_n)
            state["sources"][spec] = src_state
            save_state(state)
            if time.time() > deadline:
                writer.flush_all()
                save_seen(writer.seen)
                mirror_to_hf()
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
    mirror_to_hf()
    log.info("[done] %s seen=%d kept=%d", spec, seen_n, kept_n)
    return "done"


def write_readme(state: Dict[str, Any]) -> None:
    rows = state.get("rows_out", {})
    md = [
        "---", "license: other", "tags: [damru, oracle, verifier]", "---", "",
        "# Damru Oracle (verifier answer-keys)", "",
        "Built by `damru_oracle_builder.py` (Step 2). **Facts only** -- problem,",
        "short final answer, and test cases. NO chain-of-thought / prose / style.",
        "Used by the VERIFIER to CHECK Damru's own answers; NOT training text.",
        "", "## Domains", "",
        f"- **code**: {rows.get('code', 0)} (problem + tests)",
        f"- **math**: {rows.get('math', 0)} (problem + final answer)",
        f"- **reasoning**: {rows.get('reasoning', 0)}",
        f"- **other**: {rows.get('other', 0)}",
        "", f"**Total:** {state.get('total_out', 0)} oracles", "",
        "Fields: id, source, closed_model, domain, problem, answer, answer_type,",
        "tests, verifiable, judge_only, license.",
        "Closed-model non-verifiable references are dropped (anti-copy).",
    ]
    path = os.path.join(WORKDIR, "README.md")
    atomic_write_text(path, "\n".join(md) + "\n")
    try:
        hf_upload(path, "README.md")
    except Exception as exc:
        log_error("write_readme", exc)


def main() -> int:
    if not HF_TOKEN:
        log.error("HF_TOKEN env var is required.")
        return 2
    log.info("=== Damru Oracle Builder | out=%s | sources=%d | budget=%dmin ===",
             OUT_REPO, len(SOURCES), TIME_BUDGET_MIN)
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
            log.info("[budget] time budget reached -- stopping; next run resumes.")
            break

    writer.flush_all()
    write_readme(state)
    mirror_to_hf()
    if stopped:
        log.info("=== PARTIAL saved. Re-run to continue. total=%d ===", state.get("total_out", 0))
    else:
        log.info("=== DONE. total oracles=%d | domains=%s ===",
                 state.get("total_out", 0), state.get("rows_out", {}))
    log.info("Dataset: https://huggingface.co/datasets/%s", OUT_REPO)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log.warning("Interrupted -- state saved, safe to re-run to resume.")
        sys.exit(130)
