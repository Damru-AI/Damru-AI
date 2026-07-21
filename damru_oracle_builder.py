#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
damru_oracle_builder.py  --  STEP 2 of the Damru 14B Master Plan  (v2, rate-safe)
================================================================
Build a VERIFIER-ORACLE dataset `Damaru-ai/damru-oracle` from closed-model
traces + legal reasoning/coding datasets.

LEGAL / ANTI-COPY LINE (binding):
  Extract ONLY facts: problem statement, SHORT final answer, and test cases.
  NEVER store chain-of-thought / prose / style / full completion.
  Closed-model non-verifiable references are DROPPED. AGPL sources REFUSED.
  Damru does NOT train on this text; the VERIFIER uses it to CHECK answers.

v2 FIXES (match omni v3):
  1. COMMIT BATCHING: local-first, one throttled upload_folder commit
     (<= 1 / ORACLE_COMMIT_EVERY_SEC) -> avoids HF 429 (128 commits/hour).
  2. 429-aware retry.  3. Schema logging (first-row columns per source).
  4. Wider field matching.

RUN:
  pip install "datasets>=2.19" "huggingface_hub>=0.24"
  export HF_TOKEN=hf_xxx
  python damru_oracle_builder.py
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
COMMIT_EVERY    = int(os.environ.get("ORACLE_COMMIT_EVERY_SEC", "600"))
WORKDIR         = os.environ.get("ORACLE_WORKDIR", "./oracle_work")
MIRROR_STATE    = os.environ.get("ORACLE_MIRROR_STATE", "1") == "1"
HF_TOKEN        = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")

DEFAULT_SOURCES = [
    "greghavens/kimi-k3-coding-and-debugging-traces:code:0",
    "nvidia/Open-SWE-Traces:code:0",
    "SupraLabs/reasoning-corpus-4K-5M-v1:reasoning:0",
    "armand0e/claude-fable-5-claude-code:code:1",
    "Crownelius/Complete-FABLE.5-traces-2M:mixed:1",
    "greghavens/fable-5-coding-and-debugging-traces:code:1",
    "WithinUsAI/claude_mythos_distilled_25k:mixed:1",
]
SOURCES = [s.strip() for s in os.environ.get(
    "ORACLE_SOURCES", ",".join(DEFAULT_SOURCES)).split(",") if s.strip()]

BLOCKLIST = ("glint-research/fable-5-traces", "agpl")
DOMAINS = ("math", "code", "reasoning", "other")
MAX_RETRIES = 6

CKPT_DIR   = os.path.join(WORKDIR, "_ckpt")
DATA_DIR   = os.path.join(WORKDIR, "data")
STATE_PATH = os.path.join(CKPT_DIR, "state.json")
SEEN_PATH  = os.path.join(CKPT_DIR, "seen.txt.gz")
README_PATH = os.path.join(WORKDIR, "README.md")
ERR_PATH   = os.path.join(WORKDIR, "errors.log")

for _d in (WORKDIR, CKPT_DIR, DATA_DIR):
    os.makedirs(_d, exist_ok=True)
for _dm in DOMAINS:
    os.makedirs(os.path.join(DATA_DIR, _dm), exist_ok=True)

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


def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_PATH):
        try:
            with open(STATE_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            log_error("load_state", exc)
    return {"version": 2, "sources": {},
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
    if not force and (time.time() - _last_commit[0] < COMMIT_EVERY):
        return

    def _up():
        hf_api().upload_folder(
            folder_path=WORKDIR, repo_id=OUT_REPO, repo_type="dataset",
            allow_patterns=["data/**", "_ckpt/**", "README.md"],
            commit_message=f"oracle update {time.strftime('%Y-%m-%d %H:%M:%S')}",
        )
    try:
        retry(_up, what="commit_folder")
        _last_commit[0] = time.time()
        log.info("[commit] uploaded folder snapshot")
    except Exception as exc:
        log_error("commit_all", exc)
        log.warning("[commit] failed -- kept locally, retry later")


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
# ORACLE extraction (facts only)
# ----------------------------------------------------------------------------
BOXED   = re.compile(r"\\boxed\{([^{}]{1,200})\}")
GSM     = re.compile(r"####\s*([^\n]{1,100})")
ANSWER  = re.compile(r"(?:final answer|the answer is|answer)\s*[:=\-]?\s*([^\n]{1,120})", re.I)
NUMBER  = re.compile(r"-?\d[\d,]*\.?\d*")
ASSERT  = re.compile(r"^\s*assert\s+.+$", re.M)
CODEBLK = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.S)

P_KEYS = ("problem", "question", "prompt", "instruction", "query", "task",
          "input", "text_input", "context")
C_KEYS = ("solution", "answer", "response", "output", "completion", "chosen",
          "target", "text_output", "code")
M_KEYS = ("messages", "conversations", "conversation", "chat", "dialogue", "turns")


def _first(row: Dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _msgs(row: Dict[str, Any]):
    for mk in M_KEYS:
        if isinstance(row.get(mk), list) and row[mk]:
            return row[mk]
    return None


def extract_problem(row: Dict[str, Any]) -> Optional[str]:
    p = _first(row, P_KEYS)
    if p and len(p.strip()) >= 10:
        return p.strip()
    msgs = _msgs(row)
    if msgs:
        for m in msgs:
            if isinstance(m, dict):
                role = (m.get("role") or m.get("from") or "").lower()
                if role in ("user", "human", "prompter"):
                    c = m.get("content") or m.get("value") or m.get("text")
                    if isinstance(c, str) and len(c.strip()) >= 10:
                        return c.strip()
    return None


def completion_text(row: Dict[str, Any]) -> str:
    v = _first(row, C_KEYS)
    if v:
        return v
    msgs = _msgs(row)
    if msgs:
        for m in reversed(msgs):
            if isinstance(m, dict):
                role = (m.get("role") or m.get("from") or "").lower()
                if role in ("assistant", "gpt", "model", "bot"):
                    c = m.get("content") or m.get("value") or m.get("text")
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
        rec["judge_only"] = True
        if not closed:
            ref = (comp or "").strip()
            if ref:
                rec["answer"] = ref[:400]
                rec["answer_type"] = "rubric"
    return rec


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
        local = os.path.join(DATA_DIR, dom, fname)
        atomic_write_text(local, "\n".join(buf) + "\n")
        self.state["shard_index"][dom] = idx + 1
        self.state["rows_out"][dom] = self.state["rows_out"].get(dom, 0) + len(buf)
        self.state["total_out"] = self.state.get("total_out", 0) + len(buf)
        self.buffers[dom] = []
        save_state(self.state)
        save_seen(self.seen)
        commit_all()
        log.info("[flush] %s shard %d (+%d) | total=%d", dom, idx, len(buf), self.state["total_out"])

    def flush_all(self) -> None:
        for d in DOMAINS:
            self.flush(d)


def process_source(spec: str, writer: ShardWriter, state: Dict[str, Any], deadline: float) -> str:
    parts = spec.split(":")
    repo = parts[0]
    hint = parts[1] if len(parts) > 1 else "other"
    closed = (len(parts) > 2 and parts[2] == "1")

    low = repo.lower()
    if any(b in low for b in BLOCKLIST):
        log.warning("[refuse] %s copyleft/AGPL -- skipped (legal).", repo)
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
        log.error("[source] cannot open %s -- skipping.", spec)
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


def write_readme(state: Dict[str, Any]) -> None:
    rows = state.get("rows_out", {})
    md = [
        "---", "license: other", "tags: [damru, oracle, verifier]", "---", "",
        "# Damru Oracle (verifier answer-keys)", "",
        "Built by `damru_oracle_builder.py` (Step 2). Facts only -- problem,",
        "short final answer, test cases. NO chain-of-thought / prose. Used by",
        "the VERIFIER to CHECK Damru's own answers; NOT training text.",
        "", "## Domains", "",
        f"- **code**: {rows.get('code', 0)}",
        f"- **math**: {rows.get('math', 0)}",
        f"- **reasoning**: {rows.get('reasoning', 0)}",
        f"- **other**: {rows.get('other', 0)}",
        "", f"**Total:** {state.get('total_out', 0)} oracles",
    ]
    atomic_write_text(README_PATH, "\n".join(md) + "\n")


def main() -> int:
    if not HF_TOKEN:
        log.error("HF_TOKEN env var is required.")
        return 2
    log.info("=== Damru Oracle Builder v2 | out=%s | sources=%d | budget=%dmin | commit_every=%ds ===",
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
