#!/usr/bin/env python3
"""
Damru Code Execution Lab  (Jugad A + DPO gap-fill)
==================================================
This is what actually makes Damru's coding "ultra-pro": EXECUTION FEEDBACK.

What it does
------------
1. Streams coding problems that ship with their own test cases
   (default: nvidia/OpenCodeInstruct -- solution + tests + exec feedback).
2. Re-runs each Python solution against its tests inside an isolated,
   resource-limited subprocess (CPU + memory + wall-clock capped).
3. Keeps ONLY solutions that actually pass  -> gold "verified_coding" rows.
4. Auto-creates DEBUG-FIX training pairs: it mutates a verified solution into
   a broken one, confirms it now FAILS (captures the real error), and stores
   {"broken code + error" -> "correct fix"}.  This teaches Damru to debug.
5. Auto-creates DPO preference triples {prompt, chosen, rejected} from the
   (verified vs confirmed-broken) pair -> preference optimisation (gap #4).

Outputs (schema-safe)
---------------------
* Verified coding + debug-fix rows  -> HF_REPO  (same 6-col schema as the
  main knowledge base: question, answer, intent, lang, upvotes, created_at).
* DPO triples                       -> DPO_REPO (separate repo so the main
  dataset's schema is never broken):  prompt, chosen, rejected, intent.

Why a separate subprocess and not Docker?
-----------------------------------------
GitHub Actions runners already give us an isolated, throwaway VM, and we add
CPU/mem/time rlimits per run. For hostile code you would swap _run_code() for
llm-sandbox / Modal; the interface stays identical.

Env
---
HF_TOKEN     (required)
HF_REPO      verified rows target           (default Damaru-ai/damru-knowledge)
DPO_REPO     preference triples target       (default Damaru-ai/damru-dpo)
SRC_DATASET  problems w/ tests               (default nvidia/OpenCodeInstruct)
SRC_SPLIT    (default train)
LAB_MAX      max problems to attempt         (default 40000)
EXEC_TIMEOUT per-run wall-clock seconds       (default 8)
EXEC_MEM_MB  per-run memory cap (MB)          (default 512)
SHARD        rows per uploaded shard          (default 5000)
RUN_MIN      wall-clock budget in minutes     (default 300)
MAKE_DPO     "1" to also emit DPO triples      (default 1)
"""
import os, sys, io, re, time, json, random, tempfile, subprocess, textwrap
from datetime import datetime, timezone

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_REPO", "Damaru-ai/damru-knowledge")
DPO_REPO = os.environ.get("DPO_REPO", "Damaru-ai/damru-dpo")
SRC_DATASET = os.environ.get("SRC_DATASET", "nvidia/OpenCodeInstruct")
SRC_SPLIT = os.environ.get("SRC_SPLIT", "train")
LAB_MAX = int(os.environ.get("LAB_MAX") or "40000")
EXEC_TIMEOUT = int(os.environ.get("EXEC_TIMEOUT") or "8")
EXEC_MEM_MB = int(os.environ.get("EXEC_MEM_MB") or "512")
SHARD = int(os.environ.get("SHARD") or "5000")
RUN_MIN = int(os.environ.get("RUN_MIN") or "300")
MAKE_DPO = os.environ.get("MAKE_DPO", "1") == "1"

Q_FIELDS = ["instruction", "problem", "question", "prompt", "input"]
A_FIELDS = ["solution", "output", "response", "code", "answer"]
T_FIELDS = ["unit_tests", "test", "tests", "test_code", "testcase",
            "test_cases", "test_list"]
LANG_FIELDS = ["language", "lang", "programming_language"]


def _first(ex, fields):
    for f in fields:
        v = ex.get(f)
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = "\n".join(str(x) for x in v if str(x).strip())
        v = str(v)
        if v.strip():
            return v
    return ""


# --- code extraction --------------------------------------------------------
_FENCE = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.S)


def _extract_code(sol):
    """Return the raw python code from a solution string."""
    m = _FENCE.findall(sol or "")
    if m:
        # pick the longest fenced block
        return max(m, key=len).strip()
    return (sol or "").strip()


def _is_python(ex, code):
    lng = _first(ex, LANG_FIELDS).lower()
    if lng:
        return "py" in lng
    # heuristic: looks like python
    return bool(re.search(r"\bdef \w+\(|\bimport \w+|\bprint\(", code or ""))


# --- sandboxed execution ----------------------------------------------------
_RUNNER_PREAMBLE = (
    "import resource, sys\n"
    "try:\n"
    "    resource.setrlimit(resource.RLIMIT_AS, (%d, %d))\n"
    "    resource.setrlimit(resource.RLIMIT_CPU, (%d, %d))\n"
    "except Exception:\n"
    "    pass\n"
)


def _run_code(code, tests):
    """Execute code+tests in an isolated subprocess.
    Returns (passed: bool, err: str). err is '' on success."""
    mem = EXEC_MEM_MB * 1024 * 1024
    body = (_RUNNER_PREAMBLE % (mem, mem, EXEC_TIMEOUT + 2, EXEC_TIMEOUT + 2))
    body += code.rstrip() + "\n\n" + (tests or "").strip() + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(body)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=EXEC_TIMEOUT)
        if p.returncode == 0:
            return True, ""
        err = (p.stderr or p.stdout or "non-zero exit").strip()
        # keep last, most relevant lines of the traceback
        return False, "\n".join(err.splitlines()[-6:])[:600]
    except subprocess.TimeoutExpired:
        return False, "TimeoutError: execution exceeded %ds" % EXEC_TIMEOUT
    except Exception as e:
        return False, ("%s: %s" % (type(e).__name__, e))[:600]
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


# --- mutation to manufacture a TRUE negative (for debug + DPO) ---------------
_MUTATORS = [
    (re.compile(r"\breturn\b"), "pass  #"),          # break a return
    (re.compile(r"(\w+)\s*\+\s*(\w+)"), r"\1 - \2"),  # + -> -
    (re.compile(r"=="), "!="),                        # flip comparison
    (re.compile(r"\brange\("), "range(1,"),           # off-by-one
    (re.compile(r"<="), "<"),                          # boundary bug
]


def _mutate(code):
    """Yield plausible broken variants of `code` (first match per mutator)."""
    for rx, repl in _MUTATORS:
        if rx.search(code):
            yield rx.sub(repl, code, count=1)


# --- HF upload helpers ------------------------------------------------------
_LAST_COMMIT = [0.0]


def _throttle():
    # stay well under HF's 120 commits/hour/repo
    dt = time.time() - _LAST_COMMIT[0]
    if dt < 32:
        time.sleep(32 - dt)
    _LAST_COMMIT[0] = time.time()


def _ensure_repo(api, repo):
    from huggingface_hub import create_repo
    try:
        create_repo(repo, repo_type="dataset", token=HF_TOKEN, exist_ok=True)
    except Exception as e:
        print("  create_repo note:", str(e)[:120], flush=True)


def _hf_upload(api, **kw):
    """upload_file with retry/backoff against transient HF 5xx/429/504."""
    last = None
    for attempt in range(6):
        try:
            return api.upload_file(**kw)
        except Exception as e:
            last = e
            s = str(e)
            code = getattr(getattr(e, "response", None), "status_code", None)
            transient = (
                (code is not None and int(code) >= 500)
                or "504" in s or "503" in s or "502" in s or "500" in s
                or "429" in s or "Time-out" in s or "Timeout" in s
                or "Gateway" in s or "Service Unavailable" in s
            )
            if not transient or attempt == 5:
                raise
            wait = min(120, 8 * (2 ** attempt))
            print("  HF upload transient (%s) -> retry %d/5 in %ds"
                  % (s[:70], attempt + 1, wait), flush=True)
            time.sleep(wait)
    if last:
        raise last


def _upload(api, repo, buf, tag, idx):
    from datasets import Dataset
    local = "/tmp/%s-%d.parquet" % (tag, idx)
    Dataset.from_list(buf).to_parquet(local)
    fname = "data/%s-%d-%03d.parquet" % (tag, int(time.time()), idx)
    _throttle()
    _hf_upload(api, path_or_fileobj=local, path_in_repo=fname,
               repo_id=repo, repo_type="dataset")
    try:
        os.remove(local)
    except Exception:
        pass
    print("  uploaded %s -> %s (%d rows)" % (fname, repo, len(buf)), flush=True)


def _row(q, a, intent, uv):
    return {"question": q.strip(), "answer": a.strip(),
            "intent": intent[:80], "lang": "en", "upvotes": int(uv),
            "created_at": datetime.now(timezone.utc).isoformat()}


def main():
    if not HF_TOKEN:
        print("FATAL: HF_TOKEN missing", flush=True)
        sys.exit(1)
    from huggingface_hub import HfApi
    from datasets import load_dataset
    api = HfApi(token=HF_TOKEN)
    _ensure_repo(api, DPO_REPO)

    deadline = time.time() + RUN_MIN * 60
    print("Code Lab start | src=%s split=%s max=%d timeout=%ds" %
          (SRC_DATASET, SRC_SPLIT, LAB_MAX, EXEC_TIMEOUT), flush=True)

    ds = load_dataset(SRC_DATASET, split=SRC_SPLIT, streaming=True)
    know_buf, dpo_buf = [], []
    know_idx = dpo_idx = 0
    seen = verified = debug_pairs = dpo_pairs = 0

    for ex in ds:
        if seen >= LAB_MAX or time.time() > deadline:
            break
        seen += 1
        if seen % 2000 == 0:
            print("  scanned=%d verified=%d debug=%d dpo=%d" %
                  (seen, verified, debug_pairs, dpo_pairs), flush=True)
        try:
            q = _first(ex, Q_FIELDS)
            sol = _first(ex, A_FIELDS)
            tests = _first(ex, T_FIELDS)
            if not (q and sol and tests):
                continue
            code = _extract_code(sol)
            if len(code) < 20 or not _is_python(ex, code):
                continue
            ok, _ = _run_code(code, tests)
            if not ok:
                continue  # dataset solution doesn't pass its own tests -> drop
            verified += 1
            ans = "```python\n%s\n```" % code
            know_buf.append(_row(q, ans, "verified_coding", 10))

            # manufacture a confirmed-broken variant for debug + DPO
            if MAKE_DPO:
                for broken in _mutate(code):
                    if broken == code:
                        continue
                    bad, err = _run_code(broken, tests)
                    if bad or not err:
                        continue  # negative must genuinely FAIL with an error
                    # debug-fix training row (teaches Damru to fix bugs)
                    dq = ("This Python code fails with the error below. "
                          "Find the bug and give the corrected code.\n\n"
                          "```python\n%s\n```\n\nError:\n%s" % (broken, err))
                    da = ("The bug is fixed below.\n\n```python\n%s\n```" % code)
                    know_buf.append(_row(dq, da, "debug_fix", 10))
                    debug_pairs += 1
                    # DPO preference triple
                    dpo_buf.append({
                        "prompt": q.strip(),
                        "chosen": ans,
                        "rejected": "```python\n%s\n```" % broken,
                        "intent": "coding_dpo"})
                    dpo_pairs += 1
                    break  # one negative per problem is enough

            if len(know_buf) >= SHARD:
                _upload(api, HF_REPO, know_buf, "codelab", know_idx)
                know_idx += 1
                know_buf = []
            if len(dpo_buf) >= SHARD:
                _upload(api, DPO_REPO, dpo_buf, "dpo", dpo_idx)
                dpo_idx += 1
                dpo_buf = []
        except Exception as e:
            if seen % 5000 == 0:
                print("  row error:", str(e)[:120], flush=True)
            continue

    if know_buf:
        _upload(api, HF_REPO, know_buf, "codelab", know_idx)
    if dpo_buf:
        _upload(api, DPO_REPO, dpo_buf, "dpo", dpo_idx)
    print("LAB COMPLETE | scanned=%d verified=%d debug_pairs=%d dpo_pairs=%d" %
          (seen, verified, debug_pairs, dpo_pairs), flush=True)


if __name__ == "__main__":
    main()
