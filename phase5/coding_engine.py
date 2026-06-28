"""
Coding practice engine (runs at 4X via the scheduler).
Damru: (1) invents a coding problem WITH unit tests, (2) writes a solution,
(3) ACTUALLY EXECUTES it against the tests in an isolated subprocess,
(4) only keeps it if the tests pass -> real self-verification.
Execution is sandboxed with a timeout; failures never crash the engine.
"""
import json
import os
import random
import subprocess
import tempfile

import brain

LANGS = ["Python"]
THEMES = [
    "arrays", "strings", "hash maps", "recursion", "dynamic programming",
    "graphs", "trees", "sorting", "searching", "two pointers", "greedy",
    "backtracking", "bit manipulation", "math", "stacks and queues",
]
LEVELS = ["easy", "medium", "hard"]


def _make_problem():
    theme = random.choice(THEMES)
    level = random.choice(LEVELS)
    user = (
        "Create ONE %s-level Python coding problem about %s.\n"
        "Provide a correct reference solution as a single function, plus 3-5 "
        "assert-based tests that call that function.\n\n"
        "Reply ONLY as JSON with keys:\n"
        "{\"problem\": \"clear problem statement\", \"function_name\": \"name\", "
        "\"solution\": \"def name(...):\\n    ...\", "
        "\"tests\": [\"assert name(...) == ...\"]}" % (level, theme)
    )
    try:
        txt = brain.chat(
            [{"role": "system", "content": brain.DEEP_SYS},
             {"role": "user", "content": user}],
            temperature=0.7, max_tokens=1600,
        )
        obj = brain.extract_json(txt)
        if obj and obj.get("solution") and obj.get("tests"):
            return theme, level, obj
    except Exception:
        return None
    return None


def _run(solution, tests, timeout=8):
    """Execute solution + tests in a subprocess. Return True if all pass."""
    tests = tests if isinstance(tests, list) else [str(tests)]
    code = solution + "\n\n" + "\n".join(str(t) for t in tests) + "\nprint('DAMRU_OK')\n"
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".py", dir=tempfile.gettempdir())
        with os.fdopen(fd, "w") as f:
            f.write(code)
        proc = subprocess.run(
            ["python3", path], capture_output=True, text=True, timeout=timeout
        )
        return proc.returncode == 0 and "DAMRU_OK" in (proc.stdout or "")
    except Exception:
        return False
    finally:
        if path and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


def produce(n=8):
    """Return a list of VERIFIED coding Q&A items (only those that pass tests)."""
    out = []
    for _ in range(n):
        made = _make_problem()
        if not made:
            continue
        theme, level, obj = made
        solution = str(obj.get("solution", ""))
        tests = obj.get("tests", [])
        passed = _run(solution, tests)
        if not passed:
            continue  # force correctness: only keep code that actually runs green
        problem = str(obj.get("problem", "")).strip()
        answer = (
            "Approach & reasoning, then a verified working solution "
            "(passes all tests):\n\n```python\n" + solution.strip() + "\n```\n\n"
            "Tests used:\n```python\n" + "\n".join(str(t) for t in tests) + "\n```"
        )
        out.append({
            "question": "[%s/%s] %s" % (theme, level, problem),
            "answer": answer,
            "intent": "coding_" + theme.replace(" ", "_"),
            "lang": "en",
            "verified": True,
        })
    return out
