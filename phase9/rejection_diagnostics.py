#!/usr/bin/env python3
"""Damru Rejection Diagnostics v1 (read-only).

Diagnoses why analysis/coding/Hindi/exam workers produce +0:
- probes each configured LLM provider/model;
- generates one sample per specialist worker;
- reports JSON parsing / empty output / code execution failures;
- computes the exact heuristic score, bonus, threshold and pass/fail;
- writes rejection_diagnostics.json + rejection_diagnostics.md.

No Supabase/HF writes. Safe to run manually.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PHASE5 = ROOT / "phase5"
sys.path.insert(0, str(PHASE5))

REPORT_JSON = os.getenv("DIAG_JSON", "rejection_diagnostics.json")
REPORT_MD = os.getenv("DIAG_MD", "rejection_diagnostics.md")
PROBE_SUBJECT = os.getenv("DIAG_SUBJECT", "Algebra")


def safe_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    text = re.sub(r"(hf_|sk-|AIza)[A-Za-z0-9._-]+", "[REDACTED]", text)
    return text[:500]


def quality_details(question: str, answer: str, bonus: float, evaluator: Any) -> dict[str, Any]:
    q = (question or "").strip()
    a = (answer or "").strip()
    base = float(evaluator.heuristic_quality(q, a))
    threshold = float(evaluator.dynamic_threshold(0))
    final = min(1.0, base + bonus)
    low = a.lower()
    words = low.split()
    diversity = (len(set(words)) / len(words)) if words else 0.0
    reasons = []
    if not q:
        reasons.append("empty_question")
    if not a:
        reasons.append("empty_answer")
    if len(q) <= 15:
        reasons.append("question_too_short_for_bonus")
    if len(a) <= 120:
        reasons.append("answer_under_120_chars")
    elif len(a) <= 400:
        reasons.append("answer_under_400_chars")
    if not re.search(r"\d", a):
        reasons.append("no_numeric_content_bonus")
    if not any(w in low for w in (
        "because", "therefore", "thus", "hence", "step", "example",
        "reason", "consider", "however", "derive", "proof", "so that",
    )):
        reasons.append("no_reasoning_keyword_bonus")
    if diversity < 0.40 and words:
        reasons.append("low_word_diversity_penalty")
    if final < threshold:
        reasons.append("below_quality_threshold")
    return {
        "question_chars": len(q),
        "answer_chars": len(a),
        "word_diversity": round(diversity, 4),
        "heuristic_base": round(base, 4),
        "bonus": round(bonus, 4),
        "final_quality": round(final, 4),
        "required_threshold": round(threshold, 4),
        "would_pass": final >= threshold,
        "reasons": reasons,
    }


def probe_providers(brain: Any) -> list[dict[str, Any]]:
    configured = brain._providers()
    first_by_provider = {}
    for provider, model in configured:
        first_by_provider.setdefault(provider, model)
    results = []
    messages = [
        {"role": "system", "content": "Return exactly the word OK."},
        {"role": "user", "content": "Health probe"},
    ]
    for provider, model in first_by_provider.items():
        t0 = time.time()
        try:
            text = brain._try_one(provider, model, messages, 0.0, 20)
            results.append({
                "provider": provider, "model": model,
                "ok": bool(text and text.strip()),
                "latency_sec": round(time.time() - t0, 2),
                "reply_preview": (text or "")[:80],
            })
        except Exception as exc:
            results.append({
                "provider": provider, "model": model, "ok": False,
                "latency_sec": round(time.time() - t0, 2),
                "error": safe_error(exc),
            })
    return results


def run_code_detail(solution: str, tests: Any, timeout: int = 8) -> dict[str, Any]:
    tests = tests if isinstance(tests, list) else [str(tests)]
    code = solution + "\n\n" + "\n".join(str(t) for t in tests) + "\nprint('DAMRU_OK')\n"
    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w") as f:
            f.write(code)
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=timeout)
        ok = p.returncode == 0 and "DAMRU_OK" in (p.stdout or "")
        return {
            "ok": ok,
            "returncode": p.returncode,
            "stdout": (p.stdout or "")[-800:],
            "stderr": (p.stderr or "")[-1200:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"execution_timeout_{timeout}s"}
    except Exception as exc:
        return {"ok": False, "error": safe_error(exc)}
    finally:
        if path:
            try:
                os.remove(path)
            except Exception:
                pass


def diagnose_coding(brain: Any, coding_engine: Any, evaluator: Any) -> dict[str, Any]:
    try:
        made = coding_engine._make_problem()
    except Exception as exc:
        return {"stage": "generation_exception", "error": safe_error(exc)}
    if not made:
        return {
            "stage": "generation_or_json_failed",
            "hint": "Provider reply was empty, failed, or could not be parsed into required JSON.",
        }
    theme, level, obj = made
    solution = str(obj.get("solution") or "")
    tests = obj.get("tests") or []
    execution = run_code_detail(solution, tests)
    result = {
        "stage": "execution" if not execution.get("ok") else "quality_gate",
        "theme": theme, "level": level,
        "problem_preview": str(obj.get("problem") or "")[:500],
        "solution_preview": solution[:900],
        "test_count": len(tests) if isinstance(tests, list) else 1,
        "execution": execution,
    }
    if execution.get("ok"):
        answer = (
            "Approach & reasoning, then a verified working solution (passes all tests):\n\n"
            "```python\n" + solution.strip() + "\n```\n\nTests used:\n```python\n" +
            "\n".join(str(t) for t in tests) + "\n```"
        )
        result["quality"] = quality_details(
            "[%s/%s] %s" % (theme, level, obj.get("problem", "")),
            answer, 0.15, evaluator,
        )
    return result


def diagnose_item(name: str, producer: Any, evaluator: Any, bonus_kind: str) -> dict[str, Any]:
    t0 = time.time()
    try:
        items = producer()
    except Exception as exc:
        return {"worker": name, "stage": "producer_exception", "error": safe_error(exc)}
    if not items:
        return {
            "worker": name, "stage": "empty_output",
            "latency_sec": round(time.time() - t0, 2),
            "hint": "Provider call failed/returned empty or required JSON/self-check parsing failed.",
        }
    item = items[0]
    bonus = 0.15 if item.get("verified") else (0.05 if item.get("self_checked") else 0.0)
    return {
        "worker": name, "stage": "quality_gate",
        "latency_sec": round(time.time() - t0, 2),
        "intent": item.get("intent"), "lang": item.get("lang"),
        "question_preview": str(item.get("question") or "")[:500],
        "answer_preview": str(item.get("answer") or "")[:900],
        "quality": quality_details(item.get("question", ""), item.get("answer", ""), bonus, evaluator),
    }


def recommendations(report: dict[str, Any]) -> list[str]:
    out = []
    providers = report.get("providers", [])
    names = {x.get("provider") for x in providers}
    if "groq" not in names:
        out.append("GROQ_API_KEY is missing/not configured; add it for fast coding/reasoning fallback.")
    for p in providers:
        if not p.get("ok"):
            out.append(f"Fix provider {p.get('provider')}/{p.get('model')}: {p.get('error', 'empty reply')}")
    coding = report.get("workers", {}).get("coding", {})
    if coding.get("stage") == "generation_or_json_failed":
        out.append("Coding prompt/provider output is not valid JSON; add JSON repair/retry before execution.")
    if coding.get("stage") == "execution":
        out.append("Coding generation works but solution/tests fail execution; add repair loop using captured stderr.")
    for name in ("analysis", "hindi", "exam"):
        item = report.get("workers", {}).get(name, {})
        if item.get("stage") == "empty_output":
            out.append(f"{name} returned no parseable item; instrument provider/JSON parse retry.")
        quality = item.get("quality") or {}
        if quality and not quality.get("would_pass"):
            out.append(
                f"{name} quality {quality.get('final_quality')} is below {quality.get('required_threshold')}; fix prompt/content before lowering threshold."
            )
    if not out:
        out.append("All probes passed; +0 likely came from duplicate storage conflicts rather than generation quality.")
    return out


def self_test() -> int:
    class Eval:
        @staticmethod
        def heuristic_quality(q: str, a: str) -> float:
            return 0.65
        @staticmethod
        def dynamic_threshold(total: int) -> float:
            return 0.70
    d = quality_details("A valid long question about algebra?", "Because this answer explains step 1 with example 42 and enough useful detail." * 4, 0.05, Eval)
    assert d["would_pass"] is True
    fail = run_code_detail("def add(a,b): return a-b", ["assert add(2,3)==5"])
    assert fail["ok"] is False and fail.get("stderr")
    print("Rejection Diagnostics self-test PASS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    import brain
    import evaluator
    import coding_engine
    import analysis_engine
    import hindi_engine
    import exam_engine

    report = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "subject": PROBE_SUBJECT,
        "providers": probe_providers(brain),
        "workers": {},
    }
    report["workers"]["coding"] = diagnose_coding(brain, coding_engine, evaluator)
    report["workers"]["analysis"] = diagnose_item(
        "analysis", lambda: analysis_engine.produce(PROBE_SUBJECT, n=1), evaluator, "self_checked"
    )
    report["workers"]["hindi"] = diagnose_item(
        "hindi", lambda: hindi_engine.produce(PROBE_SUBJECT, lang="hinglish", n=1), evaluator, "none"
    )
    report["workers"]["exam"] = diagnose_item(
        "exam", lambda: exam_engine.produce(n=1), evaluator, "none"
    )
    report["recommendations"] = recommendations(report)

    Path(REPORT_JSON).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Damru Rejection Diagnostics", "", "## Recommendations"]
    lines += [f"- {x}" for x in report["recommendations"]]
    lines += ["", "## Provider probes", "", "```json", json.dumps(report["providers"], indent=2), "```"]
    lines += ["", "## Worker probes", "", "```json", json.dumps(report["workers"], indent=2, ensure_ascii=False), "```"]
    Path(REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    print("DIAGNOSTICS COMPLETE — no databases or model repos were modified.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
