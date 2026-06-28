#!/usr/bin/env python3
"""
Damru EVAL / BENCHMARK runner.

Tests a model on eval/benchmark.jsonl and grades it:
  - objective items (with 'answer_key'): exact/keyword match (no API needed)
  - open items: graded 0..1 by an LLM judge
Writes eval_report.json + eval_report.html and prints per-category accuracy.

SELF-IMPROVEMENT: also writes eval/weak_topics.json listing the SUBJECTS where
Damru scored below WEAK_THRESHOLD. The learning engine reads this and focuses
extra effort exactly there (closed feedback loop).

Use it to measure Damru's quality BEFORE vs AFTER fine-tuning (change EVAL_MODEL).

Env:
  OPENROUTER_API_KEY (required)
  EVAL_MODEL   model under test (default a free model)
  JUDGE_MODEL  judge model (default a strong free model)
  EVAL_FILE    default 'eval/benchmark.jsonl'
  WEAK_THRESHOLD  category score below this -> weak (default 0.7)
  LIMIT        optional cap on number of items
"""
import os
import re
import json
import time
import urllib.request
import urllib.error

KEY = os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_KEY", "")
EVAL_MODEL = os.environ.get("EVAL_MODEL", "qwen/qwen-2.5-72b-instruct:free")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "deepseek/deepseek-chat-v3-0324:free")
EVAL_FILE = os.environ.get("EVAL_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.jsonl"))
WEAK_THRESHOLD = float(os.environ.get("WEAK_THRESHOLD", "0.7"))
LIMIT = int(os.environ.get("LIMIT", "0"))
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
WEAK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weak_topics.json")

# Map eval categories -> curriculum subject names the engine understands.
CATEGORY_TO_SUBJECTS = {
    "math": ["Mathematics", "Algebra", "Calculus"],
    "physics": ["Physics", "Classical Mechanics", "Electromagnetism"],
    "chemistry": ["Chemistry", "Organic Chemistry", "Physical Chemistry"],
    "biology": ["Biology", "Genetics", "Human Anatomy"],
    "coding": ["Computer Science", "Algorithms", "Data Structures"],
    "cs": ["Computer Science", "Operating Systems", "Databases"],
    "reasoning": ["Critical Thinking", "Logic", "Real World Problem Solving"],
    "gk_india": ["World History", "Geography", "Political Science"],
    "hindi": ["Real World Problem Solving", "Literature"],
    "english": ["Literature", "Linguistics"],
    "general": ["Real World Problem Solving"],
}


def _chat(model, messages, temperature=0.2, max_tokens=900):
    payload = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(OR_URL, data=payload, headers={
        "Authorization": "Bearer " + KEY,
        "Content-Type": "application/json",
        "HTTP-Referer": "https://damru-ai.vercel.app",
        "X-Title": "Damru Eval",
    })
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                resp = json.loads(r.read().decode("utf-8", "ignore"))
            ch = resp.get("choices") or []
            if ch:
                return ((ch[0].get("message") or {}).get("content") or "").strip()
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503):
                time.sleep(2 ** attempt + 1)
                continue
            break
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    return ""


def _norm(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s or "").lower()).strip()


def _objective_ok(answer, key):
    a = _norm(answer)
    keys = key if isinstance(key, list) else [key]
    return any(_norm(k) and _norm(k) in a for k in keys)


def _judge(question, reference, answer):
    user = (
        "You are a strict grader. Compare the STUDENT answer to the REFERENCE.\n"
        "Score 0.0 to 1.0 for correctness + completeness.\n\n"
        "QUESTION:\n%s\n\nREFERENCE:\n%s\n\nSTUDENT:\n%s\n\n"
        "Reply ONLY as JSON: {\"score\": 0.0, \"reason\": \"...\"}"
        % (question[:1200], str(reference)[:1500], answer[:2500])
    )
    txt = _chat(JUDGE_MODEL, [{"role": "user", "content": user}], temperature=0.0, max_tokens=300)
    m = re.search(r"\{.*\}", txt, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return max(0.0, min(1.0, float(obj.get("score", 0)))), str(obj.get("reason", ""))[:200]
        except Exception:
            pass
    return 0.0, "unparseable judge reply"


def write_weak_topics(by_cat):
    weak_cats = [c for c, s in by_cat.items() if s < WEAK_THRESHOLD]
    subjects = []
    for c in weak_cats:
        for s in CATEGORY_TO_SUBJECTS.get(c, []):
            if s not in subjects:
                subjects.append(s)
    payload = {
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threshold": WEAK_THRESHOLD,
        "weak_categories": weak_cats,
        "weak_subjects": subjects,
    }
    with open(WEAK_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("Wrote weak_topics.json -> weak categories:", weak_cats or "(none, all strong!)")


def main():
    if not KEY:
        print("ERROR: OPENROUTER_API_KEY not set.")
        return
    items = []
    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    if LIMIT > 0:
        items = items[:LIMIT]
    print("Loaded %d eval items. Model under test: %s" % (len(items), EVAL_MODEL))

    results, cat = [], {}
    for i, it in enumerate(items, 1):
        q = it.get("question", "")
        ans = _chat(EVAL_MODEL, [
            {"role": "system", "content": "You are Damru. Answer accurately and concisely with reasoning."},
            {"role": "user", "content": q},
        ], temperature=0.2)
        if it.get("answer_key"):
            ok = _objective_ok(ans, it["answer_key"])
            score, reason = (1.0 if ok else 0.0), "objective match" if ok else "objective miss"
        else:
            score, reason = _judge(q, it.get("reference", ""), ans)
        c = it.get("category", "general")
        cat.setdefault(c, []).append(score)
        results.append({"id": it.get("id"), "category": c, "lang": it.get("lang", "en"),
                        "score": round(score, 3), "reason": reason})
        print("  [%2d/%2d] %-12s %.2f  %s" % (i, len(items), c, score, str(it.get("id"))))
        time.sleep(0.3)

    overall = round(sum(r["score"] for r in results) / max(1, len(results)), 3)
    by_cat = {c: round(sum(v) / len(v), 3) for c, v in cat.items()}
    report = {"model": EVAL_MODEL, "judge": JUDGE_MODEL, "n": len(results),
              "overall": overall, "by_category": by_cat, "results": results,
              "ts": time.strftime("%Y-%m-%d %H:%M:%S")}

    with open("eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    write_weak_topics(by_cat)

    rows = "".join(
        "<tr><td>%s</td><td style='text-align:right'>%.0f%%</td></tr>" % (c, s * 100)
        for c, s in sorted(by_cat.items())
    )
    htmldoc = (
        "<!doctype html><meta charset='utf-8'><title>Damru Eval</title>"
        "<body style='font-family:system-ui;background:#0f0f17;color:#eee;padding:20px'>"
        "<h1>\U0001F9EA Damru Eval Report</h1>"
        "<p>Model: <b>%s</b><br>Overall: <b style='font-size:28px;color:#a29bfe'>%.0f%%</b> "
        "(%d items)<br><span style='color:#888'>%s</span></p>"
        "<table style='border-collapse:collapse'><tr><th align='left'>Category</th><th>Score</th></tr>%s</table>"
        "</body>" % (EVAL_MODEL, overall * 100, len(results), report["ts"], rows)
    )
    with open("eval_report.html", "w", encoding="utf-8") as f:
        f.write(htmldoc)

    print("\n==== EVAL DONE ====")
    print("Overall: %.1f%%" % (overall * 100))
    for c, s in sorted(by_cat.items()):
        print("  %-14s %.1f%%" % (c, s * 100))
    print("Wrote eval_report.json + eval_report.html + weak_topics.json")


if __name__ == "__main__":
    main()
