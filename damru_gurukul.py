#!/usr/bin/env python3
"""
DAMRU GURUKUL  --  multi-teacher distillation flywheel (GPU-free).

BIG IDEA: capture the "months of work" baked into frontier models (Claude,
GPT-5.x, Gemini, GLM, DeepSeek...) INTO Damru's OWN mind -- without paying for
them at answer time and without hitting API limits.

ETHIC (matches Damru's rule -- 'apne mind se jawab'):
  * Teachers are used ONLY OFFLINE to generate training data.
  * Damru NEVER calls a teacher to answer a live user.
  * Once distilled into RAG (today) + weights (Brain Forge, later), teacher dropped.

CONSENSUS (v2 -- fixed): exact-vote for NUMERIC/short answers (strict anti-
hallucination); OVERLAP-similarity clustering for OPEN-ENDED prose (so good
teaching traces are kept even when wording differs). min_agreement strictly
gates numeric answers; prose is accepted when a substantive trace exists, with a
soft overlap score stored in the `agreement` column for later filtering.

All model/network calls are INJECTED -> pure logic, testable with mocks.
"""
import hashlib
import re
import time
from collections import Counter, defaultdict

# ---------------- curriculum: every skill Damru must master ----------------
DOMAINS = {
    "coding":            ("code",      ["data structures", "async bugs", "system design", "regex", "API design"]),
    "maths":             ("math",      ["algebra", "calculus", "probability", "number theory", "linear algebra"]),
    "science":           ("science",   ["physics laws", "chemistry reactions", "biology systems", "astronomy"]),
    "world_history":     ("gk",        ["ancient India", "world wars", "revolutions", "empires", "civilizations"]),
    "architecture":      ("reason",    ["structural design", "sustainable building", "load analysis"]),
    "3d_dimension":      ("reason",    ["3D geometry", "spatial reasoning", "CAD logic", "projections"]),
    "human_behaviour":   ("chat",      ["psychology", "motivation", "empathy", "negotiation"]),
    "conversation":      ("chat",      ["small talk", "active listening", "persuasion", "tone"]),
    "tool_building":     ("code",      ["CLI tools", "automation scripts", "agent tools", "plugins"]),
    "critical_analysis": ("reason",    ["argument evaluation", "bias detection", "fallacies", "evidence"]),
    "technology_cs":     ("reason",    ["operating systems", "networking", "databases", "security"]),
    "future_thinking":   ("reason",    ["forecasting", "scenario planning", "emerging tech", "risks"]),
    "resource_mgmt":     ("reason",    ["planning", "prioritization", "budgeting", "scheduling"]),
}

_Q_TEMPLATES = [
    "Explain {sub} step by step with a concrete example.",
    "What is the core principle behind {sub}, and why does it matter?",
    "Solve a hard {sub} problem and show your full reasoning.",
    "Compare two approaches to {sub} and justify which is better.",
    "What are common mistakes in {sub} and how to avoid them?",
]


def curriculum(n_per_domain=3, domains=None, gen_fn=None):
    out = []
    doms = domains or list(DOMAINS.keys())
    for d in doms:
        intent, subs = DOMAINS[d]
        for i in range(n_per_domain):
            sub = subs[i % len(subs)]
            tmpl = _Q_TEMPLATES[i % len(_Q_TEMPLATES)]
            q = tmpl.format(sub=sub)
            if gen_fn is not None:
                try:
                    q = gen_fn([{"role": "user", "content":
                        f"Write ONE specific, challenging {d} question about '{sub}'. Return only the question."}]).strip() or q
                except Exception:
                    pass
            out.append({"question": q, "domain": d, "intent": intent})
    return out


# ---------------- teacher pool: round-robin + cooldown (API-limit jugad) ----
class TeacherPool:
    def __init__(self, teachers, cooldown_s=30):
        self.teachers = list(teachers)
        self.cooldown_s = cooldown_s
        self._until = defaultdict(float)
        self._rr = 0

    def _available(self):
        now = time.time()
        return [(n, f) for (n, f) in self.teachers if self._until[n] <= now]

    def penalize(self, name):
        self._until[name] = time.time() + self.cooldown_s

    def ask_all(self, messages, max_teachers=None):
        avail = self._available() or self.teachers
        if avail:
            self._rr = (self._rr + 1) % len(avail)
            avail = avail[self._rr:] + avail[:self._rr]
        if max_teachers:
            avail = avail[:max_teachers]
        res, errors = [], 0
        for name, fn in avail:
            try:
                ans = fn(messages)
                if ans and ans.strip():
                    res.append({"teacher": name, "answer": ans.strip()})
                else:
                    errors += 1
            except Exception:
                errors += 1
                self.penalize(name)
        self.last_errors = errors
        return res


_TRACE_SYS = (
    "You are an expert teacher. Think step by step and SHOW your reasoning, "
    "then end with a line 'FINAL: <concise answer>'. Be correct and complete.")


def _split_trace(text):
    m = re.search(r"FINAL\s*:\s*(.+)\s*$", text, re.I | re.S)
    if m:
        return text[:m.start()].strip(), m.group(1).strip()
    lines = [l for l in text.splitlines() if l.strip()]
    return ("\n".join(lines[:-1]).strip() if len(lines) > 1 else ""), (lines[-1].strip() if lines else text.strip())


def _norm(s):
    s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else s[:80]


_SHORT_RE = re.compile(r"^[\s\d.,:/*+\-()x=%]+$", re.I)


def _words(s):
    return set(re.findall(r"[a-z0-9]+", (s or "").lower()))


def _jaccard(a, b):
    return len(a & b) / len(a | b) if (a and b) else 0.0


# ---------------- consensus v2: numeric=exact, prose=overlap ----------------
def consensus(teacher_answers, min_similarity=0.4):
    parsed = []
    for t in teacher_answers:
        reasoning, final = _split_trace(t["answer"])
        parsed.append({"teacher": t["teacher"], "reasoning": reasoning, "final": final})
    if not parsed:
        return {"ok": False, "agreement": 0.0, "n": 0, "kind": "none"}
    n = len(parsed)
    finals = [p["final"] for p in parsed]
    is_short = sum(1 for f in finals if _SHORT_RE.match(f or "") or len(f.split()) <= 2) >= (n + 1) // 2

    if is_short:  # numeric / very-short -> strict exact voting
        keys = [_norm(f) for f in finals]
        win, votes = Counter(keys).most_common(1)[0]
        winners = [p for p, k in zip(parsed, keys) if k == win]
        kind, agree = "exact", votes / n
    else:         # prose -> overlap clustering (wording-tolerant)
        ws = [_words(f) for f in finals]
        best = []
        for i in range(n):
            cl = [j for j in range(n) if _jaccard(ws[i], ws[j]) >= min_similarity]
            if len(cl) > len(best):
                best = cl
        winners = [parsed[j] for j in best] if best else parsed
        kind, agree = "overlap", (len(best) / n if best else 1.0 / n)

    top = max(winners, key=lambda p: len(p["reasoning"] or ""))
    return {"ok": True, "final": top["final"], "reasoning": top["reasoning"],
            "agreement": round(agree, 2), "n": n, "kind": kind,
            "agreed_by": [p["teacher"] for p in winners]}


def to_row(question, con, intent, lang="en"):
    return {
        "question": question,
        "answer": con["final"],
        "reasoning": con["reasoning"],
        "intent": intent,
        "lang": lang,
        "upvotes": max(1, int(round(con["agreement"] * len(con.get("agreed_by", []))))),
        "agreement": con["agreement"],
        "kind": con["kind"],
        "teachers": ",".join(con.get("agreed_by", [])),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def distill(question, intent, pool, min_agreement=0.6, lang="en"):
    msgs = [{"role": "system", "content": _TRACE_SYS},
            {"role": "user", "content": question}]
    answers = pool.ask_all(msgs)
    if not answers:
        return None, {"reason": "no_teacher"}
    con = consensus(answers)
    if not con["ok"]:
        return None, {"reason": "no_teacher"}
    substantive = len((con["reasoning"] or "") + con["final"]) >= 40
    # numeric/short -> must meet min_agreement; prose -> accept if substantive
    if con["kind"] == "exact":
        if con["agreement"] < min_agreement:
            return None, {"reason": "low_agreement", "agreement": con["agreement"], "kind": "exact"}
    else:
        if not substantive:
            return None, {"reason": "thin_answer", "agreement": con["agreement"], "kind": "overlap"}
    return to_row(question, con, intent, lang), {
        "reason": "ok", "agreement": con["agreement"], "kind": con["kind"],
        "teachers": con["agreed_by"]}


def harvest(items, pool, min_agreement=0.6, seen=None):
    seen = seen if seen is not None else set()
    rows, dropped = [], 0
    reasons, debug = Counter(), []
    for it in items:
        h = hashlib.sha1(it["question"].strip().lower().encode()).hexdigest()
        if h in seen:
            dropped += 1
            reasons["dupe"] += 1
            continue
        seen.add(h)
        row, meta = distill(it["question"], it.get("intent", "reason"), pool, min_agreement)
        reasons[meta["reason"]] += 1
        if row:
            rows.append(row)
        else:
            dropped += 1
            if len(debug) < 3:
                debug.append({"q": it["question"][:60], "meta": meta})
    return {"rows": rows, "kept": len(rows), "dropped": dropped,
            "seen": seen, "reasons": dict(reasons), "debug": debug}


if __name__ == "__main__":
    def t_num1(m):
        return "12*8: 12*8=96. FINAL: 96"
    def t_num2(m):
        return "Compute: 96. FINAL: 96"
    def t_num_bad(m):
        return "FINAL: 99"
    def t_prose1(m):
        return "Data structures organize data; arrays, stacks, queues, trees. FINAL: they organize data efficiently for fast access"
    def t_prose2(m):
        return "They structure data: arrays lists stacks trees for efficiency. FINAL: organize data for efficient access and operations"

    print("--- numeric (strict) ---")
    p = TeacherPool([("a", t_num1), ("b", t_num2), ("c", t_num_bad)])
    r, m = distill("What is 12*8?", "math", p, 0.6)
    print("kept:", bool(r), "| meta:", m)

    print("--- prose (overlap) ---")
    p2 = TeacherPool([("a", t_prose1), ("b", t_prose2)])
    r2, m2 = distill("Explain data structures.", "code", p2, 0.6)
    print("kept:", bool(r2), "| meta:", m2)
    if r2:
        print("    answer:", r2["answer"][:60], "| kind:", r2["kind"], "| agree:", r2["agreement"])

    print("--- full harvest (13 domains x1) ---")
    p3 = TeacherPool([("a", t_prose1), ("b", t_prose2)])
    res = harvest(curriculum(1), p3, 0.6)
    print("kept:", res["kept"], "dropped:", res["dropped"], "reasons:", res["reasons"])
