#!/usr/bin/env python3
"""
DAMRU GURUKUL  --  multi-teacher distillation flywheel (GPU-free).

BIG IDEA: capture the "months of work" baked into frontier models (Claude,
GPT-5.x, Gemini, GLM, DeepSeek...) INTO Damru's OWN mind -- without paying for
them at answer time and without hitting API limits.

ETHIC (matches Damru's rule -- 'apne mind se jawab'):
  * Teachers are used ONLY OFFLINE to generate training data.
  * Damru NEVER calls a teacher to answer a live user. Live answers come from
    Damru's own brain + RAG + its own web research (see app.py / damru_reason.py).
  * Once a reasoning trace is distilled into RAG memory (today) + own weights
    (Brain Forge, later), the teacher is DROPPED. Damru stands alone.

PIPELINE (one coherent flywheel):
  1. curriculum()    -> questions across every skill domain.
  2. ask_teachers()  -> rotating panel of free teachers; capture step-by-step
                        REASONING TRACE (R1-style), not just the final answer.
  3. consensus()     -> cross-check teachers; multi-teacher AGREEMENT is a free
                        quality filter that kills hallucination. Only high-
                        agreement traces graduate.
  4. to_row()        -> canonical corpus rows (+reasoning) for Supabase + HF.
  5. Payoff: RAG serves them TODAY (learn w/o GPU); Brain Forge trains on them
     when GPU returns (permanent own-mind knowledge).

JUGAD for API/limits: TeacherPool = round-robin + per-provider cooldown +
  on-disk dedupe. Free/keyless tiers only, no single provider hammered.

All model/network calls are INJECTED -> pure logic, testable with mocks,
no keys/GPU needed.
"""
import hashlib
import re
import time
from collections import Counter, defaultdict

# ---------------- curriculum: every skill Damru must master ----------------
# domain -> (intent tag, seed subtopics)
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
    """Produce curriculum questions. If gen_fn (an LLM) is given, expand seeds
    into richer questions; otherwise use templates (fully offline)."""
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
    """Rotates across free teacher endpoints; skips any on cooldown after error/
    rate-limit. Each teacher_fn(messages)->str is injected; names for logging."""

    def __init__(self, teachers, cooldown_s=30):
        # teachers: list of (name, fn)
        self.teachers = list(teachers)
        self.cooldown_s = cooldown_s
        self._until = defaultdict(float)   # name -> epoch until which it's paused
        self._rr = 0

    def _available(self):
        now = time.time()
        return [(n, f) for (n, f) in self.teachers if self._until[n] <= now]

    def penalize(self, name):
        self._until[name] = time.time() + self.cooldown_s

    def ask_all(self, messages, max_teachers=None):
        """Ask each available teacher once (round-robin start). Returns list of
        {teacher, answer}. Teachers that error are penalized (cooldown)."""
        avail = self._available()
        if not avail:
            avail = self.teachers  # cooldown over-ride if all paused
        # rotate starting point so load spreads
        if avail:
            self._rr = (self._rr + 1) % len(avail)
            avail = avail[self._rr:] + avail[:self._rr]
        if max_teachers:
            avail = avail[:max_teachers]
        res = []
        for name, fn in avail:
            try:
                ans = fn(messages)
                if ans and ans.strip():
                    res.append({"teacher": name, "answer": ans.strip()})
            except Exception:
                self.penalize(name)
        return res


# reasoning-eliciting prompt (R1 / CoT distillation)
_TRACE_SYS = (
    "You are an expert teacher. Think step by step and SHOW your reasoning, "
    "then end with a line 'FINAL: <concise answer>'. Be correct and complete.")


def _split_trace(text):
    """Split a teacher output into (reasoning, final)."""
    m = re.search(r"FINAL\s*:\s*(.+)\s*$", text, re.I | re.S)
    if m:
        final = m.group(1).strip()
        reasoning = text[:m.start()].strip()
        return reasoning, final
    # fallback: last non-empty line is the answer
    lines = [l for l in text.splitlines() if l.strip()]
    return ("\n".join(lines[:-1]).strip() if len(lines) > 1 else ""), (lines[-1].strip() if lines else text.strip())


def _norm(s):
    s = re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()
    nums = re.findall(r"-?\d+(?:\.\d+)?", s)
    return nums[-1] if nums else s[:80]


# ---------------- consensus: multi-teacher agreement = quality filter --------
def consensus(teacher_answers):
    """Group teachers by normalized FINAL answer. Returns dict with the winning
    trace, agreement ratio, and which teachers agreed. Kills lone hallucinations."""
    parsed = []
    for t in teacher_answers:
        reasoning, final = _split_trace(t["answer"])
        parsed.append({"teacher": t["teacher"], "reasoning": reasoning,
                       "final": final, "key": _norm(final)})
    if not parsed:
        return {"ok": False, "agreement": 0.0, "n": 0}
    keys = [p["key"] for p in parsed]
    win_key, votes = Counter(keys).most_common(1)[0]
    agree = votes / len(parsed)
    winners = [p for p in parsed if p["key"] == win_key]
    # pick the richest reasoning trace among agreeing teachers
    best = max(winners, key=lambda p: len(p["reasoning"]))
    return {
        "ok": True,
        "final": best["final"],
        "reasoning": best["reasoning"],
        "agreement": round(agree, 2),
        "n": len(parsed),
        "agreed_by": [p["teacher"] for p in winners],
        "dissent": [p["teacher"] for p in parsed if p["key"] != win_key],
    }


# ---------------- distill one question ----------------
def distill(question, intent, pool, min_agreement=0.6, lang="en"):
    """Ask the teacher panel, run consensus, return a corpus row or None."""
    msgs = [{"role": "system", "content": _TRACE_SYS},
            {"role": "user", "content": question}]
    answers = pool.ask_all(msgs)
    if not answers:
        return None, {"reason": "no_teacher"}
    con = consensus(answers)
    if not con["ok"] or con["agreement"] < min_agreement:
        return None, {"reason": "low_agreement", "agreement": con.get("agreement", 0)}
    row = to_row(question, con, intent, lang)
    return row, {"reason": "ok", "agreement": con["agreement"], "teachers": con["agreed_by"]}


def to_row(question, con, intent, lang="en"):
    """Canonical corpus row (+reasoning distillation columns).
    Base schema matches damru-knowledge; extra cols go to reasoning-traces set."""
    return {
        "question": question,
        "answer": con["final"],
        "reasoning": con["reasoning"],      # <- the distilled 'thinking'
        "intent": intent,
        "lang": lang,
        "upvotes": int(round(con["agreement"] * len(con.get("agreed_by", [])))),
        "agreement": con["agreement"],
        "teachers": ",".join(con.get("agreed_by", [])),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------- batch harvest with dedupe ----------------
def harvest(items, pool, min_agreement=0.6, seen=None):
    """items: list of {question,intent}. seen: set of question hashes (dedupe).
    Returns {rows, kept, dropped, seen}."""
    seen = seen if seen is not None else set()
    rows, dropped = [], 0
    for it in items:
        h = hashlib.sha1(it["question"].strip().lower().encode()).hexdigest()
        if h in seen:
            dropped += 1
            continue
        seen.add(h)
        row, meta = distill(it["question"], it.get("intent", "reason"), pool, min_agreement)
        if row:
            rows.append(row)
        else:
            dropped += 1
    return {"rows": rows, "kept": len(rows), "dropped": dropped, "seen": seen}


if __name__ == "__main__":
    # ---- offline self-test: 3 mock teachers, no keys / no GPU ----
    def t_good(msgs):
        q = msgs[-1]["content"]
        return f"Step 1: analyze '{q[:30]}'. Step 2: apply principle. FINAL: 42"
    def t_good2(msgs):
        return "Reasoning: think... verify... FINAL: 42"
    def t_halluc(msgs):
        return "Some rambling. FINAL: 999"          # lone dissenter -> filtered
    def t_flaky(msgs):
        raise RuntimeError("rate limited")            # triggers cooldown

    pool = TeacherPool([("claude", t_good), ("gpt", t_good2),
                        ("gemini", t_halluc), ("glm", t_flaky)])

    curr = curriculum(n_per_domain=2)
    print(f"curriculum questions: {len(curr)} across {len(DOMAINS)} domains")
    print("sample:", curr[0]["question"], "|", curr[0]["domain"])

    res = harvest(curr[:5], pool, min_agreement=0.6)
    print(f"\nharvest -> kept={res['kept']} dropped={res['dropped']}")
    if res["rows"]:
        r = res["rows"][0]
        print("row keys:", list(r.keys()))
        print("answer:", r["answer"], "| agreement:", r["agreement"],
              "| teachers:", r["teachers"], "| upvotes:", r["upvotes"])
        print("reasoning captured:", bool(r["reasoning"]))
    # dedupe check
    res2 = harvest(curr[:5], pool, seen=res["seen"])
    print(f"re-run same items -> kept={res2['kept']} dropped={res2['dropped']} (dedupe works)")
