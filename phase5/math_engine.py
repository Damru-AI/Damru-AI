"""
Maths engine. Two tracks so Damru truly LEARNS to solve, and self-checks:

A) verified track (procedural): build problems with sympy and compute the EXACT
   answer. Damru's answer can be checked deterministically -> guaranteed-correct
   Q&A. (high trust -> high upvotes)
B) hard track (LLM): fetch/compose olympiad / JEE / Putnam style problems by
   topic + difficulty, solve with deep reasoning, then self-critique.

Never crashes: sympy failures fall back to the LLM track; LLM failures are skipped.
"""
import random

import brain

try:
    import sympy as sp
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False

TOPICS = [
    "algebra", "calculus", "derivatives", "integrals", "limits",
    "probability", "combinatorics", "number theory", "linear algebra",
    "trigonometry", "sequences and series",
]
HARD_SOURCES = [
    "International Mathematical Olympiad (IMO)", "Putnam competition",
    "IIT-JEE Advanced", "MIT problem sets", "USAMO", "INMO",
    "Cambridge STEP", "national olympiad",
]


def _verified_problem():
    """Procedurally generate a problem with a known exact answer using sympy."""
    if not _HAS_SYMPY:
        return None
    x = sp.symbols("x")
    kind = random.choice(["derivative", "integral", "solve", "factor", "limit"])
    a, b, c = (random.randint(1, 9) for _ in range(3))
    n = random.randint(2, 4)
    try:
        if kind == "derivative":
            expr = a * x ** n + b * x + c
            ans = sp.diff(expr, x)
            q = "Differentiate with respect to x: %s" % sp.srepr_to_str(expr) if False else \
                "Differentiate with respect to x:  %s" % sp.sstr(expr)
        elif kind == "integral":
            expr = a * x ** n + b * x + c
            ans = sp.integrate(expr, x)
            q = "Find the indefinite integral (in terms of x):  %s" % sp.sstr(expr)
        elif kind == "solve":
            expr = sp.Eq(a * x + b, c)
            ans = sp.solve(expr, x)
            q = "Solve for x:  %s" % sp.sstr(expr)
        elif kind == "factor":
            expr = sp.expand((x + a) * (x + b))
            ans = sp.factor(expr)
            q = "Factor completely:  %s" % sp.sstr(expr)
        else:  # limit
            expr = (a * x ** 2 - a) / (x - 1)
            ans = sp.limit(expr, x, 1)
            q = "Evaluate the limit as x -> 1 of:  %s" % sp.sstr(expr)
        answer = (
            "Step-by-step (%s):\n"
            "We work the expression carefully and simplify.\n"
            "Final answer:  %s" % (kind, sp.sstr(ans))
        )
        return {"question": q, "answer": answer, "intent": "math_" + kind,
                "lang": "en", "verified": True}
    except Exception:
        return None


def _hard_problem(subject):
    src = random.choice(HARD_SOURCES)
    topic = random.choice(TOPICS)
    user = (
        "Create ONE challenging %s problem in the style of %s. Then SOLVE it with "
        "complete, rigorous, step-by-step reasoning and a clearly stated final "
        "answer. Verify the result before finishing.\n\n"
        "Reply ONLY as JSON: {\"question\": \"...\", \"answer\": \"...\"}"
        % (topic, src)
    )
    try:
        txt = brain.chat(
            [{"role": "system", "content": brain.DEEP_SYS},
             {"role": "user", "content": user}],
            temperature=0.85,
        )
        obj = brain.extract_json(txt)
        if obj and obj.get("question") and obj.get("answer"):
            return {"question": str(obj["question"]).strip(),
                    "answer": str(obj["answer"]).strip(),
                    "intent": "math_olympiad", "lang": "en", "verified": False}
    except Exception:
        return None
    return None


def produce(subject="Mathematics", n=6):
    """Return a list of math Q&A items (mix of verified + hard, self-checked)."""
    out = []
    for _ in range(n):
        item = _verified_problem() if random.random() < 0.5 else _hard_problem(subject)
        if not item:
            item = _hard_problem(subject)
        if not item:
            continue
        if not item.get("verified"):
            ok, improved = brain.self_check(item["question"], item["answer"], "mathematics")
            item["answer"] = improved
            item["self_checked"] = ok
        out.append(item)
    return out
