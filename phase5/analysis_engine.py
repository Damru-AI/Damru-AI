"""
Deep-analysis engine: forces Damru to THINK and solve.
Takes the current subject (+ optional real context from Wikipedia) and makes
Damru pose a substantive problem, solve it with critical analysis, then
self-check/correct it. This is the 'reasoning depth' source.

Answers are long (100-150 lines, see brain.length_clause) and get DEEPER each
rotation pass over the subject (via curriculum.depth_of).
"""
import brain
import curriculum
from sources import wikipedia


def produce(subject, n=4):
    out = []
    depth = 0
    try:
        depth = curriculum.depth_of(subject)
    except Exception:
        depth = 0
    # Pull a little real context so problems stay grounded (best-effort).
    context_pool = []
    try:
        titles = wikipedia.search_titles(subject, 6)
        for t, ex in wikipedia.extracts(titles).items():
            if ex:
                context_pool.append(ex)
    except Exception:
        context_pool = []
    for i in range(n):
        ctx = context_pool[i % len(context_pool)] if context_pool else ""
        try:
            qa = brain.generate_qa(subject, ctx, depth=depth)
            if not qa:
                continue
            ok, improved = brain.self_check(qa["question"], qa["answer"], subject)
            out.append({
                "question": qa["question"],
                "answer": improved,
                "intent": subject.lower().replace(" ", "_"),
                "lang": "en",
                "self_checked": ok,
            })
        except Exception:
            continue
    return out
