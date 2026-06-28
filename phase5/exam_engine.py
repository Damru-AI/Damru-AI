"""
Exam-syllabus engine: generates syllabus-aligned exam-style Q&A with full worked
solutions for JEE / NEET / UPSC / NCERT / SSC-Banking. A share is produced in Hindi.
Crash-proof: returns fewer items on failure; never raises to the worker loop.
"""
import random

import config
import brain
import exam_syllabus

_SYS = (
    "You are Damru, an expert exam coach for Indian competitive and board exams. "
    "You create high-quality, syllabus-accurate questions and teach the full solution "
    "step by step like a top teacher."
)


def _gen_one():
    exam, subject, topic = exam_syllabus.pick()
    in_hindi = random.random() < config.EXAM_HINDI_RATIO
    lang = "hi" if in_hindi else "en"
    lang_line = (
        "Write the ENTIRE question and solution in Hindi (Devanagari), keeping formulae/symbols as needed."
        if in_hindi
        else "Write in clear English."
    )
    style = (
        "Make it a descriptive/analytical question with a model answer"
        if exam == "UPSC"
        else "Make it a numerical/conceptual problem with a fully worked step-by-step solution and the final answer"
    )
    user = (
        "Exam: %s | Subject: %s | Topic: %s.\n\n"
        "Create ONE original, exam-standard question on this exact topic. %s. "
        "%s End with a clearly marked final answer.\n\n"
        "Reply ONLY as JSON: {\"question\": \"...\", \"answer\": \"...\"}"
        % (exam, subject, topic, style, lang_line)
    )
    try:
        txt = brain.chat(
            [{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
            temperature=0.75,
        )
    except Exception:
        return None
    obj = brain.extract_json(txt)
    if not (obj and obj.get("question") and obj.get("answer")):
        return None
    return {
        "question": str(obj["question"]).strip(),
        "answer": str(obj["answer"]).strip(),
        "intent": ("exam:%s:%s" % (exam, subject))[:80],
        "lang": lang,
        "self_checked": False,
    }


def produce(n=4):
    out = []
    for _ in range(max(1, n)):
        it = _gen_one()
        if it:
            out.append(it)
    return out
