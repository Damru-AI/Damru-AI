"""
Hindi / regional-language track for Damru.
Generates exam-grade Q&A ENTIRELY in an Indian language so Damru can help
students who study in their mother tongue. Uses the shared brain (LLM) layer.
Answers are long + comprehensive (100-150 lines, see brain.length_clause).
Crash-proof: any failure just returns fewer items; never raises to the worker loop.
"""
import random

import brain

LANG_NAMES = {
    "hi": "Hindi (\u0926\u0947\u0935\u0928\u093e\u0917\u0930\u0940 script)",
    "hinglish": "Hinglish (Hindi written in Roman/English letters, the way Indian students chat)",
    "bn": "Bengali (\u09ac\u09be\u0982\u09b2\u09be)",
    "ta": "Tamil (\u0ba4\u0bae\u0bbf\u0bb4\u0bcd)",
    "te": "Telugu (\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41)",
    "mr": "Marathi (\u092e\u0930\u093e\u0920\u0940)",
    "gu": "Gujarati (\u0a97\u0ac1\u0a9c\u0ab0\u0abe\u0aa4\u0ac0)",
    "kn": "Kannada (\u0c95\u0ca8\u0ccd\u0ca8\u0ca1)",
    "ml": "Malayalam (\u0d2e\u0d32\u0d2f\u0d3e\u0d33\u0d02)",
    "pa": "Punjabi (\u0a2a\u0a70\u0a1c\u0a3e\u0a2c\u0a40)",
}

_SYS = (
    "You are Damru, a deeply knowledgeable, patient bilingual tutor for Indian "
    "students. Think step by step with rigor and explain clearly and thoroughly."
)


def _gen_one(subject, lang):
    name = LANG_NAMES.get(lang, lang)
    user = (
        "Subject: %s.\n\n"
        "1) Pose ONE substantive, exam-relevant question or real-world problem in this subject.\n"
        "2) Solve it with clear step-by-step reasoning.\n"
        "3) %s\n"
        "4) End with a complete final answer.\n\n"
        "Write the ENTIRE question AND answer in %s. "
        "Keep standard technical terms / formulae understandable (you may keep symbols and "
        "English keywords where natural), but the explanation must be in %s.\n\n"
        "Reply ONLY as JSON: {\"question\": \"...\", \"answer\": \"...\"}"
        % (subject, brain.length_clause(), name, name)
    )
    try:
        txt = brain.chat(
            [{"role": "system", "content": _SYS}, {"role": "user", "content": user}],
            temperature=0.8,
        )
    except Exception:
        return None
    obj = brain.extract_json(txt)
    if obj and obj.get("question") and obj.get("answer"):
        return {
            "question": str(obj["question"]).strip(),
            "answer": str(obj["answer"]).strip(),
            "intent": ("hindi:" + subject)[:80],
            "lang": lang,
            "self_checked": False,
        }
    return None


def produce(subject, lang="hi", n=4):
    """Return up to n Q&A items in `lang` for `subject`."""
    out = []
    for _ in range(max(1, n)):
        it = _gen_one(subject, lang)
        if it:
            out.append(it)
    return out
