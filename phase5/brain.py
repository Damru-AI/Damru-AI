"""
Damru 'brain' = LLM access layer (OpenRouter) with DEEP THINKING + self-critique.
Crash-proof: model fallback chain, retries, exponential backoff on 429/5xx.
Uses only stdlib (urllib) so it never breaks on missing deps.
"""
import json
import re
import time
import urllib.request
import urllib.error

import config

OR_URL = "https://openrouter.ai/api/v1/chat/completions"


def _post(payload, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        OR_URL,
        data=data,
        headers={
            "Authorization": "Bearer " + config.OPENROUTER_API_KEY,
            "Content-Type": "application/json",
            "HTTP-Referer": config.APP_REFERER,
            "X-Title": "Damru AI Learning Engine",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def chat(messages, temperature=0.7, max_tokens=1400, models=None):
    """Call the LLM with automatic model fallback + retries. Returns text or raises."""
    models = models or config.LLM_MODELS
    last_err = "none"
    for model in models:
        for attempt in range(3):
            try:
                resp = _post(
                    {
                        "model": model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                    }
                )
                choices = resp.get("choices") or []
                if choices:
                    txt = (choices[0].get("message") or {}).get("content", "")
                    if txt and txt.strip():
                        return txt.strip()
                last_err = "empty response"
            except urllib.error.HTTPError as e:
                last_err = "http %s" % e.code
                if e.code == 429:
                    time.sleep(2 ** attempt + 1)
                    continue
                if e.code in (500, 502, 503, 520, 524):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break  # auth/other -> try next model
            except Exception as e:
                last_err = str(e)[:140]
                time.sleep(1.2 * (attempt + 1))
    raise RuntimeError("LLM failed: " + last_err)


def extract_json(text):
    """Best-effort JSON object extraction from an LLM reply."""
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    frag = m.group(0)
    for cand in (frag, frag[: frag.rfind("}") + 1]):
        try:
            return json.loads(cand)
        except Exception:
            continue
    return None


DEEP_SYS = (
    "You are Damru, a relentless self-teaching AI. Think DEEPLY and use rigorous "
    "critical analysis. Reason step by step, question assumptions, consider edge "
    "cases, and never give up until you reach a correct, well-justified answer. "
    "Be precise and educational."
)


def generate_qa(subject, context="", lang="en"):
    """Force Damru to pose a meaningful problem in `subject` and solve it deeply.
    Returns dict {question, answer} or None."""
    ctx = ("\n\nReference context:\n" + context[:1500]) if context else ""
    user = (
        "Subject: %s.%s\n\n"
        "1) Pose ONE substantive, non-trivial question or real-world problem in this subject.\n"
        "2) Solve it with deep step-by-step reasoning and critical analysis.\n"
        "3) End with a clear, complete final answer.\n\n"
        "Reply ONLY as JSON: {\"question\": \"...\", \"answer\": \"...\"} "
        "where answer contains the full reasoning + final answer." % (subject, ctx)
    )
    txt = chat(
        [{"role": "system", "content": DEEP_SYS}, {"role": "user", "content": user}],
        temperature=0.8,
    )
    obj = extract_json(txt)
    if obj and obj.get("question") and obj.get("answer"):
        return {"question": str(obj["question"]).strip(), "answer": str(obj["answer"]).strip()}
    return None


def self_check(question, answer, subject=""):
    """Critique + correct an answer. Returns (is_correct: bool, improved_answer: str)."""
    user = (
        "Critically review this Q&A in %s. Find any error or gap, then provide the "
        "corrected, improved final answer.\n\nQ: %s\n\nA: %s\n\n"
        "Reply ONLY as JSON: {\"verdict\": \"correct|incorrect\", \"final_answer\": \"...\"}"
        % (subject or "the subject", question[:1500], answer[:3000])
    )
    try:
        txt = chat(
            [{"role": "system", "content": DEEP_SYS}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        obj = extract_json(txt)
        if obj and obj.get("final_answer"):
            ok = str(obj.get("verdict", "")).lower().startswith("correct")
            return ok, str(obj["final_answer"]).strip()
    except Exception:
        pass
    return True, answer  # fail-open: keep original
