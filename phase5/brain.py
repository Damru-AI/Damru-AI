"""
Damru 'brain' = MULTI-PROVIDER LLM access layer with DEEP THINKING + self-critique.

Rotates across every provider whose key is set -> OpenRouter + Groq + Google Gemini.
This spreads load so free-tier rate limits (429) almost never block the engine.
Each call round-robins the starting provider and falls back through the rest.
Uses only stdlib (urllib) so it never breaks on missing deps.
"""
import json
import re
import time
import threading
import urllib.request
import urllib.error

import config

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"

_rr_lock = threading.Lock()
_rr = 0


def _providers():
    """Build the (provider, model) rotation list from whatever keys exist."""
    out = []
    if config.OPENROUTER_API_KEY:
        for m in config.LLM_MODELS:
            out.append(("openrouter", m))
    if config.GROQ_API_KEY:
        for m in config.GROQ_MODELS:
            out.append(("groq", m))
    if config.GEMINI_API_KEY:
        for m in config.GEMINI_MODELS:
            out.append(("gemini", m))
    return out


def _http_post(url, payload, headers, timeout=120):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "ignore"))


def _call_openai_compatible(url, key, model, messages, temperature, max_tokens, referer=False):
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
    if referer:
        headers["HTTP-Referer"] = config.APP_REFERER
        headers["X-Title"] = "Damru AI Learning Engine"
    resp = _http_post(url, {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, headers)
    choices = resp.get("choices") or []
    if choices:
        txt = (choices[0].get("message") or {}).get("content", "")
        if txt and txt.strip():
            return txt.strip()
    return ""


def _call_gemini(key, model, messages, temperature, max_tokens):
    sys_txt = "\n".join(m["content"] for m in messages if m.get("role") == "system")
    contents = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            continue
        contents.append({
            "role": "model" if role == "assistant" else "user",
            "parts": [{"text": m.get("content", "")}],
        })
    payload = {
        "contents": contents,
        "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
    }
    if sys_txt:
        payload["system_instruction"] = {"parts": [{"text": sys_txt}]}
    url = GEMINI_BASE + model + ":generateContent?key=" + key
    resp = _http_post(url, payload, {"Content-Type": "application/json"})
    cands = resp.get("candidates") or []
    if cands:
        parts = ((cands[0].get("content") or {}).get("parts")) or []
        txt = "".join(p.get("text", "") for p in parts)
        if txt and txt.strip():
            return txt.strip()
    return ""


def _try_one(provider, model, messages, temperature, max_tokens):
    if provider == "openrouter":
        return _call_openai_compatible(OPENROUTER_URL, config.OPENROUTER_API_KEY, model,
                                       messages, temperature, max_tokens, referer=True)
    if provider == "groq":
        return _call_openai_compatible(GROQ_URL, config.GROQ_API_KEY, model,
                                       messages, temperature, max_tokens)
    if provider == "gemini":
        return _call_gemini(config.GEMINI_API_KEY, model, messages, temperature, max_tokens)
    return ""


def chat(messages, temperature=0.7, max_tokens=1400, models=None):
    """Call an LLM with multi-provider rotation + retries. Returns text or raises."""
    global _rr
    providers = _providers()
    if not providers:
        raise RuntimeError("No LLM provider keys configured")
    with _rr_lock:
        start = _rr % len(providers)
        _rr += 1
    order = providers[start:] + providers[:start]
    last_err = "none"
    for provider, model in order:
        for attempt in range(2):
            try:
                txt = _try_one(provider, model, messages, temperature, max_tokens)
                if txt:
                    return txt
                last_err = "%s/%s empty" % (provider, model)
                break
            except urllib.error.HTTPError as e:
                last_err = "%s http %s" % (provider, e.code)
                if e.code == 429:
                    time.sleep(1.5 * (attempt + 1))
                    continue
                if e.code in (500, 502, 503, 520, 524):
                    time.sleep(1.2 * (attempt + 1))
                    continue
                break  # auth/other -> next provider
            except Exception as e:
                last_err = "%s %s" % (provider, str(e)[:100])
                time.sleep(1.0 * (attempt + 1))
    raise RuntimeError("LLM failed (all providers): " + last_err)


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
