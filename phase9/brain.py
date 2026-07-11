"""
Damru 'brain' = MULTI-PROVIDER LLM access layer with DEEP THINKING + self-critique.

Rotates across every provider whose key is set -> OpenRouter + Groq + Google Gemini.
This spreads load so free-tier rate limits (429) almost never block the engine.
Each call round-robins the starting provider and falls back through the rest.
Uses only stdlib (urllib) so it never breaks on missing deps.

DEPTH: answers are forced to be long, comprehensive explanations of roughly
config.ANSWER_MIN_LINES..ANSWER_MAX_LINES lines, and each new pass over a subject
(`depth`) pushes progressively more advanced material so topics get covered fast.
"""
import json
import re
import time
import threading
import urllib.request
import urllib.error

import config
import os
import sys
from pathlib import Path

_PHASE9 = Path(__file__).resolve().parents[1] / "phase9"
if str(_PHASE9) not in sys.path:
    sys.path.insert(0, str(_PHASE9))
try:
    from open_brain import OpenBrain
    _OPEN_BRAIN = OpenBrain()
except Exception:
    _OPEN_BRAIN = None
ENABLE_LEGACY_PROVIDERS = os.environ.get("ENABLE_LEGACY_PROVIDERS", "0") == "1"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"

_rr_lock = threading.Lock()
_rr = 0


def _providers():
    """Use gpt-oss Open Brain first; stale legacy providers only when explicitly enabled."""
    out = []
    if _OPEN_BRAIN is not None and _OPEN_BRAIN.available:
        out.append(("openbrain", "gpt-oss-router"))
    if ENABLE_LEGACY_PROVIDERS:
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


def _http_post(url, payload, headers, timeout=180):
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
    if provider == "openbrain":
        if _OPEN_BRAIN is None:
            return ""
        return _OPEN_BRAIN.complete(messages, max_tokens=max_tokens,
                                    temperature=temperature)["content"]
    if provider == "openrouter":
        return _call_openai_compatible(OPENROUTER_URL, config.OPENROUTER_API_KEY, model,
                                       messages, temperature, max_tokens, referer=True)
    if provider == "groq":
        return _call_openai_compatible(GROQ_URL, config.GROQ_API_KEY, model,
                                       messages, temperature, max_tokens)
    if provider == "gemini":
        return _call_gemini(config.GEMINI_API_KEY, model, messages, temperature, max_tokens)
    return ""


def chat(messages, temperature=0.7, max_tokens=None, models=None):
    """Call an LLM with multi-provider rotation + retries. Returns text or raises."""
    global _rr
    if max_tokens is None:
        max_tokens = config.ANSWER_MAX_TOKENS
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
    "You are Damru, a relentless self-teaching AI and master teacher. Think DEEPLY "
    "and use rigorous critical analysis. Reason step by step, question assumptions, "
    "consider edge cases, and never give up until you reach a correct, well-justified "
    "answer. Your explanations are long, structured, and genuinely educational."
)


def length_clause():
    """Shared instruction that forces long, comprehensive ~100-150 line answers."""
    return (
        "Write a COMPREHENSIVE, in-depth answer of roughly %d-%d lines. Structure it "
        "with: (a) the core idea/intuition, (b) full step-by-step derivation or reasoning, "
        "(c) at least one fully worked example, (d) important edge cases and common "
        "mistakes, and (e) a short final summary / key takeaways. Be thorough and detailed, "
        "but every line must add real value (no filler or repetition)."
        % (config.ANSWER_MIN_LINES, config.ANSWER_MAX_LINES)
    )


def _depth_clause(depth):
    if not depth or depth <= 0:
        return ("This is the FIRST pass on this subject: cover a fundamental but "
                "non-trivial question that builds a strong base.")
    return (
        "This is pass #%d on this subject. Go DEEPER than before: choose a more "
        "advanced, less obvious question (advanced sub-topics, harder problems, "
        "real research/applied angles) and AVOID basic introductory questions already "
        "likely covered in earlier passes." % (depth + 1)
    )


def generate_qa(subject, context="", lang="en", depth=0):
    """Force Damru to pose a meaningful problem in `subject` and solve it deeply.
    `depth` = how many full rotation batches this subject already completed, used to
    push progressively more advanced content. Returns dict {question, answer} or None."""
    ctx = ("\n\nReference context:\n" + context[:1500]) if context else ""
    user = (
        "Subject: %s.\n%s%s\n\n"
        "1) Pose ONE substantive, non-trivial question or real-world problem in this subject.\n"
        "2) Solve it with deep step-by-step reasoning and critical analysis.\n"
        "3) %s\n"
        "4) End with a clear, complete final answer.\n\n"
        "Reply ONLY as JSON: {\"question\": \"...\", \"answer\": \"...\"} "
        "where answer contains the full reasoning + final answer."
        % (subject, _depth_clause(depth), ctx, length_clause())
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
    """Critique + correct an answer. Returns (is_correct: bool, improved_answer: str).
    Preserves the full depth/length of the answer (never shortens a good answer)."""
    user = (
        "Critically review this Q&A in %s. Find any error or gap, then provide the "
        "corrected, improved final answer. Keep it just as detailed and long (%d-%d lines): "
        "preserve all correct reasoning, fix mistakes, and fill any gaps rather than shortening.\n\n"
        "Q: %s\n\nA: %s\n\n"
        "Reply ONLY as JSON: {\"verdict\": \"correct|incorrect\", \"final_answer\": \"...\"}"
        % (subject or "the subject", config.ANSWER_MIN_LINES, config.ANSWER_MAX_LINES,
           question[:1500], answer[:8000])
    )
    try:
        txt = chat(
            [{"role": "system", "content": DEEP_SYS}, {"role": "user", "content": user}],
            temperature=0.2,
        )
        obj = extract_json(txt)
        if obj and obj.get("final_answer"):
            ok = str(obj.get("verdict", "")).lower().startswith("correct")
            improved = str(obj["final_answer"]).strip()
            # Never let self-check shrink a good long answer into a stub.
            if len(improved) < 0.6 * len(answer or ""):
                return ok, answer
            return ok, improved
    except Exception:
        pass
    return True, answer  # fail-open: keep original
