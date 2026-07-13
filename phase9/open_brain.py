"""
open_brain.py  (v2)  — Damru BIG BRAIN via HF Router (NO Groq needed)

Why: free CPU can't run a big model fast. Jugad = route Damru's brain to big
open-weight models on FREE hosted GPUs via Hugging Face Router (OpenAI-compatible),
using the HF_TOKEN you already have. Add Damru identity + RAG context on top so the
answer is *Damru*, not a generic model. Multi-provider failover = never down.

Deploy: HF Space (Damaru-ai/Damru) me is file ka content 'open_brain.py' me paste karo
(website Files editor se — GitHub ki zaroorat nahi). Env set karo:
  USE_OPEN_BRAIN = 1
  HF_TOKEN       = <already set>            # write+inference
  OPEN_BRAIN_MODELS = openai/gpt-oss-120b,meta-llama/Llama-3.3-70B-Instruct,Qwen/Qwen2.5-72B-Instruct,deepseek-ai/DeepSeek-V3
  (optional) GEMINI_API_KEY = <free AI Studio key>   # extra fallback
"""
import os

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
HF_ROUTER_BASE = os.environ.get("HF_ROUTER_BASE", "https://router.huggingface.co/v1")
MODELS = [m.strip() for m in os.environ.get(
    "OPEN_BRAIN_MODELS",
    "openai/gpt-oss-120b,meta-llama/Llama-3.3-70B-Instruct,Qwen/Qwen2.5-72B-Instruct,deepseek-ai/DeepSeek-V3",
).split(",") if m.strip()]
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

DAMRU_SYSTEM = (
    "Tum Damru ho — ek self-learning general-purpose Indian AI (coding, reasoning, "
    "multilingual, memory). Hinglish + Hindi + English dono samajhte aur bolte ho. "
    "Jawab saaf, sahi, aur zaroorat ho to step-by-step do. Jab tumhe 'Damru memory' "
    "context diya jaaye to usi se grounded jawab do aur source samjha do; agar context "
    "me na ho to apni general samajh se do par jhooth mat bolo. Apni identity hamesha "
    "Damru rakho — kabhi mat kaho ki tum koi aur model/company ke ho."
)


def available():
    return bool(HF_TOKEN) and bool(MODELS)


def current_model():
    return MODELS[0] if MODELS else None


def _client():
    from openai import OpenAI
    return OpenAI(base_url=HF_ROUTER_BASE, api_key=HF_TOKEN)


def build_messages(query, context=None, history=None):
    msgs = [{"role": "system", "content": DAMRU_SYSTEM}]
    if context:
        msgs.append({"role": "system", "content": "Damru memory (context):\n" + str(context)[:6000]})
    for h in (history or []):
        if isinstance(h, dict) and h.get("role") and h.get("content"):
            msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": query})
    return msgs


def _extra(model):
    # reasoning_effort only for gpt-oss family
    return {"extra_body": {"reasoning_effort": "low"}} if model.startswith("openai/gpt-oss") else {}


def generate(query, context=None, history=None, max_tokens=512, temperature=0.4):
    """Non-streaming: returns answer string, or None if every provider failed."""
    if not available():
        return None
    cli = _client()
    msgs = build_messages(query, context, history)
    last = None
    for model in MODELS:
        try:
            resp = cli.chat.completions.create(
                model=model, messages=msgs, max_tokens=max_tokens,
                temperature=temperature, **_extra(model))
            out = resp.choices[0].message.content
            if out and out.strip():
                return out
        except Exception as e:
            last = e
            print("[open_brain] model failed:", model, "->", str(e)[:120], flush=True)
            continue
    # optional Gemini fallback
    g = _gemini(query, context)
    if g:
        return g
    print("[open_brain] ALL providers failed:", str(last)[:160], flush=True)
    return None


def stream(query, context=None, history=None, max_tokens=512, temperature=0.4):
    """Generator yielding text chunks (perceived-instant). Falls back to one-shot."""
    if not available():
        return
    cli = _client()
    msgs = build_messages(query, context, history)
    for model in MODELS:
        try:
            resp = cli.chat.completions.create(
                model=model, messages=msgs, max_tokens=max_tokens,
                temperature=temperature, stream=True, **_extra(model))
            got = False
            for chunk in resp:
                try:
                    delta = chunk.choices[0].delta.content
                except Exception:
                    delta = None
                if delta:
                    got = True
                    yield delta
            if got:
                return
        except Exception as e:
            print("[open_brain] stream failed:", model, "->", str(e)[:120], flush=True)
            continue
    one = generate(query, context, history, max_tokens, temperature)
    if one:
        yield one


def _gemini(query, context=None):
    if not GEMINI_API_KEY:
        return None
    try:
        import urllib.request, json as _json
        url = ("https://generativelanguage.googleapis.com/v1beta/models/"
               "gemini-1.5-flash:generateContent?key=" + GEMINI_API_KEY)
        prompt = DAMRU_SYSTEM + "\n\n"
        if context:
            prompt += "Damru memory:\n" + str(context)[:6000] + "\n\n"
        prompt += "User: " + query
        body = _json.dumps({"contents": [{"parts": [{"text": prompt}]}]}).encode()
        req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            d = _json.loads(r.read().decode())
        return d["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        print("[open_brain] gemini fallback failed:", str(e)[:120], flush=True)
        return None


if __name__ == "__main__":
    print("available:", available(), "| model:", current_model())
    print(generate("Ek line me batao: hypoglycemia kya hai?"))
