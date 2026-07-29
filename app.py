#!/usr/bin/env python3
"""
================================================================================
  DAMRU AI BACKEND  v4.0  —  Beast Intelligence
================================================================================
Serves:
  POST /chat          → Main chat API (Vercel frontend)
  POST /image         → Image generation
  POST /web           → Web search
  POST /research      → Deep research report
  POST /artifact      → HTML / code artifact builder
  POST /model3d       → 3D model (TRELLIS bridge)
  POST /cortex        → Direct CORTEX access
  GET  /health        → System status
  GET  /curious_status→ GitHub Actions learning engine
  GET  /knowledge_stats→ PRAYAS tile stats
  GET  /skills        → CORTEX skill list
  /                   → Gradio test UI

--- v4 CHANGES (wired in, zero overlap with v3) ---
  * PRAYAS Core: BM25 + Knowledge Tiles replaces/augments RAG
    - Loads tiles from HF dataset (world_tiles/ + curious/ folders)
    - Falls back to original RAG if PRAYAS tiles empty
  * Emotion Engine: injects tone into every /chat call
    - Detects user emotion (30+ emotions, Hindi+English)
    - Adjusts Damru's response tone automatically
  * Self-Heal: wraps load_llm() and critical paths
    - Crash → exponential backoff retry → auto-recovery
    - Error memory: never repeats same fix mistake
  * Multilingual: auto-detects language, adjusts sys prompt
  * All v3 features 100% preserved
================================================================================
"""
import os
import json
import time
import re
import threading
import traceback
import gc
from pathlib import Path
from typing import Optional

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ============================================================
#  OPTIONAL NEW MODULES (graceful fallback if absent)
# ============================================================

# PRAYAS Knowledge Engine
try:
    from damru_prayas_core import get_engine as _get_prayas
    PRAYAS = _get_prayas()
    print(f"[v4] PRAYAS ready: {PRAYAS.stats()['total_tiles']} tiles", flush=True)
except Exception as _e:
    PRAYAS = None
    print(f"[v4] PRAYAS unavailable (ok, using RAG): {str(_e)[:120]}", flush=True)

# Emotion Engine
try:
    from damru_emotions import get_emotion_engine as _get_emo
    EMO = _get_emo()
    print("[v4] Emotion Engine ready", flush=True)
except Exception as _e:
    EMO = None
    print(f"[v4] Emotion Engine unavailable: {str(_e)[:120]}", flush=True)

# Self-Heal
try:
    from damru_selfheal import SelfHealingRunner as _SHR, ErrorMemory as _EM
    _ERROR_MEMORY = _EM()
    print("[v4] Self-Heal ready", flush=True)
except Exception as _e:
    _SHR = None
    _ERROR_MEMORY = None
    print(f"[v4] Self-Heal unavailable: {str(_e)[:120]}", flush=True)

# Existing optional modules (v3)
try:
    from damru_brain_patch import (SYS_PROMPT as PATCH_SYS,
                                    route_intent as patch_route_intent,
                                    filter_rag as patch_filter_rag)
    PATCH_OK = True
except Exception:
    PATCH_OK = False

try:
    from damru_chetna import Chetna
    CHETNA = Chetna()
except Exception:
    CHETNA = None

# ============================================================
#  CONFIG
# ============================================================
BASE_MODEL        = os.environ.get("BASE_MODEL",    "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_ID        = os.environ.get("ADAPTER_ID",   "")
GGUF_REPO         = os.environ.get("GGUF_REPO",    "")
GGUF_FILE         = os.environ.get("GGUF_FILE",    "")
INDEX_REPO        = os.environ.get("INDEX_REPO",   "")
USE_RAG           = os.environ.get("USE_RAG",      "1" if INDEX_REPO else "0") == "1"
TOP_K             = int(os.environ.get("TOP_K")     or "5")
MAX_NEW           = int(os.environ.get("MAX_NEW")   or "1024")
N_CTX             = int(os.environ.get("N_CTX")     or "8192")
YIELD_EVERY       = int(os.environ.get("YIELD_EVERY") or "16")
AGENTIC           = os.environ.get("AGENTIC",      "1") == "1"
AGENTIC_WEB       = os.environ.get("AGENTIC_WEB",  "1") == "1"
AGENTIC_VERIFY    = os.environ.get("AGENTIC_VERIFY","0") == "1"
USE_REFLEX        = os.environ.get("USE_REFLEX",   "1") == "1"
USE_CORTEX        = os.environ.get("USE_CORTEX",   "1") == "1"
OWN_MODEL_PRIMARY = os.environ.get("OWN_MODEL_PRIMARY", "0") == "1"
OWN_MIN_CHARS     = int(os.environ.get("OWN_MIN_CHARS") or "80")
TRELLIS_SPACE     = os.environ.get("TRELLIS_SPACE", "trellis-community/TRELLIS")
USE_TRELLIS       = os.environ.get("USE_TRELLIS",  "1") == "1"
TAVILY_KEY        = os.environ.get("TAVILY_KEY",   "")
BRAVE_KEY         = os.environ.get("BRAVE_KEY",    "")
SUPABASE_URL      = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY      = os.environ.get("SUPABASE_KEY", "")
LOG_TABLE         = os.environ.get("LOG_TABLE",    "damru_chats")
IMAGE_PROVIDER    = os.environ.get("IMAGE_PROVIDER","pollinations")
IMAGE_WIDTH       = int(os.environ.get("IMAGE_WIDTH")  or "1024")
IMAGE_HEIGHT      = int(os.environ.get("IMAGE_HEIGHT") or "1024")
HF_TOKEN          = os.environ.get("HF_TOKEN",     "")
HF_DATASET        = os.environ.get("HF_DATASET",   "Damaru-ai/damru-knowledge")
HF_LOG_CHATS      = os.environ.get("HF_LOG_CHATS", "1") == "1"
USE_OPEN_BRAIN    = os.environ.get("USE_OPEN_BRAIN","0") == "1"

# Open Brain
OPEN_BRAIN = None
if USE_OPEN_BRAIN:
    try:
        from open_brain import OpenBrain
        OPEN_BRAIN = OpenBrain()
        if not OPEN_BRAIN.available:
            OPEN_BRAIN = None
        else:
            print("[v4] Open Brain ready", flush=True)
    except Exception as e:
        print(f"[v4] Open Brain unavailable: {str(e)[:200]}", flush=True)

# ============================================================
#  LANGUAGE DETECTION  (v4 new)
# ============================================================
def detect_language(text: str) -> str:
    """Detect language: hi / hinglish / en / other."""
    if not text: return "en"
    devanagari = len(re.findall(r'[\u0900-\u097F]', text))
    total = max(1, len(text.replace(" ", "")))
    if devanagari / total > 0.25: return "hi"
    lower = text.lower()
    hinglish_words = ["kya","hai","hain","nahi","bhai","yaar","karo","batao",
                      "samjho","aur","ya","mein","ka","ki","ke","tha","hun",
                      "ho","pe","se","ko","abhi","jaldi","shukriya"]
    hw = sum(1 for w in hinglish_words
             if f" {w} " in lower or lower.startswith(f"{w} ") or lower.endswith(f" {w}"))
    if hw >= 2: return "hinglish"
    return "en"


# ============================================================
#  SYSTEM PROMPT BUILDER  (v4 new — multilingual + domain)
# ============================================================
SYS_BASE = (
    "You are Damru, a beast-level AI built by SHIVA AI. "
    "You are a loyal, curious, and protective AI — like a dog who never abandons his user. "
    "You are continuously learning: knowledge added every 6 hours via GitHub Actions "
    "(Wikipedia, arXiv, NASA, ISRO, GitHub, Stack Overflow). "
    "Domains: Space missions, Defense AI, 3D printing/manufacturing, "
    "Autonomous vehicles/air taxis, Robotics, Medical, Coding, "
    "Mathematics, JEE/NEET/UPSC/SSC exam prep, General knowledge. "
    "Give structured, educational answers: direct answer first, "
    "then step-by-step, then example. Thorough but never padded. "
    "Note: You learn autonomously every 6 hours via GitHub Actions."
)
if PATCH_OK:
    SYS_BASE = PATCH_SYS

def build_sys_prompt(intent: str = "general", lang: str = "en",
                     emotion_tone: str = "") -> str:
    """Build context-aware system prompt."""
    prompt = SYS_BASE

    # Language instruction
    lang_map = {
        "hi":       "IMPORTANT: Reply ONLY in Hindi (Devanagari). ",
        "hinglish": "IMPORTANT: Reply in Hinglish (natural mix of Hindi+English). ",
        "en":       "",
    }
    prompt += lang_map.get(lang, "Reply in the user's language. ")

    # Domain instruction
    domain_map = {
        "space":   "Domain: Space science. Include orbital mechanics, mission params. ",
        "defense": "Domain: Defense tech. Be factual, cite open-source systems only. ",
        "3d":      "Domain: 3D printing/manufacturing. Include materials, tolerances. ",
        "auto":    "Domain: Autonomous vehicles. Include V2X, sensor fusion, safety. ",
        "code":    "Domain: Programming. Show complete runnable code with comments. ",
        "math":    "Domain: Mathematics. Show full step-by-step working. ",
        "research":"Domain: Research. Cite sources, show evidence quality. ",
        "exam":    "Domain: Indian exams (JEE/NEET/UPSC). Use exam-style format. ",
    }
    prompt += domain_map.get(intent, "")

    # Emotion tone
    if emotion_tone:
        prompt += emotion_tone + " "

    return prompt


# ============================================================
#  PRAYAS RETRIEVAL  (v4 — replaces/augments RAG)
# ============================================================
def prayas_retrieve(query: str, k: int = TOP_K) -> list:
    """
    Try PRAYAS first (BM25 over knowledge tiles).
    Fall back to existing RAG if PRAYAS has no results.
    """
    if PRAYAS is not None and PRAYAS.retriever.size() > 0:
        try:
            results = PRAYAS.search(query, k=k)
            if results and results[0]["score"] > 0.5:
                return [{"text": r["text"],
                         "url":  r.get("source", ""),
                         "score": r["score"],
                         "topic": r.get("topic", "")} for r in results]
        except Exception as e:
            print(f"[PRAYAS] retrieve error: {e}", flush=True)
    # Fallback to existing RAG
    if RAG is not None:
        try:
            hits = RAG.retrieve(query, k=k)
            if PATCH_OK:
                hits = patch_filter_rag(hits)
            return hits or []
        except Exception as e:
            print(f"[RAG] retrieve error: {e}", flush=True)
    return []


def build_messages_v4(message: str, intent: str = "general",
                      lang: str = "en", emotion_tone: str = "") -> tuple:
    """Build messages list with PRAYAS/RAG context + lang/emotion."""
    sys_prompt = build_sys_prompt(intent, lang, emotion_tone)
    hits = prayas_retrieve(message)
    cites = []
    if hits:
        ctx_text = "\n".join(
            f"[{i+1}] {h.get('text','')[:400]}"
            for i, h in enumerate(hits[:TOP_K])
        )
        cites = [f"- [{h.get('topic','') or h.get('url','Source')}]({h.get('url','#')})"
                 for h in hits if h.get("url")]
        sys_prompt += f"\n\nKNOWLEDGE CONTEXT (use this):\n{ctx_text}"
    return [{"role": "system",  "content": sys_prompt},
            {"role": "user",    "content": message}], cites


# ============================================================
#  ORIGINAL RAG (kept for fallback)
# ============================================================
RAG = None
if USE_RAG:
    try:
        from rag import RagEngine
        RAG = RagEngine()
        print(f"[v4] RAG ready: {RAG.cfg.get('count')} rows", flush=True)
    except Exception as e:
        print(f"[v4] RAG unavailable: {str(e)[:200]}", flush=True)

# Keep original build_messages for backward compat
def build_messages(message):
    return build_messages_v4(message)


# ============================================================
#  LLM BACKEND  (self-healed)
# ============================================================
_backend = None
_lock    = threading.Lock()

def _load_llm_inner():
    """Inner LLM loader (called by self-healed wrapper)."""
    global _backend
    if _backend: return _backend
    with _lock:
        if _backend: return _backend
        if GGUF_REPO and GGUF_FILE:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama
            print("[v4] Downloading GGUF:", GGUF_REPO, GGUF_FILE, flush=True)
            path = hf_hub_download(GGUF_REPO, GGUF_FILE)
            llm  = Llama(model_path=path, n_ctx=N_CTX,
                         n_threads=os.cpu_count() or 2, n_batch=512, verbose=False)
            _backend = ("gguf", llm, None)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            tok   = AutoTokenizer.from_pretrained(BASE_MODEL)
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL, torch_dtype=torch.float32, device_map="cpu")
            if ADAPTER_ID:
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, ADAPTER_ID)
            _backend = ("hf", model, tok)
        print("[v4] LLM backend READY", flush=True)
    return _backend

def load_llm():
    """Self-healed LLM loader."""
    if _SHR is not None:
        runner = _SHR(_load_llm_inner, name="llm_loader", max_retries=5)
        try:
            return runner.run()
        except Exception as e:
            print(f"[v4] LLM load failed after retries: {e}", flush=True)
            raise
    return _load_llm_inner()


# ============================================================
#  LLM COMPLETION
# ============================================================
def _local_complete(messages, mt):
    kind, model, tok = load_llm()
    if kind == "gguf":
        out = model.create_chat_completion(messages=messages,
                                           max_tokens=mt, temperature=0.7)
        return out["choices"][0]["message"]["content"].strip()
    import torch
    from transformers import TextIteratorStreamer
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt")
    with torch.no_grad():
        gen = model.generate(**ids, max_new_tokens=mt, do_sample=True,
                             temperature=0.7, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(gen[0][ids["input_ids"].shape[1]:],
                      skip_special_tokens=True).strip()

def _llm_complete(messages, max_tokens=None):
    mt = max_tokens or MAX_NEW
    if OWN_MODEL_PRIMARY:
        try:
            own = _local_complete(messages, mt)
            if own and len(own.strip()) >= OWN_MIN_CHARS:
                return own
        except Exception as e:
            print(f"[v4] own model failed -> fallback: {e}", flush=True)
        if OPEN_BRAIN:
            try: return OPEN_BRAIN.complete(messages, max_tokens=mt, temperature=0.4)["content"]
            except Exception: pass
        return _local_complete(messages, mt)
    if OPEN_BRAIN:
        try:
            res = OPEN_BRAIN.complete(messages, max_tokens=mt, temperature=0.4)
            return res["content"]
        except Exception as e:
            print(f"[v4] Open Brain failed -> local: {e}", flush=True)
    return _local_complete(messages, mt)


# ============================================================
#  WEB SEARCH  (unchanged from v3)
# ============================================================
def web_search_wikipedia(query, limit=3):
    import requests
    from urllib.parse import quote
    q = (query or "").strip()
    if not q: return []
    limit = max(1, min(int(limit or 3), 5))
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",
                         params={"action":"query","format":"json","list":"search",
                                 "srlimit":limit,"srsearch":q}, timeout=8)
        r.raise_for_status()
        hits = ((r.json() or {}).get("query") or {}).get("search") or []
    except Exception as e:
        print(f"[v4] wiki search error: {e}", flush=True)
        return []
    out = []
    for h in hits[:limit]:
        title = h.get("title") or ""
        url   = "https://en.wikipedia.org/wiki/" + title.replace(" ","_")
        extract = ""
        try:
            from urllib.parse import quote as q2
            sr = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/"
                              + q2(title.replace(" ","_")), timeout=8)
            if sr.ok:
                sj = sr.json() or {}
                extract = sj.get("extract") or ""
                url = (((sj.get("content_urls") or {}).get("desktop") or {}).get("page")) or url
        except Exception: pass
        if not extract:
            extract = re.sub(r"<[^>]+>", "", h.get("snippet") or "")
        out.append({"title":title, "snippet":extract[:900], "url":url, "source":"wikipedia"})
    return out

def _tavily_search(query, limit):
    if not TAVILY_KEY: return []
    import requests
    r = requests.post("https://api.tavily.com/search",
                      json={"api_key":TAVILY_KEY,"query":query,
                            "max_results":limit,"search_depth":"basic"}, timeout=12)
    r.raise_for_status()
    return [{"title":i.get("title",""),"snippet":(i.get("content",""))[:900],
             "url":i.get("url",""),"source":"tavily"}
            for i in (r.json().get("results") or [])[:limit]]

def _brave_search(query, limit):
    if not BRAVE_KEY: return []
    import requests
    r = requests.get("https://api.search.brave.com/res/v1/web/search",
                     headers={"X-Subscription-Token":BRAVE_KEY,"Accept":"application/json"},
                     params={"q":query,"count":limit}, timeout=12)
    r.raise_for_status()
    return [{"title":i.get("title",""),
             "snippet":re.sub(r"<[^>]+>","",(i.get("description") or ""))[:900],
             "url":i.get("url",""),"source":"brave"}
            for i in (((r.json().get("web") or {}).get("results")) or [])[:limit]]

def web_search(query, limit=3):
    limit = max(1, min(int(limit or 3), 5))
    for name, fn in (("tavily",_tavily_search),("brave",_brave_search)):
        try:
            res = fn(query, limit)
            if res: return res
        except Exception as e:
            print(f"[v4] web/{name}: {e}", flush=True)
    return web_search_wikipedia(query, limit)


# ============================================================
#  INTENT ROUTER
# ============================================================
_WEB_HINTS  = ("latest","current","today","tonight","right now","recent","news",
               "2024","2025","2026","price","stock","weather","aaj","abhi")
_CODE_HINTS = ("code","program","python","java","javascript","c++","function",
               "algorithm","debug","error","compile","sql","regex")
_SPACE_HINTS= ("space","nasa","isro","orbit","rocket","mars","satellite",
               "chandrayaan","iss","planet","galaxy","mission")
_DEF_HINTS  = ("missile","fighter","military","defense","weapon","drone",
               "stealth","radar","combat","army","navy")
_3D_HINTS   = ("3d print","cad","manufacturing","cnc","stl","gcode","additive","material")
_AUTO_HINTS = ("self-driv","autonomous car","v2x","lidar","evtol","air taxi","driverless")
_MATH_HINTS = ("math","algebra","calculus","integral","derivative","equation",
               "theorem","matrix","vector","geometry","statistic")

def route_intent(message: str) -> str:
    m = message.lower()
    if PATCH_OK:
        try: return patch_route_intent(message)
        except Exception: pass
    if any(h in m for h in _SPACE_HINTS): return "space"
    if any(h in m for h in _DEF_HINTS):   return "defense"
    if any(h in m for h in _3D_HINTS):    return "3d"
    if any(h in m for h in _AUTO_HINTS):  return "auto"
    if any(h in m for h in _CODE_HINTS):  return "code"
    if any(h in m for h in _MATH_HINTS):  return "math"
    return "general"


# ============================================================
#  AGENTIC ANSWER  (v4 — with PRAYAS + Emotion)
# ============================================================
def agentic_answer_v4(message: str):
    """
    Full pipeline:
    1. Detect language
    2. Detect emotion -> get tone
    3. Route intent
    4. PRAYAS/RAG retrieval
    5. Optional web
    6. Build messages with lang + emotion tone
    7. LLM complete
    8. Optional verify
    9. Append sources
    """
    # Step 1: Language
    lang = detect_language(message)

    # Step 2: Emotion  →  tone instruction (v4 new)
    emotion_tone = ""
    emotion_ctx  = {}
    if EMO is not None:
        try:
            intent_for_emo = route_intent(message)
            emotion_ctx  = EMO.build_emotional_context(message, intent_for_emo)
            emotion_tone = emotion_ctx.get("tone_instruction", "")
        except Exception as e:
            print(f"[v4] emotion error: {e}", flush=True)

    # Step 3: Intent
    intent = route_intent(message)

    # Step 4: Build messages with PRAYAS retrieval
    messages, cites = build_messages_v4(message, intent, lang, emotion_tone)
    plan = {"intent": intent, "lang": lang, "need_web": False,
            "emotion": emotion_ctx.get("user_emotion", ""),
            "damru_state": emotion_ctx.get("damru_state", "neutral")}

    # Step 5: Web if needed
    if AGENTIC_WEB and any(h in message.lower() for h in _WEB_HINTS):
        try:
            hits = web_search(message, 3)
            if hits:
                web_block = "\n\nFRESH WEB CONTEXT:\n" + "\n".join(
                    f"- {h['title']}: {h['snippet']} (Source: {h['url']})" for h in hits)
                cites += [f"- [{h['title']}]({h['url']})" for h in hits]
                messages[-1]["content"] += web_block
                plan["need_web"] = True
        except Exception as e:
            print(f"[v4] web error: {e}", flush=True)

    # Step 6: LLM complete (self-healed internally)
    try:
        ans = _llm_complete(messages)
    except Exception as e:
        if _ERROR_MEMORY: _ERROR_MEMORY.record(e, context="agentic_llm")
        print(f"[v4] LLM complete failed: {e}", flush=True)
        ans = f"\u26a0\ufe0f Sorry, model error: {str(e)[:200]}"

    # Step 7: Verify (optional)
    if AGENTIC_VERIFY and ans and len(ans) > 50:
        try:
            check = [{"role":"system","content":build_sys_prompt(intent,lang)},
                     {"role":"user","content":
                      f"Review and improve this draft. Fix errors, keep good parts.\n\n"
                      f"Q: {message}\n\nDraft:\n{ans}\n\nReturn improved answer only."}]
            improved = _llm_complete(check)
            if improved and len(improved) >= 0.6 * len(ans):
                ans = improved
        except Exception as e:
            print(f"[v4] verify error: {e}", flush=True)

    # Step 8: Citations
    if cites:
        seen = list(dict.fromkeys(cites))
        ans += "\n\n---\n**Sources**\n" + "\n".join(seen)

    return ans, plan, cites


# keep old agentic_answer for backward compat (used by Gradio UI)
def agentic_answer(message):
    return agentic_answer_v4(message)

def answer_full(message):
    messages, cites = build_messages_v4(message)
    ans = _llm_complete(messages)
    if cites:
        ans += "\n\n---\n**Sources**\n" + "\n".join(cites)
    return ans


# ============================================================
#  COGNITION (CORTEX + REFLEX)  — unchanged from v3
# ============================================================
_COGNITION = {"reflex":None,"cortex":None,"ready":False,"err":""}
_cog_lock  = threading.Lock()

def _forge_complete(messages, max_tokens=700, temperature=0.2):
    return _llm_complete(messages, max_tokens=max_tokens)

def _retrieve_for_cog(query, k=3):
    return prayas_retrieve(query, k)   # v4: uses PRAYAS first

def _web_for_cog(query):
    try: return web_search(query, 3)
    except Exception: return []

def _learn_sink(row):
    try:
        if not (SUPABASE_URL and SUPABASE_KEY): return
        import requests
        requests.post(
            SUPABASE_URL + "/rest/v1/" + os.environ.get("TRACES_TABLE","damru_reasoning_traces"),
            headers={"apikey":SUPABASE_KEY,"Authorization":"Bearer "+SUPABASE_KEY,
                     "Content-Type":"application/json","Prefer":"return=minimal"},
            data=json.dumps(row), timeout=10)
    except Exception: pass

def get_cognition():
    if _COGNITION["ready"]: return _COGNITION
    with _cog_lock:
        if _COGNITION["ready"]: return _COGNITION
        if USE_REFLEX:
            try:
                from damru_reflex import ReflexEngine, MemoryStore, ReflexConfig
                dt = None
                try:
                    from damru_reason import deep_think as _dt
                    dt = lambda q: _dt(q, _forge_complete, _retrieve_for_cog, _web_for_cog)
                except Exception: pass
                _COGNITION["reflex"] = ReflexEngine(
                    complete_fn=_forge_complete, retrieve_fn=_retrieve_for_cog,
                    web_fn=_web_for_cog, deep_think_fn=dt, memory=MemoryStore(),
                    learn_sink=_learn_sink, cfg=ReflexConfig.from_env())
            except Exception as e:
                _COGNITION["err"] += "reflex:"+str(e)[:100]+" "
        if USE_CORTEX:
            try:
                from damru_cortex import CortexEngine, SkillLibrary, CortexConfig
                _COGNITION["cortex"] = CortexEngine(
                    complete_fn=_forge_complete, retrieve_fn=_retrieve_for_cog,
                    web_fn=_web_for_cog, reflex=_COGNITION["reflex"],
                    library=SkillLibrary(), cfg=CortexConfig.from_env())
            except Exception as e:
                _COGNITION["err"] += "cortex:"+str(e)[:100]+" "
        _COGNITION["ready"] = True
    return _COGNITION

def cortex_answer(message, history=None):
    cog = get_cognition()
    cortex = cog.get("cortex")
    reflex = cog.get("reflex")
    if cortex is not None:
        res  = cortex.think(message, history=history)
        meta = {"path":"cortex","trust":res.get("trust"),"plan":res.get("plan"),
                "computed":res.get("computed"),"skills":res.get("skills")}
        return (res.get("answer") or ""), meta
    if reflex is not None:
        r = reflex.answer(message, history=history)
        return r.text, {"path":"reflex","trust":{"confidence":r.confidence}}
    ans, plan, _ = agentic_answer_v4(message)
    return ans, {"path":"agentic","plan":plan}


# ============================================================
#  LOGGING
# ============================================================
def log_chat(question, answer, source):
    if not (SUPABASE_URL and SUPABASE_KEY): return
    try:
        import requests
        requests.post(
            SUPABASE_URL + "/rest/v1/" + LOG_TABLE,
            headers={"apikey":SUPABASE_KEY,"Authorization":"Bearer "+SUPABASE_KEY,
                     "Content-Type":"application/json","Prefer":"return=minimal"},
            data=json.dumps({"question":question,"answer":answer,
                             "source":source,"ts":int(time.time())}),
            timeout=10)
    except Exception: pass

def log_to_hf_dataset(question, answer, intent=""):
    if not (HF_TOKEN and HF_LOG_CHATS): return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        record = json.dumps({"instruction":question,"output":answer,
                             "source":"live_chat","intent":intent or "general",
                             "timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())},
                            ensure_ascii=False)
        ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
        api.upload_file(
            path_or_fileobj=record.encode("utf-8"),
            path_in_repo=f"live_chats/chat_{ts}_{abs(hash(question))%10000}.jsonl",
            repo_id=HF_DATASET, repo_type="dataset",
            commit_message=f"Live chat: {question[:50]}")
    except Exception as e:
        print(f"[v4] HF log error: {e}", flush=True)


# ============================================================
#  DATASET STATS (for /health and /curious_status)
# ============================================================
_CURIOUS_STATS = {
    "dataset":HF_DATASET,"last_checked":None,
    "total_files":0,"total_qa_pairs":0,
    "last_run_time":None,"last_run_qa":0,"status":"unknown"
}
_stats_lock = threading.Lock()

def _fetch_dataset_stats():
    global _CURIOUS_STATS
    if not HF_TOKEN: return
    try:
        from huggingface_hub import HfApi
        api   = HfApi(token=HF_TOKEN)
        files = list(api.list_repo_files(HF_DATASET, repo_type="dataset"))
        # Count both curious/ and world_tiles/ folders
        data_files = [f for f in files
                      if (f.startswith("curious/") or f.startswith("world_tiles/")
                          or f.startswith("live_chats/")) and f.endswith(".jsonl")]
        total_qa, last_run_time, last_run_qa = 0, None, 0
        sorted_files = sorted(data_files, reverse=True)
        for i, fname in enumerate(sorted_files[:50]):  # sample 50 files
            try:
                import requests
                url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{fname}"
                r   = requests.get(url,
                                   headers={"Authorization":f"Bearer {HF_TOKEN}"},
                                   timeout=15)
                if r.ok:
                    count = len([l for l in r.text.strip().split("\n") if l.strip()])
                    total_qa += count
                    if i == 0:
                        last_run_qa = count
                        m = re.search(r"(\d{8})_(\d{6})", fname)
                        if m:
                            last_run_time = (f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]} "
                                            f"{m.group(2)[:2]}:{m.group(2)[2:4]} UTC")
            except Exception: pass
        with _stats_lock:
            _CURIOUS_STATS.update({
                "last_checked":  time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_files":   len(data_files),
                "total_qa_pairs":total_qa,
                "last_run_time": last_run_time,
                "last_run_qa":   last_run_qa,
                "status":        "ok" if data_files else "empty",
            })
    except Exception as e:
        print(f"[v4] dataset stats error: {e}", flush=True)
        with _stats_lock: _CURIOUS_STATS["status"] = "error"

# Load PRAYAS tiles from HF dataset in background
def _load_prayas_from_hf():
    if PRAYAS is None or not HF_TOKEN: return
    try:
        from huggingface_hub import HfApi
        import requests
        api   = HfApi(token=HF_TOKEN)
        files = list(api.list_repo_files(HF_DATASET, repo_type="dataset"))
        tile_files = [f for f in files
                      if (f.startswith("world_tiles/") or f.startswith("curious/"))
                      and f.endswith(".jsonl")]
        added_total = 0
        # Load latest 20 files
        for fname in sorted(tile_files, reverse=True)[:20]:
            try:
                url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{fname}"
                r   = requests.get(url, headers={"Authorization":f"Bearer {HF_TOKEN}"}, timeout=20)
                if r.ok:
                    for line in r.text.strip().split("\n"):
                        line = line.strip()
                        if not line: continue
                        try:
                            d = json.loads(line)
                            text = (d.get("text") or d.get("instruction") or
                                    d.get("question") or d.get("output") or "")
                            if text and len(text) > 30:
                                n = PRAYAS.ingest_text(
                                    text,
                                    topic=d.get("topic") or d.get("title", ""),
                                    domain=d.get("domain", "general"),
                                    source=d.get("source") or d.get("url", fname),
                                    lang=d.get("lang", "en")
                                )
                                added_total += n
                        except Exception: pass
            except Exception as e:
                print(f"[v4] tile load {fname}: {e}", flush=True)
        print(f"[v4] PRAYAS loaded {added_total} tiles from HF dataset", flush=True)
    except Exception as e:
        print(f"[v4] PRAYAS HF load error: {e}", flush=True)

# Background startup tasks
threading.Thread(target=_fetch_dataset_stats, daemon=True).start()
threading.Thread(target=_load_prayas_from_hf,  daemon=True).start()


# ============================================================
#  IMAGE ROUTER  (unchanged)
# ============================================================
import random
from urllib.parse import quote as _uq

def _safe_image_prompt(prompt, style="realistic"):
    p = (prompt or "").strip() or "a friendly AI mascot representing Damru AI"
    style_map = {
        "realistic": "photorealistic, natural lighting, high detail, sharp focus",
        "poster":    "cinematic poster, dramatic composition, vibrant colors",
        "logo":      "clean vector logo, centered, minimal, brand identity",
        "anime":     "high quality anime style, expressive, detailed background",
        "space":     "space art, cosmic, nebula, ultra-realistic, NASA quality",
    }
    suffix = style_map.get((style or "realistic").lower(), style_map["realistic"])
    if "text" not in p.lower() and "logo" not in p.lower():
        suffix += ", no text, no watermark"
    return f"{p}, {suffix}"

def image_router(prompt, style="realistic", aspect="1:1", seed=None):
    seed  = int(seed if seed is not None else random.randint(1, 99999999))
    final = _safe_image_prompt(prompt, style)
    sizes = {"16:9":(1280,720),"9:16":(720,1280),"4:3":(1024,768)}
    w, h  = sizes.get(aspect, (IMAGE_WIDTH, IMAGE_HEIGHT))
    url   = ("https://image.pollinations.ai/prompt/" + _uq(final)
             + f"?width={w}&height={h}&nologo=true&seed={seed}&enhance=true&model=flux")
    return {"ok":True,"provider":"pollinations","quality":"draft-free",
            "imageUrl":url,"prompt":final,"seed":seed,"width":w,"height":h}

def research_report(query, depth="quick"):
    hits    = web_search(query, 5 if depth=="deep" else 3)
    src_txt = "\n".join(f"- {h['title']}: {h['snippet']} (Source: {h['url']})" for h in hits)
    prompt  = ("DEEP RESEARCH TASK. Use web results as fresh context. "
               "Structure: Quick answer, Key facts, Sources, What to verify, Next steps.\n\n"
               f"Question: {query}\n\nWEB RESULTS:\n{src_txt}")
    try:    report = answer_full(prompt)
    except Exception as e:
        report = "\u26a0\ufe0f Research failed, but web results found.\n\n" + src_txt
    return {"ok":True,"query":query,"report":report,"results":hits}

def artifact_build(prompt, kind="html"):
    task = ("ARTIFACT BUILD TASK. If HTML requested, return complete self-contained HTML inside ```html block. "
            "If PPT, return slide-by-slide with notes. If code, return complete runnable code.\n\n"
            f"User request: {prompt}")
    ans  = answer_full(task)
    html = ""
    m    = re.search(r"```html\s*([\s\S]*?)```", ans, re.I)
    if m: html = m.group(1).strip()
    return {"ok":True,"kind":kind,"answer":ans,"html":html}


# ============================================================
#  GRADIO UI  (streaming)
# ============================================================
def stream_generate(messages):
    kind, model, tok = load_llm()
    if kind == "gguf":
        stream = model.create_chat_completion(messages=messages, max_tokens=MAX_NEW,
                                              temperature=0.7, stream=True)
        acc, since = "", 0
        for chunk in stream:
            delta = chunk["choices"][0]["delta"].get("content", "")
            if delta:
                acc += delta; since += 1
                if since >= YIELD_EVERY: since=0; yield acc
        yield acc if acc else "(No answer, retry)"
        return
    import torch
    from transformers import TextIteratorStreamer
    prompt  = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids     = tok(prompt, return_tensors="pt")
    streamer= TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    kwargs  = dict(**ids, max_new_tokens=MAX_NEW, do_sample=True, temperature=0.7,
                   top_p=0.9, pad_token_id=tok.eos_token_id, streamer=streamer)
    threading.Thread(target=model.generate, kwargs=kwargs, daemon=True).start()
    acc, since = "", 0
    for piece in streamer:
        acc += piece; since += 1
        if since >= YIELD_EVERY: since=0; yield acc
    yield acc if acc else "(No answer, retry)"

def respond(message, history):
    yield "\u23f3 ..."
    try:
        # v4: use full pipeline (lang + emotion + PRAYAS)
        messages, cites = build_messages_v4(
            message,
            intent=route_intent(message),
            lang=detect_language(message)
        )
        last = ""
        for partial in stream_generate(messages):
            last = partial; yield partial
        if cites and last:
            last += "\n\n---\n**Sources**\n" + "\n".join(cites)
            yield last
        threading.Thread(target=log_chat, args=(message,last,"gradio"), daemon=True).start()
        if HF_LOG_CHATS:
            threading.Thread(target=log_to_hf_dataset, args=(message,last,"gradio"), daemon=True).start()
    except Exception as e:
        print(traceback.format_exc(), flush=True)
        yield "\u26a0\ufe0f Error:\n\n```\n" + str(e)[:500] + "\n```"

demo = gr.ChatInterface(
    respond,
    title="\U0001f415 Damru AI v4 — Beast Intelligence",
    description="Damru backend + test UI. Frontend: damru-ai.vercel.app | Learns every 6h.",
    examples=[
        "Space mission to Mars ka plan banao",
        "Explain self-driving car sensor fusion in Hindi",
        "Write Python code for binary search",
        "JEE mein integration ke important topics kya hain?",
        "Fighter jet radar system kaise kaam karta hai?",
    ],
    cache_examples=False,
)
demo.queue(default_concurrency_limit=1)


# ============================================================
#  FASTAPI
# ============================================================
api = FastAPI(title="Damru API v4 — Beast Intelligence")
api.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])

class ChatIn(BaseModel):
    message: str

class ImageIn(BaseModel):
    prompt:str; style:str="realistic"; aspect:str="1:1"; seed:Optional[int]=None

class WebIn(BaseModel):
    query:str; limit:int=3

class ResearchIn(BaseModel):
    query:str; depth:str="quick"

class ArtifactIn(BaseModel):
    prompt:str; kind:str="html"

class CortexIn(BaseModel):
    message:str

class Model3DIn(BaseModel):
    prompt:str=""; image_url:str=""; seed:int=0;
    resolution:int=1024; decimation:int=300000; texture:int=2048


@api.get("/health")
def health():
    local_ready = _backend is not None
    cog = _COGNITION
    with _stats_lock:
        ds = dict(_CURIOUS_STATS)
    prayas_tiles = PRAYAS.stats()["total_tiles"] if PRAYAS else 0
    return {
        "ok": True,
        "version": "4.0",
        # Model info
        "model_loaded":  local_ready or bool(OPEN_BRAIN),
        "brain_ready":   local_ready or bool(OPEN_BRAIN),
        "backend":       "open_brain" if OPEN_BRAIN else ("gguf" if GGUF_FILE else "hf"),
        "rag":           bool(RAG),
        # v4 new systems
        "prayas":        bool(PRAYAS),
        "prayas_tiles":  prayas_tiles,
        "emotions":      bool(EMO),
        "selfheal":      bool(_SHR),
        "multilingual":  True,
        # Cognition
        "cortex":        bool(cog.get("cortex")),
        "reflex":        bool(cog.get("reflex")),
        # Knowledge stats
        "knowledge_dataset":   HF_DATASET,
        "total_qa_pairs":      ds.get("total_qa_pairs", 0),
        "last_learning_run":   ds.get("last_run_time"),
        "curious_engine":      "github_actions_cron_6h",
        "world_harvest":       "github_actions_cron_6h",
        # Config
        "agentic":       AGENTIC,
        "max_new":       MAX_NEW,
        "open_brain":    bool(OPEN_BRAIN),
    }


@api.get("/curious_status")
def curious_status():
    with _stats_lock: ds = dict(_CURIOUS_STATS)
    if not ds.get("last_checked"):
        threading.Thread(target=_fetch_dataset_stats, daemon=True).start()
    return {
        "ok":            True,
        "engine":        "github_actions_cron_6h",
        "dataset":       HF_DATASET,
        "github_repo":   "Damru-AI/Damru-AI",
        "workflow":      ".github/workflows/damru_curious_loop.yml",
        "schedule":      "Every 6 hours",
        "sources":       ["Wikipedia 60M+ articles","arXiv 2M+ papers",
                          "GitHub public repos","NASA Open Data",
                          "ISRO","Stack Overflow","HF Datasets"],
        "last_run_time":     ds.get("last_run_time"),
        "last_run_qa_pairs": ds.get("last_run_qa"),
        "total_files":       ds.get("total_files"),
        "total_qa_pairs":    ds.get("total_qa_pairs"),
        "dataset_status":    ds.get("status"),
        "last_checked":      ds.get("last_checked"),
    }


@api.get("/knowledge_stats")
def knowledge_stats():
    threading.Thread(target=_fetch_dataset_stats, daemon=True).start()
    with _stats_lock: ds = dict(_CURIOUS_STATS)
    prayas_stats = PRAYAS.stats() if PRAYAS else {}
    return {
        "ok":              True,
        "dataset":         HF_DATASET,
        "total_tiles":     ds.get("total_qa_pairs", 0),
        "prayas_tiles":    prayas_stats.get("total_tiles", 0),
        "total_files":     ds.get("total_files", 0),
        "domains":         prayas_stats.get("domains", []),
        "last_update":     ds.get("last_run_time"),
        "refresh_triggered": True,
        "message": (
            f"Damru has {prayas_stats.get('total_tiles',0)} PRAYAS tiles + "
            f"{ds.get('total_qa_pairs',0)} HF Q&A pairs. "
            "Knowledge grows every 6 hours."
        )
    }


@api.post("/chat")
def chat_api(body: ChatIn):
    intent, meta = None, None
    if USE_CORTEX or USE_REFLEX:
        try:
            ans, meta = cortex_answer(body.message)
            src    = "api-" + (meta or {}).get("path", "cortex")
            plan   = (meta or {}).get("plan", {})
            intent = (plan.get("intent") if isinstance(plan, dict) else None)
        except Exception as e:
            print(f"[v4] cortex failed -> v4 agentic: {e}", flush=True)
            ans, plan, _ = agentic_answer_v4(body.message)
            src    = "api-agentic-v4"
            intent = plan.get("intent") if isinstance(plan, dict) else None
            meta   = {"path":"agentic","plan":plan}
    elif AGENTIC:
        ans, plan, _ = agentic_answer_v4(body.message)
        src    = "api-agentic-v4"
        intent = plan.get("intent") if isinstance(plan, dict) else None
        meta   = {"path":"agentic","plan":plan}
    else:
        ans    = answer_full(body.message)
        src    = "api-v4"

    threading.Thread(target=log_chat, args=(body.message,ans,src), daemon=True).start()
    if HF_LOG_CHATS:
        threading.Thread(target=log_to_hf_dataset,
                         args=(body.message, ans, intent or ""),
                         daemon=True).start()
    if CHETNA:
        threading.Thread(target=CHETNA.observe,
                         kwargs={"question":body.message,"answer":ans,
                                 "meta":meta,"intent":intent,"source":src},
                         daemon=True).start()

    resp = {"answer": ans, "intent": intent}
    if meta:
        resp["trust"] = meta.get("trust")
        resp["path"]  = meta.get("path")
        # v4 extras
        if isinstance(meta.get("plan"), dict):
            resp["lang"]         = meta["plan"].get("lang", "en")
            resp["emotion"]      = meta["plan"].get("emotion", "")
            resp["damru_state"]  = meta["plan"].get("damru_state", "neutral")
    return resp


@api.post("/image")
def image_api(body: ImageIn):
    data = image_router(body.prompt, body.style, body.aspect, body.seed)
    threading.Thread(target=log_chat, args=(body.prompt,json.dumps(data)[:1000],"image"), daemon=True).start()
    return data

@api.post("/web")
def web_api(body: WebIn):
    return {"ok":True,"query":body.query,"results":web_search(body.query, body.limit)}

@api.post("/research")
def research_api(body: ResearchIn):
    data = research_report(body.query, body.depth)
    threading.Thread(target=log_chat, args=(body.query,data.get("report",""),"research"), daemon=True).start()
    return data

@api.post("/artifact")
def artifact_api(body: ArtifactIn):
    data = artifact_build(body.prompt, body.kind)
    threading.Thread(target=log_chat, args=(body.prompt,data.get("answer",""),"artifact"), daemon=True).start()
    return data

@api.post("/cortex")
def cortex_api(body: CortexIn):
    cog = get_cognition()
    cortex = cog.get("cortex")
    if cortex is None:
        return {"ok":False,"error":"cortex unavailable","detail":cog.get("err")}
    res = cortex.think(body.message)
    threading.Thread(target=log_chat, args=(body.message,res.get("answer",""),"cortex"), daemon=True).start()
    return {"ok":True,**res}

@api.get("/skills")
def skills_api():
    cog = get_cognition()
    cortex = cog.get("cortex")
    if cortex is None: return {"ok":False,"skills":[]}
    return {"ok":True,"skills":cortex.lib.names(),"count":len(cortex.lib)}

@api.post("/model3d")
def model3d_api(body: Model3DIn):
    if not USE_TRELLIS: raise HTTPException(503, "TRELLIS disabled")
    try: from trellis_bridge import image_to_glb
    except Exception as e: raise HTTPException(500, str(e)[:200])
    image = (body.image_url or "").strip()
    if not image:
        if not body.prompt.strip(): raise HTTPException(400, "provide prompt or image_url")
        img   = image_router(body.prompt, "realistic", "1:1", body.seed or None)
        image = img.get("imageUrl", "")
    if not image: raise HTTPException(400, "no image to convert")
    try:
        glb_path = image_to_glb(image, seed=body.seed, resolution=body.resolution,
                                 decimation=body.decimation, texture=body.texture)
    except Exception as e: raise HTTPException(502, str(e)[:300])
    return FileResponse(glb_path, media_type="model/gltf-binary", filename="damru-model.glb")


# Mount Gradio
app = gr.mount_gradio_app(api, demo, path="/")


if __name__ == "__main__":
    if OPEN_BRAIN is None:
        threading.Thread(target=load_llm, daemon=True).start()
    if USE_CORTEX or USE_REFLEX:
        threading.Thread(target=get_cognition, daemon=True).start()
    import uvicorn
    uvicorn.run(app, host="0.0.0.0",
                port=int(os.environ.get("PORT") or "7860"))
