#!/usr/bin/env python3
# ============================================================================
#  DAMRU COUNCIL  --  multi-solver verified data engine
#  ----------------------------------------------------------------------------
#  Kaam: oracle repo se PROBLEMS lo -> multiple free LLM solvers se solve
#  karao -> code EXECUTE karke verify karo -> Gemini se judge karao ->
#  best (shortest passing) solution = SFT row, worst = DPO rejected ->
#  HuggingFace pe push.
#
#  DESIGN RULES
#  ------------
#  * SELF-RECOVERING: koi bhi provider mare -> baaki chalte rahenge
#  * 400/401/403/404 par turant BAIL (retry storm nahi)
#  * RESUME: existing rows dedupe hokar skip hote hain
#  * PERIODIC PUSH: crash/timeout hone par bhi kaam bacha rehta hai
#  * TIME BUDGET: GitHub Actions 6h limit se pehle safe exit
#  * NO load_dataset(): damru_loader ka robust loader use hota hai
#
#  REQUIRED SECRET : HF_TOKEN
#  SOLVER SECRETS  : CEREBRAS_API_KEY | OPENROUTER_KEY | NVIDIA_NIM_KEY |
#                    GROQ_API_KEY | MISTRAL_API_KEY | GH_MODELS_TOKEN |
#                    VLLM_URL + VLLM_MODEL + VLLM_KEY
#  JUDGE SECRET    : GEMINI_KEY (optional -- na ho to code-exec hi judge hai)
# ============================================================================

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

from damru_loader import load_problems

NL = chr(10)
START_TS = time.time()


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------
def env(key, default=""):
    v = os.environ.get(key)
    if v is None:
        return default
    v = v.strip()
    return v if v else default


def envi(key, default=0):
    try:
        return int(float(env(key, str(default))))
    except Exception:
        return default


def envf(key, default=0.0):
    try:
        return float(env(key, str(default)))
    except Exception:
        return default


def log(msg):
    el = int(time.time() - START_TS)
    print("[council +" + str(el) + "s] " + str(msg), flush=True)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
HF_TOKEN = env("HF_TOKEN")

OUT_REPO = env("COUNCIL_OUT_REPO", "Damaru-ai/damru-council")
SRC_REPO = env("COUNCIL_SRC_REPO", "Damaru-ai/damru-oracle")
SRC_SPLIT = env("COUNCIL_SRC_SPLIT", "train")

MAX_PROBLEMS = envi("COUNCIL_MAX_PROBLEMS", 0)          # 0 = unlimited
TIME_BUDGET_MIN = envi("COUNCIL_TIME_BUDGET_MIN", 300)  # Actions 360 limit se safe
COMMIT_EVERY_SEC = envi("COUNCIL_COMMIT_EVERY_SEC", 600)
N_SOLVERS = envi("COUNCIL_N_SOLVERS", 3)
REQ_TIMEOUT = envi("COUNCIL_REQ_TIMEOUT", 90)
MAX_TOKENS = envi("COUNCIL_MAX_TOKENS", 1024)
TEMPERATURE = envf("COUNCIL_TEMPERATURE", 0.3)
CODE_EXEC = envi("COUNCIL_CODE_EXEC", 1)
CODE_TIMEOUT = envi("COUNCIL_CODE_TIMEOUT", 10)
JUDGE_MIN = envi("COUNCIL_JUDGE_MIN", 7)

GEMINI_KEY = env("GEMINI_KEY")
GEMINI_JUDGE_MODEL = env("GEMINI_JUDGE_MODEL", "gemini-2.0-flash")

SYS_PROMPT = (
    "You are an expert Python engineer. Solve the problem with clean, correct, "
    "efficient code. Return ONE fenced python code block. No explanation outside "
    "the code block. Include a small self-test under "
    'if __name__ == "__main__":'
)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------
def build_providers():
    """Sirf un providers ko active karo jinke secrets maujood hain."""
    provs = []

    vllm_url = env("VLLM_URL")
    if vllm_url:
        provs.append({
            "name": "vllm",
            "url": vllm_url,
            "key": env("VLLM_KEY", "damru-secret"),
            "model": env("VLLM_MODEL", "Qwen/Qwen2.5-Coder-14B-Instruct-AWQ"),
        })

    if env("CEREBRAS_API_KEY"):
        provs.append({
            "name": "cerebras",
            "url": "https://api.cerebras.ai/v1/chat/completions",
            "key": env("CEREBRAS_API_KEY"),
            "model": env("CEREBRAS_MODEL", "llama-3.3-70b"),
        })

    if env("OPENROUTER_KEY"):
        provs.append({
            "name": "openrouter",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "key": env("OPENROUTER_KEY"),
            "model": env("OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct:free"),
        })

    if env("NVIDIA_NIM_KEY"):
        provs.append({
            "name": "nim",
            "url": "https://integrate.api.nvidia.com/v1/chat/completions",
            "key": env("NVIDIA_NIM_KEY"),
            "model": env("NIM_MODEL", "qwen/qwen2.5-coder-32b-instruct"),
        })

    if env("GROQ_API_KEY"):
        provs.append({
            "name": "groq",
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "key": env("GROQ_API_KEY"),
            "model": env("GROQ_MODEL", "llama-3.3-70b-versatile"),
        })

    if env("MISTRAL_API_KEY"):
        provs.append({
            "name": "mistral",
            "url": "https://api.mistral.ai/v1/chat/completions",
            "key": env("MISTRAL_API_KEY"),
            "model": env("MISTRAL_MODEL", "mistral-large-latest"),
        })

    if env("GH_MODELS_TOKEN"):
        provs.append({
            "name": "gh_models",
            "url": "https://models.inference.ai.azure.com/chat/completions",
            "key": env("GH_MODELS_TOKEN"),
            "model": env("GH_MODELS_MODEL", "gpt-4o-mini"),
        })

    return provs


DEAD = set()   # jo provider permanently fail ho gaya


def http_post_json(url, headers, payload, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST")
    for k, v in headers.items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8", errors="ignore"))


def openai_chat(prov, messages, retries=3):
    """OpenAI-compatible chat call. Fatal error par turant None (no retry storm)."""
    name = prov["name"]
    if name in DEAD:
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + prov["key"],
    }
    if name == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/Damru-AI"
        headers["X-Title"] = "Damru Council"

    payload = {
        "model": prov["model"],
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    for attempt in range(1, retries + 1):
        try:
            data = http_post_json(prov["url"], headers, payload, REQ_TIMEOUT)
            choices = data.get("choices") or []
            if not choices:
                return None
            msg = choices[0].get("message") or {}
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for c in content:
                    if isinstance(c, dict) and isinstance(c.get("text"), str):
                        parts.append(c["text"])
                content = NL.join(parts)
            if isinstance(content, str) and content.strip():
                return content
            return None

        except urllib.error.HTTPError as e:
            code = e.code
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="ignore")[:200]
            except Exception:
                pass
            # FATAL: auth / permission / bad model / bad route -> kabhi theek nahi hoga
            if code in (400, 401, 403, 404):
                log("FATAL " + name + " HTTP " + str(code) + " -> disabling. " + detail)
                DEAD.add(name)
                return None
            log("retry " + name + " (" + str(attempt) + "/" + str(retries) + ") HTTP " + str(code))
        except Exception as e:
            log("retry " + name + " (" + str(attempt) + "/" + str(retries) + ") " + str(e)[:120])

        if attempt < retries:
            time.sleep(2 * attempt)

    return None


# ---------------------------------------------------------------------------
# judge
# ---------------------------------------------------------------------------
def first_number(text):
    num = ""
    for ch in str(text):
        if ch.isdigit():
            num += ch
        elif num:
            break
    if not num:
        return None
    try:
        return int(num)
    except Exception:
        return None


def gemini_judge(problem, code):
    """0-10 score. Gemini na ho to None (tab code-exec hi decide karega)."""
    if not GEMINI_KEY:
        return None

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        + GEMINI_JUDGE_MODEL
        + ":generateContent?key="
        + GEMINI_KEY
    )
    prompt = (
        "Rate this Python solution 0-10 for correctness, efficiency and clarity."
        + NL + "Reply with ONLY the integer."
        + NL + NL + "PROBLEM:" + NL + str(problem)[:2000]
        + NL + NL + "SOLUTION:" + NL + str(code)[:4000]
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    try:
        data = http_post_json(url, {"Content-Type": "application/json"}, payload, REQ_TIMEOUT)
        cands = data.get("candidates") or []
        if not cands:
            return None
        parts = (cands[0].get("content") or {}).get("parts") or []
        if not parts:
            return None
        n = first_number(parts[0].get("text", ""))
        if n is None:
            return None
        return max(0, min(10, n))
    except Exception as e:
        log("judge failed: " + str(e)[:120])
        return None


# ---------------------------------------------------------------------------
# code extraction + execution
# ---------------------------------------------------------------------------
def extract_code(text):
    """Fenced block se code nikaalo (regex-free)."""
    if not isinstance(text, str):
        return ""
    fence = "```"
    if fence not in text:
        return text.strip()
    chunks = text.split(fence)
    best = ""
    i = 1
    while i < len(chunks):
        block = chunks[i]
        lines = block.split(NL)
        if lines and lines[0].strip().lower() in ("python", "py", "python3", ""):
            body = NL.join(lines[1:])
        else:
            body = block
        body = body.strip()
        if len(body) > len(best):
            best = body
        i += 2
    return best if best else text.strip()


def looks_like_python(code):
    if not code or len(code.strip()) < 10:
        return False
    markers = ("def ", "class ", "import ", "for ", "while ", "return", "print(", "=")
    hits = 0
    for m in markers:
        if m in code:
            hits += 1
    return hits >= 2


def run_code(code):
    """Sandbox-ish exec. Returns (ok, output)."""
    if not CODE_EXEC:
        return True, "exec disabled"
    if not looks_like_python(code):
        return False, "not python-like"

    path = None
    try:
        fd, path = tempfile.mkstemp(suffix=".py")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(code)
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True,
            text=True,
            timeout=CODE_TIMEOUT,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode == 0, out[:1500]
    except subprocess.TimeoutExpired:
        return False, "timeout after " + str(CODE_TIMEOUT) + "s"
    except Exception as e:
        return False, str(e)[:300]
    finally:
        if path:
            try:
                os.remove(path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# dataset helpers
# ---------------------------------------------------------------------------
def hash_prompt(text):
    return hashlib.md5(str(text)[:2000].encode("utf-8", errors="ignore")).hexdigest()


def extract_prompt(row):
    if isinstance(row, str):
        return row.strip()
    if not isinstance(row, dict):
        return ""
    for k in ("prompt", "problem", "question", "instruction", "input", "text"):
        v = row.get(k)
        if isinstance(v, str) and len(v.strip()) >= 20:
            return v.strip()
    return ""


def load_existing():
    """Purane rows + unke hashes (resume ke liye). Fail = fresh start."""
    sft, dpo, done = [], [], set()
    try:
        from datasets import load_dataset
        for split_name, bucket in (("sft", sft), ("dpo", dpo)):
            try:
                d = load_dataset(OUT_REPO, split=split_name, token=HF_TOKEN)
                for r in d:
                    bucket.append(dict(r))
            except Exception:
                pass
        for r in sft:
            h = r.get("phash")
            if h:
                done.add(h)
        log("resume: sft=" + str(len(sft)) + " dpo=" + str(len(dpo)) + " done=" + str(len(done)))
    except Exception as e:
        log("resume skipped: " + str(e)[:140])
    return sft, dpo, done


def push(sft, dpo):
    if not sft and not dpo:
        log("push skipped (nothing new)")
        return False
    try:
        from datasets import Dataset, DatasetDict
        parts = {}
        if sft:
            parts["sft"] = Dataset.from_list(sft)
        if dpo:
            parts["dpo"] = Dataset.from_list(dpo)
        DatasetDict(parts).push_to_hub(OUT_REPO, token=HF_TOKEN)
        log("PUSHED -> " + OUT_REPO + " sft=" + str(len(sft)) + " dpo=" + str(len(dpo)))
        return True
    except Exception as e:
        log("PUSH FAILED: " + str(e)[:250])
        return False


# ---------------------------------------------------------------------------
# core: ek problem par council chalao
# ---------------------------------------------------------------------------
def council_round(problem, providers):
    scored = []
    used = 0

    for prov in providers:
        if used >= N_SOLVERS:
            break
        if prov["name"] in DEAD:
            continue

        raw = openai_chat(prov, [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": str(problem)[:6000]},
        ])
        if not raw:
            continue

        used += 1
        code = extract_code(raw)
        if not code:
            continue

        ok, out = run_code(code)
        score = gemini_judge(problem, code)
        if score is None:
            score = 8 if ok else 3

        scored.append({
            "solver": prov["name"],
            "model": prov["model"],
            "code": code,
            "exec_ok": bool(ok),
            "exec_out": out,
            "score": int(score),
            "length": len(code),
        })

    return scored


def main():
    if not HF_TOKEN:
        log("FATAL: HF_TOKEN missing")
        return 1

    providers = build_providers()
    if not providers:
        log("FATAL: koi solver active nahi hai.")
        log("Ye secrets me se KAM SE KAM EK daalo: CEREBRAS_API_KEY, OPENROUTER_KEY,")
        log("NVIDIA_NIM_KEY, GROQ_API_KEY, MISTRAL_API_KEY, GH_MODELS_TOKEN, VLLM_URL")
        return 1

    names = []
    for p in providers:
        names.append(p["name"] + "(" + p["model"] + ")")
    log("Active solvers: " + ", ".join(names))
    log("Judge: " + (GEMINI_JUDGE_MODEL if GEMINI_KEY else "code-exec only"))

    # ---- problems load (robust loader -- load_dataset ka schema crash nahi) ----
    ds = load_problems(SRC_REPO, SRC_SPLIT, HF_TOKEN)
    if not ds:
        log("FATAL: 0 problems loaded from " + SRC_REPO)
        return 1
    log("problems available: " + str(len(ds)))

    sft, dpo, done = load_existing()
    new_sft, new_dpo = 0, 0

    deadline = START_TS + (TIME_BUDGET_MIN * 60)
    last_push = time.time()
    processed = 0

    for row in ds:
        if time.time() >= deadline:
            log("TIME BUDGET reached -- stopping cleanly")
            break
        if MAX_PROBLEMS and processed >= MAX_PROBLEMS:
            log("MAX_PROBLEMS reached")
            break
        if all(p["name"] in DEAD for p in providers):
            log("ALL solvers dead -- stopping to save time")
            break

        problem = extract_prompt(row)
        if not problem:
            continue

        ph = hash_prompt(problem)
        if ph in done:
            continue
        done.add(ph)
        processed += 1

        scored = council_round(problem, providers)
        if not scored:
            continue

        passing = [s for s in scored if s["exec_ok"] and s["score"] >= JUDGE_MIN]

        if passing:
            # EFFICIENCY IS A FEATURE: sabse chhota passing solution jeetta hai
            best = sorted(passing, key=lambda s: (s["length"], -s["score"]))[0]
            sft.append({
                "phash": ph,
                "problem": problem,
                "solution": best["code"],
                "solver": best["solver"],
                "model": best["model"],
                "score": best["score"],
                "length": best["length"],
                "verified": True,
            })
            new_sft += 1

            if len(scored) >= 2:
                worst = sorted(scored, key=lambda s: (s["score"], -s["length"]))[0]
                if worst["code"] != best["code"]:
                    dpo.append({
                        "phash": ph,
                        "prompt": problem,
                        "chosen": best["code"],
                        "rejected": worst["code"],
                        "chosen_score": best["score"],
                        "rejected_score": worst["score"],
                    })
                    new_dpo += 1

        if processed % 10 == 0:
            log("progress: seen=" + str(processed) + " sft+=" + str(new_sft) + " dpo+=" + str(new_dpo))

        if time.time() - last_push >= COMMIT_EVERY_SEC:
            push(sft, dpo)
            last_push = time.time()

    push(sft, dpo)
    log("FINAL: processed=" + str(processed) + " new_sft=" + str(new_sft) + " new_dpo=" + str(new_dpo))
    log("totals: sft=" + str(len(sft)) + " dpo=" + str(len(dpo)))
    if DEAD:
        log("dead solvers: " + ", ".join(sorted(DEAD)))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        log("interrupted")
        sys.exit(0)
    except Exception as e:
        log("UNCAUGHT: " + str(e)[:400])
        sys.exit(1)
