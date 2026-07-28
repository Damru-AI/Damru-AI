#!/usr/bin/env python3
"""
DAMRU DATA FORGE  --  24/7 self-switching quality-data engine
=============================================================
GPT-4 ko takkar SIZE se nahi, DATA QUALITY + DISTILLATION se. Chhote model ko
strong OPEN-WEIGHT teachers se distill karo -- DeepSeek-R1 / Gemma ka raaz.

SELF-SWITCHING: multiple providers + multiple keys (comma-separated). Jab koi
provider 429/quota deta hai -> use cooldown me daal ke agla healthy pick.
15-PROVIDER COUNCIL: Cerebras, OpenRouter, GitHub Models, HF Router,
  SambaNova, Together, DeepInfra, Hyperbolic, Mistral, NVIDIA NIM,
  Cloudflare AI, Fireworks, Cohere, Chutes, Scaleway

RUNS on ANY free cloud CPU (GitHub Actions cron / HF Space) -- NO device needed.
DATA -> Hugging Face dataset (public ~5TB free).

ENV (jo mile wo set karo; forge baaki skip kar dega). Comma se multiple keys:
  HF_TOKEN               (push + hfrouter teacher)
  CEREBRAS_API_KEY       cloud.cerebras.ai
  OPENROUTER_API_KEY     openrouter.ai
  GH_MODELS_TOKEN_VAL    github PAT (models:read)
  SAMBANOVA_API_KEY      sambanova.ai/api
  TOGETHER_API_KEY       api.together.ai
  DEEPINFRA_API_KEY      deepinfra.com
  HYPERBOLIC_API_KEY     app.hyperbolic.xyz
  MISTRAL_API_KEY        console.mistral.ai
  NVIDIA_NIM_KEY         build.nvidia.com
  CF_API_TOKEN           Cloudflare Workers AI
  CF_ACCOUNT_ID          Cloudflare account ID
  FIREWORKS_API_KEY      fireworks.ai
  COHERE_API_KEY         cohere.com
  CHUTES_API_KEY         chutes.ai
  SCALEWAY_API_KEY       scaleway.com
  DAMRU_DATASET          default Damaru-ai/damru-knowledge
  DAMRU_MAX_ITERS        0=forever ; N=stop after N (CI)
  DAMRU_PUSH_EVERY=40  DAMRU_KEEP_SCORE=4  DAMRU_SLEEP=1.5
"""
import os, sys, json, time, re, random, hashlib

try:
    import requests
except Exception:
    os.system(sys.executable + " -m pip install -q requests huggingface_hub")
    import requests

HF_TOKEN     = os.environ.get("HF_TOKEN", "")
DATASET_REPO = os.environ.get("DAMRU_DATASET", "Damaru-ai/damru-knowledge")
MAX_ITERS    = int(os.environ.get("DAMRU_MAX_ITERS", "0"))
PUSH_EVERY   = int(os.environ.get("DAMRU_PUSH_EVERY", "40"))
KEEP_SCORE   = int(os.environ.get("DAMRU_KEEP_SCORE", "4"))
HTTP_TIMEOUT = int(os.environ.get("DAMRU_HTTP_TIMEOUT", "90"))
WORK_DIR     = os.environ.get("DAMRU_WORKDIR", "/tmp/damru_forge")
COOLDOWN     = int(os.environ.get("DAMRU_COOLDOWN", "300"))
os.makedirs(WORK_DIR, exist_ok=True)
OUT_JSONL = os.path.join(WORK_DIR, "damru_forge.jsonl")
SEEN_FILE = os.path.join(WORK_DIR, "seen.txt")

# === 15-PROVIDER COUNCIL ===
# (name, url, env_var, [models])
SPECS = [
    # 1. Cerebras -- fastest free inference
    ("cerebras", "https://api.cerebras.ai/v1/chat/completions", "CEREBRAS_API_KEY",
     ["gpt-oss-120b", "zai-glm-4.7", "gemma-4-31b"]),

    # 2. OpenRouter -- free tier models
    ("openrouter", "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY",
     ["openrouter/free"]),

    # 3. GitHub Models
    ("github", "https://models.github.ai/inference/chat/completions", "GH_MODELS_TOKEN_VAL",
     ["deepseek/DeepSeek-R1", "meta/Llama-3.3-70B-Instruct"]),

    # 4. HuggingFace Router
    ("hfrouter", "https://router.huggingface.co/v1/chat/completions", "HF_TOKEN",
     ["meta-llama/Llama-3.3-70B-Instruct"]),

    # 5. SambaNova -- fast 405B free
    ("sambanova", "https://api.sambanova.ai/v1/chat/completions", "SAMBANOVA_API_KEY",
     ["Meta-Llama-3.3-70B-Instruct", "Meta-Llama-3.1-405B-Instruct"]),

    # 6. Together AI
    ("together", "https://api.together.xyz/v1/chat/completions", "TOGETHER_API_KEY",
     ["meta-llama/Llama-3.3-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-R1"]),

    # 7. DeepInfra
    ("deepinfra", "https://api.deepinfra.com/v1/openai/chat/completions", "DEEPINFRA_API_KEY",
     ["meta-llama/Llama-3.3-70B-Instruct", "mistralai/Mixtral-8x22B-Instruct-v0.1"]),

    # 8. Hyperbolic
    ("hyperbolic", "https://api.hyperbolic.xyz/v1/chat/completions", "HYPERBOLIC_API_KEY",
     ["meta-llama/Llama-3.3-70B-Instruct", "deepseek-ai/DeepSeek-R1"]),

    # 9. Mistral AI
    ("mistral", "https://api.mistral.ai/v1/chat/completions", "MISTRAL_API_KEY",
     ["mistral-large-latest", "mistral-small-latest"]),

    # 10. NVIDIA NIM
    ("nvidia", "https://integrate.api.nvidia.com/v1/chat/completions", "NVIDIA_NIM_KEY",
     ["meta/llama-3.3-70b-instruct", "deepseek-ai/deepseek-r1"]),

    # 11. Fireworks AI
    ("fireworks", "https://api.fireworks.ai/inference/v1/chat/completions", "FIREWORKS_API_KEY",
     ["accounts/fireworks/models/llama-v3p3-70b-instruct",
      "accounts/fireworks/models/deepseek-r1"]),

    # 12. Cohere
    ("cohere", "https://api.cohere.com/v2/chat", "COHERE_API_KEY",
     ["command-r-plus", "command-r"]),

    # 13. Chutes AI -- truly free
    ("chutes", "https://llm.chutes.ai/v1/chat/completions", "CHUTES_API_KEY",
     ["deepseek-ai/DeepSeek-R1", "unsloth/Llama-3.3-70B-Instruct"]),

    # 14. Scaleway
    ("scaleway", "https://api.scaleway.ai/v1/chat/completions", "SCALEWAY_API_KEY",
     ["llama-3.3-70b-instruct", "deepseek-r1"]),
]
# 15. Cloudflare Workers AI (non-standard URL -- handled separately)
CF_ACCOUNT_ID = os.environ.get("CF_ACCOUNT_ID", "")
CF_API_TOKEN  = os.environ.get("CF_API_TOKEN", "")
CF_MODELS     = ["@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                 "@cf/deepseek/deepseek-r1-distill-qwen-32b"]


def build_teachers():
    teachers = []
    for name, url, env, models in SPECS:
        keys = [k.strip() for k in os.environ.get(env, "").split(",") if k.strip()]
        for i, key in enumerate(keys):
            for model in models:
                teachers.append({"name": f"{name}#{i}:{model.split('/')[-1]}",
                                 "url": url, "key": key, "model": model,
                                 "cool_until": 0.0, "fails": 0, "provider": name})
    # Cloudflare special URL
    if CF_ACCOUNT_ID and CF_API_TOKEN:
        for model in CF_MODELS:
            cf_url = (f"https://api.cloudflare.com/client/v4/accounts/"
                      f"{CF_ACCOUNT_ID}/ai/run/{model}")
            teachers.append({"name": f"cloudflare:0:{model.split('/')[-1]}",
                             "url": cf_url, "key": CF_API_TOKEN, "model": model,
                             "cool_until": 0.0, "fails": 0, "provider": "cloudflare"})
    return teachers


DOMAINS = [
    "step-by-step math reasoning", "competitive coding in Python",
    "debugging and fixing code", "science explanation (physics/bio/chem)",
    "logical & lateral reasoning puzzles", "history and geography",
    "Hindi/Hinglish tutoring for students", "essay and creative writing",
    "data analysis and SQL", "system design and architecture",
    "machine learning concepts", "real-world how-to and life advice",
    "business and finance basics", "grammar and language learning",
    "ethical dilemmas and balanced reasoning", "agentic tool-use planning",
]
AUDIENCES = ["a 10-year-old", "a college student", "an expert", "a beginner coder",
             "a Hindi-speaking student", "a busy professional"]
EVOLVE = ["make it deeper and more detailed", "make it harder and more nuanced",
          "add a real-world constraint", "require multi-step reasoning",
          "broaden it to a related edge case"]


def healthy(teachers):
    now = time.time()
    return [t for t in teachers if t["cool_until"] < now]


def mark_fail(t):
    t["fails"] += 1
    if t["fails"] >= 3:
        t["cool_until"] = time.time() + COOLDOWN
        t["fails"] = 0
        print(f"  [switch] {t['name']} -> cooldown {COOLDOWN}s")


def _chat(t, messages, max_tokens=1024, temperature=0.8):
    """Universal chat caller -- handles Cloudflare and Cohere special formats."""
    headers = {"Authorization": "Bearer " + t["key"], "Content-Type": "application/json"}

    if t.get("provider") == "cloudflare":
        body = {"messages": messages, "max_tokens": max_tokens, "temperature": temperature}
        r = requests.post(t["url"], headers=headers, data=json.dumps(body), timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        t["fails"] = 0
        return r.json().get("result", {}).get("response", "").strip()

    elif t.get("provider") == "cohere":
        # Cohere v2 returns message.content[0].text
        body = {"model": t["model"], "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature}
        r = requests.post(t["url"], headers=headers, data=json.dumps(body), timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        t["fails"] = 0
        resp = r.json()
        return resp["message"]["content"][0]["text"].strip()

    else:
        # Standard OpenAI-compatible
        body = {"model": t["model"], "messages": messages,
                "max_tokens": max_tokens, "temperature": temperature}
        r = requests.post(t["url"], headers=headers, data=json.dumps(body), timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        t["fails"] = 0
        return r.json()["choices"][0]["message"]["content"].strip()


def _norm(s):
    return re.sub(r"\s+", " ", (s or "").lower()).strip()[:400]


def _hash(s):
    return hashlib.sha1(_norm(s).encode("utf-8")).hexdigest()


def _load_seen():
    if os.path.exists(SEEN_FILE):
        with open(SEEN_FILE) as f:
            return set(x.strip() for x in f if x.strip())
    return set()


def gen_instruction(t):
    domain = random.choice(DOMAINS)
    aud = random.choice(AUDIENCES)
    evo = random.choice(EVOLVE)
    sys_p = ("You are an expert dataset author. Output ONE single high-quality "
             "instruction/question only -- no answer, no preamble, no numbering.")
    user = (f"Write one challenging instruction about '{domain}' aimed at {aud}. "
            f"Then {evo}. Some of the time write it in Hindi or Hinglish. "
            f"Return ONLY the instruction text.")
    q = _chat(t, [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
              max_tokens=200, temperature=1.0)
    return re.sub(r'^[\d\.\)\-\s"]+', "", q).strip().strip('"')


def gen_answer(t, instruction):
    sys_p = ("You are Damru, a brilliant, precise Indian AI tutor. Answer fully and "
             "correctly with clear reasoning. If code, make it runnable. Match the "
             "language of the question (Hindi/Hinglish/English).")
    return _chat(t, [{"role": "system", "content": sys_p},
                     {"role": "user", "content": instruction}], max_tokens=1400, temperature=0.6)


def verify(t, instruction, answer):
    sys_p = "You are a strict grader. Reply with ONLY an integer 1-5."
    user = ("Rate the answer's correctness, completeness, and helpfulness 1-5.\n\n"
            f"QUESTION:\n{instruction}\n\nANSWER:\n{answer}\n\nScore (1-5):")
    out = _chat(t, [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                max_tokens=5, temperature=0.0)
    m = re.search(r"[1-5]", out)
    return int(m.group(0)) if m else 0


def push_dataset(path):
    if not HF_TOKEN:
        print("[warn] no HF_TOKEN -> data saved locally only")
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN.split(",")[0].strip())
        api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        api.upload_file(path_or_fileobj=path,
                        path_in_repo=f"forge/damru_forge_{stamp}.jsonl",
                        repo_id=DATASET_REPO, repo_type="dataset")
        print("[push] ->", DATASET_REPO, stamp)
    except Exception as e:
        print("[warn] push failed:", str(e)[:200])


def safe(fn, *a):
    """Run a teacher call; on failure mark_fail + return None so loop self-switches."""
    t = a[0]
    try:
        return fn(*a)
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", 0)
        print(f"  http {code} on {t['name']}")
        mark_fail(t)
        if code == 429:
            t["cool_until"] = time.time() + COOLDOWN
        return None
    except Exception as e:
        print("  err", t["name"], str(e)[:120])
        mark_fail(t)
        return None


def main():
    teachers = build_teachers()
    if not teachers:
        print("NO API KEYS FOUND. Set any of:\n"
              "  CEREBRAS_API_KEY, OPENROUTER_API_KEY, GH_MODELS_TOKEN_VAL, HF_TOKEN,\n"
              "  SAMBANOVA_API_KEY, TOGETHER_API_KEY, DEEPINFRA_API_KEY, HYPERBOLIC_API_KEY,\n"
              "  MISTRAL_API_KEY, NVIDIA_NIM_KEY, CF_API_TOKEN+CF_ACCOUNT_ID,\n"
              "  FIREWORKS_API_KEY, COHERE_API_KEY, CHUTES_API_KEY, SCALEWAY_API_KEY")
        sys.exit(1)
    print(f"[Damru Forge] {len(teachers)} teacher slots loaded:")
    for t in teachers:
        print(f"  + {t['name']}")
    seen = _load_seen()
    made, it = 0, 0
    fout  = open(OUT_JSONL, "a", encoding="utf-8")
    fseen = open(SEEN_FILE, "a", encoding="utf-8")
    while True:
        it += 1
        if MAX_ITERS and it > MAX_ITERS:
            break
        pool = healthy(teachers)
        if not pool:
            nap = 30
            print(f"  all teachers cooling -> sleep {nap}s")
            time.sleep(nap)
            continue
        tq = random.choice(pool)
        ta = random.choice(pool)
        tv = random.choice(pool)
        q = safe(gen_instruction, tq)
        if not q or len(q) < 12:
            continue
        h = _hash(q)
        if h in seen:
            continue
        a = safe(gen_answer, ta, q)
        if not a or len(a) < 20:
            continue
        score = safe(verify, tv, q, a) or 0
        if score < KEEP_SCORE:
            print(f"  drop (score {score}) :: {q[:60]}")
            continue
        rec = {
            "messages": [{"role": "user", "content": q},
                         {"role": "assistant", "content": a}],
            "instruction": q, "output": a, "score": score,
            "teacher_q": tq["model"], "teacher_a": ta["model"],
            "ts": time.time()
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        fseen.write(h + "\n")
        fseen.flush()
        seen.add(h)
        made += 1
        print(f"[{made}] score {score} | {ta['name']} | {q[:66]}")
        if made % PUSH_EVERY == 0:
            push_dataset(OUT_JSONL)
        time.sleep(float(os.environ.get("DAMRU_SLEEP", "1.5")))
    fout.close()
    fseen.close()
    if made:
        push_dataset(OUT_JSONL)
    print(f"\nDONE. {made} quality examples this run -> {OUT_JSONL}")


if __name__ == "__main__":
    main()
