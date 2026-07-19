#!/usr/bin/env python3
"""
DAMRU DATA FORGE  --  24/7 self-switching quality-data engine
=============================================================
GPT-4 ko takkar SIZE se nahi, DATA QUALITY + DISTILLATION se. Chhote model ko
strong OPEN-WEIGHT teachers se distill karo -- DeepSeek-R1 / Gemma ka raaz.

SELF-SWITCHING: multiple providers + multiple keys (comma-separated). Jab koi
provider 429/quota deta hai -> use cooldown me daal ke agla healthy pick.
GROQ HATA DIYA (commercial email maangta). Personal-Gmail teachers only.

RUNS on ANY free cloud CPU (GitHub Actions cron / HF Space) -- NO device needed.
DATA -> Hugging Face dataset (public ~5TB free), NOT Supabase (500MB chhota).

ENV (jo mile wo set karo; forge baaki skip kar dega). Comma se multiple keys:
  HF_TOKEN               (push + hfrouter teacher)   e.g. hf_xxx,hf_yyy
  CEREBRAS_API_KEY       cloud.cerebras.ai (no card) e.g. csk_a,csk_b
  OPENROUTER_API_KEY     openrouter.ai
  GITHUB_MODELS_TOKEN    github PAT (models:read)    -> GitHub Models
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

# --- Teachers: OPEN-WEIGHT only (distillation-legal), personal-email providers.
# NOTE: model catalogs shift; multiple models per provider add resilience.
SPECS = [
    ("cerebras",   "https://api.cerebras.ai/v1/chat/completions", "CEREBRAS_API_KEY",
     ["qwen-3-235b-a22b-instruct-2507", "gpt-oss-120b", "llama-3.3-70b"]),
    ("openrouter", "https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY",
     ["deepseek/deepseek-r1:free", "meta-llama/llama-3.3-70b-instruct:free"]),
    ("github",     "https://models.github.ai/inference/chat/completions", "GITHUB_MODELS_TOKEN",
     ["deepseek/DeepSeek-R1", "meta/Llama-3.3-70B-Instruct"]),
    ("hfrouter",   "https://router.huggingface.co/v1/chat/completions", "HF_TOKEN",
     ["meta-llama/Llama-3.3-70B-Instruct"]),
]


def build_teachers():
    teachers = []
    for name, url, env, models in SPECS:
        keys = [k.strip() for k in os.environ.get(env, "").split(",") if k.strip()]
        for i, key in enumerate(keys):
            for model in models:
                teachers.append({"name": f"{name}#{i}:{model.split('/')[-1]}",
                                 "url": url, "key": key, "model": model,
                                 "cool_until": 0.0, "fails": 0})
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
    headers = {"Authorization": "Bearer " + t["key"], "Content-Type": "application/json"}
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
    domain, aud, evo = random.choice(DOMAINS), random.choice(AUDIENCES), random.choice(EVOLVE)
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
        print("[warn] no HF_TOKEN -> data saved locally only"); return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN.split(",")[0].strip())
        api.create_repo(DATASET_REPO, repo_type="dataset", exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        api.upload_file(path_or_fileobj=path, path_in_repo=f"forge/damru_forge_{stamp}.jsonl",
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
        print(f"  http {code} on {t['name']}"); mark_fail(t)
        if code == 429:
            t["cool_until"] = time.time() + COOLDOWN
        return None
    except Exception as e:
        print("  err", t["name"], str(e)[:120]); mark_fail(t); return None


def main():
    teachers = build_teachers()
    if not teachers:
        print("NO API KEYS. Set CEREBRAS_API_KEY / OPENROUTER_API_KEY / "
              "GITHUB_MODELS_TOKEN / HF_TOKEN (comma-separate multiples).")
        sys.exit(1)
    print("Teachers loaded:", len(teachers), "->", sorted({t['name'] for t in teachers}))
    seen = _load_seen()
    made, it = 0, 0
    fout = open(OUT_JSONL, "a", encoding="utf-8")
    fseen = open(SEEN_FILE, "a", encoding="utf-8")
    while True:
        it += 1
        if MAX_ITERS and it > MAX_ITERS:
            break
        pool = healthy(teachers)
        if not pool:
            nap = 30
            print(f"  all teachers cooling -> sleep {nap}s"); time.sleep(nap); continue
        tq, ta, tv = random.choice(pool), random.choice(pool), random.choice(pool)
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
            print(f"  drop (score {score}) :: {q[:60]}"); continue
        rec = {"messages": [{"role": "user", "content": q},
                            {"role": "assistant", "content": a}],
               "instruction": q, "output": a, "score": score,
               "teacher_q": tq["model"], "teacher_a": ta["model"], "ts": time.time()}
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n"); fout.flush()
        fseen.write(h + "\n"); fseen.flush(); seen.add(h)
        made += 1
        print(f"[{made}] score {score} | {ta['name']} | {q[:66]}")
        if made % PUSH_EVERY == 0:
            push_dataset(OUT_JSONL)
        time.sleep(float(os.environ.get("DAMRU_SLEEP", "1.5")))
    fout.close(); fseen.close()
    if made:
        push_dataset(OUT_JSONL)
    print(f"\nDONE. {made} quality examples this run -> {OUT_JSONL}")


if __name__ == "__main__":
    main()
