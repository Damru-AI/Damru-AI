#!/usr/bin/env python3
# Damru Council (Step 4) - cloud, tab-free, multi-solver.
# Providers are auto-enabled by whichever secrets exist. Uses OpenAI-compatible
# chat endpoints for vLLM / Cerebras / OpenRouter / NVIDIA NIM / Groq / Mistral / GH Models.
# Gemini is the judge. Reads problems from oracle, writes verified sft + dpo splits.
import os, sys, json, time, subprocess, tempfile, logging, random, hashlib

import requests
from datasets import load_dataset, Dataset, DatasetDict

NL = chr(10)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("council")


def env(k, d=None):
    v = os.environ.get(k)
    return v if v not in (None, "") else d


def envi(k, d):
    try:
        return int(os.environ.get(k, d))
    except Exception:
        return int(d)


def envf(k, d):
    try:
        return float(os.environ.get(k, d))
    except Exception:
        return float(d)


OUT_REPO = env("COUNCIL_OUT_REPO", "Damaru-ai/damru-council")
SRC_REPO = env("COUNCIL_SRC_REPO", "Damaru-ai/damru-oracle")
SRC_SPLIT = env("COUNCIL_SRC_SPLIT", "train")
MAX_PROBLEMS = envi("COUNCIL_MAX_PROBLEMS", 0)
TIME_BUDGET_MIN = envi("COUNCIL_TIME_BUDGET_MIN", 300)
COMMIT_EVERY_SEC = envi("COUNCIL_COMMIT_EVERY_SEC", 600)
N_SOLVERS = envi("COUNCIL_N_SOLVERS", 3)
REQ_TIMEOUT = envi("COUNCIL_REQ_TIMEOUT", 90)
MAX_TOKENS = envi("COUNCIL_MAX_TOKENS", 1024)
TEMPERATURE = envf("COUNCIL_TEMPERATURE", 0.3)
CODE_EXEC = envi("COUNCIL_CODE_EXEC", 1)
CODE_TIMEOUT = envi("COUNCIL_CODE_TIMEOUT", 10)
JUDGE_MIN = envf("COUNCIL_JUDGE_MIN", 7)
HF_TOKEN = env("HF_TOKEN")


def build_providers():
    p = []
    if env("VLLM_URL"):
        p.append(dict(name="vllm", base=env("VLLM_URL"), key=env("VLLM_KEY", ""),
                      model=env("VLLM_MODEL", "")))
    if env("CEREBRAS_API_KEY"):
        p.append(dict(name="cerebras", base="https://api.cerebras.ai/v1/chat/completions",
                      key=env("CEREBRAS_API_KEY"), model=env("CEREBRAS_MODEL", "llama-3.3-70b")))
    if env("OPENROUTER_KEY"):
        p.append(dict(name="openrouter", base="https://openrouter.ai/api/v1/chat/completions",
                      key=env("OPENROUTER_KEY"),
                      model=env("OPENROUTER_MODEL", "qwen/qwen-2.5-coder-32b-instruct:free")))
    if env("NVIDIA_NIM_KEY"):
        p.append(dict(name="nim", base="https://integrate.api.nvidia.com/v1/chat/completions",
                      key=env("NVIDIA_NIM_KEY"), model=env("NIM_MODEL", "qwen/qwen2.5-coder-32b-instruct")))
    if env("GROQ_API_KEY"):
        p.append(dict(name="groq", base="https://api.groq.com/openai/v1/chat/completions",
                      key=env("GROQ_API_KEY"), model=env("GROQ_MODEL", "llama-3.3-70b-versatile")))
    if env("MISTRAL_API_KEY"):
        p.append(dict(name="mistral", base="https://api.mistral.ai/v1/chat/completions",
                      key=env("MISTRAL_API_KEY"), model=env("MISTRAL_MODEL", "mistral-large-latest")))
    if env("GH_MODELS_TOKEN"):
        p.append(dict(name="gh_models", base="https://models.inference.ai.azure.com/chat/completions",
                      key=env("GH_MODELS_TOKEN"), model=env("GH_MODELS_MODEL", "gpt-4o-mini")))
    return p


def openai_chat(prov, messages, retries=3):
    headers = {"Authorization": "Bearer " + str(prov["key"]), "Content-Type": "application/json"}
    if prov["name"] == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/Damru-AI"
        headers["X-Title"] = "Damru Council"
    payload = {"model": prov["model"], "messages": messages,
               "max_tokens": MAX_TOKENS, "temperature": TEMPERATURE}
    for a in range(retries):
        try:
            r = requests.post(prov["base"], headers=headers, json=payload, timeout=REQ_TIMEOUT)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            log.warning("[retry] chat:%s failed (%d/%d): HTTP %d: %s",
                        prov["name"], a + 1, retries, r.status_code, r.text[:160])
            # permission / model / not-found errors will NOT fix by retrying -> bail out fast
            if r.status_code in (400, 401, 403, 404):
                return None
        except Exception as e:
            log.warning("[retry] chat:%s exc (%d/%d): %s",
                        prov["name"], a + 1, retries, str(e)[:150])
        time.sleep(min(2 ** a, 8) + random.random())
    return None


def gemini_judge(problem, solution):
    key = env("GEMINI_KEY") or env("GEMINI_API_KEY")
    if not key:
        return 10.0  # no judge configured -> pass verified solutions through
    model = env("GEMINI_JUDGE_MODEL", "gemini-2.0-flash")
    url = ("https://generativelanguage.googleapis.com/v1beta/models/"
           + model + ":generateContent?key=" + key)
    prompt = ("Rate from 0 to 10 how correct and complete this SOLUTION is for the PROBLEM. "
              "Reply with ONLY a number." + NL + NL + "PROBLEM:" + NL + problem[:4000]
              + NL + NL + "SOLUTION:" + NL + solution[:4000])
    try:
        r = requests.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=REQ_TIMEOUT)
        if r.status_code == 200:
            t = r.json()["candidates"][0]["content"]["parts"][0]["text"]
            n = first_number(t)
            if n is not None:
                return n
        else:
            log.warning("judge HTTP %d: %s", r.status_code, r.text[:120])
    except Exception as e:
        log.warning("judge exc: %s", str(e)[:150])
    return 0.0


def first_number(t):
    num = ""
    for ch in t:
        if ch.isdigit() or (ch == "." and num):
            num += ch
        elif num:
            break
    try:
        return float(num) if num else None
    except Exception:
        return None


def extract_code(text):
    if not text:
        return ""
    if "```" in text:
        parts = text.split("```")
        blocks = []
        for i in range(1, len(parts), 2):
            b = parts[i]
            nl = b.find(NL)
            if nl != -1 and b[:nl].strip().lower() in ("python", "py", ""):
                b = b[nl + 1:]
            blocks.append(b)
        if blocks:
            return max(blocks, key=len).strip()
    return text.strip()


def looks_like_python(code):
    if not code:
        return False
    hints = ("def ", "import ", "print(", "class ", "return ", "for ", "while ", "=")
    return any(h in code for h in hints)


def run_code(code):
    path = None
    try:
        f = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
        f.write(code)
        f.close()
        path = f.name
        pr = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=CODE_TIMEOUT)
        return pr.returncode == 0, (pr.stdout + pr.stderr)[-400:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:200]
    finally:
        if path:
            try:
                os.remove(path)
            except Exception:
                pass


def hash_prompt(s):
    return hashlib.md5(s.strip().encode("utf-8", "ignore")).hexdigest()


def extract_prompt(row):
    for k in ("problem", "prompt", "question", "instruction", "input", "text"):
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def load_existing():
    sft, dpo = [], []
    try:
        d = load_dataset(OUT_REPO, split="sft", token=HF_TOKEN)
        sft = [dict(x) for x in d]
    except Exception:
        pass
    try:
        d = load_dataset(OUT_REPO, split="dpo", token=HF_TOKEN)
        dpo = [dict(x) for x in d]
    except Exception:
        pass
    return sft, dpo


def push(sft_rows, dpo_rows):
    dd = {}
    if sft_rows:
        dd["sft"] = Dataset.from_list(sft_rows)
    if dpo_rows:
        dd["dpo"] = Dataset.from_list(dpo_rows)
    if not dd:
        return
    DatasetDict(dd).push_to_hub(OUT_REPO, token=HF_TOKEN)


def main():
    provs = build_providers()
    if not provs:
        log.error("No solver providers configured. Set at least one secret: "
                  "CEREBRAS_API_KEY, OPENROUTER_KEY, VLLM_URL, NVIDIA_NIM_KEY, "
                  "GROQ_API_KEY, MISTRAL_API_KEY, or GH_MODELS_TOKEN.")
        sys.exit(1)
    log.info("Active solvers: %s", ", ".join(p["name"] + "(" + str(p["model"]) + ")" for p in provs))
    solvers = provs[:max(1, N_SOLVERS)]

    ds = load_dataset(SRC_REPO, split=SRC_SPLIT, token=HF_TOKEN)
    log.info("Loaded %d problems from %s", len(ds), SRC_REPO)

    sft_rows, dpo_rows = load_existing()
    done = set(hash_prompt(r.get("prompt", "")) for r in sft_rows if r.get("prompt"))
    log.info("Resuming with sft=%d dpo=%d (already done=%d)", len(sft_rows), len(dpo_rows), len(done))

    start = time.time()
    last_commit = time.time()
    processed = 0

    for row in ds:
        if MAX_PROBLEMS and processed >= MAX_PROBLEMS:
            break
        if (time.time() - start) / 60.0 >= TIME_BUDGET_MIN:
            log.info("Time budget reached.")
            break
        prompt = extract_prompt(row)
        if not prompt:
            continue
        h = hash_prompt(prompt)
        if h in done:
            continue

        cands = []
        for p in solvers:
            out = openai_chat(p, [{"role": "user", "content": prompt}])
            if out:
                cands.append((p["name"], out))
        if not cands:
            continue

        scored = []
        for name, out in cands:
            code = extract_code(out)
            ok = True
            if CODE_EXEC and looks_like_python(code):
                ok, _ = run_code(code)
            score = gemini_judge(prompt, out) if ok else 0.0
            scored.append(dict(name=name, out=out, code=code, ok=ok,
                               score=score, length=len(code or out)))

        passing = [s for s in scored if s["ok"] and s["score"] >= JUDGE_MIN]
        if passing:
            # efficiency selector: shortest passing solution, tie -> highest score
            best = sorted(passing, key=lambda s: (s["length"], -s["score"]))[0]
            sft_rows.append({"prompt": prompt, "response": best["out"],
                             "source": best["name"], "score": best["score"]})
            done.add(h)
            worst = sorted(scored, key=lambda s: (s["score"], -s["length"]))[0]
            if worst["out"] != best["out"]:
                dpo_rows.append({"prompt": prompt, "chosen": best["out"], "rejected": worst["out"]})
            processed += 1

        if time.time() - last_commit >= COMMIT_EVERY_SEC:
            log.info("Commit: sft=%d dpo=%d (new this run=%d)", len(sft_rows), len(dpo_rows), processed)
            try:
                push(sft_rows, dpo_rows)
            except Exception as e:
                log.warning("push failed: %s", str(e)[:200])
            last_commit = time.time()

    log.info("FINAL commit: sft=%d dpo=%d (new this run=%d)", len(sft_rows), len(dpo_rows), processed)
    try:
        push(sft_rows, dpo_rows)
        log.info("DONE. Pushed to %s", OUT_REPO)
    except Exception as e:
        log.error("final push failed: %s", str(e)[:300])
        sys.exit(1)


if __name__ == "__main__":
    main()
