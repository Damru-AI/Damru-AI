#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 DAMRU DATA FORGE  --  15-Provider Knowledge Data Engine
================================================================================
Damru Autopilot ke liye knowledge/QA/instruction data generate karta hai.
Har 2 ghante GitHub Actions par chalta hai -> HF dataset me push karta hai.

Self-healing design:
  - 15 providers, rotation + cooldown
  - DAMRU_SKIP_MISSING=1 => missing key pe crash nahi
  - Dedup by hash, resume-safe
  - Time budget se clean exit

ENV:
  HF_TOKEN             (required) HuggingFace write token
  DAMRU_DATASET        default 'Damaru-ai/damru-knowledge'
  DAMRU_MAX_ITERS      default 1200 (0 = unlimited)
  DAMRU_PUSH_EVERY     default 40
  DAMRU_KEEP_SCORE     default 4 (min score to keep, out of 10)
  DAMRU_SKIP_MISSING   default 1 (1 = skip providers with no key, 0 = fail)

PROVIDERS (all optional but at least 1 needed):
  CEREBRAS_API_KEY, OPENROUTER_API_KEY, GH_MODELS_TOKEN_VAL,
  SAMBANOVA_API_KEY, TOGETHER_API_KEY, DEEPINFRA_API_KEY,
  HYPERBOLIC_API_KEY, MISTRAL_API_KEY, NVIDIA_NIM_KEY,
  CF_ACCOUNT_ID + CF_API_TOKEN, FIREWORKS_API_KEY,
  COHERE_API_KEY, CHUTES_API_KEY, SCALEWAY_API_KEY, HF_TOKEN (HF router)
================================================================================
"""
import os
import re
import sys
import json
import time
import random
import hashlib
import traceback
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("[FATAL] pip install requests", flush=True)
    raise

try:
    from huggingface_hub import HfApi, upload_file
    _HAS_HF = True
except ImportError:
    _HAS_HF = False


# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------
def now_iso():
    return datetime.now(timezone.utc).isoformat()

def log(*a):
    print(f"[{now_iso()}]", *a, flush=True)

def env(name, default=None):
    v = os.environ.get(name)
    return v if (v is not None and str(v).strip() != "") else default

def env_int(name, default):
    try:
        return int(str(env(name, default)).strip())
    except Exception:
        return default

def sha256(s):
    return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------
class Provider:
    def __init__(self, name, url, model, key, headers=None):
        self.name = name
        self.url = url
        self.model = model
        self.key = key
        self.extra_headers = headers or {}
        self.fails = 0
        self.cool_until = 0.0

    def healthy(self):
        return time.time() >= self.cool_until

    def chat(self, messages, temperature=0.7, max_tokens=1024, timeout=60):
        if not self.healthy():
            return None
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        h.update(self.extra_headers)
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            r = requests.post(self.url, headers=h, json=payload, timeout=timeout)
            if r.status_code == 200:
                txt = (r.json().get("choices", [{}])[0]
                              .get("message", {}).get("content"))
                if txt and str(txt).strip():
                    self.fails = 0
                    return str(txt).strip()
                self._trip("empty")
                return None
            if r.status_code in (401, 403, 404):
                log(f"  [dead] {self.name} HTTP {r.status_code} — disabling")
                self.cool_until = time.time() + 86400
                return None
            self._trip(f"http{r.status_code}")
            return None
        except Exception as e:
            self._trip(str(e)[:60])
            return None

    def _trip(self, why):
        self.fails += 1
        cool = min(1200, 60 * self.fails)
        self.cool_until = time.time() + cool
        log(f"  [cool] {self.name} {cool}s ({why}, fails={self.fails})")


def build_providers():
    skip_missing = env_int("DAMRU_SKIP_MISSING", 1)
    provs = []

    def add(name, url, model, key, headers=None):
        if not key:
            if not skip_missing:
                log(f"[WARN] {name}: no key, DAMRU_SKIP_MISSING=0 -> abort")
                sys.exit(1)
            return  # silently skip
        provs.append(Provider(name, url, model, key, headers))

    hf = env("HF_TOKEN")

    add("cerebras",
        "https://api.cerebras.ai/v1/chat/completions",
        env("CEREBRAS_MODEL", "llama-3.3-70b"),
        env("CEREBRAS_API_KEY"))

    add("openrouter",
        "https://openrouter.ai/api/v1/chat/completions",
        env("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
        env("OPENROUTER_API_KEY"),
        {"HTTP-Referer": "https://damru-ai.vercel.app", "X-Title": "Damru Forge"})

    # NOTE: GH_MODELS_TOKEN_VAL (not GITHUB_* — GitHub Actions blocks GITHUB_* prefix)
    add("gh-models",
        "https://models.github.ai/inference/chat/completions",
        env("GH_MODEL", "openai/gpt-4.1-mini"),
        env("GH_MODELS_TOKEN_VAL"))

    add("sambanova",
        "https://api.sambanova.ai/v1/chat/completions",
        env("SAMBANOVA_MODEL", "Meta-Llama-3.3-70B-Instruct"),
        env("SAMBANOVA_API_KEY"))

    add("together",
        "https://api.together.xyz/v1/chat/completions",
        env("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"),
        env("TOGETHER_API_KEY"))

    add("deepinfra",
        "https://api.deepinfra.com/v1/openai/chat/completions",
        env("DEEPINFRA_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        env("DEEPINFRA_API_KEY"))

    add("hyperbolic",
        "https://api.hyperbolic.xyz/v1/chat/completions",
        env("HYPERBOLIC_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
        env("HYPERBOLIC_API_KEY"))

    add("mistral",
        "https://api.mistral.ai/v1/chat/completions",
        env("MISTRAL_MODEL", "mistral-small-latest"),
        env("MISTRAL_API_KEY"))

    add("nvidia-nim",
        "https://integrate.api.nvidia.com/v1/chat/completions",
        env("NIM_MODEL", "meta/llama-3.3-70b-instruct"),
        env("NVIDIA_NIM_KEY"))

    # Cloudflare Workers AI
    cf_account = env("CF_ACCOUNT_ID")
    cf_token = env("CF_API_TOKEN")
    if cf_account and cf_token:
        add("cloudflare",
            f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/ai/v1/chat/completions",
            env("CF_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast"),
            cf_token)

    add("fireworks",
        "https://api.fireworks.ai/inference/v1/chat/completions",
        env("FIREWORKS_MODEL", "accounts/fireworks/models/llama-v3p3-70b-instruct"),
        env("FIREWORKS_API_KEY"))

    add("cohere",
        "https://api.cohere.com/v2/chat",
        env("COHERE_MODEL", "command-r-plus-08-2024"),
        env("COHERE_API_KEY"))

    add("chutes",
        "https://llm.chutes.ai/v1/chat/completions",
        env("CHUTES_MODEL", "unsloth/Llama-3.3-70B-Instruct"),
        env("CHUTES_API_KEY"))

    add("scaleway",
        "https://api.scaleway.ai/v1/chat/completions",
        env("SCALEWAY_MODEL", "llama-3.3-70b-instruct"),
        env("SCALEWAY_API_KEY"))

    # HF Router as fallback
    if hf:
        add("hf-router",
            "https://router.huggingface.co/v1/chat/completions",
            env("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct"),
            hf)

    return provs


# ---------------------------------------------------------------------------
# Topic generator
# ---------------------------------------------------------------------------
TOPICS = [
    "Python programming", "Machine learning basics", "Data structures",
    "Web development", "Linux commands", "Git and version control",
    "SQL databases", "APIs and REST", "Algorithms", "Mathematics",
    "Physics", "Chemistry", "Biology", "History", "Geography",
    "Economics", "Philosophy", "Psychology", "Literature", "Grammar",
    "Cooking", "Health and nutrition", "Exercise and fitness",
    "Personal finance", "Career advice", "Communication skills",
    "Critical thinking", "Problem solving", "Creativity", "Leadership",
    "Cloud computing", "Cybersecurity", "DevOps", "Kubernetes", "Docker",
    "React", "TypeScript", "Rust", "Go programming", "System design",
]

QUESTION_TYPES = [
    "explain concept", "step-by-step tutorial", "compare and contrast",
    "pros and cons", "real-world example", "common mistakes",
    "best practices", "quick overview", "detailed deep dive",
    "beginner guide", "advanced tips", "troubleshooting guide",
]


def generate_qa_prompt():
    topic = random.choice(TOPICS)
    qtype = random.choice(QUESTION_TYPES)
    return (
        f"Generate ONE high-quality question and answer pair about: {topic}\n"
        f"Style: {qtype}\n"
        "Format (strict JSON, no extra text):\n"
        '{"question": "<clear question>", "answer": "<detailed helpful answer>"}'
    )


def generate_instruction_prompt():
    topic = random.choice(TOPICS)
    return (
        f"Create ONE instruction-following example about: {topic}\n"
        "The instruction should require a helpful, detailed response.\n"
        "Format (strict JSON, no extra text):\n"
        '{"instruction": "<task instruction>", "response": "<high-quality response>"}'
    )


# ---------------------------------------------------------------------------
# Brain (round-robin + heal)
# ---------------------------------------------------------------------------
class Brain:
    def __init__(self, providers):
        self.providers = providers
        self.idx = 0

    def alive(self):
        return any(p.healthy() for p in self.providers)

    def ask(self, messages, temperature=0.7, max_tokens=1024):
        n = len(self.providers)
        if n == 0:
            return None
        for _ in range(n * 2):
            p = self.providers[self.idx % n]
            self.idx += 1
            if not p.healthy():
                continue
            out = p.chat(messages, temperature=temperature, max_tokens=max_tokens)
            if out:
                return out
        return None


# ---------------------------------------------------------------------------
# JSON extractor
# ---------------------------------------------------------------------------
def extract_json(text):
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # fenced block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # first balanced brace
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except Exception:
                        break
        start = text.find("{", start + 1)
    return None


# ---------------------------------------------------------------------------
# HF push
# ---------------------------------------------------------------------------
def hf_push(dataset, records, hf_token):
    if not _HAS_HF or not hf_token or not records:
        if records:
            log(f"[hf] skip push ({len(records)} records — no token or lib)")
        return
    try:
        api = HfApi(token=hf_token)
        try:
            api.create_repo(dataset, repo_type="dataset", exist_ok=True, private=True)
        except Exception:
            pass
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        fname = f"/tmp/forge_{stamp}.jsonl"
        with open(fname, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        upload_file(
            path_or_fileobj=fname,
            path_in_repo=f"forge/forge_{stamp}.jsonl",
            repo_id=dataset, repo_type="dataset", token=hf_token,
        )
        log(f"[hf] pushed {len(records)} records -> {dataset}/forge/forge_{stamp}.jsonl")
        os.unlink(fname)
    except Exception as e:
        log(f"[hf] push failed: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    hf_token = env("HF_TOKEN")
    if not hf_token:
        log("[WARN] HF_TOKEN missing — data will not be pushed")

    dataset = env("DAMRU_DATASET", "Damaru-ai/damru-knowledge")
    max_iters = env_int("DAMRU_MAX_ITERS", 1200)
    push_every = env_int("DAMRU_PUSH_EVERY", 40)
    keep_score = env_int("DAMRU_KEEP_SCORE", 4)  # min answer length in words
    time_budget_min = env_int("DAMRU_TIME_BUDGET_MIN", 50)  # autopilot has 55min
    t_start = time.time()
    deadline = t_start + time_budget_min * 60

    log("=" * 60)
    log("DAMRU DATA FORGE starting")
    log(f"  dataset={dataset} max_iters={max_iters} push_every={push_every}")

    providers = build_providers()
    if not providers:
        log("[FATAL] No providers active. Set at least one API key secret.")
        log("  Needed: CEREBRAS_API_KEY | OPENROUTER_API_KEY | GH_MODELS_TOKEN_VAL")
        log("  | SAMBANOVA_API_KEY | TOGETHER_API_KEY | etc.")
        sys.exit(1)

    log(f"  active providers ({len(providers)}): " +
        ", ".join(f"{p.name}" for p in providers))
    brain = Brain(providers)

    seen = set()
    records = []
    total = 0
    pushed = 0

    for iteration in range(1, max_iters + 1 if max_iters else 10**9):
        if time.time() >= deadline:
            log(f"[done] time budget {time_budget_min}min reached")
            break
        if not brain.alive():
            log("[heal] all providers cooling, sleeping 60s")
            time.sleep(60)
            continue

        try:
            # Alternate between QA and instruction formats
            if iteration % 2 == 0:
                prompt = generate_qa_prompt()
                mode = "qa"
            else:
                prompt = generate_instruction_prompt()
                mode = "instr"

            raw = brain.ask(
                [{"role": "user", "content": prompt}],
                temperature=random.uniform(0.6, 0.9),
                max_tokens=1024,
            )
            if not raw:
                continue

            data = extract_json(raw)
            if not data:
                continue

            if mode == "qa":
                q = str(data.get("question", "")).strip()
                a = str(data.get("answer", "")).strip()
                if not q or not a:
                    continue
                if len(a.split()) < keep_score * 5:  # too short
                    continue
                h = sha256(q)
                if h in seen:
                    continue
                seen.add(h)
                records.append({
                    "type": "qa",
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a},
                    ],
                    "ts": now_iso(),
                })
            else:
                ins = str(data.get("instruction", "")).strip()
                resp = str(data.get("response", "")).strip()
                if not ins or not resp:
                    continue
                if len(resp.split()) < keep_score * 5:
                    continue
                h = sha256(ins)
                if h in seen:
                    continue
                seen.add(h)
                records.append({
                    "type": "instruction",
                    "messages": [
                        {"role": "user", "content": ins},
                        {"role": "assistant", "content": resp},
                    ],
                    "ts": now_iso(),
                })

            total += 1
            if total % 10 == 0:
                log(f"  iter={iteration} total={total} pending_push={len(records)-pushed}")

            if len(records) - pushed >= push_every:
                hf_push(dataset, records[pushed:], hf_token)
                pushed = len(records)

        except Exception:
            log(f"[iter {iteration}] error (continuing):")
            log(traceback.format_exc()[:500])

    # final push
    if len(records) > pushed:
        hf_push(dataset, records[pushed:], hf_token)

    log("=" * 60)
    log(f"DAMRU DATA FORGE done. total_generated={total} total_records={len(records)}")
    active_now = sum(1 for p in providers if p.healthy())
    log(f"  providers alive at end: {active_now}/{len(providers)}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        log("[FATAL]")
        log(traceback.format_exc())
        sys.exit(1)
