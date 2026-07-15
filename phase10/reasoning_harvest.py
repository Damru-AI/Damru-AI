#!/usr/bin/env python3
"""
Nightly GURUKUL harvest runner (GitHub Action).
Wires FREE teacher endpoints -> damru_gurukul -> pushes reasoning traces to HF.

ETHIC: OFFLINE ONLY. Teachers generate training data; Damru never calls them
live. Once distilled into RAG (today) + weights (Brain Forge), teacher dropped.

JUGAD for API/limits: keyless Pollinations with MULTIPLE models = multiple
teachers from ONE endpoint; OpenRouter free tier added only if key present;
TeacherPool round-robin + cooldown spreads load.

Env:
  HF_TOKEN        (required to push)   HF WRITE token
  OPENROUTER_KEY  (optional)           adds OpenRouter free teachers
  TRACES_REPO     (default Damaru-ai/damru-reasoning-traces)
  PER_DOMAIN      (default 2)          curriculum questions per domain
  MIN_AGREEMENT   (default 0.6)        consensus threshold
  DRY_RUN         (=1)                 mock teachers, no network, no push (local test)
"""
import os
import sys
import json
import time
import traceback
import urllib.request

# import damru_gurukul.py from repo root (this file lives in phase10/)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from damru_gurukul import curriculum, TeacherPool, harvest  # noqa: E402

POLL_URL = "https://text.pollinations.ai/openai"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"


def _post(url, payload, headers, timeout=90):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def make_pollinations_teacher(model):
    def fn(messages):
        out = _post(POLL_URL,
                    {"model": model, "messages": messages, "temperature": 0.4,
                     "seed": int(time.time()) % 100000, "referrer": "damru-gurukul"},
                    {"Content-Type": "application/json"})
        return out["choices"][0]["message"]["content"]
    return fn


def make_openrouter_teacher(model, key):
    def fn(messages):
        out = _post(OR_URL,
                    {"model": model, "messages": messages,
                     "temperature": 0.4, "max_tokens": 1024},
                    {"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}"})
        return out["choices"][0]["message"]["content"]
    return fn


def build_pool():
    teachers = []
    # keyless jugad: one endpoint, multiple models = multiple teachers
    for m in ["openai", "mistral", "deepseek"]:
        teachers.append((f"poll:{m}", make_pollinations_teacher(m)))
    key = os.environ.get("OPENROUTER_KEY")
    if key:
        for m in ["meta-llama/llama-3.3-70b-instruct:free",
                  "google/gemini-2.0-flash-exp:free",
                  "deepseek/deepseek-r1:free"]:
            teachers.append((f"or:{m.split('/')[-1][:16]}",
                             make_openrouter_teacher(m, key)))
    print(f"[gurukul] {len(teachers)} teachers wired: {[t[0] for t in teachers]}")
    return TeacherPool(teachers, cooldown_s=60)


def push_to_hf(rows, repo, token):
    from datasets import Dataset, concatenate_datasets, load_dataset
    new_ds = Dataset.from_list(rows)
    try:
        old = load_dataset(repo, split="train", token=token)
        merged = concatenate_datasets([old, new_ds])
    except Exception:
        merged = new_ds  # first run: repo empty / missing
    merged.push_to_hub(repo, token=token, private=True)
    return len(merged)


def main():
    per_domain = int(os.environ.get("PER_DOMAIN", "2"))
    min_agree = float(os.environ.get("MIN_AGREEMENT", "0.6"))
    repo = os.environ.get("TRACES_REPO", "Damaru-ai/damru-reasoning-traces")
    dry = os.environ.get("DRY_RUN") == "1"

    if dry:
        def good(m):
            return "Step 1: reason. Step 2: verify. FINAL: 42"
        def good2(m):
            return "Analyze then conclude. FINAL: 42"
        def bad(m):
            return "FINAL: 999"
        pool = TeacherPool([("t1", good), ("t2", good2), ("t3", bad)])
    else:
        pool = build_pool()

    items = curriculum(n_per_domain=per_domain)
    print(f"[gurukul] {len(items)} curriculum questions; dry={dry}")
    res = harvest(items, pool, min_agreement=min_agree)
    print(f"[gurukul] kept={res['kept']} dropped={res['dropped']}")

    if dry:
        print(json.dumps(res["rows"][:2], indent=2)[:900])
        return
    if not res["rows"]:
        print("[gurukul] nothing passed consensus; nothing to push")
        return
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[gurukul] ERROR: HF_TOKEN missing, cannot push")
        sys.exit(1)
    total = push_to_hf(res["rows"], repo, token)
    print(f"[gurukul] pushed {res['kept']} new rows -> {repo} (total {total})")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
