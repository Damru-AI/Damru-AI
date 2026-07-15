#!/usr/bin/env python3
"""
Nightly GURUKUL harvest runner (GitHub Action)  --  v3.
Wires FREE teacher endpoints -> damru_gurukul -> pushes reasoning traces to HF.

ETHIC: OFFLINE ONLY. Teachers generate training data; Damru never calls them live.

v3 fixes: browser User-Agent (CDN 403 fix), detailed per-teacher error probe,
Pollinations GET fallback, tolerant response parsing, OpenRouter referer headers,
and CURRICULUM=research switch to harvest the PDF mind-map topics (A-to-Z).

Env:
  HF_TOKEN        (required to push)   HF WRITE token
  OPENROUTER_KEY  (optional)           adds OpenRouter free teachers
  TRACES_REPO     (default Damaru-ai/damru-reasoning-traces)
  CURRICULUM      generic | research   (research = PDF mind-map topics)
  PER_DOMAIN      (default 2)          generic curriculum only
  ANGLES          (default 3, max 6)   research curriculum depth per topic
  MIN_AGREEMENT   (default 0.6)
  DRY_RUN         (=1)                 mock teachers, no network, no push
"""
import os
import sys
import json
import time
import traceback
import urllib.request
import urllib.parse
import urllib.error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from damru_gurukul import curriculum, TeacherPool, harvest  # noqa: E402

POLL_URL = "https://text.pollinations.ai/openai"
POLL_GET = "https://text.pollinations.ai/"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
UA = "Mozilla/5.0 (compatible; DamruGurukul/1.0; +https://github.com/Damru-AI)"


def _post(url, payload, headers, timeout=90):
    hdr = {"Content-Type": "application/json", "Accept": "application/json",
           "User-Agent": UA}
    hdr.update(headers or {})
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=hdr, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")[:200]
        raise RuntimeError(f"HTTP {e.code}: {body}")


def _get(url, timeout=90):
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors="ignore")


def _extract(out):
    """Tolerant parse across provider response shapes."""
    if isinstance(out, str):
        return out
    if "choices" in out and out["choices"]:
        ch = out["choices"][0]
        return (ch.get("message", {}) or {}).get("content") or ch.get("text", "")
    for k in ("content", "text", "response", "output"):
        if out.get(k):
            return out[k]
    return ""


def make_pollinations_teacher(model):
    def fn(messages):
        try:
            out = _post(POLL_URL,
                        {"model": model, "messages": messages, "temperature": 0.4,
                         "seed": int(time.time()) % 100000, "referrer": "damru-gurukul"},
                        {})
            txt = _extract(out)
            if txt:
                return txt
            raise RuntimeError("empty POST body")
        except Exception:
            # GET fallback (classic keyless pollinations text API)
            prompt = "\n\n".join(m["content"] for m in messages)
            url = POLL_GET + urllib.parse.quote(prompt[:1500]) + "?model=" + model
            return _get(url)
    return fn


def make_openrouter_teacher(model, key):
    def fn(messages):
        out = _post(OR_URL,
                    {"model": model, "messages": messages,
                     "temperature": 0.4, "max_tokens": 1024},
                    {"Authorization": f"Bearer {key}",
                     "HTTP-Referer": "https://github.com/Damru-AI",
                     "X-Title": "Damru Gurukul"})
        return _extract(out)
    return fn


def build_pool():
    teachers = []
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


def probe(pool):
    """Diagnostic: call EACH teacher directly, print exact result/error."""
    print("[gurukul] --- teacher probe (per-teacher) ---")
    msgs = [{"role": "system", "content": "You are a teacher. End with 'FINAL: <answer>'."},
            {"role": "user", "content": "What is 12*8? Explain briefly."}]
    ok = 0
    for name, fn in pool.teachers:
        try:
            r = fn(msgs)
            if r and r.strip():
                ok += 1
                print(f"   [{name}] OK len={len(r)} :: {r.strip()[:90]!r}")
            else:
                print(f"   [{name}] EMPTY response")
        except Exception as e:
            print(f"   [{name}] ERR {type(e).__name__}: {str(e)[:150]}")
    print(f"[gurukul] probe: {ok}/{len(pool.teachers)} teachers OK")
    return ok


def get_items(dry):
    which = os.environ.get("CURRICULUM", "generic")
    if which == "research":
        from damru_research_curriculum import research_curriculum, stats
        angles = int(os.environ.get("ANGLES", "3"))
        items = research_curriculum(angles_per_topic=angles)
        print(f"[gurukul] research curriculum: {stats()} -> {len(items)} questions ({angles} angles)")
        return items
    per_domain = int(os.environ.get("PER_DOMAIN", "2"))
    items = curriculum(n_per_domain=per_domain)
    print(f"[gurukul] generic curriculum: {len(items)} questions")
    return items


def push_to_hf(rows, repo, token):
    from datasets import Dataset, concatenate_datasets, load_dataset
    new_ds = Dataset.from_list(rows)
    try:
        old = load_dataset(repo, split="train", token=token)
        merged = concatenate_datasets([old, new_ds])
    except Exception:
        merged = new_ds
    merged.push_to_hub(repo, token=token, private=True)
    return len(merged)


def main():
    min_agree = float(os.environ.get("MIN_AGREEMENT", "0.6"))
    repo = os.environ.get("TRACES_REPO", "Damaru-ai/damru-reasoning-traces")
    dry = os.environ.get("DRY_RUN") == "1"

    if dry:
        def good(m):
            return "Data structures organize data: arrays, stacks, trees. FINAL: organize data for efficient access"
        def good2(m):
            return "They structure data efficiently via arrays lists trees. FINAL: organize data for efficient access and ops"
        pool = TeacherPool([("t1", good), ("t2", good2)])
    else:
        pool = build_pool()
        if probe(pool) == 0:
            print("[gurukul] ABORT: no teacher endpoint responded (see errors above)")
            sys.exit(1)

    items = get_items(dry)
    print(f"[gurukul] harvesting {len(items)} questions; min_agreement={min_agree}; dry={dry}")
    res = harvest(items, pool, min_agreement=min_agree)
    print(f"[gurukul] kept={res['kept']} dropped={res['dropped']} reasons={res['reasons']}")
    if res["kept"] == 0 and res.get("debug"):
        print("[gurukul] sample drops:", json.dumps(res["debug"], indent=2)[:600])

    if dry:
        print(json.dumps(res["rows"][:1], indent=2)[:700])
        return
    if not res["rows"]:
        print("[gurukul] nothing passed consensus")
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
