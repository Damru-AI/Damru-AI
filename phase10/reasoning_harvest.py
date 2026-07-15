#!/usr/bin/env python3
"""
Nightly GURUKUL harvest runner (GitHub Action)  --  v4 (crash-safe / time-boxed).
Wires FREE teacher endpoints -> damru_gurukul -> pushes reasoning traces to HF.

ETHIC: OFFLINE ONLY. Teachers generate training data; Damru never calls them live.

v4 fixes (why run #5 was CANCELED at 90 min):
  - TIME BUDGET: stop harvesting at HARVEST_BUDGET_MIN and push what we have.
  - INCREMENTAL PUSH: push after every BATCH_SIZE so a timeout never loses data.
  - Shorter per-call HTTP timeout (was 90s -> 30s) so one slow teacher can't stall.
  - ANGLES default lowered to 3 (420 -> 210 questions per run).
Also keeps v3: browser UA (CDN 403 fix), per-teacher error probe, Pollinations
GET fallback, tolerant parsing, OpenRouter referer, CURRICULUM=research switch.

Env:
  HF_TOKEN        (required to push)   HF WRITE token
  OPENROUTER_KEY  (optional)           adds OpenRouter free teachers
  TRACES_REPO     (default Damaru-ai/damru-reasoning-traces)
  CURRICULUM      generic | research   (research = PDF mind-map topics)
  PER_DOMAIN      (default 2)          generic curriculum only
  ANGLES          (default 3, max 6)   research curriculum depth per topic
  MIN_AGREEMENT   (default 0.6)
  HARVEST_BUDGET_MIN (default 70)      hard time budget before graceful stop
  BATCH_SIZE      (default 15)         push cadence (crash-safe)
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
from collections import Counter

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from damru_gurukul import curriculum, TeacherPool, harvest  # noqa: E402

POLL_URL = "https://text.pollinations.ai/openai"
POLL_GET = "https://text.pollinations.ai/"
OR_URL = "https://openrouter.ai/api/v1/chat/completions"
UA = "Mozilla/5.0 (compatible; DamruGurukul/1.0; +https://github.com/Damru-AI)"
HTTP_TIMEOUT = int(os.environ.get("HTTP_TIMEOUT", "30"))


def _post(url, payload, headers, timeout=HTTP_TIMEOUT):
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


def _get(url, timeout=HTTP_TIMEOUT):
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode(errors="ignore")


def _strip_ads(t):
    """Remove Pollinations free-tier promo so it never pollutes training data."""
    if not t:
        return t
    import re
    t = re.sub(r"\n?[^\n]*Powered by Pollinations[^\n]*", "", t, flags=re.I)
    t = re.sub(r"\n?[^\n]*Support (?:our mission|Pollinations)[^\n]*", "", t, flags=re.I)
    t = re.sub(r"\n?\s*\U0001F338?\s*Ad\s*\U0001F338?\s*", "\n", t)
    return t.strip()


def _extract(out):
    if isinstance(out, str):
        return _strip_ads(out)
    if "choices" in out and out["choices"]:
        ch = out["choices"][0]
        return _strip_ads((ch.get("message", {}) or {}).get("content") or ch.get("text", ""))
    for k in ("content", "text", "response", "output"):
        if out.get(k):
            return _strip_ads(out[k])
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
            prompt = "\n\n".join(m["content"] for m in messages)
            url = POLL_GET + urllib.parse.quote(prompt[:1500]) + "?model=" + model
            return _strip_ads(_get(url))
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


def load_old(repo, token):
    try:
        from datasets import load_dataset
        return load_dataset(repo, split="train", token=token)
    except Exception:
        return None


def push_to_hf(new_rows, repo, token, old_ds):
    from datasets import Dataset, concatenate_datasets
    ds = Dataset.from_list(new_rows)
    merged = concatenate_datasets([old_ds, ds]) if old_ds is not None else ds
    merged.push_to_hub(repo, token=token, private=True)
    return len(merged)


def main():
    min_agree = float(os.environ.get("MIN_AGREEMENT", "0.6"))
    repo = os.environ.get("TRACES_REPO", "Damaru-ai/damru-reasoning-traces")
    budget = float(os.environ.get("HARVEST_BUDGET_MIN", "70"))
    batch_size = int(os.environ.get("BATCH_SIZE", "15"))
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
    token = os.environ.get("HF_TOKEN")
    if not dry and not token:
        print("[gurukul] ERROR: HF_TOKEN missing, cannot push")
        sys.exit(1)

    old_ds = None if dry else load_old(repo, token)
    if old_ds is not None:
        print(f"[gurukul] existing dataset rows: {len(old_ds)}")

    # ---- crash-safe batched harvest with time budget ----
    start = time.time()
    seen, all_rows = set(), []
    reasons = Counter()
    pushed = 0
    stopped_early = False
    for i in range(0, len(items), batch_size):
        if (time.time() - start) / 60.0 >= budget:
            print(f"[gurukul] TIME BUDGET {budget}min reached at item {i}; graceful stop")
            stopped_early = True
            break
        batch = items[i:i + batch_size]
        res = harvest(batch, pool, min_agreement=min_agree, seen=seen)
        seen = res["seen"]
        for k, v in res["reasons"].items():
            reasons[k] += v
        if res["rows"]:
            all_rows.extend(res["rows"])
        done = min(i + batch_size, len(items))
        print(f"[gurukul] batch {done}/{len(items)} | kept_total={len(all_rows)} "
              f"| {(time.time()-start)/60:.1f}min | reasons={dict(reasons)}")
        # incremental push so a later timeout never loses this batch
        if not dry and all_rows:
            try:
                total = push_to_hf(all_rows, repo, token, old_ds)
                pushed = len(all_rows)
                print(f"[gurukul]   pushed {pushed} new rows -> {repo} (total {total})")
            except Exception as e:
                print(f"[gurukul]   push failed (will retry next batch): {str(e)[:150]}")

    print(f"[gurukul] DONE kept={len(all_rows)} pushed={pushed} "
          f"reasons={dict(reasons)} stopped_early={stopped_early}")
    if len(all_rows) == 0:
        print("[gurukul] nothing passed consensus (see probe + reasons above)")
    if dry:
        print(json.dumps(all_rows[:1], indent=2)[:700])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
