#!/usr/bin/env python3
"""
DAMRU BULK HARVESTER (phase7)  --  source -> dedup filter -> DIRECT to HuggingFace.

THE PLAN (yours, hardened):
  1. STREAM big, genuine open Q&A / reasoning datasets from the HF Hub
     (streaming = no full download, low memory, low disk).
  2. MIDDLE FILTER: a persistent BLOOM filter (phase7/dedup_bloom.py) stored on
     HF guarantees NO duplicate / copied row is ever written again -- across
     every run AND across the live engine track.
  3. Write kept rows as parquet SHARDS straight into the HF dataset repo
     (data/bulk-*.parquet). SUPABASE IS BYPASSED for bulk -> the 500MB NANO
     buffer is no longer the bottleneck, so we scale to tens of millions.
  4. Fully AUTOMATIC + RESUMABLE: a state file on HF records which datasets are
     done; a scheduled GitHub Action re-runs this and continues where it left
     off. concurrency=1 keeps the bloom filter consistent.

load_dataset("Damaru-ai/damru-knowledge") reads ALL shards (live + bulk) together.

USAGE
  pip install datasets huggingface_hub requests
  HF_TOKEN=... python phase7/bulk_harvest.py

KNOBS (env)
  HF_REPO         default Damaru-ai/damru-knowledge
  PER_DATASET     max KEPT rows per dataset per pass     (default 1200000)
  SHARD_SIZE      rows per parquet shard                 (default 100000)
  RUN_BUDGET_MIN  soft wall-clock budget for this run    (default 320)
  SCAN_MULT       max scanned = PER_DATASET*SCAN_MULT     (default 6)
  ONLY            comma-substring filter of dataset ids  (optional)
  MIN_Q / MIN_A   min question / answer length           (default 8 / 40)
  BLOOM_CAPACITY  expected unique items                  (default 60000000)
  BLOOM_ERROR     bloom false-positive rate              (default 0.01)
"""
import os
import re
import io
import sys
import json
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from dedup_bloom import BloomFilter, normalize  # noqa: E402

HF_TOKEN = os.environ.get("HF_TOKEN", "")
HF_REPO = os.environ.get("HF_REPO", "Damaru-ai/damru-knowledge")
PER_DATASET = int(os.environ.get("PER_DATASET", "1200000"))
SHARD_SIZE = int(os.environ.get("SHARD_SIZE", "100000"))
RUN_BUDGET_MIN = int(os.environ.get("RUN_BUDGET_MIN", "320"))
SCAN_MULT = int(os.environ.get("SCAN_MULT", "6"))
MIN_Q = int(os.environ.get("MIN_Q", "8"))
MIN_A = int(os.environ.get("MIN_A", "40"))
ONLY = [s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()]
BLOOM_CAP = int(os.environ.get("BLOOM_CAPACITY", "60000000"))
BLOOM_ERR = float(os.environ.get("BLOOM_ERROR", "0.01"))
BLOOM_FILE = "_dedup.bloom.gz"
STATE_FILE = "_bulk_state.json"

# ---------------------------------------------------------------------------
# Curated sources -> millions of GENUINE rows. Wrong field guesses just SKIP
# (graceful), so the list can be ambitious. kind: qa | chat | mcq | medmcqa.
# ---------------------------------------------------------------------------
DATASETS = [
    # ===== MATH (deep, step-by-step reasoning) =====
    {"id": "nvidia/OpenMathInstruct-2", "kind": "qa",
     "q": ["problem", "question"], "a": ["generated_solution", "solution"], "intent": "math"},
    {"id": "meta-math/MetaMathQA", "kind": "qa",
     "q": ["query", "original_question"], "a": ["response"], "intent": "math"},
    {"id": "TIGER-Lab/MathInstruct", "kind": "qa",
     "q": ["instruction"], "a": ["output"], "intent": "math"},
    {"id": "openai/gsm8k", "config": "main", "kind": "qa",
     "q": ["question"], "a": ["answer"], "intent": "math_word"},
    # ===== REASONING / SCIENCE INSTRUCTION (PhD-level thinking) =====
    {"id": "TIGER-Lab/WebInstructSub", "kind": "qa",
     "q": ["question"], "a": ["answer"], "intent": "reasoning"},
    {"id": "open-thoughts/OpenThoughts-114k", "kind": "chat",
     "conv": "conversations", "intent": "reasoning"},
    {"id": "Open-Orca/OpenOrca", "kind": "qa",
     "q": ["question"], "a": ["response"], "intent": "reasoning"},
    {"id": "garage-bAInd/Open-Platypus", "kind": "qa",
     "q": ["instruction"], "a": ["output"], "intent": "reasoning"},
    {"id": "teknium/OpenHermes-2.5", "kind": "chat",
     "conv": "conversations", "intent": "reasoning"},
    # ===== CODING (make it competitive) =====
    {"id": "nvidia/OpenCodeInstruct", "kind": "qa",
     "q": ["input", "question", "instruction"], "a": ["output", "response", "solution"], "intent": "coding"},
    {"id": "ise-uiuc/Magicoder-Evol-Instruct-110K", "kind": "qa",
     "q": ["instruction"], "a": ["response"], "intent": "coding"},
    {"id": "glaiveai/glaive-code-assistant", "kind": "qa",
     "q": ["question"], "a": ["answer"], "intent": "coding"},
    # ===== SCIENCE TUTOR DIALOGUES (CAMEL ~20k each) =====
    {"id": "camel-ai/physics", "kind": "qa", "q": ["message_1"], "a": ["message_2"], "intent": "physics"},
    {"id": "camel-ai/chemistry", "kind": "qa", "q": ["message_1"], "a": ["message_2"], "intent": "chemistry"},
    {"id": "camel-ai/biology", "kind": "qa", "q": ["message_1"], "a": ["message_2"], "intent": "biology"},
    {"id": "camel-ai/math", "kind": "qa", "q": ["message_1"], "a": ["message_2"], "intent": "math"},
    {"id": "sciq", "kind": "mcq", "q": ["question"], "a": ["correct_answer"], "support": "support", "intent": "science"},
    # ===== MEDICAL / NURSING (Indian exams + nursing) =====
    {"id": "openlifescienceai/medmcqa", "kind": "medmcqa", "intent": "medical"},
    {"id": "NevenaD/MedNurse-QA", "kind": "qa",
     "q": ["question", "instruction", "Question"], "a": ["answer", "output", "Answer"], "intent": "nursing"},
    # ===== INDIAN COMPETITIVE EXAMS =====
    {"id": "169Pi/exambench", "kind": "qa",
     "q": ["question", "instruction", "prompt", "input"],
     "a": ["answer", "solution", "response", "output", "explanation"], "intent": "exam"},
]


def _api():
    from huggingface_hub import HfApi
    return HfApi(token=HF_TOKEN)


def _first(ex, fields):
    for f in fields or []:
        v = ex.get(f)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _from_chat(ex, conv_field):
    conv = ex.get(conv_field) or ex.get("messages") or ex.get("conversations") or []
    q, a = "", ""
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("from") or turn.get("role") or "").lower()
        val = (turn.get("value") or turn.get("content") or "").strip()
        if not val:
            continue
        if not q and role in ("human", "user", "prompter"):
            q = val
        elif q and not a and role in ("gpt", "assistant", "model", "bot"):
            a = val
            break
    return q, a


def _from_medmcqa(ex):
    q = (ex.get("question") or "").strip()
    opts = [ex.get("opa"), ex.get("opb"), ex.get("opc"), ex.get("opd")]
    opts = [str(o).strip() for o in opts if o is not None and str(o).strip()]
    cop, exp = ex.get("cop"), (ex.get("exp") or "").strip()
    if not q or cop is None or len(opts) < 2:
        return "", ""
    try:
        ci = int(cop)
    except Exception:
        return "", ""
    if ci < 0 or ci >= len(opts):
        return "", ""
    letters = ["A", "B", "C", "D"]
    qfull = q + "\nOptions:\n" + "\n".join(
        "%s) %s" % (letters[i], opts[i]) for i in range(len(opts)))
    ans = "The correct answer is %s) %s." % (letters[ci], opts[ci])
    if exp:
        ans += " " + exp
    return qfull, ans


def _pair(ex, spec):
    kind = spec.get("kind", "qa")
    if kind == "chat":
        return _from_chat(ex, spec.get("conv", "conversations"))
    if kind == "medmcqa":
        return _from_medmcqa(ex)
    q = _first(ex, spec.get("q"))
    a = _first(ex, spec.get("a"))
    if kind == "mcq" and spec.get("support"):
        sup = str(ex.get(spec["support"], "")).strip()
        if sup:
            a = (a + ". " + sup) if a else sup
    return q, a


def make_row(q, a, intent, lang="en", quality=0.75):
    uv = int(round(max(0.0, min(1.0, quality)) * 10))
    return {
        "question": q.strip(),
        "answer": a.strip(),
        "intent": (intent or "general")[:80],
        "lang": lang or "en",
        "upvotes": uv,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _safe_tag(spec):
    base = spec["id"] + ("-" + spec["config"] if spec.get("config") else "")
    return re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_")[:60]


def _flush(api, buf, tag, idx):
    from datasets import Dataset
    local = "/tmp/%s-%d.parquet" % (tag, idx)
    Dataset.from_list(buf).to_parquet(local)
    fname = "data/bulk-%s-%d-%03d.parquet" % (tag, int(time.time()), idx)
    api.upload_file(path_or_fileobj=local, path_in_repo=fname,
                    repo_id=HF_REPO, repo_type="dataset")
    try:
        os.remove(local)
    except Exception:
        pass
    print("    uploaded shard %s (%d rows)" % (fname, len(buf)), flush=True)


def load_bloom():
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(HF_REPO, BLOOM_FILE, repo_type="dataset", token=HF_TOKEN)
        with open(p, "rb") as f:
            bf = BloomFilter.from_bytes(f.read())
        print("Loaded bloom: m=%d k=%d n~=%d" % (bf.m, bf.k, bf.n), flush=True)
        return bf
    except Exception as e:
        print("No bloom yet -> new:", str(e)[:100], flush=True)
        return BloomFilter(capacity=BLOOM_CAP, error_rate=BLOOM_ERR)


def save_bloom(api, bf):
    raw = bf.to_bytes()
    api.upload_file(path_or_fileobj=io.BytesIO(raw), path_in_repo=BLOOM_FILE,
                    repo_id=HF_REPO, repo_type="dataset")
    print("  saved bloom (%.1f MB, n~=%d)" % (len(raw) / 1e6, bf.n), flush=True)


def read_state():
    from huggingface_hub import hf_hub_download
    try:
        p = hf_hub_download(HF_REPO, STATE_FILE, repo_type="dataset", token=HF_TOKEN)
        with open(p) as f:
            return json.load(f)
    except Exception:
        return {"done": [], "total": 0}


def write_state(api, st):
    buf = json.dumps(st).encode()
    api.upload_file(path_or_fileobj=io.BytesIO(buf), path_in_repo=STATE_FILE,
                    repo_id=HF_REPO, repo_type="dataset")


def process_dataset(api, spec, bf, deadline):
    """Returns (inserted, completed). completed=False if stopped by budget."""
    from datasets import load_dataset
    name = spec["id"] + ("/" + spec["config"] if spec.get("config") else "")
    tag = _safe_tag(spec)
    print("\n=== %s (cap %d) ===" % (name, PER_DATASET), flush=True)
    try:
        ds = load_dataset(spec["id"], spec.get("config"),
                          split=spec.get("split", "train"),
                          streaming=True, trust_remote_code=True)
    except Exception as e:
        print("  SKIP (load failed):", str(e)[:160], flush=True)
        return 0, True   # treat as done so we don't retry forever
    buf, inserted, scanned, idx, completed = [], 0, 0, 0, True
    scan_cap = PER_DATASET * SCAN_MULT
    try:
        for ex in ds:
            scanned += 1
            if inserted >= PER_DATASET or scanned > scan_cap:
                break
            if not isinstance(ex, dict):
                continue
            q, a = _pair(ex, spec)
            if len(q) < MIN_Q or len(a) < MIN_A:
                continue
            if not bf.add(normalize(q)):       # already seen -> skip
                continue
            buf.append(make_row(q, a, spec.get("intent", "general"),
                                lang=spec.get("lang", "en")))
            inserted += 1
            if len(buf) >= SHARD_SIZE:
                _flush(api, buf, tag, idx)
                idx += 1
                buf = []
                save_bloom(api, bf)            # persist dedup right after upload
                if inserted % 200000 == 0:
                    print("    ...%d kept (scanned %d)" % (inserted, scanned), flush=True)
                if time.time() > deadline:
                    completed = False
                    break
    except Exception as e:
        print("  stopped early:", str(e)[:160], flush=True)
    if buf:
        _flush(api, buf, tag, idx)
        save_bloom(api, bf)
    print("  DONE %s -> +%d genuine rows (scanned %d, completed=%s)"
          % (name, inserted, scanned, completed), flush=True)
    return inserted, completed


def main():
    if not HF_TOKEN:
        print("ERROR: set HF_TOKEN")
        sys.exit(1)
    from huggingface_hub import login
    login(HF_TOKEN)
    api = _api()
    bf = load_bloom()
    st = read_state()
    done = set(st.get("done", []))
    pool = [d for d in DATASETS if (not ONLY or any(o in d["id"] for o in ONLY))]
    print("Damru BULK harvest | datasets=%d | per=%d | budget=%dmin | bypass=Supabase"
          % (len(pool), PER_DATASET, RUN_BUDGET_MIN), flush=True)
    deadline = time.time() + RUN_BUDGET_MIN * 60
    for spec in pool:
        key = spec["id"] + "::" + str(spec.get("config", "")) + "::" + spec.get("split", "train")
        if key in done:
            print("skip (done):", key, flush=True)
            continue
        if time.time() > deadline:
            print("budget reached; will resume next run.", flush=True)
            break
        ins, completed = process_dataset(api, spec, bf, deadline)
        st["total"] = st.get("total", 0) + ins
        save_bloom(api, bf)
        if completed:
            done.add(key)
            st["done"] = sorted(done)
        write_state(api, st)
        print("== %s -> +%d (running total ~%d) ==" % (key, ins, st["total"]), flush=True)
    print("\nRUN COMPLETE. cumulative bulk total ~%d rows" % st.get("total", 0), flush=True)


if __name__ == "__main__":
    main()
