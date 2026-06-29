#!/usr/bin/env python3
"""
DAMRU FAST-FILL: ingest MILLIONS of genuine Q&A from curated open datasets.

WHY THIS EXISTS
---------------
Generating 5,000,000 deep Q&A from scratch on free LLMs would take ~a year
(rate limits). The real-world way every big model is built is to AGGREGATE
existing high-quality open datasets, then ADD your own unique data on top.

This script STREAMS well-known, genuinely high-quality open Q&A / instruction
datasets from the Hugging Face Hub (no full download -> low memory), maps each
example into Damru's schema, and pushes it through the SAME store.insert_batch()
used by the live engine -- so it gets:
  * local sqlite dedup, and
  * DB-level dedup (ON CONFLICT DO NOTHING via the qnorm unique constraint).

Result: hundreds of thousands -> millions of real, diverse, genuine Q&A in
HOURS instead of a year. Your LLM engine keeps running in parallel to add the
unique Indian-exam / Hindi / self-checked-reasoning flavour on top.

USAGE
-----
  pip install datasets huggingface_hub requests
  SUPABASE_URL=... SUPABASE_KEY=... python phase6/ingest_open_datasets.py

KNOBS (env):
  PER_DATASET   max rows to pull from EACH dataset (default 50000)
  GLOBAL_CAP    stop after this many total inserts (default 2000000)
  ONLY          comma-sep substring filter of dataset ids to run (optional)
  MIN_Q / MIN_A min question / answer length (defaults 8 / 40)
"""
import os
import sys
import time

# Reuse the engine's store + config (dedup + Supabase insert live there).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "phase5"))
import store  # noqa: E402

PER_DATASET = int(os.environ.get("PER_DATASET", "50000"))
GLOBAL_CAP = int(os.environ.get("GLOBAL_CAP", "2000000"))
MIN_Q = int(os.environ.get("MIN_Q", "8"))
MIN_A = int(os.environ.get("MIN_A", "40"))
ONLY = [s.strip() for s in os.environ.get("ONLY", "").split(",") if s.strip()]
BATCH = 200

# -------------------------------------------------------------------------
# Curated sources. Each entry:
#   id        : HF dataset id
#   config    : dataset config name (optional)
#   split     : split to stream (default "train")
#   kind      : "qa" | "chat" | "mcq"
#   q / a     : candidate field names for question / answer (qa & mcq)
#   conv      : conversation field name (chat)
#   intent    : Damru intent tag
#   lang      : language code
# Only widely-used, reliably-structured datasets are listed (low guess risk).
# -------------------------------------------------------------------------
DATASETS = [
    # ---- Math (huge + genuine, step-by-step) ----
    {"id": "openai/gsm8k", "config": "main", "kind": "qa",
     "q": ["question"], "a": ["answer"], "intent": "math_word", "lang": "en"},
    {"id": "meta-math/MetaMathQA", "kind": "qa",
     "q": ["query", "original_question"], "a": ["response"], "intent": "math", "lang": "en"},
    {"id": "TIGER-Lab/MathInstruct", "kind": "qa",
     "q": ["instruction"], "a": ["output"], "intent": "math", "lang": "en"},
    {"id": "nvidia/OpenMathInstruct-2", "kind": "qa",
     "q": ["problem", "question"], "a": ["generated_solution", "solution"],
     "intent": "math", "lang": "en"},
    # ---- Reasoning / general instruction ----
    {"id": "Open-Orca/OpenOrca", "kind": "qa",
     "q": ["question"], "a": ["response"], "intent": "reasoning", "lang": "en"},
    {"id": "garage-bAInd/Open-Platypus", "kind": "qa",
     "q": ["instruction"], "a": ["output"], "intent": "reasoning", "lang": "en"},
    {"id": "teknium/OpenHermes-2.5", "kind": "chat",
     "conv": "conversations", "intent": "reasoning", "lang": "en"},
    # ---- Coding (genuine solutions) ----
    {"id": "ise-uiuc/Magicoder-Evol-Instruct-110K", "kind": "qa",
     "q": ["instruction"], "a": ["response"], "intent": "coding", "lang": "en"},
    {"id": "glaiveai/glaive-code-assistant", "kind": "qa",
     "q": ["question"], "a": ["answer"], "intent": "coding", "lang": "en"},
    # ---- Science (CAMEL = genuine tutor dialogues, ~20k each) ----
    {"id": "camel-ai/physics", "kind": "qa",
     "q": ["message_1"], "a": ["message_2"], "intent": "physics", "lang": "en"},
    {"id": "camel-ai/chemistry", "kind": "qa",
     "q": ["message_1"], "a": ["message_2"], "intent": "chemistry", "lang": "en"},
    {"id": "camel-ai/biology", "kind": "qa",
     "q": ["message_1"], "a": ["message_2"], "intent": "biology", "lang": "en"},
    {"id": "camel-ai/math", "kind": "qa",
     "q": ["message_1"], "a": ["message_2"], "intent": "math", "lang": "en"},
    # ---- Science MCQ w/ explanation ----
    {"id": "sciq", "kind": "mcq",
     "q": ["question"], "a": ["correct_answer"], "support": "support",
     "intent": "science", "lang": "en"},
]


def _first(example, fields):
    for f in fields or []:
        v = example.get(f)
        if v is not None and str(v).strip():
            return str(v).strip()
    return ""


def _from_chat(example, conv_field):
    """Extract (question, answer) from a conversations list (ShareGPT style)."""
    conv = example.get(conv_field) or example.get("messages") or []
    q, a = "", ""
    for turn in conv:
        if not isinstance(turn, dict):
            continue
        role = (turn.get("from") or turn.get("role") or "").lower()
        val = (turn.get("value") or turn.get("content") or "").strip()
        if not val:
            continue
        if not q and role in ("human", "user"):
            q = val
        elif q and not a and role in ("gpt", "assistant", "model"):
            a = val
            break
    return q, a


def _to_item(example, spec):
    kind = spec.get("kind", "qa")
    if kind == "chat":
        q, a = _from_chat(example, spec.get("conv", "conversations"))
    else:
        q = _first(example, spec.get("q"))
        a = _first(example, spec.get("a"))
        if kind == "mcq" and spec.get("support"):
            sup = str(example.get(spec["support"], "")).strip()
            if sup:
                a = (a + ". " + sup) if a else sup
    q, a = q.strip(), a.strip()
    if len(q) < MIN_Q or len(a) < MIN_A:
        return None
    return store.make_row(q, a, spec.get("intent", "general"),
                          lang=spec.get("lang", "en"), quality=0.75)


def ingest_one(spec, remaining):
    from datasets import load_dataset
    name = spec["id"] + ("/" + spec["config"] if spec.get("config") else "")
    print("\n=== %s (cap %d) ===" % (name, min(PER_DATASET, remaining)), flush=True)
    try:
        ds = load_dataset(spec["id"], spec.get("config"),
                          split=spec.get("split", "train"), streaming=True)
    except Exception as e:
        print("  SKIP (load failed):", str(e)[:160], flush=True)
        return 0
    inserted, seen, batch = 0, 0, []
    cap = min(PER_DATASET, remaining)
    try:
        for ex in ds:
            if seen >= cap:
                break
            seen += 1
            item = _to_item(ex, spec)
            if not item:
                continue
            batch.append(item)
            if len(batch) >= BATCH:
                inserted += store.insert_batch(batch)
                batch = []
                if inserted and inserted % 2000 == 0:
                    print("  ...%d inserted (scanned %d)" % (inserted, seen), flush=True)
        if batch:
            inserted += store.insert_batch(batch)
    except Exception as e:
        print("  stopped early:", str(e)[:160], flush=True)
    print("  DONE %s -> +%d genuine rows (scanned %d)" % (name, inserted, seen), flush=True)
    return inserted


def main():
    if not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_KEY")):
        print("ERROR: set SUPABASE_URL and SUPABASE_KEY")
        sys.exit(1)
    pool = DATASETS
    if ONLY:
        pool = [d for d in DATASETS if any(o in d["id"] for o in ONLY)]
    print("Damru open-dataset ingest | datasets=%d | per=%d | global_cap=%d"
          % (len(pool), PER_DATASET, GLOBAL_CAP), flush=True)
    t0 = time.time()
    total = 0
    for spec in pool:
        if total >= GLOBAL_CAP:
            print("Global cap reached.")
            break
        total += ingest_one(spec, GLOBAL_CAP - total)
        print(">> running total inserted: %d (%.1f min)" % (total, (time.time() - t0) / 60.0),
              flush=True)
    print("\nALL DONE. Total genuine rows added: %d in %.1f min"
          % (total, (time.time() - t0) / 60.0), flush=True)


if __name__ == "__main__":
    main()
