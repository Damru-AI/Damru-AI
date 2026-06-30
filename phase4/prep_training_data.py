#!/usr/bin/env python3
"""
Damru TRAINING DATA PREP  (decontaminate -> dedup -> balance -> split)
======================================================================
Turns the raw harvested `damru-knowledge` dataset into a CLEAN, training-ready
set so nothing sabotages fine-tuning:

  1. DECONTAMINATE  drop any row whose question matches a public benchmark
                    TEST item (HumanEval / MBPP / MMLU / MedMCQA / GSM8K)
                    -> eval scores stay REAL, not memorised.
  2. DEDUP          normalised-question Bloom hashing (bounded RAM).
  3. BALANCE        cap dominant domains (coding) + UPSAMPLE minority
                    domains (medical/nursing!) so the exam domain isn't
                    drowned by 7M coding rows.
  4. FORMAT         ChatML `messages` + flat question/answer (loss-maskable).
  5. SPLIT          deterministic, clean held-out val + test (never in train).

Fully streaming -> safe on a 7 GB GitHub Actions runner.

Env:
  HF_TOKEN         (required)
  SRC_REPO         Damaru-ai/damru-knowledge
  OUT_REPO         Damaru-ai/damru-train     (created if missing)
  VAL_FRAC         0.01
  TEST_FRAC        0.01
  MAX_PER_DOMAIN   900000     (cap the dominant domain in TRAIN)
  SHARD_SIZE       50000
  DECON            1
  UPSAMPLE_MEDICAL 3          (repeat medical/nursing rows in TRAIN)
  UPSAMPLE_HOLY    2
  UPSAMPLE_AGENTIC 2
  MAX_ROWS         0          (0 = all)
"""
import os
import re
import io
import json
import math
import time
import hashlib
from collections import Counter

HF_TOKEN = os.environ.get("HF_TOKEN", "")
SRC_REPO = os.environ.get("SRC_REPO", "Damaru-ai/damru-knowledge")
OUT_REPO = os.environ.get("OUT_REPO", "Damaru-ai/damru-train")
VAL_FRAC = float(os.environ.get("VAL_FRAC") or "0.01")
TEST_FRAC = float(os.environ.get("TEST_FRAC") or "0.01")
MAX_PER_DOMAIN = int(os.environ.get("MAX_PER_DOMAIN") or "900000")
SHARD_SIZE = int(os.environ.get("SHARD_SIZE") or "50000")
DECON = (os.environ.get("DECON", "1") == "1")
MAX_ROWS = int(os.environ.get("MAX_ROWS") or "0")
SYS_PROMPT = os.environ.get(
    "SYS_PROMPT",
    "You are Damru, a careful, exam-grade tutor. Answer accurately, show "
    "reasoning when useful, and stay grounded in facts.")

UPSAMPLE = {
    "medical": int(os.environ.get("UPSAMPLE_MEDICAL") or "3"),
    "holy": int(os.environ.get("UPSAMPLE_HOLY") or "2"),
    "agentic": int(os.environ.get("UPSAMPLE_AGENTIC") or "2"),
}


class Bloom:
    """Tiny, dependency-free Bloom filter (bounded RAM dedup/decon)."""
    def __init__(self, capacity, error=0.01):
        self.m = max(1024, int(-capacity * math.log(error) / (math.log(2) ** 2)))
        self.k = max(1, int(self.m / capacity * math.log(2)))
        self.bits = bytearray((self.m + 7) // 8)

    def _pos(self, item):
        h = hashlib.blake2b(item.encode("utf-8"), digest_size=16).digest()
        a = int.from_bytes(h[:8], "big")
        b = int.from_bytes(h[8:], "big") | 1
        for i in range(self.k):
            yield (a + i * b) % self.m

    def add(self, item):
        for p in self._pos(item):
            self.bits[p >> 3] |= (1 << (p & 7))

    def seen_then_add(self, item):
        """True if probably already present; else add and return False."""
        present = True
        for p in self._pos(item):
            byte, mask = p >> 3, 1 << (p & 7)
            if not (self.bits[byte] & mask):
                present = False
                self.bits[byte] |= mask
        return present

    def __contains__(self, item):
        return all(self.bits[p >> 3] & (1 << (p & 7)) for p in self._pos(item))


def norm_q(q):
    return re.sub(r"\s+", " ",
                  re.sub(r"[^a-z0-9 ]", " ", (q or "").lower())).strip()


def domain_of(intent):
    s = (intent or "").lower()
    def has(*ks):
        return any(k in s for k in ks)
    if has("cod", "program", "python", "algorithm", "competitive", "devops"):
        return "coding"
    if has("nurs", "med", "clinic", "disease", "anatom", "physio", "pharma",
           "patho", "health", "surg", "biolog", "nutri"):
        return "medical"
    if has("physic", "chem", "math", "reason", "logic", "calcul", "science"):
        return "stem"
    if has("veda", "gita", "bible", "quran", "holy", "itihasa", "mahabharat",
           "ramayan", "upanishad", "verse", "distilled"):
        return "holy"
    if has("agent", "tool", "plan"):
        return "agentic"
    return "general"


def split_of(qn):
    h = int.from_bytes(hashlib.blake2b(qn.encode("utf-8"), digest_size=8)
                       .digest(), "big") / 2 ** 64
    if h < TEST_FRAC:
        return "test"
    if h < TEST_FRAC + VAL_FRAC:
        return "val"
    return "train"


def load_decon():
    bloom = Bloom(400000)
    if not DECON:
        print("DECON disabled", flush=True)
        return bloom
    from datasets import load_dataset
    specs = [
        ("openai/openai_humaneval", None, "test", ["prompt"]),
        ("google-research-datasets/mbpp", "full", "test", ["text", "prompt"]),
        ("cais/mmlu", "all", "test", ["question"]),
        ("openlifescienceai/medmcqa", None, "validation", ["question"]),
        ("openai/gsm8k", "main", "test", ["question"]),
    ]
    n = 0
    for repo, cfg, split, fields in specs:
        try:
            ds = (load_dataset(repo, cfg, split=split, streaming=True)
                  if cfg else load_dataset(repo, split=split, streaming=True))
            c = 0
            for ex in ds:
                for f in fields:
                    v = ex.get(f)
                    if v:
                        bloom.add(norm_q(str(v)))
                        c += 1
                        break
                if c >= 50000:
                    break
            n += c
            print("  decon loaded %d from %s" % (c, repo), flush=True)
        except Exception as e:
            print("  decon SKIP %s (%s)" % (repo, str(e)[:70]), flush=True)
    print("DECON set ~%d benchmark questions" % n, flush=True)
    return bloom


class ShardWriter:
    def __init__(self, split):
        self.split = split
        self.buf = []
        self.part = 0
        self.rows = 0

    def add(self, row):
        self.buf.append(row)
        self.rows += 1
        if len(self.buf) >= SHARD_SIZE:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq
        from huggingface_hub import HfApi
        cols = {k: [r.get(k) for r in self.buf] for k in
                ("question", "answer", "messages", "intent", "domain",
                 "lang", "text")}
        tbl = pa.table(cols)
        sink = io.BytesIO()
        pq.write_table(tbl, sink, compression="zstd")
        sink.seek(0)
        path = "%s/part-%05d.parquet" % (self.split, self.part)
        api = HfApi(token=HF_TOKEN)
        for attempt in range(8):
            try:
                api.upload_file(path_or_fileobj=sink, path_in_repo=path,
                                repo_id=OUT_REPO, repo_type="dataset")
                break
            except Exception as e:
                s = str(e)
                if attempt == 7:
                    raise
                wait = 1900 if ("per hour" in s or "repository commits" in s) \
                    else min(120, 8 * (2 ** attempt))
                print("  upload retry %ds (%s)" % (wait, s[:60]), flush=True)
                time.sleep(wait)
        print("  wrote %s (%d rows)" % (path, len(self.buf)), flush=True)
        self.part += 1
        self.buf = []


def to_messages(q, a):
    return [{"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": q},
            {"role": "assistant", "content": a}]


def render_chatml(msgs):
    out = []
    for m in msgs:
        out.append("<|im_start|>%s\n%s<|im_end|>" % (m["role"], m["content"]))
    return "\n".join(out)


def main():
    assert HF_TOKEN, "HF_TOKEN required"
    from datasets import load_dataset
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    try:
        api.create_repo(OUT_REPO, repo_type="dataset", exist_ok=True,
                        private=False)
    except Exception as e:
        print("create_repo:", str(e)[:80], flush=True)

    decon = load_decon()
    dedup = Bloom(int(os.environ.get("DEDUP_CAPACITY") or "80000000"))
    writers = {s: ShardWriter(s) for s in ("train", "val", "test")}
    cap = Counter()           # train rows kept per domain (for capping)
    dom_in = Counter()        # domains seen
    kept = Counter()          # per split
    dropped = Counter()
    rows = 0
    t0 = time.time()

    ds = load_dataset(SRC_REPO, split="train", streaming=True)
    for ex in ds:
        rows += 1
        if MAX_ROWS and rows > MAX_ROWS:
            break
        q = (ex.get("question") or "").strip()
        a = (ex.get("answer") or "").strip()
        if len(q) < 8 or len(a) < 20:
            dropped["short"] += 1
            continue
        qn = norm_q(q)
        if not qn:
            dropped["empty"] += 1
            continue
        if qn in decon:
            dropped["contaminated"] += 1
            continue
        if dedup.seen_then_add(qn):
            dropped["dup"] += 1
            continue
        intent = (ex.get("intent") or "").strip()
        dom = domain_of(intent)
        dom_in[dom] += 1
        lang = (ex.get("lang") or "en").strip() or "en"
        split = split_of(qn)
        msgs = to_messages(q, a)
        row = {"question": q, "answer": a, "messages": json.dumps(msgs),
               "intent": intent, "domain": dom, "lang": lang,
               "text": render_chatml(msgs)}
        if split != "train":
            writers[split].add(row)
            kept[split] += 1
            continue
        # TRAIN: apply domain cap + upsample minority
        if cap[dom] >= MAX_PER_DOMAIN:
            dropped["capped"] += 1
            continue
        reps = UPSAMPLE.get(dom, 1)
        for _ in range(reps):
            if cap[dom] >= MAX_PER_DOMAIN:
                break
            writers["train"].add(row)
            cap[dom] += 1
            kept["train"] += 1
        if rows % 200000 == 0:
            print("scanned %d | kept train=%d val=%d test=%d | %.0fs"
                  % (rows, kept["train"], kept["val"], kept["test"],
                     time.time() - t0), flush=True)

    for w in writers.values():
        w.flush()

    report = {
        "src": SRC_REPO, "out": OUT_REPO, "scanned": rows,
        "kept": dict(kept), "dropped": dict(dropped),
        "domain_unique": dict(dom_in), "train_per_domain": dict(cap),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_per_domain": MAX_PER_DOMAIN, "upsample": UPSAMPLE,
        "decontaminated": DECON,
    }
    raw = json.dumps(report, indent=2).encode("utf-8")
    try:
        api.upload_file(path_or_fileobj=io.BytesIO(raw),
                        path_in_repo="prep_report.json", repo_id=OUT_REPO,
                        repo_type="dataset")
    except Exception as e:
        print("report upload:", str(e)[:80], flush=True)
    print("DONE.", json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
