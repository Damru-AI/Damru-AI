#!/usr/bin/env python3
"""
Damru DEDUP + QUALITY AUDIT  (low-memory streaming -- no OOM)
============================================================
The old audit loaded everything into RAM and OOM'd. This streams the whole
`damru-knowledge` dataset with a bounded-RAM Bloom filter and lightweight
counters, so it never blows up.

Reports (no rewrite -- read-only):
  * duplicate rate (normalised-question)
  * rows per domain / per intent (top 40)
  * language distribution
  * answer/question length buckets
  * empty / too-short counts

Pushes `audit_report.json` to the dataset repo.

Env:
  HF_TOKEN          (required to push report)
  SRC_REPO          Damaru-ai/damru-knowledge
  DEDUP_CAPACITY    80000000
  MAX_ROWS          0  (0 = all)
  PUSH_REPORT       1
"""
import os
import io
import re
import json
import math
import time
import hashlib
from collections import Counter

HF_TOKEN = os.environ.get("HF_TOKEN", "")
SRC_REPO = os.environ.get("SRC_REPO", "Damaru-ai/damru-knowledge")
DEDUP_CAPACITY = int(os.environ.get("DEDUP_CAPACITY") or "80000000")
MAX_ROWS = int(os.environ.get("MAX_ROWS") or "0")
PUSH_REPORT = ((os.environ.get("PUSH_REPORT") or "1") == "1")


class Bloom:
    def __init__(self, capacity, error=0.01):
        self.m = max(1024, int(-capacity * math.log(error) / (math.log(2) ** 2)))
        self.k = max(1, int(self.m / capacity * math.log(2)))
        self.bits = bytearray((self.m + 7) // 8)
        print("Bloom m=%d bits (%.0f MB), k=%d"
              % (self.m, self.m / 8 / 1e6, self.k), flush=True)

    def _pos(self, item):
        h = hashlib.blake2b(item.encode("utf-8"), digest_size=16).digest()
        a = int.from_bytes(h[:8], "big")
        b = int.from_bytes(h[8:], "big") | 1
        for i in range(self.k):
            yield (a + i * b) % self.m

    def seen_then_add(self, item):
        present = True
        for p in self._pos(item):
            byte, mask = p >> 3, 1 << (p & 7)
            if not (self.bits[byte] & mask):
                present = False
                self.bits[byte] |= mask
        return present


def norm_q(q):
    return re.sub(r"\s+", " ",
                  re.sub(r"[^a-z0-9 ]", " ", (q or "").lower())).strip()


def len_bucket(n):
    for hi in (20, 50, 100, 200, 400, 800, 1600, 3200):
        if n < hi:
            return "<%d" % hi
    return ">=3200"


def main():
    from datasets import load_dataset
    dedup = Bloom(DEDUP_CAPACITY)
    rows = dups = empties = shorts = 0
    by_domain = Counter()
    by_intent = Counter()
    by_lang = Counter()
    q_len = Counter()
    a_len = Counter()
    t0 = time.time()

    # Read ALL shards (train-* + bulk-*); README config otherwise hides bulk-*.
    ds = load_dataset(SRC_REPO, data_files="data/*.parquet",
                      split="train", streaming=True)
    for ex in ds:
        rows += 1
        if MAX_ROWS and rows > MAX_ROWS:
            break
        q = (ex.get("question") or "").strip()
        a = (ex.get("answer") or "").strip()
        intent = (ex.get("intent") or "").strip()
        lang = (ex.get("lang") or "").strip() or "unknown"
        if not q or not a:
            empties += 1
            continue
        if len(q) < 8 or len(a) < 20:
            shorts += 1
        qn = norm_q(q)
        if qn and dedup.seen_then_add(qn):
            dups += 1
        by_intent[intent or "(none)"] += 1
        by_lang[lang] += 1
        q_len[len_bucket(len(q))] += 1
        a_len[len_bucket(len(a))] += 1
        if rows % 200000 == 0:
            print("scanned %d | dup %.1f%% | %.0fs"
                  % (rows, 100.0 * dups / max(1, rows), time.time() - t0),
                  flush=True)

    report = {
        "src": SRC_REPO,
        "rows": rows,
        "duplicates": dups,
        "dup_rate": round(dups / max(1, rows), 4),
        "empty": empties,
        "too_short": shorts,
        "unique_estimate": rows - dups - empties,
        "top_intents": dict(by_intent.most_common(40)),
        "languages": dict(by_lang.most_common(20)),
        "question_len_buckets": dict(q_len),
        "answer_len_buckets": dict(a_len),
        "audited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("AUDIT", json.dumps(report, indent=2), flush=True)
    if PUSH_REPORT and HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.upload_file(
                path_or_fileobj=io.BytesIO(json.dumps(report, indent=2)
                                           .encode("utf-8")),
                path_in_repo="audit_report.json", repo_id=SRC_REPO,
                repo_type="dataset")
            print("pushed audit_report.json", flush=True)
        except Exception as e:
            print("push failed:", str(e)[:80], flush=True)


if __name__ == "__main__":
    main()
