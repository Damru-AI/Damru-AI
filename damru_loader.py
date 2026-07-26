# ============================================================================
#  damru_loader.py  --  Damru AI robust dataset loader
#  REPO ROOT ME UPLOAD KARO (damru_council.py ke saath same folder me)
# ----------------------------------------------------------------------------
#  KYUN?
#  -----
#  HuggingFace ka load_dataset() ek repo ki SAARI files ko ek hi schema me
#  merge karta hai. damru-oracle repo me data ke saath manifest/state file
#  bhi padi hai (keys: version / sources / shard_index / rows_out / total_out)
#  -> "column names don't match" -> DatasetGenerationError -> run dead.
#
#  YE LOADER
#  ---------
#  * load_dataset() ko poora bypass karta hai
#  * har data file ALAG se padhta hai (pandas)
#  * manifest / state / readme / config files SKIP karta hai
#  * ek file kharab -> sirf wahi skip, run zinda rehta hai
#  * har row ko common shape { "prompt": str, "source": str, ...original } me
#    normalize karta hai
#
#  damru_council.py ME SIRF 2 LINE BADALNI HAI
#  -------------------------------------------
#  (A) file ke top ke imports me ye add karo:
#          from damru_loader import load_problems
#
#  (B) main() ke andar (line ~234) ye line:
#          ds = load_dataset(SRC_REPO, split=SRC_SPLIT, token=HF_TOKEN)
#      ko badal do:
#          ds = load_problems(SRC_REPO, SRC_SPLIT, HF_TOKEN)
#
#  Bas. Neeche ka `for row in ds:` waise ka waisa chalega (list[dict] milti hai).
#  Agar kahin ds.select(a, b) ho to ds[a:b] kar dena.
#
#  TEST MODE
#  ---------
#  Repo secret/env me LOADER_MAX_ROWS=50 daal do -> 2 min me pata chal jayega.
#  Kaam karne lage to hata do (ya 0 kar do = unlimited).
# ============================================================================

import json
import os
import sys

LOADER_MAX_FILES = int(os.environ.get("LOADER_MAX_FILES", "0") or 0)   # 0 = all
LOADER_MAX_ROWS = int(os.environ.get("LOADER_MAX_ROWS", "0") or 0)     # 0 = all
LOADER_MIN_LEN = int(os.environ.get("LOADER_MIN_LEN", "20") or 20)

# ye naam kabhi asli data nahi hote -> hamesha skip
_SKIP_TOKENS = (
    "manifest",
    "state",
    "stats",
    "_meta",
    "metadata",
    "readme",
    "dataset_infos",
    "gitattributes",
    "checkpoint",
    "progress",
    "_index",
    "config",
    "scorecard",
)

_DATA_EXT = (".parquet", ".jsonl", ".ndjson", ".json", ".csv", ".tsv")

_PROMPT_KEYS = (
    "problem",
    "prompt",
    "question",
    "instruction",
    "query",
    "task",
    "input",
    "text",
    "content",
    "body",
    "title",
)


def _log(msg):
    print("[loader] " + str(msg), flush=True)


def _looks_like_data_file(path, split):
    low = str(path).lower()
    if not low.endswith(_DATA_EXT):
        return False
    base = low.rsplit("/", 1)[-1]
    for bad in _SKIP_TOKENS:
        if bad in base:
            return False
    if split:
        s = str(split).lower()
        if s in low:
            return True
        for other in ("train", "test", "validation", "val", "dev"):
            if other == s:
                continue
            if ("/" + other) in low or base.startswith(other):
                return False
    return True


def _pick_prompt(row):
    if not isinstance(row, dict):
        return None

    for k in _PROMPT_KEYS:
        v = row.get(k)
        if isinstance(v, str) and len(v.strip()) >= LOADER_MIN_LEN:
            return v.strip()

    lowmap = {}
    for k in row.keys():
        if isinstance(k, str):
            lowmap[k.lower()] = k
    for k in _PROMPT_KEYS:
        real = lowmap.get(k)
        if real is None:
            continue
        v = row.get(real)
        if isinstance(v, str) and len(v.strip()) >= LOADER_MIN_LEN:
            return v.strip()

    msgs = row.get("messages")
    if msgs is None:
        msgs = row.get("conversations")
    if isinstance(msgs, (list, tuple)):
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role") or m.get("from") or "").lower()
            if role in ("user", "human"):
                v = m.get("content")
                if v is None:
                    v = m.get("value")
                if isinstance(v, str) and len(v.strip()) >= LOADER_MIN_LEN:
                    return v.strip()
    return None


def _read_one_file(local_path):
    low = str(local_path).lower()
    try:
        import pandas as pd
    except Exception as e:
        _log("pandas missing: " + str(e))
        return []

    try:
        if low.endswith(".parquet"):
            df = pd.read_parquet(local_path)
        elif low.endswith((".jsonl", ".ndjson")):
            df = pd.read_json(local_path, lines=True)
        elif low.endswith(".json"):
            with open(local_path, "r", encoding="utf-8", errors="ignore") as fh:
                raw = json.load(fh)
            if isinstance(raw, dict):
                for key in ("data", "rows", "items", "examples", "problems"):
                    if isinstance(raw.get(key), list):
                        return [r for r in raw[key] if isinstance(r, dict)]
                return []
            if isinstance(raw, list):
                return [r for r in raw if isinstance(r, dict)]
            return []
        elif low.endswith(".tsv"):
            df = pd.read_csv(local_path, sep="\t")
        else:
            df = pd.read_csv(local_path)
    except Exception as e:
        _log("SKIP unreadable " + str(local_path) + " :: " + str(e)[:160])
        return []

    try:
        return df.to_dict(orient="records")
    except Exception as e:
        _log("SKIP convert-failed " + str(local_path) + " :: " + str(e)[:160])
        return []


def load_problems(repo, split="train", token=None):
    """load_dataset() ka self-healing replacement. Returns list[dict]."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except Exception as e:
        _log("FATAL: huggingface_hub import failed :: " + str(e))
        return []

    if token is None:
        token = os.environ.get("HF_TOKEN")

    api = HfApi(token=token)

    try:
        files = api.list_repo_files(repo_id=repo, repo_type="dataset")
    except Exception as e:
        _log("FATAL: repo list failed for " + str(repo) + " :: " + str(e))
        return []

    candidates = sorted([f for f in files if _looks_like_data_file(f, split)])
    if not candidates:
        _log("no file matched split=" + str(split) + " -> retry without split filter")
        candidates = sorted([f for f in files if _looks_like_data_file(f, None)])

    if LOADER_MAX_FILES > 0:
        candidates = candidates[:LOADER_MAX_FILES]

    _log("repo=" + str(repo) + " split=" + str(split))
    _log("total files=" + str(len(files)) + " | data files=" + str(len(candidates)))

    out = []
    seen = set()
    ok_files = 0
    bad_files = 0

    for rel in candidates:
        try:
            local = hf_hub_download(
                repo_id=repo,
                filename=rel,
                repo_type="dataset",
                token=token,
            )
        except Exception as e:
            bad_files += 1
            _log("SKIP download-failed " + str(rel) + " :: " + str(e)[:160])
            continue

        rows = _read_one_file(local)
        if not rows:
            bad_files += 1
            continue

        kept = 0
        for r in rows:
            p = _pick_prompt(r)
            if not p:
                continue
            key = hash(p[:400])
            if key in seen:
                continue
            seen.add(key)
            item = dict(r)
            item["prompt"] = p
            item["source"] = rel
            out.append(item)
            kept += 1
            if LOADER_MAX_ROWS > 0 and len(out) >= LOADER_MAX_ROWS:
                break

        ok_files += 1
        _log(str(rel) + " -> rows=" + str(len(rows)) + " kept=" + str(kept) + " total=" + str(len(out)))

        if LOADER_MAX_ROWS > 0 and len(out) >= LOADER_MAX_ROWS:
            _log("LOADER_MAX_ROWS reached, stopping early")
            break

    _log("DONE files_ok=" + str(ok_files) + " files_skipped=" + str(bad_files) + " problems=" + str(len(out)))

    if not out:
        _log("WARNING: 0 problems loaded. Check repo name / HF_TOKEN / split.")

    return out


# alias -- purane code se compatibility ke liye
load_dataset_safe = load_problems


if __name__ == "__main__":
    repo_arg = sys.argv[1] if len(sys.argv) > 1 else "Damaru-ai/damru-oracle"
    split_arg = sys.argv[2] if len(sys.argv) > 2 else "train"
    probs = load_problems(repo_arg, split_arg, os.environ.get("HF_TOKEN"))
    print("loaded " + str(len(probs)) + " problems")
    for sample in probs[:2]:
        print("---")
        print(str(sample.get("source")))
        print(str(sample.get("prompt"))[:300])
