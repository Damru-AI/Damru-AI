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
    # OpenThoughts-114k: long reasoning traces. Safe now via the low-mem parquet
    # reader + byte-budget flush (previously OOM-killed the runner).
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
    # ===== SCIENCE TUTOR DIALOGUES =====
    # camel-ai/* and sciq REMOVED: they are old loading-script datasets (no
    # parquet on the Hub), so on datasets v4 they error or HANG the runner.
    # The load timeout in open_rows() also guards against such hangs.
    # ===== ALL-SUBJECTS MCQ (MMLU 57 subjects; MMLU-Pro 14 incl engineering,
    #       physics, math, chemistry, biology, economics, business, etc.) =====
    {"id": "TIGER-Lab/MMLU-Pro", "kind": "choices", "split": "test",
     "q": ["question"], "opts": ["options"], "ans": ["answer_index", "answer"],
     "exp": ["cot_content"], "intent": "exam"},
    {"id": "cais/mmlu", "config": "all", "split": "test", "kind": "choices",
     "q": ["question"], "opts": ["choices"], "ans": ["answer"], "intent": "exam"},
    {"id": "allenai/ai2_arc", "config": "ARC-Challenge", "kind": "choices",
     "q": ["question"], "opts": ["choices"], "ans": ["answerKey"], "intent": "science"},
    {"id": "allenai/ai2_arc", "config": "ARC-Easy", "kind": "choices",
     "q": ["question"], "opts": ["choices"], "ans": ["answerKey"], "intent": "science"},
    {"id": "allenai/openbookqa", "config": "main", "kind": "choices",
     "q": ["question_stem", "question"], "opts": ["choices"], "ans": ["answerKey"], "intent": "science"},
    # ===== SCIENCE JOURNALS + EXPERT Q&A (every subject: physics, chemistry,
    #       biology, all engineering, earth science / oceanography, economics,
    #       resource management, etc. -- StackExchange network + PubMed) =====
    {"id": "lvwerra/stack-exchange-paired", "kind": "qa",
     "q": ["question"], "a": ["response_j"], "intent": "reasoning"},
    {"id": "qiaojin/PubMedQA", "config": "pqa_artificial", "kind": "qa",
     "q": ["question"], "a": ["long_answer"], "intent": "medical"},
    # ===== BROAD INSTRUCTION (all subjects + management + general knowledge) =====
    {"id": "arcee-ai/The-Tome", "kind": "chat", "conv": "conversations", "intent": "reasoning"},
    {"id": "databricks/databricks-dolly-15k", "kind": "qa",
     "q": ["instruction"], "a": ["response"], "context": "context", "intent": "general"},
    {"id": "STEM-AI-mtl/Electrical-engineering", "kind": "qa",
     "q": ["input", "instruction", "question", "Question"],
     "a": ["output", "response", "answer", "Answer"], "intent": "engineering"},
    # ===== EVERY-SUBJECT KNOWLEDGE (universe, pollution, farming, water, hunger,
    #       history, culture, ecology, economy, development -- explanations) =====
    {"id": "sentence-transformers/eli5", "kind": "qa",
     "q": ["question", "title"], "a": ["answer", "response"], "intent": "general"},
    {"id": "yahma/alpaca-cleaned", "kind": "qa",
     "q": ["instruction"], "a": ["output"], "intent": "general"},
    {"id": "WizardLMTeam/WizardLM_evol_instruct_V2_196k", "kind": "chat",
     "conv": "conversations", "intent": "reasoning"},
    {"id": "HuggingFaceH4/no_robots", "kind": "chat",
     "conv": "messages", "intent": "general"},
    # ===== SPACE AGENCIES + RESEARCH PAPERS (live APIs, resumable) =====
    # NASA NTRS: 645k+ public NASA technical reports / papers across astrophysics,
    # planetary science, aerodynamics, propulsion, life sciences, earth science.
    {"id": "nasa-ntrs", "loader": "ntrs", "kind": "paper",
     "paper_src": "NASA NTRS", "intent": "nasa_space"},
    # arXiv: research from scientists across NASA/ESA/ISRO/JAXA/ROSCOSMOS/etc.
    {"id": "arxiv-astro-ph", "loader": "arxiv", "arxiv_cat": "astro-ph",
     "kind": "paper", "paper_src": "arXiv astrophysics", "intent": "astrophysics"},
    {"id": "arxiv-space-ph", "loader": "arxiv", "arxiv_cat": "physics.space-ph",
     "kind": "paper", "paper_src": "arXiv space physics", "intent": "space"},
    {"id": "arxiv-geo-ph", "loader": "arxiv", "arxiv_cat": "physics.geo-ph",
     "kind": "paper", "paper_src": "arXiv geophysics / earth science", "intent": "earth_science"},
    {"id": "arxiv-gr-qc", "loader": "arxiv", "arxiv_cat": "gr-qc",
     "kind": "paper", "paper_src": "arXiv relativity / cosmology", "intent": "physics"},
    # ===== MEDICAL / NURSING (Indian exams + nursing) =====
    {"id": "openlifescienceai/medmcqa", "kind": "medmcqa", "intent": "medical"},
    {"id": "NevenaD/MedNurse-QA", "kind": "qa",
     "q": ["question", "instruction", "Question"], "a": ["answer", "output", "Answer"], "intent": "nursing"},
    # ===== INDIAN COMPETITIVE EXAMS =====
    {"id": "169Pi/exambench", "kind": "qa",
     "q": ["question", "instruction", "prompt", "input"],
     "a": ["answer", "solution", "response", "output", "explanation"], "intent": "exam"},
]

# ---- Programmatic bulk expansion (huge volume + breadth) -------------------
# arXiv categories: millions of genuine research abstracts across every science
# (CS/AI, math, biology, physics, chemistry, earth & ocean, economics, etc.).
_ARXIV_CATS = {
    "cs.AI": "artificial intelligence", "cs.LG": "machine learning",
    "cs.CL": "natural language processing", "cs.CV": "computer vision",
    "cs.CR": "cryptography & security", "cs.RO": "robotics",
    "cs.DC": "distributed computing", "cs.NE": "neural & evolutionary computing",
    "math.OC": "optimization & control", "math.PR": "probability",
    "math.NA": "numerical analysis", "math.ST": "statistics theory",
    "math.DS": "dynamical systems", "stat.ML": "statistical machine learning",
    "stat.AP": "applied statistics", "q-bio.BM": "biomolecules",
    "q-bio.NC": "neuroscience", "q-bio.PE": "populations & evolution",
    "q-bio.GN": "genomics", "cond-mat.mtrl-sci": "materials science",
    "cond-mat.stat-mech": "statistical mechanics",
    "cond-mat.supr-con": "superconductivity",
    "hep-ph": "high-energy particle physics", "hep-th": "theoretical physics",
    "nucl-th": "nuclear theory", "quant-ph": "quantum physics",
    "physics.med-ph": "medical physics", "physics.bio-ph": "biophysics",
    "physics.chem-ph": "chemical physics", "physics.flu-dyn": "fluid dynamics",
    "physics.ao-ph": "atmospheric & oceanic physics",
    "physics.optics": "optics", "physics.plasm-ph": "plasma physics",
    "physics.app-ph": "applied physics", "physics.soc-ph": "physics of society",
    "eess.SP": "signal processing", "eess.SY": "systems & control",
    "eess.IV": "image & video processing", "econ.GN": "general economics",
    "q-fin.GN": "general finance",
}
for _c, _d in _ARXIV_CATS.items():
    DATASETS.append({"id": "arxiv-" + _c.replace(".", "-"), "loader": "arxiv",
                     "arxiv_cat": _c, "kind": "paper",
                     "paper_src": "arXiv " + _d, "intent": "research"})

# OpenAlex: 250M+ scholarly works, FREE, no key. Institution-filtered pulls give
# the EXACT research output of MIT / IIT / AIIMS / world-famous institutes.
_OA_INSTITUTES = [
    "Massachusetts Institute of Technology", "Stanford University",
    "Harvard University", "California Institute of Technology",
    "University of Oxford", "University of Cambridge",
    "Princeton University", "University of California, Berkeley",
    "ETH Zurich", "Indian Institute of Science",
    "Indian Institute of Technology Bombay",
    "Indian Institute of Technology Delhi",
    "Indian Institute of Technology Madras",
    "Indian Institute of Technology Kanpur",
    "Indian Institute of Technology Kharagpur",
    "All India Institute of Medical Sciences",
    "Tsinghua University", "University of Tokyo", "Max Planck Society",
    "European Organization for Nuclear Research",
]
for _inst in _OA_INSTITUTES:
    _slug = re.sub(r"[^a-z0-9]+", "-", _inst.lower()).strip("-")[:40]
    DATASETS.append({"id": "openalex-" + _slug, "loader": "openalex",
                     "kind": "paper", "oa_inst": _inst,
                     "paper_src": _inst, "intent": "research_institute"})
# Broad top-cited science across ALL fields (institution-agnostic).
DATASETS.append({"id": "openalex-top-science", "loader": "openalex",
                 "kind": "paper",
                 "paper_src": "OpenAlex (top-cited global science)",
                 "intent": "research"})


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


def _from_choices(ex, spec):
    """Generic MCQ across ALL subjects. Handles MMLU (choices list + int answer),
    MMLU-Pro (options list + letter/answer_index), ARC & OpenBookQA
    (choices={text,label} + answerKey letter/number)."""
    q = _first(ex, spec.get("q", ["question", "question_stem"]))
    if not q:
        return "", ""
    raw = None
    for cf in (spec.get("opts") or ["choices", "options"]):
        if ex.get(cf) is not None:
            raw = ex.get(cf)
            break
    if raw is None:
        return "", ""
    texts, labels = [], []
    if isinstance(raw, dict):
        texts = [str(t).strip() for t in (raw.get("text") or raw.get("choices") or [])]
        labels = [str(l).strip() for l in (raw.get("label") or [])]
    elif isinstance(raw, (list, tuple)):
        texts = [str(t).strip() for t in raw]
    texts = [t for t in texts if t]
    if len(texts) < 2:
        return "", ""
    letters = [chr(65 + i) for i in range(len(texts))]
    if not labels or len(labels) != len(texts):
        labels = letters
    av = None
    for af in (spec.get("ans") or ["answer_index", "answerKey", "answer", "label"]):
        if ex.get(af) is not None and str(ex.get(af)).strip() != "":
            av = ex.get(af)
            break
    if av is None or isinstance(av, bool):
        return "", ""
    ans_idx = None
    if isinstance(av, int):
        ans_idx = av
    else:
        s = str(av).strip()
        if s in labels:
            ans_idx = labels.index(s)
        elif s.isdigit():
            ans_idx = int(s)
            if ans_idx not in range(len(texts)) and (ans_idx - 1) in range(len(texts)):
                ans_idx -= 1
        elif len(s) == 1 and s.upper() in letters:
            ans_idx = letters.index(s.upper())
    if ans_idx is None or ans_idx < 0 or ans_idx >= len(texts):
        return "", ""
    qfull = q + "\nOptions:\n" + "\n".join(
        "%s) %s" % (letters[i], texts[i]) for i in range(len(texts)))
    ans = "The correct answer is %s) %s." % (letters[ans_idx], texts[ans_idx])
    for ef in (spec.get("exp") or ["cot_content", "explanation", "support", "exp"]):
        e = ex.get(ef)
        if e and str(e).strip():
            ans += " " + str(e).strip()
            break
    return qfull, ans


def _from_paper(ex, spec):
    """Build a knowledge Q/A from a research-paper record (NASA NTRS / arXiv).
    Uses the abstract when present; otherwise a factual citation entry."""
    title = re.sub(r"\s+", " ", str(ex.get("title") or "")).strip()
    if len(title) < 6:
        return "", ""
    abs_ = re.sub(r"\s+", " ", str(ex.get("abstract") or ex.get("summary") or "")).strip()
    names = []
    aa = ex.get("authors") or ex.get("authorAffiliations")
    if isinstance(aa, list):
        for x in aa:
            if isinstance(x, str):
                names.append(x.strip())
            elif isinstance(x, dict):
                m = (x.get("meta") or {}).get("author") or {}
                names.append((m.get("name") or x.get("name") or "").strip())
    names = [n for n in names if n][:6]
    pub = ""
    pl = ex.get("publications")
    if isinstance(pl, list) and pl and isinstance(pl[0], dict):
        pub = (pl[0].get("publicationName") or "").strip()
    src = spec.get("paper_src", "space research")
    q = "Explain the research paper: %s" % title
    meta = []
    if names:
        meta.append("Authors: " + ", ".join(names))
    if pub:
        meta.append("Published in: " + pub)
    meta.append("Source: " + src)
    if abs_:
        a = abs_ + " (" + "; ".join(meta) + ")"
    else:
        a = "%s. A %s document%s%s." % (
            title, src,
            (" authored by " + ", ".join(names)) if names else "",
            (", published in " + pub) if pub else "")
    return q, a


def _pair(ex, spec):
    kind = spec.get("kind", "qa")
    if kind == "chat":
        return _from_chat(ex, spec.get("conv", "conversations"))
    if kind == "medmcqa":
        return _from_medmcqa(ex)
    if kind == "choices":
        return _from_choices(ex, spec)
    if kind == "paper":
        return _from_paper(ex, spec)
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
    _throttle_commit()
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
    _throttle_commit()
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
    _throttle_commit()
    api.upload_file(path_or_fileobj=io.BytesIO(buf), path_in_repo=STATE_FILE,
                    repo_id=HF_REPO, repo_type="dataset")


_SENTINEL = object()
FLUSH_BYTES = int(os.environ.get("FLUSH_BYTES", str(96 * 1024 * 1024)))

# HuggingFace allows ~128 repo commits/hour. Every shard / bloom / state upload
# is a commit, so we self-throttle to stay safely under the limit (never 429).
_COMMITS = []
COMMIT_LIMIT = int(os.environ.get("COMMIT_LIMIT", "115"))


def _throttle_commit():
    now = time.time()
    while _COMMITS and now - _COMMITS[0] > 3600:
        _COMMITS.pop(0)
    if len(_COMMITS) >= COMMIT_LIMIT:
        wait = int(3600 - (now - _COMMITS[0])) + 5
        print("  HF commit-budget guard: sleeping %ds (stay < 128/hr)" % wait, flush=True)
        time.sleep(max(1, wait))
        now = time.time()
        while _COMMITS and now - _COMMITS[0] > 3600:
            _COMMITS.pop(0)
    _COMMITS.append(time.time())


def _lowmem_parquet_iter(repo_id, config, split):
    """Yield dict rows by reading the dataset's parquet files in SMALL batches
    via HfFileSystem -> bounded RAM even for huge / long-text datasets."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem(token=HF_TOKEN)
    split = split or "train"
    cands = list(fs.glob("datasets/%s/**/*.parquet" % repo_id))
    if not cands:
        raise RuntimeError("no parquet files for %s" % repo_id)
    sel = [f for f in cands if (("/%s/" % split) in f or ("/%s-" % split) in f
                                or f.endswith("/%s.parquet" % split))]
    if config:
        cfg = [f for f in sel if ("/%s/" % config) in f or ("/%s-" % config) in f]
        if cfg:
            sel = cfg
    for path in (sel or cands):
        with fs.open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    yield row


def _run_with_timeout(fn, seconds):
    """Run fn() in a daemon thread; raise TimeoutError if it overruns. Keeps a
    hanging / script-based dataset from freezing the whole run."""
    import threading
    box = {}

    def _w():
        try:
            box["r"] = fn()
        except BaseException as e:   # noqa
            box["e"] = e

    t = threading.Thread(target=_w, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError("timed out after %ss" % seconds)
    if "e" in box:
        raise box["e"]
    return box.get("r")


def _ntrs_iter(start=0):
    """Stream NASA NTRS citations (645k+ public records) via the public API,
    starting at offset `start` (resumable). Yields raw citation dicts."""
    import json as _json
    import urllib.request as _u
    import urllib.parse as _up
    frm = int(start)
    hard = int(os.environ.get("NTRS_MAX", "200000"))
    size = 100
    while frm < hard:
        url = "https://ntrs.nasa.gov/api/citations/search?" + _up.urlencode(
            {"page.size": size, "page.from": frm})
        req = _u.Request(url, headers={"User-Agent": "Mozilla/5.0 (DamruBot)"})
        try:
            d = _json.load(_u.urlopen(req, timeout=60))
        except Exception as e:
            print("  ntrs stop @%d: %s" % (frm, str(e)[:80]), flush=True)
            break
        res = d.get("results") or []
        if not res:
            break
        for r in res:
            yield r
        frm += size
        time.sleep(0.3)


def _arxiv_iter(cat, start=0):
    """Stream arXiv abstracts for a category via the export API (resumable).
    Covers space/physics/earth research from scientists across NASA, ESA, ISRO,
    JAXA, ROSCOSMOS, etc. Yields {title, abstract, authors}."""
    import urllib.request as _u
    import urllib.parse as _up
    import xml.etree.ElementTree as ET
    ns = {"a": "http://www.w3.org/2005/Atom"}
    frm = int(start)
    hard = int(os.environ.get("ARXIV_MAX", "50000"))
    size = 100
    empties = 0
    while frm < hard:
        url = "http://export.arxiv.org/api/query?" + _up.urlencode({
            "search_query": "cat:%s" % cat, "start": frm, "max_results": size,
            "sortBy": "submittedDate", "sortOrder": "descending"})
        mail = os.environ.get("OPENALEX_MAILTO", "research@damru.ai")
        req = _u.Request(url, headers={
            "User-Agent": "DamruAI/1.0 (+https://huggingface.co/Damaru-ai; mailto:%s)" % mail})
        raw = None
        for attempt in range(6):
            try:
                raw = _u.urlopen(req, timeout=90).read().decode("utf-8", "ignore")
                break
            except Exception as e:
                code = getattr(e, "code", None)
                if code in (429, 403, 503, 500):
                    w = 15 * (attempt + 1)
                    print("  arxiv %s @%d -> backoff %ds" % (code, frm, w), flush=True)
                    time.sleep(w)
                    continue
                print("  arxiv stop @%d: %s" % (frm, str(e)[:80]), flush=True)
                raw = None
                break
        if raw is None:
            break
        try:
            root = ET.fromstring(raw)
        except Exception as e:
            print("  arxiv parse stop @%d: %s" % (frm, str(e)[:80]), flush=True)
            break
        ents = root.findall("a:entry", ns)
        if not ents:
            empties += 1
            if empties >= 3:
                break
            time.sleep(3)
            continue
        empties = 0
        for e in ents:
            tt = e.find("a:title", ns)
            ss = e.find("a:summary", ns)
            names = [(a.find("a:name", ns).text or "").strip()
                     for a in e.findall("a:author", ns)]
            yield {"title": (tt.text if tt is not None else "") or "",
                   "abstract": (ss.text if ss is not None else "") or "",
                   "authors": names}
        frm += size
        time.sleep(3)   # arXiv asks ~3s between calls


def _oa_abstract(inv):
    """Reconstruct an abstract from an OpenAlex abstract_inverted_index."""
    if not isinstance(inv, dict) or not inv:
        return ""
    pos = []
    for word, idxs in inv.items():
        if isinstance(idxs, list):
            for i in idxs:
                pos.append((i, word))
    pos.sort()
    return " ".join(w for _, w in pos)


def _oa_get(url):
    """GET a JSON OpenAlex page with a descriptive UA + 429 backoff."""
    import json as _json
    import urllib.request as _u
    mail = os.environ.get("OPENALEX_MAILTO", "research@damru.ai")
    req = _u.Request(url, headers={
        "User-Agent": "DamruAI/1.0 (+https://huggingface.co/Damaru-ai; mailto:%s)" % mail})
    for attempt in range(6):
        try:
            return _json.load(_u.urlopen(req, timeout=90))
        except Exception as e:
            code = getattr(e, "code", None)
            if code in (429, 403, 503, 500):
                w = 15 * (attempt + 1)
                print("  openalex %s -> backoff %ds" % (code, w), flush=True)
                time.sleep(w)
                continue
            print("  openalex stop: %s" % str(e)[:90], flush=True)
            return None
    return None


def _oa_ror(name):
    """Resolve an institution name -> its ROR id via OpenAlex (best match)."""
    import urllib.parse as _up
    mail = os.environ.get("OPENALEX_MAILTO", "research@damru.ai")
    url = "https://api.openalex.org/institutions?" + _up.urlencode(
        {"search": name, "per-page": 1, "mailto": mail})
    d = _oa_get(url)
    res = (d or {}).get("results") or []
    if res:
        rid = res[0].get("ror") or res[0].get("id") or ""
        return rid, (res[0].get("display_name") or name)
    return "", name


def _openalex_iter(spec, start_cursor="*"):
    """Stream works from OpenAlex (250M+ scholarly works, free, no key). Filter
    by institution (MIT / IIT / AIIMS / any world institute) and/or concept;
    rebuild the abstract from its inverted index. Cursor-resumable: writes
    spec['_oa_cursor'] as it advances. Yields {title, abstract, authors,
    publications}."""
    import urllib.parse as _up
    mail = os.environ.get("OPENALEX_MAILTO", "research@damru.ai")
    hard = int(os.environ.get("OPENALEX_MAX", "300000"))
    filters = ["has_abstract:true"]
    if spec.get("oa_inst"):
        ror, disp = _oa_ror(spec["oa_inst"])
        if not ror:
            print("  openalex: no ROR for %s -> skip" % spec["oa_inst"], flush=True)
            return
        filters.append("institutions.ror:" + ror)
        print("  [OpenAlex %s -> %s]" % (disp, ror), flush=True)
    if spec.get("oa_concept"):
        filters.append("concepts.id:" + spec["oa_concept"])
    cursor = start_cursor or "*"
    seen = 0
    while cursor and seen < hard:
        url = "https://api.openalex.org/works?" + _up.urlencode({
            "filter": ",".join(filters), "per-page": 200, "cursor": cursor,
            "mailto": mail, "sort": "cited_by_count:desc"})
        d = _oa_get(url)
        if not d:
            return
        res = d.get("results") or []
        if not res:
            break
        for w in res:
            names = []
            for au in (w.get("authorships") or [])[:6]:
                nm = ((au.get("author") or {}).get("display_name") or "").strip()
                if nm:
                    names.append(nm)
            src = (w.get("primary_location") or {}).get("source") or {}
            venue = (src.get("display_name") or "").strip()
            yield {
                "title": w.get("title") or w.get("display_name") or "",
                "abstract": _oa_abstract(w.get("abstract_inverted_index")),
                "authors": names,
                "publications": [{"publicationName": venue}] if venue else [],
            }
            seen += 1
        cursor = (d.get("meta") or {}).get("next_cursor")
        spec["_oa_cursor"] = cursor or ""
        time.sleep(0.2)


def open_rows(spec, start=0):
    """Unified, memory-safe row iterator. Live-API loaders (NASA NTRS, arXiv)
    first; then the low-mem parquet reader (bounded RAM); then streaming
    load_dataset. HF paths are wrapped in a load timeout so a hanging dataset is
    skipped instead of stalling for hours."""
    loader = spec.get("loader")
    if loader == "ntrs":
        print("  [NASA NTRS API loader @ %d]" % start, flush=True)
        return _ntrs_iter(start)
    if loader == "arxiv":
        print("  [arXiv API loader %s @ %d]" % (spec.get("arxiv_cat"), start), flush=True)
        return _arxiv_iter(spec.get("arxiv_cat", "astro-ph"), start)
    if loader == "openalex":
        sc = start if (isinstance(start, str) and start) else "*"
        return _openalex_iter(spec, start_cursor=sc)
    rid, cfg = spec["id"], spec.get("config")
    split = spec.get("split", "train")
    try:
        it = _lowmem_parquet_iter(rid, cfg, split)
        first = _run_with_timeout(lambda: next(it, _SENTINEL), 90)
        if first is _SENTINEL:
            return iter(())
        print("  [low-mem parquet reader]", flush=True)

        def _gen():
            yield first
            for r in it:
                yield r
        return _gen()
    except Exception as e:
        print("  low-mem reader off (%s); trying stream" % str(e)[:90], flush=True)
    from datasets import load_dataset
    ds = _run_with_timeout(
        lambda: load_dataset(rid, cfg, split=split, streaming=True), 150)
    return iter(ds)


def _resume_val(spec, base, scanned):
    """Next-run resume token. OpenAlex advances an opaque cursor (written to
    spec['_oa_cursor']); offset loaders (NTRS / arXiv) use an integer offset."""
    if spec.get("loader") == "openalex":
        return spec.get("_oa_cursor") or "*"
    try:
        return int(base) + scanned
    except Exception:
        return scanned


def process_dataset(api, spec, bf, deadline, st=None, key=None):
    """Returns (inserted, completed). completed=False if stopped by budget.
    For live-API loaders, a cursor in `st` makes scanning resumable across runs
    so we never re-fetch the same API pages."""
    name = spec["id"] + ("/" + spec["config"] if spec.get("config") else "")
    tag = _safe_tag(spec)
    is_api = bool(spec.get("loader"))
    base = ((st or {}).get("cursor", {}) or {}).get(key, 0) if is_api else 0
    print("\n=== %s (cap %d) ===" % (name, PER_DATASET), flush=True)
    try:
        row_source = open_rows(spec, start=base)
    except Exception as e:
        print("  SKIP (load failed):", str(e)[:160], flush=True)
        return 0, True   # treat as done so we don't retry forever
    buf, inserted, scanned, idx, completed = [], 0, 0, 0, True
    buf_bytes = 0
    scan_cap = PER_DATASET * SCAN_MULT
    try:
        for ex in row_source:
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
            buf_bytes += len(q) + len(a)
            inserted += 1
            # flush on row-count OR byte-budget (keeps long-text datasets safe)
            if len(buf) >= SHARD_SIZE or buf_bytes >= FLUSH_BYTES:
                _flush(api, buf, tag, idx)
                idx += 1
                buf = []
                buf_bytes = 0
                # persist dedup + cursor PERIODICALLY (every 4 shards) to keep
                # well under the HF 128-commits/hour budget
                if idx % 4 == 0:
                    save_bloom(api, bf)
                    if is_api and st is not None and key is not None:
                        st.setdefault("cursor", {})[key] = _resume_val(spec, base, scanned)
                        write_state(api, st)
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
    if is_api and st is not None and key is not None:
        if completed:
            (st.get("cursor", {}) or {}).pop(key, None)
        else:
            st.setdefault("cursor", {})[key] = _resume_val(spec, base, scanned)
        write_state(api, st)
    print("  DONE %s -> +%d genuine rows (scanned %d, completed=%s)"
          % (name, inserted, scanned, completed), flush=True)
    return inserted, completed


def seed_bloom_from_existing(api, bf, st):
    """One-time: teach the bloom EVERY question already on HF (live track +
    phase6 ingested rows) so the bulk track never re-creates a duplicate of
    what is already there. Runs only once (state['seeded'])."""
    if st.get("seeded") or os.environ.get("SEED_FROM_HF", "true").lower() != "true":
        return
    from datasets import load_dataset
    print("Seeding bloom from existing HF dataset (one-time)...", flush=True)
    n = 0
    try:
        ds = load_dataset(HF_REPO, split="train", streaming=True)
        for ex in ds:
            if not isinstance(ex, dict):
                continue
            q = (ex.get("question") or "").strip()
            if q:
                bf.add(normalize(q))
                n += 1
                if n % 50000 == 0:
                    print("  seeded %d existing rows" % n, flush=True)
    except Exception as e:
        print("  seed skipped (will dedup at training instead):", str(e)[:160], flush=True)
        return
    save_bloom(api, bf)
    st["seeded"] = True
    write_state(api, st)
    print("Seeded bloom with %d existing questions." % n, flush=True)


def main():
    if not HF_TOKEN:
        print("ERROR: set HF_TOKEN")
        sys.exit(1)
    from huggingface_hub import login
    login(HF_TOKEN)
    api = _api()
    bf = load_bloom()
    st = read_state()
    seed_bloom_from_existing(api, bf, st)
    done = set(st.get("done", []))
    tried = set(st.get("tried", []))
    # Self-heal: a dataset STARTED last run but never finished almost certainly
    # killed the runner (OOM / "operation was canceled"). Quarantine it so a
    # resume steps PAST the poison pill instead of dying on it forever.
    quarantined = tried - done
    for q in sorted(quarantined):
        print("quarantine (crashed a previous run -> skipping):", q, flush=True)
    skip_set = done | quarantined
    pool = [d for d in DATASETS if (not ONLY or any(o in d["id"] for o in ONLY))]
    print("Damru BULK harvest | datasets=%d | per=%d | budget=%dmin | bypass=Supabase"
          % (len(pool), PER_DATASET, RUN_BUDGET_MIN), flush=True)
    deadline = time.time() + RUN_BUDGET_MIN * 60
    for spec in pool:
        key = spec["id"] + "::" + str(spec.get("config", "")) + "::" + spec.get("split", "train")
        if key in skip_set:
            print("skip:", key, flush=True)
            continue
        if time.time() > deadline:
            print("budget reached; will resume next run.", flush=True)
            break
        # Breadcrumb on HF BEFORE the risky load. A hard OOM kill can't be caught
        # in Python, so this persisted marker lets the next run quarantine it.
        tried.add(key)
        st["tried"] = sorted(tried)
        write_state(api, st)
        ins, completed = process_dataset(api, spec, bf, deadline, st=st, key=key)
        st["total"] = st.get("total", 0) + ins
        save_bloom(api, bf)
        tried.discard(key)            # survived the load -> clear breadcrumb
        if completed:
            done.add(key)
        st["done"] = sorted(done)
        st["tried"] = sorted(tried)
        write_state(api, st)
        print("== %s -> +%d (running total ~%d) ==" % (key, ins, st["total"]), flush=True)
    print("\nRUN COMPLETE. cumulative bulk total ~%d rows" % st.get("total", 0), flush=True)


if __name__ == "__main__":
    main()
