#!/usr/bin/env python3
"""
Damru WIRE CHECK -- one-shot connectivity + health self-test
============================================================
Verifies every external dependency Damru needs is reachable and configured, so
you can see at a glance whether HF <-> GitHub <-> Supabase <-> the Space are all
wired together.

Checks:
  * env         -- which secrets are set (HF/Supabase/OpenBrain/WebSearch)
  * hf.token    -- HF_TOKEN valid (whoami)
  * hf.dataset  -- damru-knowledge / damru-train / damru-rag-index reachable
  * hf.model    -- 14b-lora / 14b-gguf / tutor-lora / coder-lora reachable
  * rag.index   -- config.json + index.faiss + meta.parquet present (+ count/dim)
  * supabase    -- log table reachable (chat logging + hot RAG source)
  * github.repo -- Damru-AI/Damru-AI reachable (Kaggle clones this)
  * space.health-- live Space /health (backend / rag / brain_ready)

Read-only + safe: NO writes, NO side effects. Prints a PASS/WARN/FAIL report and
exits non-zero only if a REQUIRED check FAILS (WARN = optional / not configured).
Run locally, on the Space, or in CI. Deps auto-install if missing.

Env: HF_TOKEN, SUPABASE_URL, SUPABASE_KEY / SUPABASE_SERVICE_KEY, LOG_TABLE,
     SPACE_URL, INDEX_REPO, HF_DATASET, TRAIN_REPO, OWN_MODEL_REPO,
     GROQ_API_KEY, HF_ROUTER_URL, TAVILY_KEY, BRAVE_KEY, GITHUB_REPO

Built by Shiva AI for Damru.
"""
import os
import json

GITHUB_REPO   = os.environ.get("GITHUB_REPO", "Damru-AI/Damru-AI")
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
SRC_REPO      = os.environ.get("HF_DATASET", "Damaru-ai/damru-knowledge")
INDEX_REPO    = os.environ.get("INDEX_REPO", "Damaru-ai/damru-rag-index")
TRAIN_REPO    = os.environ.get("TRAIN_REPO", "Damaru-ai/damru-train")
SPACE_URL     = os.environ.get("SPACE_URL", "")
SUPABASE_URL  = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY  = (os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", ""))
LOG_TABLE     = os.environ.get("LOG_TABLE", "damru_chats")
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
HF_ROUTER_URL = os.environ.get("HF_ROUTER_URL", "")
TAVILY_KEY    = os.environ.get("TAVILY_KEY", "")
BRAVE_KEY     = os.environ.get("BRAVE_KEY", "")

MODEL_REPOS = [r for r in [
    os.environ.get("OWN_MODEL_REPO", ""),
    "Damaru-ai/damru-14b-lora", "Damaru-ai/damru-14b-gguf",
    "Damaru-ai/damru-tutor-lora", "Damaru-ai/damru-coder-lora",
] if r]

_R = []  # list of (status, name, detail)


def _ensure_deps():
    import importlib.util
    missing = [pip for mod, pip in [("requests", "requests"),
               ("huggingface_hub", "huggingface_hub")]
               if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    import subprocess, sys
    print(">> installing:", missing, flush=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", *missing])


def _add(status, name, detail=""):
    _R.append((status, name, detail))
    icon = {"PASS": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]", "SKIP": "[ -- ]"}.get(status, "[ ?? ]")
    print("%s %-20s %s" % (icon, name, detail), flush=True)


def check_env():
    _add("PASS" if HF_TOKEN else "WARN", "env.HF_TOKEN",
         "set" if HF_TOKEN else "MISSING (required to push/pull private repos)")
    _add("PASS" if (SUPABASE_URL and SUPABASE_KEY) else "WARN", "env.SUPABASE",
         "url+key set" if (SUPABASE_URL and SUPABASE_KEY) else "not configured (logs / hot-RAG off)")
    _add("PASS" if (GROQ_API_KEY or HF_ROUTER_URL) else "WARN", "env.OpenBrain",
         "configured" if (GROQ_API_KEY or HF_ROUTER_URL) else "no Groq / HF router")
    _add("PASS" if (TAVILY_KEY or BRAVE_KEY) else "WARN", "env.WebSearch",
         "configured" if (TAVILY_KEY or BRAVE_KEY) else "no Tavily / Brave key")


def check_hf():
    try:
        from huggingface_hub import HfApi
        who = HfApi(token=HF_TOKEN or None).whoami()
        name = who.get("name") if isinstance(who, dict) else str(who)
        _add("PASS", "hf.token", "user=%s" % name)
    except Exception as e:
        _add("FAIL" if HF_TOKEN else "WARN", "hf.token", str(e)[:90])
        return
    from huggingface_hub import HfApi
    targets = [(SRC_REPO, "dataset"), (TRAIN_REPO, "dataset"), (INDEX_REPO, "dataset")]
    targets += [(m, "model") for m in MODEL_REPOS]
    for repo, rtype in targets:
        try:
            HfApi(token=HF_TOKEN or None).repo_info(repo_id=repo, repo_type=rtype)
            _add("PASS", "hf.%s" % rtype, repo)
        except Exception as e:
            _add("WARN", "hf.%s" % rtype, "%s : %s" % (repo, str(e)[:55]))


def check_index():
    try:
        from huggingface_hub import HfApi, hf_hub_download
        files = HfApi(token=HF_TOKEN or None).list_repo_files(INDEX_REPO, repo_type="dataset")
        need = ["config.json", "index.faiss", "meta.parquet"]
        miss = [f for f in need if f not in files]
        _add("WARN" if miss else "PASS", "rag.index_files",
             ("missing %s" % miss) if miss else "all 3 present")
        if "config.json" in files:
            p = hf_hub_download(INDEX_REPO, "config.json", repo_type="dataset", token=HF_TOKEN or None)
            with open(p) as f:
                cfg = json.load(f)
            _add("PASS", "rag.config", "count=%s dim=%s model=%s" %
                 (cfg.get("count"), cfg.get("dim"), cfg.get("embed_model")))
    except Exception as e:
        _add("WARN", "rag.index", str(e)[:90])


def check_supabase():
    if not (SUPABASE_URL and SUPABASE_KEY):
        _add("SKIP", "supabase", "not configured")
        return
    import requests
    try:
        r = requests.get(SUPABASE_URL + "/rest/v1/" + LOG_TABLE,
                         headers={"apikey": SUPABASE_KEY,
                                  "Authorization": "Bearer " + SUPABASE_KEY,
                                  "Range": "0-0"}, timeout=12)
        if r.status_code < 400:
            _add("PASS", "supabase.table", "%s reachable (HTTP %d)" % (LOG_TABLE, r.status_code))
        else:
            _add("WARN", "supabase.table", "%s -> HTTP %d" % (LOG_TABLE, r.status_code))
    except Exception as e:
        _add("FAIL", "supabase", str(e)[:90])


def check_github():
    import requests
    try:
        r = requests.get("https://raw.githubusercontent.com/%s/main/README.md" % GITHUB_REPO, timeout=12)
        if r.status_code < 400:
            _add("PASS", "github.repo", "%s reachable" % GITHUB_REPO)
            return
        r2 = requests.get("https://api.github.com/repos/%s" % GITHUB_REPO, timeout=12)
        _add("PASS" if r2.status_code < 400 else "WARN", "github.repo",
             "%s -> HTTP %d" % (GITHUB_REPO, r2.status_code))
    except Exception as e:
        _add("FAIL", "github.repo", str(e)[:90])


def check_space():
    if not SPACE_URL:
        _add("SKIP", "space.health", "set SPACE_URL to ping /health")
        return
    import requests
    try:
        r = requests.get(SPACE_URL.rstrip("/") + "/health", timeout=25)
        j = r.json()
        _add("PASS" if r.status_code < 400 else "WARN", "space.health",
             "backend=%s rag=%s brain=%s" % (j.get("backend"), j.get("rag"), j.get("brain_ready")))
    except Exception as e:
        _add("WARN", "space.health", str(e)[:90])


def main():
    print("== Damru WIRE CHECK ==", flush=True)
    _ensure_deps()
    check_env()
    check_hf()
    check_index()
    check_supabase()
    check_github()
    check_space()
    fails = [x for x in _R if x[0] == "FAIL"]
    warns = [x for x in _R if x[0] == "WARN"]
    print("", flush=True)
    print("== SUMMARY ==  %d checks | %d FAIL | %d WARN" % (len(_R), len(fails), len(warns)), flush=True)
    if fails:
        print(">> Blocking issues:", flush=True)
        for _, n, d in fails:
            print("   - %s: %s" % (n, d), flush=True)
    print(">> WARN = optional / not-configured; safe to ignore if that feature is unused.", flush=True)
    import sys
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
