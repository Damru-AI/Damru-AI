#!/usr/bin/env python3
"""
Damru RAG INDEX  --  RESUME RUNNER (v5)
=======================================
rebuild_rag.py (v5) jab apna 11h30m ka time budget khatam karke "PARTIAL"
bolta hai, tab YE file chalao. Ye kya karta hai:

  1. INDEX_REPO se state.json padhta hai aur progress report print karta hai
     (kitne vectors, kitni files done, kaun si file adhuri thi, kis row par).
  2. Wahi v5 engine load karta hai -- pehle local rebuild_rag.py, warna
     INDEX_REPO/builder_src.py (bilkul wahi version jisne state banaya tha),
     warna GitHub raw se.
  3. RESUME=1 + TIME_BUDGET_SEC=41400 (11h30m) ke saath usko chala deta hai.
     Engine partial index.faiss + meta.parquet + dedup hashes download karke
     USI index me aage add karta hai -- ek bhi row dobara embed nahi hoti.

Jab tak log "STATUS: COMPLETE" na bole, ye file dobara-dobara chalate raho.
Har run apne aap 11h30m par ruk kar progress upload kar deta hai, isliye
Kaggle ka 12 ghante ka hard kill (exit 137) kabhi kaam barbaad nahi karega.

KAGGLE ME KAISE CHALAYE
  1. New Notebook -> Settings: Accelerator = GPU T4 x2, Internet = ON
  2. Add-ons -> Secrets -> HF_TOKEN (WRITE)  [Attach karo]
  3. Ye poori file EK cell me paste karke run / Save Version.

ENV (sab optional)
  HF_TOKEN           required (WRITE)  -- Kaggle Secret se bhi utha lega
  INDEX_REPO         Damaru-ai/damru-rag-index
  TIME_BUDGET_SEC    41400   (11h30m) -- self-stop budget
  UPLOAD_RESERVE_SEC 1500    finish + upload ke liye reserve
  CKPT_EVERY_SEC     21600   mid-run safety checkpoint (0 = off)
  ENGINE_URL         GitHub raw fallback URL
  ALLOW_FRESH        1       state na mile to fresh build chalu kar do
  MAX_INDEX / DRY_RUN / BATCH ... -- engine ke saare env yahan bhi chalte hain
"""
import os
import sys
import json
import urllib.request

ENGINE_FILE = "rebuild_rag.py"


def _env(k, d):
    v = os.environ.get(k)
    return v if v not in (None, "") else d


INDEX_REPO = _env("INDEX_REPO", "Damaru-ai/damru-rag-index")
RAW_URL    = _env("ENGINE_URL",
                  "https://raw.githubusercontent.com/Damru-AI/Damru-AI/"
                  "main/phase4/rebuild_rag.py")
BUDGET     = int(_env("TIME_BUDGET_SEC", "41400"))     # 11h30m
RESERVE    = int(_env("UPLOAD_RESERVE_SEC", "1500"))
CKPT       = int(_env("CKPT_EVERY_SEC", "21600"))
ALLOW_FRESH = _env("ALLOW_FRESH", "1") == "1"


def _log(*a):
    print("[rag-resume]", *a, flush=True)


def _hms(sec):
    sec = int(max(0, sec))
    return "%dh%02dm%02ds" % (sec // 3600, (sec % 3600) // 60, sec % 60)


def _resolve_hf_token():
    """WRITE HF token env se ya Kaggle Secrets se auto-nikaalo (Save Version
    run me interactive env nahi hota, isliye secret bhi padho)."""
    for _k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN",
               "HF_API_TOKEN", "HUGGINGFACE_TOKEN"):
        _v = (os.environ.get(_k) or "").strip()
        if _v:
            return _v
    try:
        from kaggle_secrets import UserSecretsClient
        _c = UserSecretsClient()
        for _k in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            try:
                _v = (_c.get_secret(_k) or "").strip()
                if _v:
                    return _v
            except Exception:
                pass
    except Exception:
        pass
    return ""


def _ensure_hub():
    import importlib.util as _u
    if _u.find_spec("huggingface_hub") is None:
        import subprocess
        _log("installing huggingface_hub ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                        "huggingface_hub"])


def _fetch_state(token):
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(INDEX_REPO, "state.json", repo_type="dataset",
                            token=token or None)
        with open(p, "r", encoding="utf-8") as f:
            st = json.load(f)
        return st if isinstance(st, dict) else {}
    except Exception as e:
        _log("state.json fetch failed:", str(e)[:120])
        return {}


def _report(st):
    done = int(st.get("files_done") or len(st.get("done_files") or []))
    tot = int(st.get("files_total") or 0)
    pct = (100.0 * done / tot) if tot else 0.0
    _log("-" * 58)
    _log("PREVIOUS PROGRESS (state.json v%s)" % st.get("version"))
    _log("  complete       :", st.get("complete"))
    _log("  vectors indexed: %s  (meta rows %s)"
         % (st.get("count"), st.get("meta_rows")))
    _log("  index type     : %s  dim=%s  model=%s"
         % (st.get("index_type"), st.get("dim"), st.get("embed_model")))
    _log("  files done     : %d / %d  (%.1f%%)" % (done, tot, pct))
    part = st.get("partial") or {}
    if part:
        _log("  adhuri file    : %s  @ row %s"
             % (part.get("file"), part.get("rows_done")))
    _log("  scanned / kept : %s / %s" % (st.get("scanned"), st.get("kept")))
    _log("  last stop      : %s   (updated %s)"
         % (st.get("stop_reason"), st.get("updated_at")))
    for r in (st.get("runs") or [])[-5:]:
        _log("   run:", json.dumps(r, ensure_ascii=False)[:200])
    _log("-" * 58)


def _local_candidates():
    cands = []
    try:
        cands.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  ENGINE_FILE))
    except Exception:
        pass
    cands.append(os.path.join(os.getcwd(), ENGINE_FILE))
    cands.append("/kaggle/working/" + ENGINE_FILE)
    try:
        for dp, _dn, fn in os.walk("/kaggle/input"):
            if ENGINE_FILE in fn:
                cands.append(os.path.join(dp, ENGINE_FILE))
    except Exception:
        pass
    return cands


def _looks_like_engine(src):
    return bool(src) and "def main(" in src and "STATE_VER" in src


def _engine_source(token):
    """v5 engine source laao: local file -> HF builder_src.py -> GitHub raw."""
    for p in _local_candidates():
        try:
            if p and os.path.exists(p):
                with open(p, "r", encoding="utf-8") as f:
                    src = f.read()
                if _looks_like_engine(src):
                    _log("engine source: local", p, "(%d bytes)" % len(src))
                    return src
        except Exception:
            pass
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(INDEX_REPO, "builder_src.py", repo_type="dataset",
                            token=token or None)
        with open(p, "r", encoding="utf-8") as f:
            src = f.read()
        if _looks_like_engine(src):
            _log("engine source: %s/builder_src.py (%d bytes)"
                 % (INDEX_REPO, len(src)))
            return src
    except Exception as e:
        _log("builder_src.py not on HF (%s)" % str(e)[:90])
    try:
        req = urllib.request.Request(RAW_URL,
                                     headers={"User-Agent": "damru-resume/5"})
        with urllib.request.urlopen(req, timeout=90) as r:
            src = r.read().decode("utf-8", "ignore")
        if _looks_like_engine(src):
            _log("engine source: GitHub raw (%d bytes)" % len(src))
            return src
        _log("GitHub raw content engine jaisa nahi laga")
    except Exception as e:
        _log("GitHub raw fetch failed:", str(e)[:120])
    return ""


def main():
    token = _resolve_hf_token()
    assert token, (
        "HF_TOKEN (WRITE) nahi mila. Kaggle: Add-ons -> Secrets -> 'HF_TOKEN' "
        "(WRITE) add + Attach karo, ya pehle cell me "
        "import os; os.environ['HF_TOKEN']='hf_xxx'"
    )
    os.environ.setdefault("HF_TOKEN", token)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)
    _ensure_hub()

    st = _fetch_state(token)
    if st:
        _report(st)
        if st.get("complete"):
            _log("pichhla state COMPLETE tha -> sirf NAYI files add hongi "
                 "(daily learning ka fresh data).")
        os.environ["RESUME"] = "1"
    else:
        _log("INDEX_REPO me state.json nahi mila:", INDEX_REPO)
        if not ALLOW_FRESH:
            _log("ALLOW_FRESH=0 -> ruk gaya. Pehle rebuild_rag.py chalao.")
            return
        _log("-> resume ke bajaye FRESH build shuru karta hoon (safe hai, "
             "budget ke saath).")
        os.environ["RESUME"] = "auto"
    os.environ["FRESH"] = "0"
    os.environ["TIME_BUDGET_SEC"] = str(BUDGET)
    os.environ["UPLOAD_RESERVE_SEC"] = str(RESERVE)
    os.environ.setdefault("CKPT_EVERY_SEC", str(CKPT))

    src = _engine_source(token)
    assert src, (
        "v5 engine source nahi mila. Fix: (A) is cell se pehle GitHub se "
        "phase4/rebuild_rag.py ka content ek file me likh do, ya (B) ENGINE_URL "
        "env set karo, ya (C) rebuild_rag.py hi seedha paste karke chala do "
        "(usme resume built-in hai)."
    )
    try:
        with open(ENGINE_FILE, "w", encoding="utf-8") as f:
            f.write(src)
    except Exception as e:
        _log("engine local write warn:", str(e)[:100])
    _log("starting engine | budget %s | reserve %s | ckpt %s"
         % (_hms(BUDGET), _hms(RESERVE), _hms(int(os.environ["CKPT_EVERY_SEC"]))))
    g = {"__name__": "__main__", "__file__": os.path.abspath(ENGINE_FILE),
         "__builtins__": __builtins__}
    exec(compile(src, ENGINE_FILE, "exec"), g)
    _log("resume run finished -- upar ka 'STATUS:' line dekho "
         "(COMPLETE = ho gaya, PARTIAL = ye file dobara chala do).")


if __name__ == "__main__":
    main()
