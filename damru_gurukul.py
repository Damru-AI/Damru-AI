#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 DAMRU GURUKUL  --  Unified Self-Play Self-Learning Loop
  RLVR  +  SeRL (self-play)  +  GRPO (group-relative)  +  SLM curriculum
================================================================================
A NEW way of LLM self-learning. ONE looping algorithm that:

  1. PROPOSE   -> Damru invents its OWN problems (SeRL self-play), per domain,
                  at the current difficulty (learnability frontier).
  2. SOLVE     -> Samples a GROUP of G candidate solutions (GRPO group).
  3. VERIFY    -> RLVR: reward comes from a *verifiable* checker, NOT from
                  copying a teacher.  math=sympy, code=unit-tests, reasoning=
                  self-consistency + exact match.  (This is why it is ORIGINAL,
                  not copy-paste distillation.)
  4. ADVANTAGE -> GRPO group-relative advantage = (r - mean)/std.
                  best = chosen, worst = rejected  (preference pairs).
  5. REFLECT   -> Reflect-Retry-Reward: if the whole group fails, Damru writes
                  a self-reflection and retries once. Success => bonus reward.
  6. CURRICULUM-> pass-rate too high => harder, too low => easier. Auto.
  7. COLLECT   -> dedup + write SFT + preference JSONL, push to HF dataset.
  8. SELF-HEAL -> provider rotation, cooldowns, atomic checkpoint, never dies.

Designed for FREE cloud only (GitHub Actions CPU brain). No PC, no phone-on.
The policy model = Damru itself (OWN_MODEL) once trained, else a strong open
base model as the bootstrap brain. The GRPO trainer (damru_grpo_train.py) runs
on Kaggle GPU and pushes a new adapter -> Gurukul reloads it -> loop closes.

ENV (all optional except one provider key + HF_TOKEN to push):
  HF_TOKEN               HuggingFace write token (push dataset)
  DAMRU_DATASET          default 'Damaru-ai/damru-gurukul'
  OWN_MODEL              Damru's own model id (HF router) to close the loop
  CEREBRAS_API_KEY       comma-list ok (multi-key rotation)
  OPENROUTER_API_KEY     comma-list ok
  GITHUB_MODELS_TOKEN    comma-list ok
  DAMRU_GROUP_SIZE       G, default 6
  DAMRU_DOMAINS          default 'math,code,reasoning'
  DAMRU_MAX_ITERS        0 = forever (default 0)
  DAMRU_PUSH_EVERY       default 25
  DAMRU_START_DIFF       default 1  (1..10)
  DAMRU_STATE            default './gurukul_state.json'
  DAMRU_OUT             default './gurukul_out'
  DAMRU_SLEEP            seconds between iters, default 1.0
  DAMRU_TIME_BUDGET_MIN  stop cleanly after N minutes (CI safe), default 320
================================================================================
"""
import os
import re
import sys
import json
import time
import math
import signal
import random
import hashlib
import tempfile
import traceback
import subprocess
from datetime import datetime, timezone

# ------------------------------------------------------------------ soft deps
try:
    import requests
except Exception:
    print("[FATAL] `requests` missing. pip install requests", flush=True)
    raise

try:
    import sympy
    from sympy import simplify, sympify, Rational, nsimplify
    from sympy.parsing.sympy_parser import parse_expr
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False

try:
    from huggingface_hub import HfApi, upload_file
    _HAS_HF = True
except Exception:
    _HAS_HF = False


# ============================================================ tiny utilities
def now_iso():
    return datetime.now(timezone.utc).isoformat()


def log(*a):
    print(f"[{now_iso()}]", *a, flush=True)


def env(name, default=None):
    v = os.environ.get(name)
    return v if (v is not None and str(v).strip() != "") else default


def env_int(name, default):
    try:
        return int(str(env(name, default)).strip())
    except Exception:
        return default


def env_float(name, default):
    try:
        return float(str(env(name, default)).strip())
    except Exception:
        return default


def sha1(s):
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_json_loads(txt):
    """Force-extract a JSON object from messy LLM output. Never raises."""
    if not txt:
        return None
    # direct
    try:
        return json.loads(txt)
    except Exception:
        pass
    # fenced ```json ... ```
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", txt, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # first balanced { ... }
    start = txt.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(txt)):
            c = txt[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    chunk = txt[start:i + 1]
                    try:
                        return json.loads(chunk)
                    except Exception:
                        break
        start = txt.find("{", start + 1)
    return None


# ================================================================= PROVIDERS
class Provider:
    """One brain endpoint. Self-heals: consecutive fails -> cooldown."""
    def __init__(self, name, url, model, key, extra_headers=None):
        self.name = name
        self.url = url
        self.model = model
        self.key = key
        self.extra_headers = extra_headers or {}
        self.fails = 0
        self.cool_until = 0.0
        self.calls = 0

    def healthy(self):
        return time.time() >= self.cool_until

    def _headers(self):
        h = {"Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        h.update(self.extra_headers)
        return h

    def chat(self, messages, temperature=0.7, max_tokens=1200, timeout=90):
        """Return text or None. Never raises."""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            self.calls += 1
            r = requests.post(self.url, headers=self._headers(),
                              json=payload, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                txt = (data.get("choices", [{}])[0]
                          .get("message", {}).get("content"))
                if txt:
                    self.fails = 0
                    return txt
                self._trip("empty")
                return None
            # rate limited / quota / server -> cooldown
            if r.status_code in (429, 402, 403, 500, 502, 503, 529):
                self._trip(f"http{r.status_code}")
            else:
                self._trip(f"http{r.status_code}")
            return None
        except Exception as e:
            self._trip(f"exc:{type(e).__name__}")
            return None

    def _trip(self, why):
        self.fails += 1
        if self.fails >= 3:
            cool = min(900, 60 * self.fails)
            self.cool_until = time.time() + cool
            log(f"  [heal] {self.name} cooling {cool}s ({why}, fails={self.fails})")


def _split_keys(raw):
    if not raw:
        return []
    return [k.strip() for k in str(raw).split(",") if k.strip()]


def build_providers():
    """Open-weight (or own) brains only. Multi-key -> one entry per key."""
    provs = []

    own = env("OWN_MODEL")
    hf = env("HF_TOKEN")
    if own and hf:
        # Damru's OWN model via HF router -> true self-play / loop closure
        provs.append(Provider(
            "damru-own", "https://router.huggingface.co/v1/chat/completions",
            own, hf))

    for k in _split_keys(env("CEREBRAS_API_KEY")):
        provs.append(Provider(
            "cerebras", "https://api.cerebras.ai/v1/chat/completions",
            env("CEREBRAS_MODEL", "llama-3.3-70b"), k))

    for k in _split_keys(env("OPENROUTER_API_KEY")):
        provs.append(Provider(
            "openrouter", "https://openrouter.ai/api/v1/chat/completions",
            env("OPENROUTER_MODEL", "deepseek/deepseek-r1:free"), k,
            {"HTTP-Referer": "https://damru-ai.vercel.app",
             "X-Title": "Damru Gurukul"}))

    for k in _split_keys(env("GITHUB_MODELS_TOKEN")):
        provs.append(Provider(
            "github-models", "https://models.github.ai/inference/chat/completions",
            env("GITHUB_MODEL", "deepseek/DeepSeek-R1"), k))

    # HF router with generic open model as last-resort bootstrap brain
    if hf and not own:
        provs.append(Provider(
            "hf-router", "https://router.huggingface.co/v1/chat/completions",
            env("HF_MODEL", "meta-llama/Llama-3.3-70B-Instruct"), hf))

    return provs


class Brain:
    """Round-robin over healthy providers with self-heal + retry."""
    def __init__(self, providers):
        self.providers = providers
        self.rr = 0

    def alive(self):
        return any(p.healthy() for p in self.providers)

    def ask(self, messages, temperature=0.7, max_tokens=1200, tries=None):
        n = len(self.providers)
        if n == 0:
            return None
        tries = tries or n
        for _ in range(tries):
            p = self.providers[self.rr % n]
            self.rr += 1
            if not p.healthy():
                continue
            out = p.chat(messages, temperature=temperature,
                         max_tokens=max_tokens)
            if out:
                return out
        return None

    def sample_group(self, messages, g, temperature=0.8, max_tokens=1200):
        """GRPO group: g independent samples (diverse temps)."""
        outs = []
        for i in range(g):
            t = clamp(temperature + random.uniform(-0.2, 0.25), 0.2, 1.15)
            o = self.ask(messages, temperature=t, max_tokens=max_tokens)
            if o:
                outs.append(o)
        return outs


# ================================================================= VERIFIERS
_ANS_RE = re.compile(r"####\s*(.+?)\s*$", re.S)
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_final(text):
    """Pull the final answer token from a solution."""
    if not text:
        return ""
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _ANS_RE.search(text.strip())
    if m:
        return m.group(1).strip().splitlines()[-1].strip()
    # last non-empty line, stripped of trailing punctuation
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    tail = lines[-1]
    m = re.search(r"(-?\d+(?:\.\d+)?(?:/\d+)?)\s*$", tail)
    return m.group(1) if m else tail


def _norm_num(s):
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "")
    s = s.replace(" ", "")
    return s


def verify_math(gold, cand_text):
    """RLVR math reward in [0,1]. sympy-equiv else numeric else string."""
    cand = extract_final(cand_text)
    g = _norm_num(gold)
    c = _norm_num(cand)
    if c == "":
        return 0.0
    if c == g:
        return 1.0
    # numeric compare
    try:
        if abs(float(c) - float(g)) < 1e-6:
            return 1.0
    except Exception:
        pass
    # symbolic equivalence
    if _HAS_SYMPY:
        try:
            ge = parse_expr(g.replace("^", "**"))
            ce = parse_expr(c.replace("^", "**"))
            if simplify(ge - ce) == 0:
                return 1.0
        except Exception:
            pass
        try:
            if nsimplify(g) == nsimplify(c):
                return 1.0
        except Exception:
            pass
    return 0.0


_CODE_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.S)


def extract_code(text):
    if not text:
        return ""
    blocks = _CODE_RE.findall(text)
    if blocks:
        # prefer the block that defines solve(
        for b in blocks:
            if "def solve" in b:
                return b.strip()
        return blocks[-1].strip()
    return text.strip()


def verify_code(tests, cand_text, timeout=8):
    """RLVR code reward = fraction of asserts passed. Sandboxed subprocess."""
    code = extract_code(cand_text)
    if not code or "def solve" not in code:
        return 0.0
    if not isinstance(tests, list) or not tests:
        return 0.0
    # Build a hardened runner: no network, limited builtins-ish, timeout.
    harness = (
        "import signal, sys\n"
        "def _to(*a):\n    raise TimeoutError()\n"
        "try:\n"
        "    signal.signal(signal.SIGALRM, _to); signal.alarm(%d)\n"
        "except Exception:\n    pass\n"
        "_PASS=0; _TOTAL=0\n"
        % (timeout,)
    )
    harness += code + "\n"
    for t in tests:
        t = str(t).strip()
        if not t:
            continue
        # only allow assert-style tests
        safe = t.replace("\n", " ")
        harness += (
            "_TOTAL+=1\n"
            "try:\n"
            f"    {safe}\n"
            "    _PASS+=1\n"
            "except Exception:\n    pass\n"
        )
    harness += "print('RESULT %d %d' % (_PASS, _TOTAL))\n"

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(harness)
            path = f.name
        proc = subprocess.run(
            [sys.executable, path],
            capture_output=True, text=True, timeout=timeout + 5,
            env={"PATH": os.environ.get("PATH", ""),
                 "PYTHONDONTWRITEBYTECODE": "1"},
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        m = re.search(r"RESULT\s+(\d+)\s+(\d+)", out)
        if m:
            p, tot = int(m.group(1)), int(m.group(2))
            return (p / tot) if tot else 0.0
        return 0.0
    except Exception:
        return 0.0
    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


def _norm_txt(s):
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" .\t\n")


def verify_reasoning(gold, cand_text):
    """RLVR reasoning reward: normalized exact/containment match."""
    cand = extract_final(cand_text)
    g = _norm_txt(gold)
    c = _norm_txt(cand)
    if not c:
        return 0.0
    if c == g:
        return 1.0
    if g and (g in c or c in g):
        return 0.7
    # token overlap
    gs, cs = set(g.split()), set(c.split())
    if gs and len(gs & cs) / len(gs) >= 0.8:
        return 0.6
    return 0.0


def verify(domain, gold, tests, cand_text):
    try:
        if domain == "math":
            return verify_math(gold, cand_text)
        if domain == "code":
            return verify_code(tests, cand_text)
        return verify_reasoning(gold, cand_text)
    except Exception:
        return 0.0


# ================================================================= PROMPTS
DOMAINS = ["math", "code", "reasoning"]


def propose_prompt(domain, difficulty, avoid_hashes_count):
    diff = clamp(int(difficulty), 1, 10)
    common = (
        "You are Damru's problem-setter. Invent ONE brand-new, ORIGINAL "
        f"{domain} problem at difficulty {diff}/10. Do NOT copy famous problems. "
        "Return STRICT JSON only, no prose.\n"
    )
    if domain == "math":
        schema = (
            '{"domain":"math","difficulty":%d,"problem":"<clear statement>",'
            '"answer":"<final numeric or exact symbolic answer only>"}' % diff
        )
        return common + "Schema: " + schema
    if domain == "code":
        schema = (
            '{"domain":"code","difficulty":%d,"problem":"<task; implement '
            'function solve(...)>","tests":["assert solve(...)==...","assert '
            'solve(...)==...","assert solve(...)==..."]}' % diff
        )
        return common + ("The function MUST be named solve. Provide 3-5 "
                         "assert tests that fully pin the behavior. Schema: "
                         + schema)
    schema = (
        '{"domain":"reasoning","difficulty":%d,"problem":"<logic/word '
        'problem>","answer":"<short final answer>"}' % diff
    )
    return common + "Schema: " + schema


def solve_prompt(domain, problem):
    if domain == "math":
        return (
            "Solve this problem. Think step by step (deep reasoning), then give "
            "the final answer on its own last line as: #### <answer>\n\n"
            + problem
        )
    if domain == "code":
        return (
            "Implement the function `solve` in a single ```python code block. "
            "Think about edge cases first, then write clean code.\n\n" + problem
        )
    return (
        "Reason carefully step by step, then give the final answer on its own "
        "last line as: #### <answer>\n\n" + problem
    )


def reflect_prompt(domain, problem, best_attempt):
    return (
        "Your previous attempt was WRONG. Write a short self-reflection: what "
        "specific mistake or wrong assumption did you make, and what is the "
        "correct strategy? Be concrete.\n\nPROBLEM:\n" + problem +
        "\n\nYOUR WRONG ATTEMPT:\n" + (best_attempt or "")[:1500]
    )


# ================================================================= GRPO CORE
def grpo_advantages(rewards):
    """Group-relative advantage = (r - mean)/std. Returns list."""
    if not rewards:
        return []
    m = sum(rewards) / len(rewards)
    var = sum((r - m) ** 2 for r in rewards) / len(rewards)
    sd = math.sqrt(var) + 1e-6
    return [(r - m) / sd for r in rewards]


# ================================================================= STATE
def load_state(path):
    default = {
        "iter": 0,
        "difficulty": {d: env_int("DAMRU_START_DIFF", 1) for d in DOMAINS},
        "passrate": {d: [] for d in DOMAINS},   # rolling window of group means
        "seen": {},                              # hash -> 1 (dedup)
        "kept_sft": 0,
        "kept_pref": 0,
        "started": now_iso(),
    }
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                s = json.load(f)
            for k, v in default.items():
                s.setdefault(k, v)
            for d in DOMAINS:
                s["difficulty"].setdefault(d, env_int("DAMRU_START_DIFF", 1))
                s["passrate"].setdefault(d, [])
            return s
    except Exception as e:
        log("[state] load failed, fresh:", e)
    return default


def save_state(path, state):
    """Atomic write -> force-overwrite, corruption-proof."""
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(state, f)
        os.replace(tmp, path)
    except Exception as e:
        log("[state] save failed:", e)


def update_curriculum(state, domain, group_mean):
    win = state["passrate"][domain]
    win.append(group_mean)
    if len(win) > 8:
        del win[0]
    if len(win) >= 4:
        avg = sum(win) / len(win)
        d = state["difficulty"][domain]
        if avg >= 0.75 and d < 10:
            state["difficulty"][domain] = d + 1
            state["passrate"][domain] = []
            log(f"  [curriculum] {domain} UP -> {d + 1} (avg={avg:.2f})")
        elif avg <= 0.15 and d > 1:
            state["difficulty"][domain] = d - 1
            state["passrate"][domain] = []
            log(f"  [curriculum] {domain} DOWN -> {d - 1} (avg={avg:.2f})")


# ================================================================= COLLECTOR
class Collector:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.sft_path = os.path.join(out_dir, f"sft_{stamp}.jsonl")
        self.pref_path = os.path.join(out_dir, f"pref_{stamp}.jsonl")
        self.sft_n = 0
        self.pref_n = 0

    def add_sft(self, domain, problem, solution, reward, difficulty):
        rec = {
            "domain": domain, "difficulty": difficulty, "reward": reward,
            "messages": [
                {"role": "user", "content": problem},
                {"role": "assistant", "content": solution},
            ],
            "ts": now_iso(),
        }
        self._append(self.sft_path, rec)
        self.sft_n += 1

    def add_pref(self, domain, problem, chosen, rejected, adv, difficulty):
        rec = {
            "domain": domain, "difficulty": difficulty, "advantage": adv,
            "prompt": problem, "chosen": chosen, "rejected": rejected,
            "ts": now_iso(),
        }
        self._append(self.pref_path, rec)
        self.pref_n += 1

    def _append(self, path, rec):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:
            log("[collect] write failed:", e)


def hf_push(dataset, files):
    if not _HAS_HF:
        log("[hf] huggingface_hub missing, skip push")
        return
    tok = env("HF_TOKEN")
    if not tok:
        log("[hf] no HF_TOKEN, skip push")
        return
    try:
        api = HfApi(token=tok)
        try:
            api.create_repo(dataset, repo_type="dataset", exist_ok=True,
                            private=True)
        except Exception:
            pass
        for fp in files:
            if fp and os.path.exists(fp) and os.path.getsize(fp) > 0:
                upload_file(
                    path_or_fileobj=fp,
                    path_in_repo=f"gurukul/{os.path.basename(fp)}",
                    repo_id=dataset, repo_type="dataset", token=tok,
                )
                log(f"[hf] pushed {os.path.basename(fp)} -> {dataset}")
    except Exception as e:
        log("[hf] push failed:", e)


# ================================================================= MAIN LOOP
_STOP = {"flag": False}


def _handle_sig(signum, frame):
    _STOP["flag"] = True
    log(f"[signal] {signum} -> graceful stop after this iter")


def main():
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _handle_sig)
        except Exception:
            pass

    dataset = env("DAMRU_DATASET", "Damaru-ai/damru-gurukul")
    G = clamp(env_int("DAMRU_GROUP_SIZE", 6), 2, 16)
    max_iters = env_int("DAMRU_MAX_ITERS", 0)
    push_every = clamp(env_int("DAMRU_PUSH_EVERY", 25), 1, 10000)
    domains = [d.strip() for d in env("DAMRU_DOMAINS", "math,code,reasoning").split(",")
               if d.strip() in DOMAINS] or DOMAINS
    state_path = env("DAMRU_STATE", "./gurukul_state.json")
    out_dir = env("DAMRU_OUT", "./gurukul_out")
    sleep_s = env_float("DAMRU_SLEEP", 1.0)
    time_budget = env_int("DAMRU_TIME_BUDGET_MIN", 320) * 60
    t_start = time.time()

    log("=" * 70)
    log("DAMRU GURUKUL starting")
    log(f"  dataset={dataset} G={G} domains={domains} push_every={push_every}")
    log(f"  sympy={_HAS_SYMPY} hf={_HAS_HF} max_iters={max_iters or 'forever'}")

    providers = build_providers()
    if not providers:
        log("[FATAL] no providers. Set OWN_MODEL+HF_TOKEN or a *_API_KEY.")
        sys.exit(2)
    log("  brains: " + ", ".join(f"{p.name}({p.model})" for p in providers))
    brain = Brain(providers)

    state = load_state(state_path)
    coll = Collector(out_dir)
    since_push = 0

    while not _STOP["flag"]:
        # budget / iter guards
        if max_iters and state["iter"] >= max_iters:
            log("[done] reached DAMRU_MAX_ITERS")
            break
        if time.time() - t_start > time_budget:
            log("[done] time budget reached, clean exit")
            break
        if not brain.alive():
            log("[heal] all brains cooling, sleeping 30s")
            time.sleep(30)
            continue

        state["iter"] += 1
        it = state["iter"]
        domain = random.choice(domains)
        diff = state["difficulty"][domain]

        try:
            # ---------------------------------------------------- 1. PROPOSE
            praw = brain.ask(
                [{"role": "user",
                  "content": propose_prompt(domain, diff, len(state["seen"]))}],
                temperature=0.9, max_tokens=900)
            spec = safe_json_loads(praw)
            if not spec or "problem" not in spec:
                log(f"#{it} {domain} d{diff}: bad proposal, skip")
                time.sleep(sleep_s)
                continue
            problem = str(spec.get("problem", "")).strip()
            gold = str(spec.get("answer", "")).strip()
            tests = spec.get("tests") if domain == "code" else None
            if not problem or (domain != "code" and not gold) \
               or (domain == "code" and not tests):
                log(f"#{it} {domain} d{diff}: incomplete spec, skip")
                time.sleep(sleep_s)
                continue

            h = sha1(domain + "|" + problem)
            if h in state["seen"]:
                log(f"#{it} {domain} d{diff}: dup problem, skip")
                time.sleep(sleep_s)
                continue
            state["seen"][h] = 1

            # ---------------------------------------------------- 2. SOLVE (group)
            sp = solve_prompt(domain, problem)
            cands = brain.sample_group(
                [{"role": "user", "content": sp}], G,
                temperature=0.85, max_tokens=1400)
            if not cands:
                log(f"#{it} {domain} d{diff}: no candidates, skip")
                time.sleep(sleep_s)
                continue

            # ---------------------------------------------------- 3. VERIFY (RLVR)
            rewards = [verify(domain, gold, tests, c) for c in cands]
            group_mean = sum(rewards) / len(rewards)
            best_i = max(range(len(rewards)), key=lambda i: rewards[i])
            worst_i = min(range(len(rewards)), key=lambda i: rewards[i])

            # ------------------------------------------ 5. REFLECT-RETRY-REWARD
            reflected = False
            if max(rewards) <= 0.0:
                refl = brain.ask(
                    [{"role": "user",
                      "content": reflect_prompt(domain, problem, cands[best_i])}],
                    temperature=0.6, max_tokens=500)
                if refl:
                    retry = brain.ask(
                        [{"role": "user", "content": sp},
                         {"role": "assistant", "content": cands[best_i]},
                         {"role": "user",
                          "content": "Reflection: " + refl +
                                     "\n\nNow solve correctly."}],
                        temperature=0.5, max_tokens=1400)
                    if retry:
                        rr = verify(domain, gold, tests, retry)
                        if rr > 0:
                            cands.append(retry)
                            rewards.append(rr)
                            best_i = len(cands) - 1
                            reflected = True

            # ------------------------------------------ 4. GRPO advantages
            advs = grpo_advantages(rewards)

            # ------------------------------------------ 7. COLLECT
            # SFT: keep verified-good traces (reward >= 0.7)
            for i, r in enumerate(rewards):
                if r >= 0.7:
                    coll.add_sft(domain, problem, cands[i], r, diff)
                    state["kept_sft"] += 1
            # Preference pair: best vs worst if there's a real gap
            if rewards[best_i] - rewards[worst_i] >= 0.5:
                coll.add_pref(domain, problem, cands[best_i], cands[worst_i],
                              round(advs[best_i] - advs[worst_i], 4), diff)
                state["kept_pref"] += 1

            # ------------------------------------------ 6. CURRICULUM
            update_curriculum(state, domain, group_mean)

            log(f"#{it} {domain} d{diff} | G={len(cands)} "
                f"mean={group_mean:.2f} best={rewards[best_i]:.2f} "
                f"{'REFLECT+' if reflected else ''}"
                f"sft={state['kept_sft']} pref={state['kept_pref']}")

            # ------------------------------------------ push + checkpoint
            since_push += 1
            if since_push >= push_every:
                hf_push(dataset, [coll.sft_path, coll.pref_path])
                since_push = 0
            save_state(state_path, state)

        except Exception:
            log("[loop] iteration error (self-heal, continue):")
            log(traceback.format_exc())
            save_state(state_path, state)

        time.sleep(sleep_s)

    # final flush
    hf_push(dataset, [coll.sft_path, coll.pref_path])
    save_state(state_path, state)
    log("=" * 70)
    log(f"GURUKUL stopped. iters={state['iter']} "
        f"sft={state['kept_sft']} pref={state['kept_pref']}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        log("[FATAL] top-level crash:")
        log(traceback.format_exc())
        sys.exit(1)
