#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 DAMRU GRPO TRAINER v3 RESUME  --  Resume from last LoRA checkpoint
================================================================================
YE FILE RESUME KARNE KE LIYE HAI:
  - GRPO trainer ka last adapter load karta hai
  - Fresh LoRA attach nahi karta (resume guard)
  - 11h time budget -- Kaggle 12h se 60min pehle graceful stop

Kaggle Secrets:
  HF_TOKEN           required
  RESUME_ADAPTER     default 'Damaru-ai/damru-grpo-lora-14b'
  DAMRU_DATASET      default 'Damaru-ai/damru-gurukul'
  OUT_LORA           default 'Damaru-ai/damru-grpo-lora-14b'
  TIME_BUDGET_SEC    default 39600 (11h)
================================================================================
"""
import os
import re
import json
import sys
import threading
import time
import traceback


def log(*a):
    print(f"[grpo-resume +{int(time.time() - _START)}s]", *a, flush=True)


_START = time.time()


def env(n, d=None):
    v = os.environ.get(n)
    return v if (v is not None and str(v).strip() != "") else d


def load_hf_token():
    tok = env("HF_TOKEN")
    if tok:
        return tok
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
        if tok:
            os.environ["HF_TOKEN"] = tok
            return tok
    except Exception as e:
        log("kaggle secret failed:", e)
    return None


# ---- verifiable reward -------------------------------------------------------
try:
    from sympy import simplify
    from sympy.parsing.sympy_parser import parse_expr
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False

_BOXED = re.compile(r'\\boxed\{([^}]*)\}')
_HASH  = re.compile(r'####\s*(.+?)\s*$', re.S)


def extract_final(text):
    if not text:
        return ""
    m = _BOXED.search(text)
    if m:
        return m.group(1).strip()
    m = _HASH.search(text.strip())
    if m:
        return m.group(1).strip().splitlines()[-1].strip()
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    m = re.search(r'(-?\d+(?:\.\d+)?(?:/\d+)?)\s*$', lines[-1])
    return m.group(1) if m else lines[-1]


def _norm(s):
    return str(s).strip().replace(",", "").replace("$", "").replace(" ", "")


def reward_correct(gold, text):
    c = _norm(extract_final(text))
    g = _norm(gold)
    if not c:
        return 0.0
    if c == g:
        return 1.0
    try:
        if abs(float(c) - float(g)) < 1e-6:
            return 1.0
    except Exception:
        pass
    if _HAS_SYMPY:
        try:
            if simplify(parse_expr(g.replace("^", "**")) -
                        parse_expr(c.replace("^", "**"))) == 0:
                return 1.0
        except Exception:
            pass
    return 0.0


def reward_format(text):
    if not text:
        return 0.0
    return 0.2 if ("####" in text or _BOXED.search(text)) else 0.0


def _ensure_deps():
    """Fresh Kaggle/Colab kernels don't always ship unsloth/trl/sympy -- auto-install
    anything missing (Internet must be ON). No-op when the stack is already present.
    sympy is re-imported after install so the verifiable reward stays enabled."""
    import importlib.util
    need = [m for m in ("unsloth", "trl", "peft", "bitsandbytes", "datasets", "sympy")
            if importlib.util.find_spec(m) is None]
    if need:
        import subprocess
        log(">> installing GRPO stack (missing: %s) -- one-time, ~2-4 min" % ",".join(need))
        cmds = [
            [sys.executable, "-m", "pip", "install", "-q", "-U", "unsloth"],
            [sys.executable, "-m", "pip", "install", "-q",
             "trl>=0.9", "peft", "accelerate", "bitsandbytes", "datasets",
             "sentencepiece", "protobuf", "sympy"],
        ]
        if any(subprocess.run(c).returncode != 0 for c in cmds):
            log(">> AUTO-INSTALL failed -- Kaggle: Settings > Internet = ON, then re-run")
            raise RuntimeError("dependency install failed -- enable Kaggle Internet and re-run")
        log(">> GRPO stack installed OK")
    global _HAS_SYMPY, simplify, parse_expr
    if not _HAS_SYMPY:
        try:
            from sympy import simplify as _sm
            from sympy.parsing.sympy_parser import parse_expr as _pe
            simplify, parse_expr, _HAS_SYMPY = _sm, _pe, True
            log(">> sympy verifiable reward enabled")
        except Exception:
            pass


# ---- TIME BUDGET CALLBACK ----------------------------------------------------
class TimeBudgetCallback:
    def __init__(self, budget_sec, lock, model_ref, tok_ref, out_lora, token):
        self.deadline = _START + budget_sec
        self.lock = lock
        self.model_ref = model_ref
        self.tok_ref = tok_ref
        self.out_lora = out_lora
        self.token = token
        self.saved = False
        log(f"[TimeBudget] Budget={budget_sec/3600:.2f}h deadline="
            f"{time.strftime('%H:%M:%S', time.localtime(self.deadline))}")

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - _START
        if time.time() >= self.deadline and not self.saved:
            log(f"[TimeBudget] {elapsed/3600:.2f}h -- stop at step {state.global_step}")
            self._save_push()
            control.should_training_stop = True
        elif state.global_step % 20 == 0:
            log(f"step={state.global_step} "
                f"elapsed={elapsed/3600:.2f}h "
                f"remain={(self.deadline-time.time())/3600:.2f}h")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self.saved:
            self._save_push()

    def _save_push(self):
        self.saved = True
        m = self.model_ref[0]
        t = self.tok_ref[0]
        if not m or not self.token:
            return
        try:
            with self.lock:
                log("[TimeBudget] Saving...")
                m.save_pretrained("damru_grpo_checkpoint")
                t.save_pretrained("damru_grpo_checkpoint")
                m.push_to_hub(self.out_lora, token=self.token,
                               commit_message="grpo-resume-timebudget")
                t.push_to_hub(self.out_lora, token=self.token)
                log(f"[TimeBudget] Pushed -> {self.out_lora}")
        except Exception as e:
            log(f"[TimeBudget] push failed: {e}")


# ---- dataset -----------------------------------------------------------------
def load_prompts(dataset, token, max_prompts):
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    rows = []
    try:
        files = list(api.list_repo_files(dataset, repo_type="dataset"))
    except Exception as e:
        log("list_repo_files failed:", e)
        files = []
    jsonls = [f for f in files
              if f.startswith("gurukul/") and f.endswith(".jsonl")]
    log(f"found {len(jsonls)} jsonl shards")
    for fn in jsonls:
        try:
            p = hf_hub_download(dataset, fn, repo_type="dataset", token=token)
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    dom = r.get("domain")
                    if dom not in ("math", "reasoning"):
                        continue
                    prompt = None
                    gold = None
                    if "messages" in r and r["messages"]:
                        prompt = r["messages"][0].get("content")
                        gold = extract_final(
                            r["messages"][-1].get("content", ""))
                    elif "prompt" in r:
                        prompt = r.get("prompt")
                        gold = extract_final(r.get("chosen", ""))
                    if prompt and gold:
                        rows.append({"prompt": prompt, "gold": gold})
                        if len(rows) >= max_prompts:
                            return rows
        except Exception as e:
            log("shard failed:", fn, e)
    return rows


# ---- main --------------------------------------------------------------------
def main():
    _ensure_deps()
    token = load_hf_token()
    if not token:
        log("[FATAL] no HF_TOKEN")
        sys.exit(2)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    time_budget_sec = int(env("TIME_BUDGET_SEC", "39600"))  # 11h
    log("=" * 60)
    log("DAMRU GRPO RESUME starting")
    log(f"Time budget: {time_budget_sec/3600:.2f}h  |  Safety: "
        f"{(43200-time_budget_sec)/60:.0f}min for save+push")
    log("=" * 60)

    dataset     = env("DAMRU_DATASET",  "Damaru-ai/damru-gurukul")
    resume      = env("RESUME_ADAPTER", "Damaru-ai/damru-grpo-lora-14b")
    out_lora    = env("OUT_LORA",       "Damaru-ai/damru-grpo-lora-14b")
    out_gguf    = env("OUT_GGUF",       "Damaru-ai/damru-gguf")
    max_prompts = int(env("MAX_PROMPTS", "400"))
    G           = int(env("GRPO_G",     "6"))
    max_steps   = int(env("MAX_STEPS",  "200"))
    max_seq     = int(env("MAX_SEQ",    "2048"))
    push_h      = float(env("PUSH_EVERY_H", "1.0"))

    log(f"resume={resume} G={G} steps={max_steps}")

    # Hourly push thread
    _stop  = threading.Event()
    _lock  = threading.Lock()
    _model = [None]
    _tok   = [None]

    def _pusher():
        interval = int(push_h * 3600)
        time.sleep(interval)
        while not _stop.is_set():
            if _model[0] and token:
                try:
                    with _lock:
                        s = time.strftime("%Y%m%d-%H%M")
                        _model[0].push_to_hub(
                            out_lora, token=token,
                            commit_message=f"grpo-resume-hourly-{s}")
                        _tok[0].push_to_hub(out_lora, token=token)
                        log(f"HourlyPush -> {out_lora}")
                except Exception as e:
                    log("HourlyPush warn:", str(e)[:120])
            _stop.wait(timeout=interval)

    threading.Thread(target=_pusher, daemon=True).start()

    # Load resumed model
    from unsloth import FastLanguageModel
    import torch

    log(f"Loading RESUMED model: {resume}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resume,
        max_seq_length=max_seq,
        load_in_4bit=True,
        token=token,
    )
    log("Resumed LoRA loaded -- get_peft_model SKIPPED (resume guard)")
    log(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    _model[0] = model
    _tok[0]   = tokenizer

    # Load data
    prompts = load_prompts(dataset, token, max_prompts)
    if len(prompts) < 8:
        log(f"[FATAL] only {len(prompts)} prompts -- run Gurukul first!")
        sys.exit(3)
    log(f"Loaded {len(prompts)} RL prompts")

    from datasets import Dataset

    SYS = ("Think step by step, then give the final answer on its own last "
           "line as: #### <answer>")

    def to_row(r):
        return {
            "prompt": [
                {"role": "system",  "content": SYS},
                {"role": "user",    "content": r["prompt"]},
            ],
            "gold": r["gold"],
        }

    ds = Dataset.from_list([to_row(r) for r in prompts])

    def r_correct(prompts=None, completions=None, gold=None, **kw):
        outs = []
        for comp, g in zip(completions, gold):
            text = comp[-1]["content"] if isinstance(comp, list) else str(comp)
            outs.append(reward_correct(g, text))
        return outs

    def r_format(prompts=None, completions=None, **kw):
        outs = []
        for comp in completions:
            text = comp[-1]["content"] if isinstance(comp, list) else str(comp)
            outs.append(reward_format(text))
        return outs

    time_cb = TimeBudgetCallback(
        budget_sec=time_budget_sec,
        lock=_lock, model_ref=_model, tok_ref=_tok,
        out_lora=out_lora, token=token,
    )

    from trl import GRPOConfig, GRPOTrainer

    cfg = GRPOConfig(
        output_dir="damru_grpo_resume_out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=G,
        max_prompt_length=1024,
        max_completion_length=1024,
        learning_rate=5e-6,     # lower LR for resume
        logging_steps=5,
        max_steps=max_steps,
        save_steps=100000,
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer,
        reward_funcs=[r_correct, r_format],
        args=cfg, train_dataset=ds,
        callbacks=[time_cb],
    )
    log("GRPO resume training start...")
    trainer.train()
    log("GRPO done.")
    _stop.set()

    if not time_cb.saved:
        try:
            with _lock:
                model.push_to_hub(out_lora, token=token)
                tokenizer.push_to_hub(out_lora, token=token)
                log(f"Final push -> {out_lora}")
        except Exception as e:
            log("final push failed:", e)

    elapsed = time.time() - _START
    if elapsed < 41000:
        try:
            model.push_to_hub_gguf(out_gguf, tokenizer,
                                    quantization_method="q4_k_m", token=token)
            log(f"GGUF -> {out_gguf}")
        except Exception as e:
            log("GGUF skip:", e)
    log(f"[DONE] Total: {elapsed/3600:.2f}h")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        log("[FATAL]")
        log(traceback.format_exc())
        sys.exit(1)
