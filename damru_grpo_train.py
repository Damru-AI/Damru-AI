#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 DAMRU GRPO TRAINER v3  --  TIME BUDGET FIXED
================================================================================
CHANGES in v3:
  [1] TIME BUDGET 11.5h (41400s) -- Kaggle 12h CellTimeoutError FIXED
      Default: TIME_BUDGET_SEC=39600 (11h) -> 60min safety margin for save+push
  [2] TimeBudgetCallback -- graceful stop before limit, saves + pushes model
  [3] Hourly push thread -- crash safe
  [4] BASE_MODEL default -> Qwen2.5-14B
  [5] Resume guard: no LoRA double-attach

KAGGLE LIMIT: 12h = 43200s
We stop training at 11h (39600s) -> 60min left to save + push GGUF safely.
Set TIME_BUDGET_SEC=41400 for 11.5h if you want to use more time.

ENV / Kaggle secrets:
  HF_TOKEN           required
  DAMRU_DATASET      default 'Damaru-ai/damru-gurukul'
  BASE_MODEL         default 'unsloth/Qwen2.5-14B-Instruct-bnb-4bit'
  RESUME_ADAPTER     optional -- previous LoRA to continue from
  OUT_LORA           default 'Damaru-ai/damru-grpo-lora-14b'
  OUT_GGUF           default 'Damaru-ai/damru-gguf'
  MAX_PROMPTS        default 400
  GRPO_G             generations per prompt, default 6
  MAX_STEPS          default 200
  PUSH_EVERY_H       hourly push interval, default 1.0
  TIME_BUDGET_SEC    default 39600 (11h) -- stop training this many seconds in
================================================================================
"""
import os
import re
import json
import sys
import math
import threading
import time
import traceback


def log(*a):
    print(f"[grpo +{int(time.time() - _START)}s]", *a, flush=True)


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
        log("kaggle secret load failed:", e)
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
            if simplify(
                parse_expr(g.replace("^", "**")) -
                parse_expr(c.replace("^", "**"))
            ) == 0:
                return 1.0
        except Exception:
            pass
    return 0.0


def reward_format(text):
    if not text:
        return 0.0
    return 0.2 if ("####" in text or _BOXED.search(text)) else 0.0


# ---- TIME BUDGET CALLBACK ----------------------------------------------------
class TimeBudgetCallback:
    """
    Stops GRPOTrainer gracefully when TIME_BUDGET_SEC is reached.
    On stop: saves checkpoint locally + pushes LoRA adapter to HuggingFace.
    This ensures we never hit Kaggle's hard 12h CellTimeoutError.

    Timeline example (TIME_BUDGET_SEC=39600 = 11h):
      0h        : training starts
      11h       : TimeBudgetCallback triggers, saves + pushes model
      11h-12h   : GGUF export (if time allows)
      12h       : Kaggle kills the notebook (we are already done!)
    """

    def __init__(self, budget_sec, lock, model_ref, tok_ref, out_lora, token):
        self.deadline = _START + budget_sec
        self.lock = lock
        self.model_ref = model_ref
        self.tok_ref = tok_ref
        self.out_lora = out_lora
        self.token = token
        self.saved = False
        log(f"[TimeBudget] Budget = {budget_sec/3600:.2f}h, "
            f"deadline = {time.strftime('%H:%M:%S', time.localtime(self.deadline))}")

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - _START
        remain  = self.deadline - time.time()
        if remain <= 0 and not self.saved:
            log(f"[TimeBudget] {elapsed/3600:.2f}h elapsed -- "
                f"graceful stop at step {state.global_step}")
            self._save_and_push()
            control.should_training_stop = True
        elif state.global_step % 50 == 0:
            log(f"[TimeBudget] step={state.global_step} "
                f"elapsed={elapsed/3600:.2f}h remain={remain/3600:.2f}h")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self.saved:
            self._save_and_push()

    def _save_and_push(self):
        self.saved = True
        m = self.model_ref[0]
        t = self.tok_ref[0]
        if not m or not self.token:
            log("[TimeBudget] No model/token -- skip push")
            return
        try:
            with self.lock:
                log("[TimeBudget] Saving local checkpoint...")
                m.save_pretrained("damru_grpo_checkpoint")
                t.save_pretrained("damru_grpo_checkpoint")
                log("[TimeBudget] Pushing LoRA to HuggingFace...")
                m.push_to_hub(
                    self.out_lora,
                    token=self.token,
                    commit_message="grpo-timebudget-checkpoint",
                )
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
                    gold   = None
                    if "messages" in r and r["messages"]:
                        prompt = r["messages"][0].get("content")
                        gold   = extract_final(
                            r["messages"][-1].get("content", ""))
                    elif "prompt" in r:
                        prompt = r.get("prompt")
                        gold   = extract_final(r.get("chosen", ""))
                    if prompt and gold:
                        rows.append({"prompt": prompt, "gold": gold})
                        if len(rows) >= max_prompts:
                            return rows
        except Exception as e:
            log("shard load failed:", fn, e)
    return rows


# ---- main --------------------------------------------------------------------
def main():
    token = load_hf_token()
    if not token:
        log("[FATAL] no HF_TOKEN")
        sys.exit(2)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    # ---- TIME BUDGET (key fix) ----
    # Kaggle hard limit = 12h = 43200s
    # Default: stop at 11h = 39600s  ->  60min left for GGUF export
    # Set TIME_BUDGET_SEC=41400 for 11.5h (30min left for export)
    time_budget_sec = int(env("TIME_BUDGET_SEC", "39600"))  # 11h default
    log("=" * 60)
    log(f"DAMRU GRPO TRAINER v3 starting")
    log(f"Time budget : {time_budget_sec/3600:.2f}h ({time_budget_sec}s)")
    log(f"Kaggle limit: 12.00h (43200s)")
    log(f"Save window : {(43200 - time_budget_sec)/60:.0f}min for save+GGUF push")
    log("=" * 60)

    dataset     = env("DAMRU_DATASET",  "Damaru-ai/damru-gurukul")
    base        = env("BASE_MODEL",     "unsloth/Qwen2.5-14B-Instruct-bnb-4bit")
    resume      = env("RESUME_ADAPTER")
    out_lora    = env("OUT_LORA",       "Damaru-ai/damru-grpo-lora-14b")
    out_gguf    = env("OUT_GGUF",       "Damaru-ai/damru-gguf")
    max_prompts = int(env("MAX_PROMPTS", "400"))
    G           = int(env("GRPO_G",     "6"))
    max_steps   = int(env("MAX_STEPS",  "200"))
    max_seq     = int(env("MAX_SEQ",    "2048"))
    push_h      = float(env("PUSH_EVERY_H", "1.0"))

    log(f"base={base} resume={resume or '-'} G={G} steps={max_steps}")

    # ---- Hourly push thread ----
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
                            commit_message=f"grpo-hourly-{s}")
                        _tok[0].push_to_hub(out_lora, token=token)
                        log(f"HourlyPush done -> {out_lora}")
                except Exception as e:
                    log("HourlyPush warn:", str(e)[:120])
            _stop.wait(timeout=interval)

    threading.Thread(target=_pusher, daemon=True).start()
    log("Hourly push thread started")

    # ---- model load (Unsloth 4-bit) ----
    from unsloth import FastLanguageModel
    import torch

    log(f"Loading model: {resume or base}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resume or base,
        max_seq_length=max_seq,
        load_in_4bit=True,
    )

    # KEY FIX: Resume guard -- no LoRA double-attach
    if not resume:
        model = FastLanguageModel.get_peft_model(
            model, r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )
        log("Fresh LoRA attached")
    else:
        log("Resumed model -- LoRA already baked in (get_peft_model SKIP)")

    _model[0] = model
    _tok[0]   = tokenizer
    log(f"VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # ---- data ----
    prompts = load_prompts(dataset, token, max_prompts)
    if len(prompts) < 8:
        log(f"[FATAL] only {len(prompts)} prompts -- run Gurukul data forge first!")
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

    # ---- reward funcs ----
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

    # ---- TIME BUDGET CALLBACK (key fix v3) ----
    time_cb = TimeBudgetCallback(
        budget_sec=time_budget_sec,
        lock=_lock,
        model_ref=_model,
        tok_ref=_tok,
        out_lora=out_lora,
        token=token,
    )

    # ---- GRPO ----
    from trl import GRPOConfig, GRPOTrainer

    cfg = GRPOConfig(
        output_dir="damru_grpo_out",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=G,
        max_prompt_length=1024,
        max_completion_length=1024,
        learning_rate=1e-5,
        logging_steps=5,
        max_steps=max_steps,
        save_steps=100000,
        optim="adamw_8bit",
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[r_correct, r_format],
        args=cfg,
        train_dataset=ds,
        callbacks=[time_cb],   # v3 FIX: time budget callback
    )
    log("GRPO training start...")
    trainer.train()
    log("GRPO training done.")

    # ---- stop hourly push ----
    _stop.set()

    # ---- push adapter (if not already done by TimeBudgetCallback) ----
    if not time_cb.saved:
        try:
            with _lock:
                model.push_to_hub(out_lora, token=token)
                tokenizer.push_to_hub(out_lora, token=token)
                log(f"pushed adapter -> {out_lora}")
        except Exception as e:
            log("adapter push failed:", e)
    else:
        log("Adapter already pushed by TimeBudgetCallback -- skip final push.")

    # ---- push GGUF (best-effort, only if time remains) ----
    elapsed = time.time() - _START
    # GGUF export takes ~5-10min; only attempt if < 41000s elapsed
    if elapsed < 41000:
        try:
            model.push_to_hub_gguf(
                out_gguf, tokenizer,
                quantization_method="q4_k_m", token=token)
            log(f"pushed GGUF -> {out_gguf}")
        except Exception as e:
            log("gguf push skipped:", e)
    else:
        log(f"[SKIP] GGUF export skipped -- "
            f"{elapsed/3600:.2f}h elapsed, not enough time left")

    log(f"[DONE] Total: {elapsed/3600:.2f}h")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        log("[FATAL] crash:")
        log(traceback.format_exc())
        sys.exit(1)
