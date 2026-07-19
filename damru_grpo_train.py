#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
 DAMRU GRPO TRAINER  --  RLVR self-play training on Kaggle GPU (T4/P100)
================================================================================
Consumes the Gurukul dataset (SFT + preference JSONL produced by
damru_gurukul.py) and trains Damru with GROUP RELATIVE POLICY OPTIMIZATION
(GRPO) using verifiable rewards -- the SAME reward logic as the Gurukul, so
the loop is consistent end to end.

This is the ORIGINAL-intelligence step: Damru is NOT copying a teacher, it is
rewarded only for VERIFIABLY correct answers (math=sympy, format).

Pipeline:
  1. Load base (or previous adapter for CONTINUAL learning) with Unsloth 4-bit.
  2. Load prompts from the Gurukul dataset (math/reasoning have gold answers).
  3. GRPO: sample G completions per prompt, reward each, optimize group-relative.
  4. Push new LoRA adapter + merged GGUF back to the Hub -> Gurukul reloads it.

Run on Kaggle (enable GPU + Internet). Add HF_TOKEN as a Kaggle secret.

ENV / Kaggle secrets:
  HF_TOKEN            required (pull dataset + push model)
  DAMRU_DATASET       default 'Damaru-ai/damru-gurukul'
  BASE_MODEL          default 'unsloth/Qwen2.5-7B-Instruct-bnb-4bit'
  RESUME_ADAPTER      optional previous LoRA to continue from (continual RL)
  OUT_LORA            default 'Damaru-ai/damru-gurukul-lora'
  OUT_GGUF            default 'Damaru-ai/damru-gguf'
  MAX_PROMPTS         default 400
  GRPO_G              generations per prompt, default 6
  MAX_STEPS           default 200
================================================================================
"""
import os
import re
import json
import sys
import math
import traceback


def log(*a):
    print("[grpo]", *a, flush=True)


def env(n, d=None):
    v = os.environ.get(n)
    return v if (v is not None and str(v).strip() != "") else d


# ---- Kaggle secret bridge (HF_TOKEN) ---------------------------------------
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


# ---- verifiable reward (mirror of gurukul) ---------------------------------
try:
    from sympy import simplify, nsimplify
    from sympy.parsing.sympy_parser import parse_expr
    _HAS_SYMPY = True
except Exception:
    _HAS_SYMPY = False

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_HASH = re.compile(r"####\s*(.+?)\s*$", re.S)


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
    m = re.search(r"(-?\d+(?:\.\d+)?(?:/\d+)?)\s*$", lines[-1])
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
    """Small shaping reward for producing a #### final answer."""
    if not text:
        return 0.0
    return 0.2 if ("####" in text or _BOXED.search(text)) else 0.0


# ---- dataset ---------------------------------------------------------------
def load_prompts(dataset, token, max_prompts):
    """Pull gurukul/*.jsonl, keep records that have a checkable gold answer."""
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(token=token)
    rows = []
    try:
        files = api.list_repo_files(dataset, repo_type="dataset")
    except Exception as e:
        log("list_repo_files failed:", e)
        files = []
    jsonls = [f for f in files if f.startswith("gurukul/") and f.endswith(".jsonl")]
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
                    # SFT record: reconstruct prompt + gold from messages
                    dom = r.get("domain")
                    if dom not in ("math", "reasoning"):
                        continue  # code needs exec-reward; keep RL to checkable
                    prompt = None
                    gold = None
                    if "messages" in r and r["messages"]:
                        prompt = r["messages"][0].get("content")
                        gold = extract_final(r["messages"][-1].get("content", ""))
                    elif "prompt" in r:
                        prompt = r.get("prompt")
                        gold = extract_final(r.get("chosen", ""))
                    if prompt and gold:
                        rows.append({"prompt": prompt, "gold": gold})
                        if len(rows) >= max_prompts:
                            return rows
        except Exception as e:
            log("shard load failed:", fn, e)
    return rows


# ---- main ------------------------------------------------------------------
def main():
    token = load_hf_token()
    if not token:
        log("[FATAL] no HF_TOKEN (env or Kaggle secret)")
        sys.exit(2)
    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)

    dataset = env("DAMRU_DATASET", "Damaru-ai/damru-gurukul")
    base = env("BASE_MODEL", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    resume = env("RESUME_ADAPTER")
    out_lora = env("OUT_LORA", "Damaru-ai/damru-gurukul-lora")
    out_gguf = env("OUT_GGUF", "Damaru-ai/damru-gguf")
    max_prompts = int(env("MAX_PROMPTS", "400"))
    G = int(env("GRPO_G", "6"))
    max_steps = int(env("MAX_STEPS", "200"))
    max_seq = int(env("MAX_SEQ", "2048"))

    log(f"dataset={dataset} base={base} resume={resume or '-'} G={G} steps={max_steps}")

    # ---- model (Unsloth 4-bit, OOM-proof) ----
    from unsloth import FastLanguageModel
    import torch

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=resume or base,
        max_seq_length=max_seq,
        load_in_4bit=True,
        fast_inference=False,
    )
    if not resume:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
            random_state=3407,
        )

    # ---- data ----
    prompts = load_prompts(dataset, token, max_prompts)
    if len(prompts) < 8:
        log(f"[FATAL] not enough checkable prompts ({len(prompts)}). "
            f"Let the Gurukul forge more data first.")
        sys.exit(3)
    log(f"loaded {len(prompts)} RL prompts")

    from datasets import Dataset

    SYS = ("Think step by step, then give the final answer on its own last "
           "line as: #### <answer>")

    def to_row(r):
        return {
            "prompt": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": r["prompt"]},
            ],
            "gold": r["gold"],
        }

    ds = Dataset.from_list([to_row(r) for r in prompts])

    # ---- reward funcs (TRL GRPO signature) ----
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
    )
    log("GRPO training start...")
    trainer.train()
    log("GRPO training done.")

    # ---- push adapter ----
    try:
        model.push_to_hub(out_lora, token=token)
        tokenizer.push_to_hub(out_lora, token=token)
        log(f"pushed adapter -> {out_lora}")
    except Exception as e:
        log("adapter push failed:", e)

    # ---- push GGUF (best-effort) ----
    try:
        model.push_to_hub_gguf(out_gguf, tokenizer,
                               quantization_method="q4_k_m", token=token)
        log(f"pushed GGUF -> {out_gguf}")
    except Exception as e:
        log("gguf push skipped/failed:", e)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        log("[FATAL] crash:")
        log(traceback.format_exc())
        sys.exit(1)
