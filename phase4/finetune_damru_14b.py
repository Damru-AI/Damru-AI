#!/usr/bin/env python3
"""
Damru 14B BRAIN -- Unsloth QLoRA SFT -> GGUF -> HF   (phase 4, the real brain)
=============================================================================
Upgrades Damru from the 3B tutor (finetune_damru.py) to a 14B PRIMARY brain.

Why Unsloth: a 14B QLoRA fits in ~13-15 GB, so it trains on ONE Kaggle T4 (16 GB)
or Colab. Vanilla transformers+bnb (finetune_damru.py) cannot fit 14B on a T4 --
that script stays for the 3B path; THIS one is the 14B path.

Reuses the CLEAN, decontaminated, balanced `Damaru-ai/damru-train` split built by
prep_training_data.py (same ChatML `messages`), COMPLETION-ONLY loss (learn the
answer, not the prompt), then exports GGUF (q4_k_m + q5_k_m) and pushes so the HF
Space can serve it with OWN_MODEL_PRIMARY=1.

Kaggle: GPU = T4 (or T4x2 -- Unsloth uses one), Internet ON, add HF_TOKEN secret.
Install:
  pip install -U "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install -U "trl>=0.9" "transformers>=4.44" peft accelerate bitsandbytes datasets

Env (all overridable):
  HF_TOKEN, BASE_MODEL, TRAIN_REPO, LORA_REPO, GGUF_REPO,
  EPOCHS, LR, MAX_SEQ, BATCH, GRAD_ACCUM, LORA_R, LORA_ALPHA, LORA_DROPOUT,
  MAX_TRAIN_ROWS (0=all), QUANTS (csv), MERGE_16BIT (0/1), SKIP_TRAIN (0/1 -> export only)

Built by Shiva AI for Damru.
"""
import os

CFG = {
    "base_model": os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-14B-Instruct-bnb-4bit"),
    "train_repo": os.environ.get("TRAIN_REPO", "Damaru-ai/damru-train"),
    "lora_repo":  os.environ.get("LORA_REPO",  "Damaru-ai/damru-14b-lora"),
    "gguf_repo":  os.environ.get("GGUF_REPO",  "Damaru-ai/damru-14b-gguf"),
    "epochs":     float(os.environ.get("EPOCHS") or "1"),
    "lr":         float(os.environ.get("LR") or "2e-4"),
    "max_seq":    int(os.environ.get("MAX_SEQ") or "2048"),
    "batch":      int(os.environ.get("BATCH") or "2"),
    "grad_accum": int(os.environ.get("GRAD_ACCUM") or "8"),
    "lora_r":     int(os.environ.get("LORA_R") or "16"),
    "lora_alpha": int(os.environ.get("LORA_ALPHA") or "32"),
    "lora_dropout": float(os.environ.get("LORA_DROPOUT") or "0.0"),
    "max_train_rows": int(os.environ.get("MAX_TRAIN_ROWS") or "0"),
    "quants":     [q.strip() for q in (os.environ.get("QUANTS") or "q4_k_m,q5_k_m").split(",") if q.strip()],
    "merge_16bit": os.environ.get("MERGE_16BIT", "0") == "1",
}
HF_TOKEN = os.environ.get("HF_TOKEN", "")
SKIP_TRAIN = os.environ.get("SKIP_TRAIN", "0") == "1"
SEED = int(os.environ.get("SEED") or "3407")


def _ensure_deps():
    """Fresh Kaggle/Colab kernels don't ship unsloth/trl -- auto-install if missing.
    Internet must be ON. No-op when the stack is already present."""
    import importlib.util
    if all(importlib.util.find_spec(m) is not None
           for m in ("unsloth", "trl", "peft", "bitsandbytes", "datasets")):
        return
    import subprocess, sys
    print(">> installing training stack (unsloth/trl/peft/...) -- one-time, ~2-4 min", flush=True)
    cmds = [
        [sys.executable, "-m", "pip", "install", "-q", "-U", "unsloth"],
        [sys.executable, "-m", "pip", "install", "-q",
         "trl>=0.9", "peft", "accelerate", "bitsandbytes", "datasets",
         "sentencepiece", "protobuf"],
    ]
    if any(subprocess.run(c).returncode != 0 for c in cmds):
        print(">> AUTO-INSTALL failed. Kaggle: Settings > Internet = ON, phir dobara run karo;", flush=True)
        print(">>   ya pehle: pip install -U unsloth 'trl>=0.9' peft accelerate bitsandbytes datasets", flush=True)
        raise RuntimeError("dependency install failed -- enable Kaggle Internet and re-run")
    print(">> training stack installed OK", flush=True)


def _auto_vram():
    """Pick safe seq/batch for the detected GPU (T4 16GB -> 2048/2, smaller -> 1024/1)."""
    try:
        import torch
        if not torch.cuda.is_available():
            print("no CUDA -- running config only", flush=True)
            return
        gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print("GPU: %s  %.1f GB" % (torch.cuda.get_device_name(0), gb), flush=True)
        if gb < 15 and "MAX_SEQ" not in os.environ:
            CFG["max_seq"] = 1024
        if gb < 15 and "BATCH" not in os.environ:
            CFG["batch"] = 1
    except Exception as e:
        print("vram check skipped:", e, flush=True)


def load_model():
    from unsloth import FastLanguageModel
    model, tok = FastLanguageModel.from_pretrained(
        model_name=CFG["base_model"], max_seq_length=CFG["max_seq"],
        dtype=None, load_in_4bit=True)
    return model, tok


def add_lora(model):
    from unsloth import FastLanguageModel
    return FastLanguageModel.get_peft_model(
        model, r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"], bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=SEED)


def load_splits(tok):
    """Reuse the CLEAN damru-train split; render ChatML via the tokenizer template."""
    import json
    from datasets import load_dataset
    train = load_dataset(CFG["train_repo"], data_dir="train", split="train")
    try:
        val = load_dataset(CFG["train_repo"], data_dir="val", split="train")
    except Exception:
        val = None
    if CFG["max_train_rows"]:
        train = train.select(range(min(CFG["max_train_rows"], len(train))))

    def fmt(ex):
        msgs = ex.get("messages")
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        if not msgs:
            msgs = [{"role": "user", "content": ex.get("question", "")},
                    {"role": "assistant", "content": ex.get("answer", "")}]
        return {"text": tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)}

    train = train.map(fmt, remove_columns=[c for c in train.column_names if c != "text"])
    if val is not None:
        val = val.map(fmt, remove_columns=[c for c in val.column_names if c != "text"])
    return train, val


def train():
    from trl import SFTTrainer, SFTConfig
    from unsloth import is_bfloat16_supported
    from unsloth.chat_templates import train_on_responses_only
    import inspect

    model, tok = load_model()
    model = add_lora(model)
    train_ds, val_ds = load_splits(tok)
    print("train rows:", len(train_ds), "| val rows:",
          (len(val_ds) if val_ds is not None else 0), flush=True)

    # --- version-robust SFTConfig -------------------------------------------
    # TRL/Transformers APIs drift a lot. Newer TRL (Transformers 5.x) renamed
    # SFTConfig's `max_seq_length` -> `max_length` and dropped a few old kwargs.
    # Build every kwarg we want, then keep ONLY the ones THIS installed SFTConfig
    # actually accepts, so a future version bump can never crash the run again.
    _sft_ok = set(inspect.signature(SFTConfig.__init__).parameters)
    _sft_kwargs = dict(
        output_dir="damru-14b-lora",
        num_train_epochs=CFG["epochs"],
        per_device_train_batch_size=CFG["batch"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["lr"], warmup_ratio=0.03, weight_decay=0.0,
        lr_scheduler_type="cosine", logging_steps=20,
        optim="adamw_8bit", seed=SEED,
        bf16=is_bfloat16_supported(), fp16=not is_bfloat16_supported(),
        packing=False,
        dataset_text_field="text",
        eval_strategy=("steps" if val_ds is not None else "no"),
        eval_steps=int(os.environ.get("EVAL_STEPS") or "500"),
        save_steps=int(os.environ.get("SAVE_STEPS") or "500"),
        report_to="none",
    )
    # sequence-length kwarg moved: max_seq_length (old TRL) -> max_length (new TRL)
    if "max_seq_length" in _sft_ok:
        _sft_kwargs["max_seq_length"] = CFG["max_seq"]
    elif "max_length" in _sft_ok:
        _sft_kwargs["max_length"] = CFG["max_seq"]
    # older transformers spelled it evaluation_strategy instead of eval_strategy
    if "eval_strategy" not in _sft_ok and "evaluation_strategy" in _sft_ok:
        _sft_kwargs["evaluation_strategy"] = _sft_kwargs.pop("eval_strategy")
    # newer TRL SFTConfig ships a placeholder eos_token ("<EOS_TOKEN>") that is NOT a real
    # Qwen token; SFTTrainer validates it vs the vocab and crashes. Pin eos/pad to the
    # tokenizer's ACTUAL specials (Qwen2.5 -> eos <|im_end|>) when the kwargs are accepted.
    if "eos_token" in _sft_ok and getattr(tok, "eos_token", None):
        _sft_kwargs["eos_token"] = tok.eos_token
    if "pad_token" in _sft_ok and (getattr(tok, "pad_token", None) or getattr(tok, "eos_token", None)):
        _sft_kwargs["pad_token"] = tok.pad_token or tok.eos_token
    # finally drop anything this installed version does not accept (never crash on a kwarg)
    _sft_kwargs = {k: v for k, v in _sft_kwargs.items() if k in _sft_ok}
    args = SFTConfig(**_sft_kwargs)

    # SFTTrainer renamed `tokenizer` -> `processing_class` in newer TRL; use what exists.
    _tr_ok = set(inspect.signature(SFTTrainer.__init__).parameters)
    if "processing_class" in _tr_ok:
        _tok_kw = "processing_class"
    elif "tokenizer" in _tr_ok:
        _tok_kw = "tokenizer"
    else:
        _tok_kw = "processing_class"
    trainer = SFTTrainer(model=model, train_dataset=train_ds,
                         eval_dataset=val_ds, args=args, **{_tok_kw: tok})
    # COMPLETION-ONLY: mask the user turn, learn only the assistant answer (Qwen ChatML).
    trainer = train_on_responses_only(
        trainer, instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n")
    stats = trainer.train()
    print("train done:", getattr(stats, "metrics", stats), flush=True)
    model.save_pretrained("damru-14b-lora")
    tok.save_pretrained("damru-14b-lora")
    if HF_TOKEN:
        try:
            model.push_to_hub(CFG["lora_repo"], token=HF_TOKEN)
            tok.push_to_hub(CFG["lora_repo"], token=HF_TOKEN)
            print("pushed LoRA ->", CFG["lora_repo"], flush=True)
        except Exception as e:
            print("lora push failed:", str(e)[:120], flush=True)
    return model, tok


def load_trained_for_export():
    """SKIP_TRAIN path: load an already-trained adapter repo/dir for GGUF export."""
    from unsloth import FastLanguageModel
    src = os.environ.get("ADAPTER", CFG["lora_repo"])
    print("loading trained adapter:", src, flush=True)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=src, max_seq_length=CFG["max_seq"], dtype=None, load_in_4bit=True)
    return model, tok


def export_gguf(model, tok):
    """Merge + convert to GGUF (q4_k_m/q5_k_m) and push so the Space can serve it."""
    quants = CFG["quants"] or ["q4_k_m"]
    if HF_TOKEN:
        try:
            model.push_to_hub_gguf(CFG["gguf_repo"], tok,
                                   quantization_method=quants, token=HF_TOKEN)
            print("pushed GGUF ->", CFG["gguf_repo"], quants, flush=True)
            return
        except Exception as e:
            print("gguf push failed, saving locally:", str(e)[:120], flush=True)
    model.save_pretrained_gguf("damru-14b-gguf", tok, quantization_method=quants[0])
    print("saved GGUF locally -> damru-14b-gguf/ (set HF_TOKEN to push)", flush=True)
    if CFG["merge_16bit"] and HF_TOKEN:
        try:
            model.push_to_hub_merged(CFG["lora_repo"] + "-merged", tok,
                                     save_method="merged_16bit", token=HF_TOKEN)
            print("pushed merged 16bit", flush=True)
        except Exception as e:
            print("merged push failed:", str(e)[:120], flush=True)


def main():
    print("CONFIG:", CFG, "| SKIP_TRAIN:", SKIP_TRAIN, flush=True)
    _ensure_deps()
    _auto_vram()
    if SKIP_TRAIN:
        model, tok = load_trained_for_export()
    else:
        model, tok = train()
    export_gguf(model, tok)
    print("DONE. 14B brain -> GGUF:", CFG["gguf_repo"], flush=True)
    print("Next: set HF Space env OWN_MODEL_PRIMARY=1 and point the GGUF loader at",
          CFG["gguf_repo"], flush=True)


if __name__ == "__main__":
    main()
