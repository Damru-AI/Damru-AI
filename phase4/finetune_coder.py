#!/usr/bin/env python3
"""
Damru Specialist Coder Fine-tune  (Jugad B + gap #5)
====================================================
Trains a CODE-SPECIALIST Damru on top of a code-pretrained base
(Qwen2.5-Coder), which is far stronger at programming than a general 3B.

Two stages
----------
1. SFT (QLoRA): learn from EXECUTION-VERIFIED coding rows + debug-fix rows
   produced by phase7/code_lab.py  (intents: verified_coding, debug_fix,
   coding, coding_reasoning, competitive_coding).
2. DPO (optional): preference-optimise on {prompt, chosen, rejected} triples
   from DPO_REPO so the model prefers code that actually passed tests.

Runs on a free Colab/Kaggle T4 (Unsloth 4-bit). Mobile-friendly: just set the
secrets and run. Deps auto-install on first run (Internet ON). Produces a LoRA
adapter you can merge or load at inference.

Env / config
------------
HF_TOKEN     (required to read private data + push adapter)
BASE_MODEL   default unsloth/Qwen2.5-Coder-3B-Instruct
DATA_REPO    verified coding knowledge   (default Damaru-ai/damru-knowledge)
DPO_REPO     preference triples          (default Damaru-ai/damru-dpo)
OUT_REPO     where to push the adapter   (default Damaru-ai/damru-coder-lora)
MAX_ROWS     cap SFT rows                (default 200000)
DO_DPO       "1" to run the DPO stage      (default 1)
EPOCHS       SFT epochs                  (default 1)
MAXLEN       sequence length             (default 2048)
"""
import os

HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_MODEL = os.environ.get("BASE_MODEL", "unsloth/Qwen2.5-Coder-3B-Instruct")
DATA_REPO = os.environ.get("DATA_REPO", "Damaru-ai/damru-knowledge")
DPO_REPO = os.environ.get("DPO_REPO", "Damaru-ai/damru-dpo")
OUT_REPO = os.environ.get("OUT_REPO", "Damaru-ai/damru-coder-lora")
MAX_ROWS = int(os.environ.get("MAX_ROWS", "200000"))
DO_DPO = (os.environ.get("DO_DPO") or "1") == "1"
EPOCHS = float(os.environ.get("EPOCHS", "1"))
MAXLEN = int(os.environ.get("MAXLEN", "2048"))

CODE_INTENTS = {"verified_coding", "debug_fix", "coding", "coding_reasoning",
                "competitive_coding", "tool_use", "agent_planning"}


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


def _load_sft():
    from datasets import load_dataset
    ds = load_dataset(DATA_REPO, split="train", streaming=True)
    rows = []
    for ex in ds:
        if (ex.get("intent") or "") not in CODE_INTENTS:
            continue
        q, a = (ex.get("question") or "").strip(), (ex.get("answer") or "").strip()
        if len(q) < 8 or len(a) < 10:
            continue
        rows.append({"q": q, "a": a})
        if len(rows) >= MAX_ROWS:
            break
    print("SFT rows:", len(rows), flush=True)
    return rows


def main():
    assert HF_TOKEN, "HF_TOKEN required"
    _ensure_deps()
    import inspect
    from unsloth import FastLanguageModel
    import torch
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    model, tok = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=MAXLEN, load_in_4bit=True,
        dtype=None)
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth", random_state=3407)

    def fmt(ex):
        msgs = [{"role": "user", "content": ex["q"]},
                {"role": "assistant", "content": ex["a"]}]
        return {"text": tok.apply_chat_template(msgs, tokenize=False)}

    sft = Dataset.from_list(_load_sft()).map(fmt)

    # --- version-robust SFTConfig (max_seq_length -> max_length on new TRL) --
    _sft_ok = set(inspect.signature(SFTConfig.__init__).parameters)
    _sft_kwargs = dict(
        per_device_train_batch_size=2, gradient_accumulation_steps=8,
        warmup_steps=20, num_train_epochs=EPOCHS, learning_rate=2e-4,
        logging_steps=20, optim="adamw_8bit", weight_decay=0.01,
        lr_scheduler_type="cosine", seed=3407, output_dir="out_sft",
        dataset_text_field="text")
    if "max_seq_length" in _sft_ok:
        _sft_kwargs["max_seq_length"] = MAXLEN
    elif "max_length" in _sft_ok:
        _sft_kwargs["max_length"] = MAXLEN
    # newer TRL SFTConfig ships a placeholder eos_token ("<EOS_TOKEN>") that is NOT a real
    # Qwen token; SFTTrainer validates it vs the vocab and crashes. Pin eos/pad to the
    # tokenizer's ACTUAL specials (Qwen2.5 -> eos <|im_end|>) when the kwargs are accepted.
    if "eos_token" in _sft_ok and getattr(tok, "eos_token", None):
        _sft_kwargs["eos_token"] = tok.eos_token
    if "pad_token" in _sft_ok and (getattr(tok, "pad_token", None) or getattr(tok, "eos_token", None)):
        _sft_kwargs["pad_token"] = tok.pad_token or tok.eos_token
    _sft_kwargs = {k: v for k, v in _sft_kwargs.items() if k in _sft_ok}
    _sft_args = SFTConfig(**_sft_kwargs)

    # --- version-robust SFTTrainer (tokenizer -> processing_class on new TRL) --
    _tr_ok = set(inspect.signature(SFTTrainer.__init__).parameters)
    _tr_kwargs = dict(model=model, train_dataset=sft, args=_sft_args)
    if "processing_class" in _tr_ok:
        _tr_kwargs["processing_class"] = tok
    elif "tokenizer" in _tr_ok:
        _tr_kwargs["tokenizer"] = tok
    _tr_kwargs = {k: v for k, v in _tr_kwargs.items() if k in _tr_ok}
    SFTTrainer(**_tr_kwargs).train()
    print("SFT done.", flush=True)

    if DO_DPO:
        try:
            from trl import DPOTrainer, DPOConfig
            from datasets import load_dataset
            dpo_raw = load_dataset(DPO_REPO, split="train")
            dpo = dpo_raw.map(lambda e: {
                "prompt": e["prompt"], "chosen": e["chosen"],
                "rejected": e["rejected"]})
            FastLanguageModel.for_training(model)
            _dcfg_ok = set(inspect.signature(DPOConfig.__init__).parameters)
            _dcfg_kwargs = dict(
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8, warmup_steps=10,
                num_train_epochs=1, learning_rate=5e-6, beta=0.1,
                logging_steps=10, optim="adamw_8bit", seed=3407,
                output_dir="out_dpo", max_length=MAXLEN,
                max_prompt_length=MAXLEN // 2)
            _dcfg_kwargs = {k: v for k, v in _dcfg_kwargs.items() if k in _dcfg_ok}
            _dpo_args = DPOConfig(**_dcfg_kwargs)
            _dtr_ok = set(inspect.signature(DPOTrainer.__init__).parameters)
            _dtr_kwargs = dict(model=model, ref_model=None, train_dataset=dpo,
                               args=_dpo_args)
            if "processing_class" in _dtr_ok:
                _dtr_kwargs["processing_class"] = tok
            elif "tokenizer" in _dtr_ok:
                _dtr_kwargs["tokenizer"] = tok
            _dtr_kwargs = {k: v for k, v in _dtr_kwargs.items() if k in _dtr_ok}
            DPOTrainer(**_dtr_kwargs).train()
            print("DPO done.", flush=True)
        except Exception as e:
            print("DPO skipped:", str(e)[:200], flush=True)

    model.push_to_hub(OUT_REPO, token=HF_TOKEN)
    tok.push_to_hub(OUT_REPO, token=HF_TOKEN)
    print("Pushed adapter ->", OUT_REPO, flush=True)


if __name__ == "__main__":
    main()
