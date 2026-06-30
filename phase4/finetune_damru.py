#!/usr/bin/env python3
"""
Damru GENERAL FINE-TUNE  (QLoRA SFT on the prepped, balanced dataset)
====================================================================
Trains the general + exam brain (Nursing/medical/STEM/reasoning) on the CLEAN
`damru-train` split produced by prep_training_data.py. Coder model is trained
separately by finetune_coder.py; this is the broad tutor.

Key correctness pieces (the stuff people forget):
  * ChatML chat template via tokenizer.apply_chat_template
  * COMPLETION-ONLY loss masking (loss on the answer, NOT the prompt)
  * train on `train` split, EVAL on the held-out `val` split each epoch
  * QLoRA 4-bit -> fits a 3B on a single T4/Colab; preserves base knowledge
  * 1-2 epochs (avoid over-fit / catastrophic forgetting)

Designed for Colab / Kaggle T4 (NOT GitHub Actions -- no GPU there).

Install:
  pip install -U "transformers>=4.44" "trl>=0.9" peft accelerate \
      bitsandbytes datasets

Env / config (all overridable):
  HF_TOKEN, BASE_MODEL, TRAIN_REPO, OUT_REPO,
  EPOCHS, LR, MAX_SEQ, BATCH, GRAD_ACCUM, LORA_R, LORA_ALPHA, LORA_DROPOUT,
  RUN_DPO, DPO_REPO
"""
import os

CFG = {
    "base_model": os.environ.get("BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
    "train_repo": os.environ.get("TRAIN_REPO", "Damaru-ai/damru-train"),
    "out_repo": os.environ.get("OUT_REPO", "Damaru-ai/damru-tutor-lora"),
    "epochs": float(os.environ.get("EPOCHS") or "1"),
    "lr": float(os.environ.get("LR") or "2e-4"),
    "max_seq": int(os.environ.get("MAX_SEQ") or "4096"),
    "batch": int(os.environ.get("BATCH") or "2"),
    "grad_accum": int(os.environ.get("GRAD_ACCUM") or "8"),
    "lora_r": int(os.environ.get("LORA_R") or "16"),
    "lora_alpha": int(os.environ.get("LORA_ALPHA") or "32"),
    "lora_dropout": float(os.environ.get("LORA_DROPOUT") or "0.05"),
    "warmup_ratio": 0.03,
    "weight_decay": 0.0,
    "save_steps": int(os.environ.get("SAVE_STEPS") or "500"),
    "eval_steps": int(os.environ.get("EVAL_STEPS") or "500"),
}
HF_TOKEN = os.environ.get("HF_TOKEN", "")
RUN_DPO = (os.environ.get("RUN_DPO", "0") == "1")
DPO_REPO = os.environ.get("DPO_REPO", "Damaru-ai/damru-dpo")


def load_splits(tok):
    from datasets import load_dataset
    train = load_dataset(CFG["train_repo"], data_dir="train", split="train")
    try:
        val = load_dataset(CFG["train_repo"], data_dir="val", split="train")
    except Exception:
        val = None
    import json

    def fmt(ex):
        msgs = ex.get("messages")
        if isinstance(msgs, str):
            msgs = json.loads(msgs)
        if not msgs:
            msgs = [{"role": "user", "content": ex.get("question", "")},
                    {"role": "assistant", "content": ex.get("answer", "")}]
        return {"text": tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False)}

    train = train.map(fmt, remove_columns=[c for c in train.column_names
                                           if c != "text"])
    if val is not None:
        val = val.map(fmt, remove_columns=[c for c in val.column_names
                                           if c != "text"])
    return train, val


def sft():
    import torch
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from trl import DataCollatorForCompletionOnlyLM

    tok = AutoTokenizer.from_pretrained(CFG["base_model"],
                                        trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(
        CFG["base_model"], quantization_config=bnb, device_map="auto",
        trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)

    lora = LoraConfig(
        r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
        lora_dropout=CFG["lora_dropout"], bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    train, val = load_splits(tok)
    print("train rows:", len(train), "| val rows:",
          (len(val) if val is not None else 0))

    # COMPLETION-ONLY: mask everything up to the assistant turn so loss is
    # computed on the answer only (ChatML assistant marker).
    resp_template = "<|im_start|>assistant\n"
    collator = DataCollatorForCompletionOnlyLM(resp_template, tokenizer=tok)

    args = SFTConfig(
        output_dir="damru-tutor-lora",
        num_train_epochs=CFG["epochs"],
        per_device_train_batch_size=CFG["batch"],
        gradient_accumulation_steps=CFG["grad_accum"],
        learning_rate=CFG["lr"],
        warmup_ratio=CFG["warmup_ratio"],
        weight_decay=CFG["weight_decay"],
        lr_scheduler_type="cosine",
        logging_steps=20,
        save_steps=CFG["save_steps"],
        eval_steps=CFG["eval_steps"],
        eval_strategy=("steps" if val is not None else "no"),
        bf16=False, fp16=True,
        max_seq_length=CFG["max_seq"],
        packing=False,                 # packing off so completion-mask works
        gradient_checkpointing=True,
        report_to="none",
        dataset_text_field="text",
        push_to_hub=bool(HF_TOKEN),
        hub_model_id=CFG["out_repo"],
        hub_token=HF_TOKEN or None,
    )
    trainer = SFTTrainer(
        model=model, args=args, train_dataset=train,
        eval_dataset=val, peft_config=lora,
        data_collator=collator, processing_class=tok)
    trainer.train()
    trainer.save_model("damru-tutor-lora")
    if HF_TOKEN:
        trainer.push_to_hub()
    print("SFT DONE -> damru-tutor-lora")
    return tok


def dpo(tok):
    """Optional preference-alignment stage on damru-dpo (chosen/rejected)."""
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig
    from trl import DPOTrainer, DPOConfig
    ds = load_dataset(DPO_REPO, split="train")
    model = AutoModelForCausalLM.from_pretrained(
        CFG["base_model"], torch_dtype=torch.float16, device_map="auto",
        trust_remote_code=True)
    lora = LoraConfig(r=CFG["lora_r"], lora_alpha=CFG["lora_alpha"],
                      lora_dropout=CFG["lora_dropout"], bias="none",
                      task_type="CAUSAL_LM")
    args = DPOConfig(output_dir="damru-tutor-dpo", beta=0.1,
                     per_device_train_batch_size=1,
                     gradient_accumulation_steps=8,
                     learning_rate=5e-6, num_train_epochs=1,
                     logging_steps=20, fp16=True, report_to="none",
                     push_to_hub=bool(HF_TOKEN),
                     hub_model_id=CFG["out_repo"] + "-dpo",
                     hub_token=HF_TOKEN or None)
    trainer = DPOTrainer(model=model, args=args, train_dataset=ds,
                         peft_config=lora, processing_class=tok)
    trainer.train()
    if HF_TOKEN:
        trainer.push_to_hub()
    print("DPO DONE -> damru-tutor-dpo")


if __name__ == "__main__":
    print("CONFIG:", CFG, "| RUN_DPO:", RUN_DPO)
    t = sft()
    if RUN_DPO:
        dpo(t)
