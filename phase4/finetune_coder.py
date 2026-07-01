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
secrets and run. Produces a LoRA adapter you can merge or load at inference.

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
    SFTTrainer(
        model=model, tokenizer=tok, train_dataset=sft,
        args=SFTConfig(
            per_device_train_batch_size=2, gradient_accumulation_steps=8,
            warmup_steps=20, num_train_epochs=EPOCHS, learning_rate=2e-4,
            logging_steps=20, optim="adamw_8bit", weight_decay=0.01,
            lr_scheduler_type="cosine", seed=3407, output_dir="out_sft",
            dataset_text_field="text", max_seq_length=MAXLEN),
    ).train()
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
            DPOTrainer(
                model=model, ref_model=None, tokenizer=tok, train_dataset=dpo,
                args=DPOConfig(
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=8, warmup_steps=10,
                    num_train_epochs=1, learning_rate=5e-6, beta=0.1,
                    logging_steps=10, optim="adamw_8bit", seed=3407,
                    output_dir="out_dpo", max_length=MAXLEN,
                    max_prompt_length=MAXLEN // 2),
            ).train()
            print("DPO done.", flush=True)
        except Exception as e:
            print("DPO skipped:", str(e)[:200], flush=True)

    model.push_to_hub(OUT_REPO, token=HF_TOKEN)
    tok.push_to_hub(OUT_REPO, token=HF_TOKEN)
    print("Pushed adapter ->", OUT_REPO, flush=True)


if __name__ == "__main__":
    main()
