#!/usr/bin/env python3
"""
DAMRU 10X TRAINER  --  Unsloth + QLoRA (Kaggle T4 / free tier)
============================================================
WHY your 8-hour run died: you did a FULL fine-tune of a 7B model on a 14.5GB
T4. At the loss step the full-vocab logits blew past VRAM -> CUDA OOM. That is
not a small-GPU problem, it is a WRONG-METHOD problem.

THE 10X FIX (verified best free-tier method, 2026):
  * QLoRA 4-bit         -> ~70% less VRAM (7B fits in ~6-8GB)
  * Unsloth kernels     -> 2-5x faster, FUSED cross-entropy (kills the exact
                           OOM line you hit)
  * Unsloth grad ckpt   -> another 30% VRAM off, longer context
  * 8-bit optimizer     -> optimizer state halved

HOW TO RUN (mobile friendly):
  1. Kaggle -> New Notebook
  2. Settings: Accelerator = GPU T4 x2  |  Internet = ON
  3. Add-ons -> Secrets -> add  HF_TOKEN  (your Hugging Face write token)
  4. Paste this WHOLE file into ONE cell -> Run.

Env overrides (optional, via os.environ before run):
  DAMRU_BASE, DAMRU_MAXSEQ, DAMRU_PUSH_REPO, DAMRU_DATASET, DAMRU_EPOCHS
"""
import os, sys, subprocess


def sh(*a):
    print(">>", " ".join(a))
    subprocess.run(list(a), check=False)


# ---------- 0. Install (Kaggle already ships torch/CUDA) ----------
sh(sys.executable, "-m", "pip", "install", "-q", "--no-deps", "unsloth", "unsloth_zoo")
sh(sys.executable, "-m", "pip", "install", "-q",
   "bitsandbytes", "accelerate", "peft", "trl", "datasets",
   "sentencepiece", "protobuf", "hf_transfer", "xformers")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ---------- 1. Config ----------
BASE_MODEL   = os.environ.get("DAMRU_BASE", "unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
MAX_SEQ      = int(os.environ.get("DAMRU_MAXSEQ", "2048"))
EPOCHS       = float(os.environ.get("DAMRU_EPOCHS", "1"))
OUT_DIR      = "/kaggle/working/damru-lora"
PUSH_REPO    = os.environ.get("DAMRU_PUSH_REPO", "Damaru-ai/damru-tutor-lora")
DATASET_REPO = os.environ.get("DAMRU_DATASET", "Damaru-ai/damru-knowledge")

# HF token from Kaggle Secrets or env
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as e:
        print("[warn] no HF_TOKEN secret found (training still works, push will skip):", e)
os.environ["HF_TOKEN"] = HF_TOKEN or ""

# ---------- 2. Load model in 4-bit (the OOM killer) ----------
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    max_seq_length = MAX_SEQ,
    dtype          = None,      # auto bf16/fp16
    load_in_4bit   = True,      # <-- QLoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",   # 30% less VRAM, long ctx
    random_state = 3407,
    use_rslora = False,
)

# ---------- 3. Data (accepts many schemas) ----------
from datasets import load_dataset, Dataset


def to_messages(ex):
    if ex.get("messages"):
        return {"messages": ex["messages"]}
    pairs = (("instruction", "output"), ("prompt", "response"),
             ("question", "answer"), ("input", "output"))
    for uk, ak in pairs:
        if ex.get(uk):
            msgs = []
            if ex.get("system"):
                msgs.append({"role": "system", "content": str(ex["system"])})
            msgs.append({"role": "user", "content": str(ex[uk])})
            if ex.get(ak):
                msgs.append({"role": "assistant", "content": str(ex[ak])})
            return {"messages": msgs}
    return {"messages": [{"role": "user", "content": str(ex.get("text", ""))}]}


try:
    raw = load_dataset(DATASET_REPO, split="train", token=HF_TOKEN or None)
    print("Loaded dataset", DATASET_REPO, "rows:", len(raw))
except Exception as e:
    print("[warn] dataset load failed, using tiny built-in sample:", e)
    raw = Dataset.from_list([
        {"instruction": "Damru kaun hai?",
         "output": "Main Damru hoon -- ek Bhartiya AI, tere dwara banaya gaya."},
    ] * 100)

ds = raw.map(to_messages, remove_columns=list(raw.column_names))


def fmt(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}


ds = ds.map(fmt)

# ---------- 4. Train ----------
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = ds,
    args = SFTConfig(
        dataset_text_field = "text",
        max_seq_length = MAX_SEQ,
        packing = True,                       # more tokens/step -> faster
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,      # effective batch = 8
        warmup_steps = 10,
        num_train_epochs = EPOCHS,
        learning_rate = 2e-4,
        logging_steps = 5,
        optim = "adamw_8bit",                  # 8-bit optimizer = less VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = OUT_DIR,
        save_strategy = "steps",
        save_steps = 200,
        report_to = "none",
    ),
)

g = torch.cuda.get_device_properties(0)
print(f"GPU: {g.name}  {g.total_memory/1e9:.1f} GB")
trainer.train()

# ---------- 5. Save + push adapter ----------
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print("Saved LoRA ->", OUT_DIR)

if HF_TOKEN:
    try:
        model.push_to_hub(PUSH_REPO, token=HF_TOKEN)
        tokenizer.push_to_hub(PUSH_REPO, token=HF_TOKEN)
        print("Pushed adapter ->", PUSH_REPO)
    except Exception as e:
        print("[warn] push failed:", e)

# ---------- 6. (optional) GGUF export for your HF Space / llama.cpp ----------
try:
    model.save_pretrained_gguf(OUT_DIR + "-gguf", tokenizer,
                               quantization_method="q4_k_m")
    print("GGUF saved -> upload to Damaru-ai/damru-gguf, set GGUF_REPO/GGUF_FILE")
except Exception as e:
    print("[warn] GGUF export skipped:", e)

print("\n==== DAMRU TRAINING DONE ====")
