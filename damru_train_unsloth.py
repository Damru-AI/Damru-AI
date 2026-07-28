#!/usr/bin/env python3
"""
DAMRU BRAIN FORGE v4  --  Qwen2.5-14B QLoRA (Kaggle T4 x2)
============================================================
SHIFT: 7B -> 14B (Qwen2.5-14B-Instruct-bnb-4bit)
FIX:   group_by_length removed (deprecated in new trl/transformers)
FIX:   SFTConfig used instead of TrainingArguments (correct API)
FIX:   Smart dataset sampling (11M rows -> sample wisely)
FIX:   T4 x2 VRAM optimized settings for 14B model

HOW TO RUN (Kaggle):
  1. Kaggle -> New Notebook
  2. Settings: Accelerator = GPU T4 x2 | Internet = ON
  3. Add-ons -> Secrets -> add HF_TOKEN
  4. Paste this whole file into ONE cell -> Run All

Env overrides:
  DAMRU_BASE        default unsloth/Qwen2.5-14B-Instruct-bnb-4bit
  DAMRU_MAXSEQ      default 2048
  DAMRU_PUSH_REPO   default Damaru-ai/damru-tutor-lora
  DAMRU_DATASET     default Damaru-ai/damru-knowledge
  DAMRU_EPOCHS      default 1
  DAMRU_SAMPLE      default 180000 (rows to sample from dataset)
"""
import os, sys, subprocess


def sh(*a):
    print(">>", " ".join(a))
    subprocess.run(list(a), check=False)


# ========== 0. Install ==========
print("[STEP 0] Installing dependencies...")
sh(sys.executable, "-m", "pip", "install", "-q",
   "--upgrade", "unsloth", "unsloth_zoo")
sh(sys.executable, "-m", "pip", "install", "-q",
   "bitsandbytes", "accelerate", "peft",
   "trl>=0.12.0",            # SFTConfig API
   "transformers>=4.46.0",   # no group_by_length
   "datasets", "sentencepiece", "protobuf",
   "hf_transfer", "xformers")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ========== 1. Config ==========
BASE_MODEL   = os.environ.get("DAMRU_BASE",
                              "unsloth/Qwen2.5-14B-Instruct-bnb-4bit")  # 14B SHIFT!
MAX_SEQ      = int(os.environ.get("DAMRU_MAXSEQ", "2048"))
EPOCHS       = float(os.environ.get("DAMRU_EPOCHS", "1"))
SAMPLE_ROWS  = int(os.environ.get("DAMRU_SAMPLE", "180000"))
OUT_DIR      = "/kaggle/working/damru-lora-14b"
PUSH_REPO    = os.environ.get("DAMRU_PUSH_REPO", "Damaru-ai/damru-tutor-lora")
DATASET_REPO = os.environ.get("DAMRU_DATASET", "Damaru-ai/damru-knowledge")

print(f"[CONFIG] Model: {BASE_MODEL}")
print(f"[CONFIG] Dataset: {DATASET_REPO} | Sample: {SAMPLE_ROWS}")
print(f"[CONFIG] Max seq: {MAX_SEQ} | Epochs: {EPOCHS}")

# HF Token
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        print("[OK] HF_TOKEN loaded from Kaggle Secrets")
    except Exception as e:
        print("[warn] HF_TOKEN not found:", e)
os.environ["HF_TOKEN"] = HF_TOKEN or ""

# ========== 2. Load 14B Model ==========
print("[STEP 1] Loading Qwen2.5-14B with 4-bit QLoRA...")
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    max_seq_length = MAX_SEQ,
    dtype          = None,      # auto bf16/fp16
    load_in_4bit   = True,      # QLoRA -- fits 14B in ~12GB
)
print(f"[OK] Model loaded. VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

print("[STEP 2] Attaching LoRA adapter...")
model = FastLanguageModel.get_peft_model(
    model,
    r                      = 16,
    target_modules         = ["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
    lora_alpha             = 16,
    lora_dropout           = 0,
    bias                   = "none",
    use_gradient_checkpointing = "unsloth",  # 30% less VRAM
    random_state           = 3407,
    use_rslora             = False,
)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[OK] LoRA attached. Trainable params: {total_params:,}")

# ========== 3. Dataset ==========
print(f"[STEP 3] Loading dataset from {DATASET_REPO}...")
from datasets import load_dataset, Dataset
import random


def to_messages(ex):
    """Universal schema converter -> messages format."""
    if ex.get("messages") and isinstance(ex["messages"], list):
        return {"messages": ex["messages"]}
    pairs = (("instruction", "output"), ("prompt", "response"),
             ("question", "answer"), ("input", "output"))
    for uk, ak in pairs:
        if ex.get(uk) and ex.get(ak):
            msgs = []
            if ex.get("system"):
                msgs.append({"role": "system", "content": str(ex["system"])})
            msgs.append({"role": "user", "content": str(ex[uk])})
            msgs.append({"role": "assistant", "content": str(ex[ak])})
            return {"messages": msgs}
    text = str(ex.get("text", "") or ex.get("content", ""))
    return {"messages": [{"role": "user", "content": text}]}


try:
    raw = load_dataset(DATASET_REPO, split="train",
                       token=HF_TOKEN or None,
                       streaming=False)
    total = len(raw)
    print(f"[OK] Total rows available: {total:,}")
    # Smart sampling -- shuffle + take top N
    if total > SAMPLE_ROWS:
        indices = random.sample(range(total), SAMPLE_ROWS)
        raw = raw.select(indices)
        print(f"[OK] Sampled: {SAMPLE_ROWS:,} rows")
except Exception as e:
    print(f"[warn] Dataset load failed ({e}), using fallback sample")
    raw = Dataset.from_list([
        {"instruction": "Damru kaun hai?",
         "output": ("Main Damru hoon -- ek Bhartiya AI, tere liye bana, "
                    "GPT-4 se takkar lene wala.")},
    ] * 200)

print("[STEP 4] Formatting dataset...")
ds = raw.map(to_messages, remove_columns=list(raw.column_names))


def fmt(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}


ds = ds.map(fmt, remove_columns=["messages"])
# Filter out empty/too-short examples
ds = ds.filter(lambda x: len(x["text"]) > 50)
print(f"[OK] After format & filter: {len(ds):,} examples ready")

# ========== 4. Train (SFTConfig -- no group_by_length) ==========
print("[STEP 5] Starting training (14B QLoRA)...")
from trl import SFTTrainer, SFTConfig

# T4 x2 = 2x 14.5GB = 29GB total VRAM
# 14B 4-bit ~ 8-9GB, LoRA overhead ~ 2GB => ~11GB used, ~18GB free for activations
MAX_STEPS = int((len(ds) / (2 * 4)) * EPOCHS)  # approx
print(f"[CONFIG] Approx training steps: {MAX_STEPS}")

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = ds,
    args          = SFTConfig(
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ,
        packing                     = False,   # False is safer for 14B on T4x2
        dataset_num_proc            = 2,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,       # effective batch = 16
        warmup_steps                = 20,
        num_train_epochs            = EPOCHS,
        learning_rate               = 2e-4,
        logging_steps               = 10,
        optim                       = "adamw_8bit",
        weight_decay                = 0.01,
        lr_scheduler_type           = "cosine",
        seed                        = 3407,
        output_dir                  = OUT_DIR,
        save_strategy               = "steps",
        save_steps                  = 200,
        fp16                        = not torch.cuda.is_bf16_supported(),
        bf16                        = torch.cuda.is_bf16_supported(),
        report_to                   = "none",
        # NOTE: group_by_length REMOVED -- deprecated in transformers>=4.46
    ),
)

g = torch.cuda.get_device_properties(0)
print(f"[GPU] {g.name}  {g.total_memory/1e9:.1f} GB")
trainer.train()
print("[OK] Training complete!")

# ========== 5. Save + Push ==========
print("[STEP 6] Saving LoRA adapter...")
model.save_pretrained(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)
print(f"[OK] Saved -> {OUT_DIR}")

if HF_TOKEN:
    try:
        model.push_to_hub(PUSH_REPO, token=HF_TOKEN)
        tokenizer.push_to_hub(PUSH_REPO, token=HF_TOKEN)
        print(f"[OK] Pushed adapter -> {PUSH_REPO}")
    except Exception as e:
        print("[warn] Push failed:", e)
else:
    print("[warn] No HF_TOKEN -- adapter saved locally only")

# ========== 6. GGUF Export ==========
print("[STEP 7] Exporting GGUF (q4_k_m)...")
try:
    gguf_dir = OUT_DIR + "-gguf"
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
    print(f"[OK] GGUF saved -> {gguf_dir}")
    print("[INFO] Upload to Damaru-ai/damru-gguf on HuggingFace")
except Exception as e:
    print("[warn] GGUF export skipped:", e)

print("\n" + "="*50)
print("  DAMRU 14B TRAINING COMPLETE! GPT-4 ko takkar!")
print("="*50)
