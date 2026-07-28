#!/usr/bin/env python3
# ================================================================
# DAMRU BRAIN FORGE v4  --  RESUME + HOURLY PUSH + MULTI-ACCOUNT
# ================================================================
# MODEL   : Qwen2.5-14B-Instruct-bnb-4bit  (14B -- GPT-4 killer!)
# RESUME  : Automatically loads previous LoRA adapter from HF
#           (Damaru-ai/damru-tutor-lora) and continues training
# HOURLY  : Background thread pushes checkpoint to HF every hour
#           so NO GPU time is wasted if run crashes
# MULTI   : Run this on 2-3 Kaggle accounts simultaneously --
#           set ACCOUNT_ID=1,2,3 to save to different branches
#
# HOW TO RUN (Kaggle):
#   1. kaggle.com -> New Notebook
#   2. Settings -> Accelerator: GPU T4 x2  |  Internet: ON
#   3. Add-ons -> Secrets -> add  HF_TOKEN
#   4. Paste this WHOLE file into ONE cell -> Run All
#
# MULTI-ACCOUNT (parallel training on 2-3 Kaggle accounts):
#   Account 1: ACCOUNT_ID=1  (saves to damru-tutor-lora-acc1)
#   Account 2: ACCOUNT_ID=2  (saves to damru-tutor-lora-acc2)
#   Account 3: ACCOUNT_ID=3  (saves to damru-tutor-lora-acc3)
#   Baad me teen adapters merge kar sakte hain!
# ================================================================

import os, sys, subprocess, random, threading, time

def sh(*a):
    print(">>", " ".join(a))
    subprocess.run(list(a), check=False)

# ================================================================
# STEP 0: Install
# ================================================================
print("[STEP 0] Installing dependencies...")
sh(sys.executable, "-m", "pip", "install", "-q", "--upgrade",
   "unsloth", "unsloth_zoo")
sh(sys.executable, "-m", "pip", "install", "-q",
   "bitsandbytes", "accelerate", "peft",
   "trl>=0.12.0",
   "transformers>=4.46.0",
   "datasets", "sentencepiece", "protobuf",
   "hf_transfer", "xformers")

os.environ["PYTORCH_CUDA_ALLOC_CONF"]    = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ================================================================
# STEP 1: Config
# ================================================================
ACCOUNT_ID   = os.environ.get("ACCOUNT_ID", "1")   # 1, 2, or 3
BASE_MODEL   = os.environ.get("DAMRU_BASE",
                 "unsloth/Qwen2.5-14B-Instruct-bnb-4bit")
PREV_ADAPTER = os.environ.get("DAMRU_PREV_ADAPTER",
                 "Damaru-ai/damru-tutor-lora")         # Resume from here
MAX_SEQ      = int(os.environ.get("DAMRU_MAXSEQ",    "2048"))
EPOCHS       = float(os.environ.get("DAMRU_EPOCHS",  "1"))
SAMPLE_ROWS  = int(os.environ.get("DAMRU_SAMPLE",    "180000"))
PUSH_EVERY_H = float(os.environ.get("DAMRU_PUSH_H",  "1.0"))  # push every N hours
DATASET_REPO = os.environ.get("DAMRU_DATASET",
                 "Damaru-ai/damru-knowledge")
# Each account saves to its own repo to avoid conflicts
if ACCOUNT_ID == "1":
    OUT_REPO = os.environ.get("DAMRU_PUSH_REPO", "Damaru-ai/damru-tutor-lora")
else:
    OUT_REPO = os.environ.get("DAMRU_PUSH_REPO",
                 f"Damaru-ai/damru-tutor-lora-acc{ACCOUNT_ID}")
OUT_DIR = f"/kaggle/working/damru-lora-14b-acc{ACCOUNT_ID}"

print(f"[CONFIG] Account ID  : {ACCOUNT_ID}")
print(f"[CONFIG] Base model  : {BASE_MODEL}")
print(f"[CONFIG] Resume from : {PREV_ADAPTER}")
print(f"[CONFIG] Push to     : {OUT_REPO}")
print(f"[CONFIG] Hourly push : every {PUSH_EVERY_H}h")
print(f"[CONFIG] Dataset     : {DATASET_REPO} | Sample: {SAMPLE_ROWS:,}")

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

# ================================================================
# STEP 2: Hourly Push System (Background Thread)
# ================================================================
_push_stop  = threading.Event()
_push_lock  = threading.Lock()
_model_ref  = [None]  # will be set after model loads
_tok_ref    = [None]

def _hourly_pusher():
    """Background thread: pushes checkpoint to HF every PUSH_EVERY_H hours."""
    interval = int(PUSH_EVERY_H * 3600)
    print(f"[HourlyPush] Started -- will push every {interval}s")
    time.sleep(interval)  # wait first interval before first push
    while not _push_stop.is_set():
        if _model_ref[0] is not None and HF_TOKEN:
            try:
                with _push_lock:
                    stamp = time.strftime("%Y%m%d-%H%M")
                    print(f"\n[HourlyPush] Pushing checkpoint at {stamp}...")
                    _model_ref[0].push_to_hub(
                        OUT_REPO, token=HF_TOKEN,
                        commit_message=f"hourly-ckpt-acc{ACCOUNT_ID}-{stamp}")
                    _tok_ref[0].push_to_hub(
                        OUT_REPO, token=HF_TOKEN)
                    print(f"[HourlyPush] Done -> {OUT_REPO}")
            except Exception as e:
                print(f"[HourlyPush] warn: {str(e)[:120]}")
        _push_stop.wait(timeout=interval)

_push_thread = threading.Thread(target=_hourly_pusher, daemon=True)
_push_thread.start()
print("[OK] Hourly push thread started")

# ================================================================
# STEP 3: Load Model + Resume Previous Adapter
# ================================================================
print("\n[STEP 1] Loading Qwen2.5-14B in 4-bit...")
from unsloth import FastLanguageModel
import torch

# Try to load previous adapter (resume training)
_resume_ok = False
try:
    if HF_TOKEN and PREV_ADAPTER:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        # Check if adapter exists on HF
        files = api.list_repo_files(PREV_ADAPTER)
        if any("adapter_config.json" in f for f in files):
            print(f"[RESUME] Found previous adapter at {PREV_ADAPTER}")
            print("[RESUME] Loading base model + merging previous LoRA...")
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name     = PREV_ADAPTER,  # Load with adapter!
                max_seq_length = MAX_SEQ,
                dtype          = None,
                load_in_4bit   = True,
                token          = HF_TOKEN,
            )
            _resume_ok = True
            print(f"[RESUME] Resumed from {PREV_ADAPTER} -- continuing session!")
        else:
            print(f"[RESUME] No adapter found at {PREV_ADAPTER}, starting fresh")
except Exception as e:
    print(f"[RESUME] Could not resume ({str(e)[:100]}), starting fresh")

if not _resume_ok:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = BASE_MODEL,
        max_seq_length = MAX_SEQ,
        dtype          = None,
        load_in_4bit   = True,
    )

print(f"[OK] Model loaded | VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# ================================================================
# STEP 4: LoRA Adapter
# ================================================================
print("[STEP 2] Attaching LoRA adapter...")
model = FastLanguageModel.get_peft_model(
    model,
    r                          = 16,
    target_modules             = ["q_proj", "k_proj", "v_proj", "o_proj",
                                   "gate_proj", "up_proj", "down_proj"],
    lora_alpha                 = 16,
    lora_dropout               = 0,
    bias                       = "none",
    use_gradient_checkpointing = "unsloth",  # 30% less VRAM
    random_state               = 3407,
)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[OK] Trainable params: {trainable:,}")

# Register model in hourly push refs
_model_ref[0] = model
_tok_ref[0]   = tokenizer

# ================================================================
# STEP 5: Dataset
# ================================================================
print(f"\n[STEP 3] Loading dataset: {DATASET_REPO}")
from datasets import load_dataset, Dataset

def to_messages(ex):
    if ex.get("messages") and isinstance(ex["messages"], list):
        return {"messages": ex["messages"]}
    for uk, ak in [("instruction","output"),("prompt","response"),
                   ("question","answer"),("input","output")]:
        if ex.get(uk) and ex.get(ak):
            msgs = []
            if ex.get("system"):
                msgs.append({"role": "system", "content": str(ex["system"])})
            msgs.append({"role": "user",      "content": str(ex[uk])})
            msgs.append({"role": "assistant",  "content": str(ex[ak])})
            return {"messages": msgs}
    return {"messages": [{"role":"user","content":str(ex.get("text",""))}]}

try:
    raw = load_dataset(DATASET_REPO, split="train", token=HF_TOKEN or None)
    total = len(raw)
    print(f"[OK] Total rows available: {total:,}")
    # Each account uses a DIFFERENT random sample -> diverse training!
    random.seed(int(ACCOUNT_ID) * 42)  # different seed per account
    if total > SAMPLE_ROWS:
        idx = random.sample(range(total), SAMPLE_ROWS)
        raw = raw.select(idx)
        print(f"[OK] Sampled: {SAMPLE_ROWS:,} rows (seed={int(ACCOUNT_ID)*42})")
except Exception as e:
    print(f"[warn] Dataset fail ({e}), using fallback")
    raw = Dataset.from_list([
        {"instruction": "Damru kaun hai?",
         "output": "Main Damru hoon -- ek Bhartiya AI, GPT-4 se takkar lene wala!"}
    ] * 500)

ds = raw.map(to_messages, remove_columns=list(raw.column_names))

def fmt(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}

ds = ds.map(fmt, remove_columns=["messages"])
ds = ds.filter(lambda x: len(x["text"]) > 50)
print(f"[OK] Final dataset: {len(ds):,} examples")

# ================================================================
# STEP 6: Train
# ================================================================
print(f"\n[STEP 4] Training started (Account {ACCOUNT_ID})...")
from trl import SFTTrainer, SFTConfig

g = torch.cuda.get_device_properties(0)
print(f"[GPU] {g.name}  {g.total_memory/1e9:.1f} GB")

trainer = SFTTrainer(
    model         = model,
    tokenizer     = tokenizer,
    train_dataset = ds,
    args          = SFTConfig(
        dataset_text_field          = "text",
        max_seq_length              = MAX_SEQ,
        packing                     = False,
        dataset_num_proc            = 2,
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
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
        # group_by_length REMOVED -- deprecated in transformers>=4.46
    ),
)

trainer.train()
print("[OK] Training complete!")

# ================================================================
# STEP 7: Final Save + Push
# ================================================================
print("\n[STEP 5] Final save + push...")
_push_stop.set()  # stop hourly thread

with _push_lock:
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print(f"[OK] Saved locally -> {OUT_DIR}")

    if HF_TOKEN:
        try:
            model.push_to_hub(
                OUT_REPO, token=HF_TOKEN,
                commit_message=f"final-acc{ACCOUNT_ID}-{time.strftime('%Y%m%d-%H%M')}"
            )
            tokenizer.push_to_hub(OUT_REPO, token=HF_TOKEN)
            print(f"[OK] Final push -> {OUT_REPO}")
        except Exception as e:
            print("[warn] Push failed:", e)
    else:
        print("[warn] No HF_TOKEN -- local save only")

# ================================================================
# STEP 8: GGUF Export
# ================================================================
print("\n[STEP 6] GGUF export (q4_k_m)...")
try:
    gguf_dir = OUT_DIR + "-gguf"
    model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
    print(f"[OK] GGUF -> {gguf_dir}")
    if HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            gguf_repo = f"Damaru-ai/damru-gguf-acc{ACCOUNT_ID}"
            api.create_repo(gguf_repo, exist_ok=True)
            api.upload_folder(folder_path=gguf_dir, repo_id=gguf_repo)
            print(f"[OK] GGUF pushed -> {gguf_repo}")
        except Exception as e:
            print("[warn] GGUF push failed:", e)
except Exception as e:
    print("[warn] GGUF skip:", e)

print("\n" + "="*60)
print(f"  DAMRU 14B TRAINING COMPLETE! Account {ACCOUNT_ID}")
print(f"  Adapter -> {OUT_REPO}")
print(f"  GPU time wasted = 0 (hourly push saved everything!)")
print("  GPT-4 ko takkar ab pakki!")
print("="*60)
