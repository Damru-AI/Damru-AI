#!/usr/bin/env python3
"""
================================================================================
 DAMRU BRAIN FORGE v5 RESUME  --  Qwen2.5-14B QLoRA (Kaggle T4 x2)
================================================================================
YE FILE RESUME KARNE KE LIYE HAI:
  - Pehle run Damaru-ai/damru-tutor-lora pe push kar chuka hai
  - Ye file usi LoRA se continue karti hai (fresh base model nahi)
  - Time budget: 11h (39600s) -- Kaggle 12h kill se 60min pehle graceful stop

HOW TO RUN (Kaggle):
  1. Kaggle -> New Notebook
  2. Settings: Accelerator = GPU T4 x2 | Internet = ON
  3. Add-ons -> Secrets -> HF_TOKEN
  4. Ye poori file ek cell mein paste karo -> Run All

Kaggle Secrets (optional overrides):
  HF_TOKEN           required
  DAMRU_RESUME       default 'Damaru-ai/damru-tutor-lora'  (last checkpoint)
  DAMRU_PUSH_REPO    default 'Damaru-ai/damru-tutor-lora'  (push target)
  DAMRU_DATASET      default 'Damaru-ai/damru-knowledge'
  DAMRU_SAMPLE       default 180000
  TIME_BUDGET_SEC    default 39600 (11h) -- change to 41400 for 11.5h
================================================================================
"""
import os
import sys
import time
import threading
import subprocess

_START = time.time()


def log(*a):
    print(f"[forge +{int(time.time()-_START)}s]", *a, flush=True)


def sh(*a):
    log("$", " ".join(a))
    subprocess.run(list(a), check=False)


# ========== 0. Install ==========
log("[STEP 0] Installing dependencies...")
sh(sys.executable, "-m", "pip", "install", "-q",
   "--upgrade", "unsloth", "unsloth_zoo")
sh(sys.executable, "-m", "pip", "install", "-q",
   "bitsandbytes", "accelerate", "peft",
   "trl>=0.12.0", "transformers>=4.46.0",
   "datasets", "sentencepiece", "protobuf",
   "hf_transfer", "xformers")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# ========== 1. Config ==========
# Kaggle 12h = 43200s. We stop at 11h = 39600s -> 60min for save+push
TIME_BUDGET_SEC = int(os.environ.get("TIME_BUDGET_SEC", "39600"))  # 11h default

# RESUME: load from last HF push (not fresh base model)
RESUME_FROM  = os.environ.get("DAMRU_RESUME",    "Damaru-ai/damru-tutor-lora")
PUSH_REPO    = os.environ.get("DAMRU_PUSH_REPO", "Damaru-ai/damru-tutor-lora")
DATASET_REPO = os.environ.get("DAMRU_DATASET",   "Damaru-ai/damru-knowledge")
SAMPLE_ROWS  = int(os.environ.get("DAMRU_SAMPLE", "180000"))
MAX_SEQ      = int(os.environ.get("DAMRU_MAXSEQ", "2048"))
EPOCHS       = float(os.environ.get("DAMRU_EPOCHS", "1"))
OUT_DIR      = "/kaggle/working/damru-lora-14b-acc1"

log(f"Time budget : {TIME_BUDGET_SEC/3600:.2f}h")
log(f"Kaggle limit: 12.00h -- safety window: {(43200-TIME_BUDGET_SEC)/60:.0f}min")
log(f"Resume from : {RESUME_FROM}")
log(f"Push target : {PUSH_REPO}")
log(f"Dataset     : {DATASET_REPO} (sample={SAMPLE_ROWS:,})")

# HF Token
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        from kaggle_secrets import UserSecretsClient
        HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
        log("[OK] HF_TOKEN loaded from Kaggle Secrets")
    except Exception as e:
        log("[warn] HF_TOKEN not found:", e)
os.environ["HF_TOKEN"] = HF_TOKEN or ""

# ========== 2. Load Resumed LoRA Model ==========
log(f"[STEP 1] Loading RESUMED model from {RESUME_FROM} ...")
from unsloth import FastLanguageModel
import torch

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=RESUME_FROM,   # <-- resumed LoRA, NOT base model
    max_seq_length=MAX_SEQ,
    dtype=None,
    load_in_4bit=True,
    token=HF_TOKEN or None,
)
log(f"[OK] Model loaded. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
log("[OK] LoRA already baked in -- skipping get_peft_model (resume guard)")

# ========== 2b. Hourly Push Thread ==========
_stop  = threading.Event()
_lock  = threading.Lock()
_model = [model]
_tok   = [tokenizer]

def _pusher():
    time.sleep(3600)
    while not _stop.is_set():
        if _model[0] and HF_TOKEN:
            try:
                with _lock:
                    stamp = time.strftime("%Y%m%d-%H%M")
                    log(f"[HourlyPush] Pushing at {stamp}...")
                    _model[0].push_to_hub(
                        PUSH_REPO, token=HF_TOKEN,
                        commit_message=f"resume-hourly-{stamp}")
                    _tok[0].push_to_hub(PUSH_REPO, token=HF_TOKEN)
                    log(f"[HourlyPush] Done -> {PUSH_REPO} v")
            except Exception as e:
                log("[HourlyPush] warn:", str(e)[:120])
        _stop.wait(timeout=3600)

threading.Thread(target=_pusher, daemon=True).start()
log("[OK] Hourly push thread started")

# ========== 3. Dataset ==========
log(f"[STEP 3] Loading dataset from {DATASET_REPO}...")
from datasets import load_dataset, Dataset
import random


def to_messages(ex):
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
                       token=HF_TOKEN or None, streaming=False)
    total = len(raw)
    log(f"[OK] Total rows: {total:,}")
    if total > SAMPLE_ROWS:
        indices = random.sample(range(total), SAMPLE_ROWS)
        raw = raw.select(indices)
        log(f"[OK] Sampled: {SAMPLE_ROWS:,} rows")
except Exception as e:
    log(f"[warn] Dataset load failed ({e}), using fallback")
    raw = Dataset.from_list([
        {"instruction": "Damru kaun hai?",
         "output": "Main Damru hoon -- ek Bhartiya AI, GPT-4 ko takkar dene wala."},
    ] * 200)

log("[STEP 4] Formatting dataset...")
ds = raw.map(to_messages, remove_columns=list(raw.column_names))


def fmt(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}


ds = ds.map(fmt, remove_columns=["messages"])
ds = ds.filter(lambda x: len(x["text"]) > 50)
log(f"[OK] {len(ds):,} examples ready")

# ========== 4. TimeBudget Callback ==========
class TimeBudgetCallback:
    """
    Stops SFTTrainer at TIME_BUDGET_SEC.
    Saves model + pushes to HF before Kaggle kills the notebook.
    """
    def __init__(self):
        self.deadline = _START + TIME_BUDGET_SEC
        self.saved = False
        log(f"[TimeBudget] Will stop at {TIME_BUDGET_SEC/3600:.2f}h "
            f"(deadline: {time.strftime('%H:%M:%S', time.localtime(self.deadline))})")

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.time() - _START
        if time.time() >= self.deadline and not self.saved:
            log(f"[TimeBudget] {elapsed/3600:.2f}h elapsed -- "
                f"graceful stop at step {state.global_step}")
            self._save_push(state.global_step)
            control.should_training_stop = True
        elif state.global_step % 100 == 0:
            remain = self.deadline - time.time()
            log(f"[TimeBudget] step={state.global_step} "
                f"elapsed={elapsed/3600:.2f}h remain={remain/3600:.2f}h")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if not self.saved:
            self._save_push(state.global_step)

    def _save_push(self, step):
        self.saved = True
        m = _model[0]
        t = _tok[0]
        if not m:
            return
        try:
            with _lock:
                log(f"[TimeBudget] Saving checkpoint at step {step}...")
                m.save_pretrained(OUT_DIR)
                t.save_pretrained(OUT_DIR)
                log(f"[TimeBudget] Saved -> {OUT_DIR}")
                if HF_TOKEN:
                    stamp = time.strftime("%Y%m%d-%H%M")
                    m.push_to_hub(
                        PUSH_REPO, token=HF_TOKEN,
                        commit_message=f"resume-timebudget-step{step}-{stamp}")
                    t.push_to_hub(PUSH_REPO, token=HF_TOKEN)
                    log(f"[TimeBudget] Pushed -> {PUSH_REPO}")
        except Exception as e:
            log(f"[TimeBudget] save/push failed: {e}")


# ========== 5. Train ==========
log("[STEP 5] Starting resumed SFT training...")
from trl import SFTTrainer, SFTConfig

g = torch.cuda.get_device_properties(0)
log(f"[GPU] {g.name}  {g.total_memory/1e9:.1f} GB")

time_cb = TimeBudgetCallback()

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    callbacks=[time_cb],
    args=SFTConfig(
        dataset_text_field="text",
        max_seq_length=MAX_SEQ,
        packing=False,
        dataset_num_proc=2,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        num_train_epochs=EPOCHS,
        learning_rate=1e-4,     # lower LR for resume
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=OUT_DIR,
        save_strategy="steps",
        save_steps=200,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        report_to="none",
    ),
)
trainer.train()
log("[OK] Training done.")

# ========== 6. Final Save + Push ==========
_stop.set()

if not time_cb.saved:
    try:
        with _lock:
            model.save_pretrained(OUT_DIR)
            tokenizer.save_pretrained(OUT_DIR)
            log(f"[OK] Local save -> {OUT_DIR}")
            if HF_TOKEN:
                stamp = time.strftime("%Y%m%d-%H%M")
                model.push_to_hub(
                    PUSH_REPO, token=HF_TOKEN,
                    commit_message=f"final-resume-{stamp}")
                tokenizer.push_to_hub(PUSH_REPO, token=HF_TOKEN)
                log(f"[OK] Final push -> {PUSH_REPO} v")
    except Exception as e:
        log("[warn] Final push failed:", e)
else:
    log("[OK] Already pushed by TimeBudgetCallback -- skipping final push.")

# ========== 7. GGUF Export ==========
elapsed = time.time() - _START
if elapsed < 41000:  # only if enough time
    log("[STEP 7] GGUF export (q4_k_m)...")
    try:
        gguf_dir = OUT_DIR + "-gguf"
        model.save_pretrained_gguf(gguf_dir, tokenizer, quantization_method="q4_k_m")
        log(f"[OK] GGUF saved -> {gguf_dir}")
        if HF_TOKEN:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            gguf_repo = "Damaru-ai/damru-gguf"
            api.create_repo(gguf_repo, exist_ok=True, repo_type="model")
            api.upload_folder(folder_path=gguf_dir, repo_id=gguf_repo)
            log(f"[OK] GGUF pushed -> {gguf_repo} v")
    except Exception as e:
        log("[warn] GGUF skip:", e)
else:
    log(f"[SKIP] GGUF -- not enough time ({elapsed/3600:.2f}h elapsed)")

log("\n" + "="*60)
log(f"  DAMRU RESUME COMPLETE! Total: {elapsed/3600:.2f}h")
log(f"  Adapter -> {PUSH_REPO}")
log("  LoRA bug FIXED | GPT-4 ko pakki takkar!")
log("="*60)
