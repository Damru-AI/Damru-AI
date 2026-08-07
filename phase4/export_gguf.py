#!/usr/bin/env python3
"""
Damru GGUF EXPORT -- standalone merge -> GGUF -> HF   (resume-friendly)
======================================================================
Kaggle/Colab sessions die mid-run. This exports GGUF from an ALREADY-trained
LoRA adapter WITHOUT retraining, and can add extra quant formats.

Works for both the 14B adapter (Damaru-ai/damru-14b-lora) and the 3B tutor
adapter (Damaru-ai/damru-tutor-lora): Unsloth loads the adapter, merges to fp16,
builds llama.cpp, quantizes, and pushes GGUF.

Deps auto-install on first run (Internet ON); no separate install cell needed.

Run (Kaggle T4, Internet ON, HF_TOKEN secret):
  ADAPTER=Damaru-ai/damru-14b-lora GGUF_REPO=Damaru-ai/damru-14b-gguf python export_gguf.py

Env:
  HF_TOKEN (push), ADAPTER (LoRA repo/dir), GGUF_REPO (out),
  MAX_SEQ (2048), QUANTS (csv, default q4_k_m,q5_k_m)

Built by Shiva AI for Damru.
"""
import os

ADAPTER = os.environ.get("ADAPTER", "Damaru-ai/damru-14b-lora")
GGUF_REPO = os.environ.get("GGUF_REPO", "Damaru-ai/damru-14b-gguf")
MAX_SEQ = int(os.environ.get("MAX_SEQ") or "2048")
QUANTS = [q.strip() for q in (os.environ.get("QUANTS") or "q4_k_m,q5_k_m").split(",") if q.strip()]
HF_TOKEN = os.environ.get("HF_TOKEN", "")


def _ensure_deps():
    """Fresh Kaggle/Colab kernels don't ship unsloth -- auto-install if missing.
    Internet must be ON. No-op when the stack is already present."""
    import importlib.util
    if all(importlib.util.find_spec(m) is not None
           for m in ("unsloth", "trl", "peft", "bitsandbytes", "datasets")):
        return
    import subprocess, sys
    print(">> installing export stack (unsloth/trl/peft/...) -- one-time, ~2-4 min", flush=True)
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
    print(">> export stack installed OK", flush=True)


def main():
    _ensure_deps()
    from unsloth import FastLanguageModel
    print("loading adapter:", ADAPTER, flush=True)
    model, tok = FastLanguageModel.from_pretrained(
        model_name=ADAPTER, max_seq_length=MAX_SEQ, dtype=None, load_in_4bit=True)
    quants = QUANTS or ["q4_k_m"]
    print("exporting GGUF:", quants, "->", GGUF_REPO, flush=True)
    if HF_TOKEN:
        model.push_to_hub_gguf(GGUF_REPO, tok, quantization_method=quants, token=HF_TOKEN)
        print("pushed ->", GGUF_REPO, flush=True)
    else:
        model.save_pretrained_gguf("damru-gguf", tok, quantization_method=quants[0])
        print("saved locally -> damru-gguf/ (set HF_TOKEN to push)", flush=True)


if __name__ == "__main__":
    main()
