#!/usr/bin/env python3
"""
Damru NURSING / MEDICAL EVAL  (the real July-3 exam metric)
===========================================================
Coding has HumanEval/MBPP; the ACTUAL target (BSc Nursing / medical) had no
eval. This measures multiple-choice accuracy on held-out medical benchmarks so
we can track exam-readiness instead of guessing.

Benchmarks:
  * MedMCQA            (validation)  -- Indian medical entrance style
  * MMLU medical subjects (test)     -- clinical_knowledge, anatomy,
                                        professional_medicine, college_medicine,
                                        medical_genetics, college_biology

Loads a model with transformers, asks each MCQ, parses the chosen letter, and
computes accuracy overall + per subject. Pushes a scorecard JSON.

Run AFTER training (set MODEL_ID to the merged model or base+adapter).

Env:
  HF_TOKEN     (required to push scorecard / gated models)
  MODEL_ID     model to eval        (default Qwen/Qwen2.5-3B-Instruct)
  ADAPTER_ID   optional LoRA adapter repo to attach
  CARD_REPO    Damaru-ai/damru-scorecards
  N_PER_SET    questions per benchmark   (default 500)
  MAX_NEW      max new tokens             (default 8)
  DEVICE       auto|cpu|cuda             (default auto)
"""
import os
import io
import re
import json
import time

HF_TOKEN = os.environ.get("HF_TOKEN", "")
MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_ID = os.environ.get("ADAPTER_ID", "")
CARD_REPO = os.environ.get("CARD_REPO", "Damaru-ai/damru-scorecards")
N_PER_SET = int(os.environ.get("N_PER_SET") or "500")
MAX_NEW = int(os.environ.get("MAX_NEW") or "8")
DEVICE = os.environ.get("DEVICE", "auto")
LETTERS = ["A", "B", "C", "D", "E"]


def _load_model():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    dev = DEVICE
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if dev == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=dtype, trust_remote_code=True,
        device_map=("auto" if dev == "cuda" else None))
    if ADAPTER_ID:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, ADAPTER_ID)
    if dev == "cpu":
        model = model.to("cpu")
    model.eval()
    return tok, model, dev


def _ask(tok, model, dev, question, options):
    import torch
    opt_txt = "\n".join("%s. %s" % (LETTERS[i], o)
                        for i, o in enumerate(options))
    prompt = ("Answer the medical multiple-choice question. Reply with ONLY "
              "the single correct option letter.\n\nQuestion: %s\n%s\n\n"
              "Answer:" % (question, opt_txt))
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").to(model.device if dev == "cuda"
                                            else "cpu")
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAX_NEW, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    gen = tok.decode(out[0][ids["input_ids"].shape[1]:],
                     skip_special_tokens=True)
    m = re.search(r"[A-E]", gen.upper())
    return m.group(0) if m else ""


def _iter_medmcqa():
    from datasets import load_dataset
    ds = load_dataset("openlifescienceai/medmcqa", split="validation",
                      streaming=True)
    n = 0
    for ex in ds:
        opts = [ex.get("opa"), ex.get("opb"), ex.get("opc"), ex.get("opd")]
        if not all(opts):
            continue
        gold = LETTERS[int(ex.get("cop", 0))]
        yield ("medmcqa", ex.get("question", ""), opts, gold)
        n += 1
        if n >= N_PER_SET:
            break


def _iter_mmlu():
    from datasets import load_dataset
    subs = ["clinical_knowledge", "anatomy", "professional_medicine",
            "college_medicine", "medical_genetics", "college_biology"]
    per = max(20, N_PER_SET // len(subs))
    for sub in subs:
        try:
            ds = load_dataset("cais/mmlu", sub, split="test", streaming=True)
        except Exception as e:
            print("  mmlu skip %s (%s)" % (sub, str(e)[:50]), flush=True)
            continue
        n = 0
        for ex in ds:
            opts = ex.get("choices") or []
            if len(opts) < 2:
                continue
            gold = LETTERS[int(ex.get("answer", 0))]
            yield ("mmlu_" + sub, ex.get("question", ""), list(opts), gold)
            n += 1
            if n >= per:
                break


def main():
    tok, model, dev = _load_model()
    print("loaded %s on %s%s" % (MODEL_ID, dev,
          (" + " + ADAPTER_ID) if ADAPTER_ID else ""), flush=True)
    per = {}
    total_ok = total = 0
    t0 = time.time()
    for src, q, opts, gold in list(_iter_medmcqa()) + list(_iter_mmlu()):
        pred = _ask(tok, model, dev, q, opts)
        ok = (pred == gold)
        d = per.setdefault(src, [0, 0])
        d[1] += 1
        d[0] += 1 if ok else 0
        total += 1
        total_ok += 1 if ok else 0
        if total % 100 == 0:
            print("  %d done | running acc %.1f%% | %.0fs"
                  % (total, 100.0 * total_ok / total, time.time() - t0),
                  flush=True)
    card = {
        "model": MODEL_ID, "adapter": ADAPTER_ID or None,
        "overall_acc": round(total_ok / max(1, total), 4),
        "n": total,
        "per_benchmark": {k: {"acc": round(v[0] / max(1, v[1]), 4),
                              "n": v[1]} for k, v in per.items()},
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print("SCORECARD", json.dumps(card, indent=2), flush=True)
    if HF_TOKEN:
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            api.create_repo(CARD_REPO, repo_type="dataset", exist_ok=True)
            name = "nursing_%s.json" % time.strftime("%Y%m%d_%H%M%S")
            api.upload_file(
                path_or_fileobj=io.BytesIO(json.dumps(card, indent=2)
                                           .encode("utf-8")),
                path_in_repo=name, repo_id=CARD_REPO, repo_type="dataset")
            print("pushed scorecard ->", name, flush=True)
        except Exception as e:
            print("scorecard push failed:", str(e)[:80], flush=True)


if __name__ == "__main__":
    main()
