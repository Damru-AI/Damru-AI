#!/usr/bin/env python3
"""
Damru Coding Eval Harness  (Jugad D + gap #2: measurability)
============================================================
"Ultra-pro coding" means nothing unless it is MEASURED. This harness scores a
Damru model on the two standard code benchmarks and executes every generated
program in a sandboxed subprocess to compute real pass@1.

Benchmarks
----------
* HumanEval  (openai_humaneval): 164 function-completion tasks w/ unit tests.
* MBPP       (mbpp):             ~500 basic Python tasks w/ assert tests.

Usage
-----
MODEL=Damaru-ai/damru-qwen-coder BENCH=both MAX_TASKS=200 python coding_eval.py

Env
---
MODEL        HF id / local path of the model to score
             (default: unsloth/Qwen2.5-Coder-3B-Instruct -- baseline)
BENCH        humaneval | mbpp | both           (default both)
MAX_TASKS    cap tasks per benchmark            (default 0 = all)
MAX_NEW      max new tokens to generate         (default 512)
EXEC_TIMEOUT per-program wall-clock seconds      (default 10)
DEVICE       cuda | cpu | auto                  (default auto)
RESULT_REPO  optional HF dataset repo to push the JSON scorecard to
HF_TOKEN     needed only if RESULT_REPO is set / model is private
"""
import os, re, sys, json, time, tempfile, subprocess
from datetime import datetime, timezone

MODEL = os.environ.get("MODEL", "unsloth/Qwen2.5-Coder-3B-Instruct")
BENCH = os.environ.get("BENCH", "both").lower()
MAX_TASKS = int(os.environ.get("MAX_TASKS", "0"))
MAX_NEW = int(os.environ.get("MAX_NEW", "512"))
EXEC_TIMEOUT = int(os.environ.get("EXEC_TIMEOUT", "10"))
DEVICE = os.environ.get("DEVICE", "auto")
RESULT_REPO = os.environ.get("RESULT_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

_FENCE = re.compile(r"```[a-zA-Z0-9_+\-]*\n(.*?)```", re.S)


def _extract(text):
    m = _FENCE.findall(text or "")
    if m:
        return max(m, key=len).strip()
    return (text or "").strip()


def _run(program):
    """Run a full python program (code + tests). Return (passed, err)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        p = subprocess.run([sys.executable, path], capture_output=True,
                           text=True, timeout=EXEC_TIMEOUT)
        if p.returncode == 0:
            return True, ""
        return False, (p.stderr or p.stdout or "exit!=0").splitlines()[-1][:200]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)[:200]
    finally:
        try:
            os.remove(path)
        except Exception:
            pass


class Gen:
    """Lazy model wrapper (transformers chat generation)."""
    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading model:", MODEL, flush=True)
        self.tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype="auto",
            device_map=(DEVICE if DEVICE != "auto" else "auto"),
            trust_remote_code=True)
        self.torch = torch

    def __call__(self, instruction):
        msgs = [{"role": "system", "content":
                 "You are an expert programmer. Reply with correct Python code "
                 "only, inside a single ```python code block."},
                {"role": "user", "content": instruction}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        ins = self.tok(text, return_tensors="pt").to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(**ins, max_new_tokens=MAX_NEW,
                                       do_sample=False,
                                       pad_token_id=self.tok.eos_token_id)
        gen = out[0][ins["input_ids"].shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)


def eval_humaneval(gen, limit):
    from datasets import load_dataset
    data = load_dataset("openai_humaneval", split="test")
    n = ok = 0
    for ex in data:
        if limit and n >= limit:
            break
        n += 1
        prompt = ex["prompt"]
        instruction = "Complete this Python function:\n\n%s" % prompt
        comp = _extract(gen(instruction))
        # ensure the function signature is present
        program = comp if ex["entry_point"] in comp else prompt + "\n" + comp
        program += "\n\n" + ex["test"] + "\n\ncheck(%s)\n" % ex["entry_point"]
        passed, _ = _run(program)
        ok += int(passed)
        if n % 20 == 0:
            print("  HumanEval %d/%d pass@1=%.1f%%" %
                  (n, len(data), 100.0 * ok / n), flush=True)
    return ok, n


def eval_mbpp(gen, limit):
    from datasets import load_dataset
    data = load_dataset("mbpp", split="test")
    n = ok = 0
    for ex in data:
        if limit and n >= limit:
            break
        n += 1
        instruction = ex["text"] + "\n\nTests:\n" + "\n".join(ex["test_list"])
        comp = _extract(gen(instruction))
        program = comp + "\n\n" + "\n".join(ex["test_list"]) + "\n"
        passed, _ = _run(program)
        ok += int(passed)
        if n % 50 == 0:
            print("  MBPP %d pass@1=%.1f%%" % (n, 100.0 * ok / n), flush=True)
    return ok, n


def main():
    gen = Gen()
    report = {"model": MODEL, "when": datetime.now(timezone.utc).isoformat(),
              "benchmarks": {}}
    lim = MAX_TASKS or 0
    if BENCH in ("humaneval", "both"):
        ok, n = eval_humaneval(gen, lim)
        report["benchmarks"]["humaneval"] = {
            "pass@1": round(100.0 * ok / max(1, n), 2), "solved": ok, "total": n}
    if BENCH in ("mbpp", "both"):
        ok, n = eval_mbpp(gen, lim)
        report["benchmarks"]["mbpp"] = {
            "pass@1": round(100.0 * ok / max(1, n), 2), "solved": ok, "total": n}
    print("\n==== DAMRU CODING SCORECARD ====")
    print(json.dumps(report, indent=2), flush=True)
    with open("coding_scorecard.json", "w") as f:
        json.dump(report, f, indent=2)
    if RESULT_REPO and HF_TOKEN:
        try:
            from huggingface_hub import HfApi, create_repo
            create_repo(RESULT_REPO, repo_type="dataset", token=HF_TOKEN,
                        exist_ok=True)
            HfApi(token=HF_TOKEN).upload_file(
                path_or_fileobj="coding_scorecard.json",
                path_in_repo="scorecards/%d.json" % int(time.time()),
                repo_id=RESULT_REPO, repo_type="dataset")
            print("Pushed scorecard to", RESULT_REPO, flush=True)
        except Exception as e:
            print("scorecard push failed:", str(e)[:160], flush=True)


if __name__ == "__main__":
    main()
