#!/usr/bin/env python3
"""
DAMRU KAGGLE AUTO-TRAINER  --  self-training trigger (device-free)
==================================================================
Kaggle ka "Save & Run All" server pe chalta hai -- tera phone/net band bhi.
Ye script Kaggle API se training kernel ko PUSH + RUN karta hai. GitHub Actions
cron isse din me chalata hai => Damru khud-ba-khud train hota rehta hai.

SELF-SWITCHING accounts: KAGGLE_USERNAMES / KAGGLE_KEYS comma-separated do to
har account ka 30h/week quota rotate hota hai (ek khatam -> agla).

ONE-TIME setup per Kaggle account (mobile browser se):
  1. Notebook me HF_TOKEN ko Add-ons -> Secrets me daalo (naam: HF_TOKEN).
     (Kernel secrets version ke sath persist rehte hain, aage API re-push reuse karega.)
  2. Account -> Settings -> Create New API Token -> kaggle.json -> username+key.

ENV:
  KAGGLE_USERNAME / KAGGLE_KEY            (single account)  OR
  KAGGLE_USERNAMES / KAGGLE_KEYS         (comma lists, multi-account rotation)
  DAMRU_TRAIN_SCRIPT   default damru_train_unsloth.py (kernel ka code)
  DAMRU_KERNEL_SLUG    default damru-autotrain
"""
import os, sys, json, shutil, tempfile, subprocess, time

TRAIN_SCRIPT = (os.environ.get("DAMRU_TRAIN_SCRIPT")
                or os.environ.get("KERNEL_CODE_FILE")
                or "damru_train_unsloth.py")
KERNEL_SLUG  = os.environ.get("DAMRU_KERNEL_SLUG", "damru-autotrain")


def ensure_cli():
    try:
        import kaggle  # noqa
    except Exception:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"], check=True)


def accounts():
    users = os.environ.get("KAGGLE_USERNAMES") or os.environ.get("KAGGLE_USERNAME", "")
    keys  = os.environ.get("KAGGLE_KEYS") or os.environ.get("KAGGLE_KEY", "")
    us = [u.strip() for u in users.split(",") if u.strip()]
    ks = [k.strip() for k in keys.split(",") if k.strip()]
    return list(zip(us, ks))


def push(username, key):
    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key
    if not os.path.exists(TRAIN_SCRIPT):
        print("!! train script missing:", TRAIN_SCRIPT); return False
    work = tempfile.mkdtemp(prefix="damru_kernel_")
    shutil.copy(TRAIN_SCRIPT, os.path.join(work, os.path.basename(TRAIN_SCRIPT)))
    meta = {
        "id": f"{username}/{KERNEL_SLUG}",
        "title": "Damru AutoTrain",
        "code_file": os.path.basename(TRAIN_SCRIPT),
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,       # T4 x2
        "enable_internet": True,  # HF pull/push chahiye
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    with open(os.path.join(work, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Pushing (Save & Run All):", meta["id"])
    r = subprocess.run(["kaggle", "kernels", "push", "-p", work],
                       capture_output=True, text=True)
    print(r.stdout.strip()); print(r.stderr.strip())
    return r.returncode == 0 and "error" not in (r.stdout + r.stderr).lower()


def main():
    ensure_cli()
    accs = accounts()
    if not accs:
        print("No Kaggle creds. Set KAGGLE_USERNAME + KAGGLE_KEY "
              "(ya KAGGLE_USERNAMES/KAGGLE_KEYS comma-lists)."); sys.exit(1)
    h = time.gmtime().tm_hour % len(accs)          # rotate quota by hour
    order = accs[h:] + accs[:h]
    for u, k in order:
        try:
            if push(u, k):
                print("[trained-triggered] account:", u); return
        except Exception as e:
            print("account failed", u, str(e)[:140])
    print("All Kaggle accounts failed / quota"); sys.exit(1)


if __name__ == "__main__":
    main()
