#!/usr/bin/env python3
"""
DAMRU KAGGLE AUTO-TRIGGER
Oracle se Kaggle pe automatically training trigger karta hai.
Cron se call hota hai every 12h.
"""
import os
import json
import subprocess
import logging
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [KAGGLE-TRIGGER] %(message)s")
log = logging.getLogger()

# Secrets
secrets_file = Path("/etc/damru/secrets.env")
for line in secrets_file.read_text().splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

KAGGLE_USER = os.environ["KAGGLE_USERNAME"]
KAGGLE_KEY  = os.environ["KAGGLE_KEY"]
HF_TOKEN    = os.environ["HF_TOKEN"]

# Kaggle API
kaggle_json = {
    "username": KAGGLE_USER,
    "key": KAGGLE_KEY
}
Path("/root/.kaggle").mkdir(exist_ok=True)
Path("/root/.kaggle/kaggle.json").write_text(json.dumps(kaggle_json))
os.chmod("/root/.kaggle/kaggle.json", 0o600)


def trigger_sft():
    """SFT resume trainer trigger."""
    log.info("Triggering Kaggle SFT run via API...")
    payload = {
        "source": {
            "sourceType": "SCRIPT",
            "scriptContent": Path("/opt/damru/damru_train_unsloth_resume.py").read_text(),
        },
        "kernel_type": "script",
        "language": "python",
        "enable_gpu": True,
        "enable_internet": True,
        "environment_variables": [
            {"key": "HF_TOKEN", "value": HF_TOKEN},
            {"key": "TIME_BUDGET_SEC", "value": "39600"},
        ],
    }
    try:
        import requests
        r = requests.post(
            f"https://www.kaggle.com/api/v1/kernels",
            json=payload,
            auth=(KAGGLE_USER, KAGGLE_KEY),
            timeout=30,
        )
        log.info(f"SFT trigger: HTTP {r.status_code}")
        return r.status_code == 200
    except Exception as e:
        log.error(f"SFT trigger failed: {e}")
        # Fallback: kaggle CLI
        try:
            result = subprocess.run(
                ["/opt/damru/venv/bin/kaggle", "kernels", "push",
                 "-p", "/opt/damru/"],
                capture_output=True, text=True
            )
            log.info(f"kaggle CLI: {result.stdout}")
        except Exception as e2:
            log.error(f"kaggle CLI failed: {e2}")
    return False


if __name__ == "__main__":
    log.info(f"Kaggle trigger at {datetime.utcnow().isoformat()}")
    trigger_sft()
    log.info("Done.")
