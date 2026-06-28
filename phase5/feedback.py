"""
Eval-driven self-improvement loop.

The eval workflow writes eval/weak_topics.json listing the SUBJECTS where Damru
scored low. With probability FOCUS_RATIO, workers focus on those weak subjects so
Damru spends extra effort exactly where it is weakest -> a closed self-improving loop.
Crash-proof: if the file is missing/invalid, it simply has no effect.
"""
import json
import random
import time

import config

_weak = []
_loaded_at = 0.0
_TTL = 600  # re-read the file every 10 min so fresh eval results take effect


def _load():
    global _weak, _loaded_at
    now = time.time()
    if now - _loaded_at < _TTL:
        return
    _loaded_at = now
    try:
        with open(config.WEAK_TOPICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        subs = data.get("weak_subjects") or []
        _weak = [str(s) for s in subs if str(s).strip()]
    except Exception:
        _weak = []


def pick_subject(fallback):
    """Return a weak subject (sometimes) or the given fallback subject."""
    _load()
    if _weak and random.random() < config.FOCUS_RATIO:
        return random.choice(_weak)
    return fallback


def weak_subjects():
    _load()
    return list(_weak)
