"""
Self-evaluation. Two parts:
1) heuristic_quality(): fast 0..1 score (no API) used to gate every item.
2) dynamic_threshold(): the quality bar RISES as total grows -> Damru's
   self-evaluation standard keeps increasing over time.
"""
import re

import config

_REASON_WORDS = (
    "because", "therefore", "thus", "hence", "step", "example",
    "reason", "consider", "however", "derive", "proof", "so that",
)
_BAD_STARTS = ("i don't", "i cannot", "sorry", "as an ai", "i'm not able", "unknown")


def heuristic_quality(question, answer):
    q = (question or "").strip()
    a = (answer or "").strip()
    if not q or not a:
        return 0.0
    score = 0.40
    if len(q) > 15:
        score += 0.08
    if len(a) > 120:
        score += 0.15
    if len(a) > 400:
        score += 0.10
    if len(a) > 900:
        score += 0.05
    if re.search(r"\d", a):
        score += 0.04
    low = a.lower()
    if any(w in low for w in _REASON_WORDS):
        score += 0.10
    if low.startswith(_BAD_STARTS):
        score -= 0.45
    words = low.split()
    if words:
        diversity = len(set(words)) / len(words)
        if diversity < 0.40:
            score -= 0.20
    return max(0.0, min(1.0, score))


def dynamic_threshold(total_done):
    rise = (total_done / 10000.0) * config.QUALITY_RISE_PER_10K
    return min(config.MAX_QUALITY, config.START_QUALITY + rise)


def passes(question, answer, total_done, bonus=0.0):
    """Return (ok, quality)."""
    ql = min(1.0, heuristic_quality(question, answer) + bonus)
    return ql >= dynamic_threshold(total_done), ql
