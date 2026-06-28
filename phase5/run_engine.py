#!/usr/bin/env python3
"""
Damru Learning Engine - orchestrator.

Runs many workers in parallel, each crash-isolated (one failure never stops the
engine). Workers:
  - general : multi-source harvest (Wikipedia + arXiv + Stack Exchange) on the
              current subject (one subject at a time -> mastery -> next).
  - analysis: deep reasoning + self-check on the current subject.
  - math    : verified (sympy) + olympiad/JEE/MIT-style self-solved+checked.
  - coding  : 4X workers; problems solved AND executed to verify correctness.
  - hindi   : exam-grade Q&A generated in Hindi / regional languages.
  - exam    : syllabus-aligned Q&A for JEE/NEET/UPSC/NCERT/SSC-Banking.

Multi-provider brain (OpenRouter + Groq + Gemini) beats rate limits.
Eval-driven feedback makes workers focus on Damru's weak subjects.
Sharding (SHARD_TOTAL>1) lets several engines run in parallel without overlap.

Deploy continuously:  python3 run_engine.py
Or cron-chunked: set MAX_RUNTIME_MIN=110.
"""
import os
import sys
import time
import random
import signal
import threading
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import store
import evaluator
import curriculum
import feedback
import math_engine
import coding_engine
import analysis_engine
import hindi_engine
import exam_engine
from sources import wikipedia, arxiv, stackexchange

_STOP = threading.Event()
_START = time.time()


class Stats:
    def __init__(self):
        self._lock = threading.Lock()
        self.total = 0
        self.by_kind = {}

    def add(self, kind, n):
        if n <= 0:
            return
        with self._lock:
            self.total += n
            self.by_kind[kind] = self.by_kind.get(kind, 0) + n

    def get_total(self):
        with self._lock:
            return self.total


STATS = Stats()


def log(msg):
    el = time.time() - _START
    print("[%7.0fs] %s" % (el, msg), flush=True)


def _subject():
    """Curriculum subject, sometimes overridden by a weak subject (self-improvement)."""
    return feedback.pick_subject(curriculum.current_subject())


def _accept_and_store(items, kind, subject=None):
    """Gate items through the rising self-eval bar, then store. Returns inserted count."""
    total = STATS.get_total()
    rows, qsum = [], 0.0
    for it in items:
        q = it.get("question", "")
        a = it.get("answer", "")
        bonus = 0.15 if it.get("verified") else (0.05 if it.get("self_checked") else 0.0)
        ok, ql = evaluator.passes(q, a, total, bonus=bonus)
        if not ok:
            continue
        rows.append(store.make_row(q, a, it.get("intent"), it.get("lang", "en"),
                                   quality=ql, upvotes=it.get("upvotes")))
        qsum += ql
    n = store.insert_batch(rows)
    if n > 0:
        STATS.add(kind, n)
        if subject:
            mastered = curriculum.record(subject, n, qsum / max(1, len(rows)))
            if mastered:
                log("MASTERED subject: %s -> moving to next" % subject)
    return n


def general_worker(wid):
    srcs = [wikipedia, arxiv, stackexchange]
    while not _STOP.is_set():
        try:
            subject = _subject()
            src = random.choice(srcs)
            items = src.harvest(subject)
            n = _accept_and_store(items, "general", subject=subject)
            log("general#%d %-26s +%d (src=%s)" % (wid, subject[:26], n, src.__name__.split('.')[-1]))
            time.sleep(0.5)
        except Exception as e:
            log("general#%d ERR %s" % (wid, str(e)[:120]))
            time.sleep(2)


def analysis_worker(wid):
    while not _STOP.is_set():
        try:
            subject = _subject()
            items = analysis_engine.produce(subject, n=4)
            n = _accept_and_store(items, "analysis", subject=subject)
            log("analysis#%d %-24s +%d" % (wid, subject[:24], n))
            time.sleep(0.3)
        except Exception as e:
            log("analysis#%d ERR %s" % (wid, str(e)[:120]))
            time.sleep(2)


def math_worker(wid):
    while not _STOP.is_set():
        try:
            items = math_engine.produce("Mathematics", n=6)
            n = _accept_and_store(items, "math")
            log("math#%d +%d" % (wid, n))
            time.sleep(0.3)
        except Exception as e:
            log("math#%d ERR %s" % (wid, str(e)[:120]))
            time.sleep(2)


def coding_worker(wid):
    while not _STOP.is_set():
        try:
            items = coding_engine.produce(n=6)
            n = _accept_and_store(items, "coding")
            log("coding#%d +%d (verified)" % (wid, n))
            time.sleep(0.2)
        except Exception as e:
            log("coding#%d ERR %s" % (wid, str(e)[:120]))
            time.sleep(2)


def hindi_worker(wid):
    while not _STOP.is_set():
        try:
            subject = _subject()
            lang = random.choice(config.REGIONAL_LANGS) if config.REGIONAL_LANGS else "hi"
            items = hindi_engine.produce(subject, lang=lang, n=4)
            n = _accept_and_store(items, "hindi", subject=subject)
            log("hindi#%d %-20s [%s] +%d" % (wid, subject[:20], lang, n))
            time.sleep(0.4)
        except Exception as e:
            log("hindi#%d ERR %s" % (wid, str(e)[:120]))
            time.sleep(2)


def exam_worker(wid):
    while not _STOP.is_set():
        try:
            items = exam_engine.produce(n=4)
            n = _accept_and_store(items, "exam")
            log("exam#%d +%d" % (wid, n))
            time.sleep(0.3)
        except Exception as e:
            log("exam#%d ERR %s" % (wid, str(e)[:120]))
            time.sleep(2)


def reporter():
    while not _STOP.is_set():
        for _ in range(30):
            if _STOP.is_set():
                return
            time.sleep(2)
        el_h = max(1e-6, (time.time() - _START) / 3600.0)
        total = STATS.get_total()
        rate_day = total / el_h * 24
        weak = feedback.weak_subjects()
        log("== TOTAL %d | %.0f/hr | proj %.0f/day | target %d | bar=%.3f | %s"
            % (total, total / el_h, rate_day, config.DAILY_TARGET,
               evaluator.dynamic_threshold(total), dict(STATS.by_kind)))
        if weak:
            log("   focusing weak subjects: %s" % ", ".join(weak[:8]))
        for subj, cnt, mastered in curriculum.snapshot():
            log("   %-28s %6d %s" % (subj, cnt, "\u2713" if mastered else ""))


def _handle_signal(signum, frame):
    log("signal %d -> graceful shutdown" % signum)
    _STOP.set()


def main():
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)
    log("Damru Learning Engine starting. target=%d/day" % config.DAILY_TARGET)
    log("shard %d/%d | workers: general=%d analysis=%d math=%d coding=%d hindi=%d exam=%d"
        % (config.SHARD_ID, config.SHARD_TOTAL, config.GENERAL_WORKERS,
           config.ANALYSIS_WORKERS, config.MATH_WORKERS, config.CODING_WORKERS,
           config.HINDI_WORKERS, config.EXAM_WORKERS))
    provs = []
    if config.OPENROUTER_API_KEY:
        provs.append("OpenRouter")
    if config.GROQ_API_KEY:
        provs.append("Groq")
    if config.GEMINI_API_KEY:
        provs.append("Gemini")
    log("LLM providers active: %s" % (", ".join(provs) or "NONE (general harvest only)"))

    threads = []
    for i in range(config.GENERAL_WORKERS):
        threads.append(threading.Thread(target=general_worker, args=(i,), daemon=True))
    for i in range(config.ANALYSIS_WORKERS):
        threads.append(threading.Thread(target=analysis_worker, args=(i,), daemon=True))
    for i in range(config.MATH_WORKERS):
        threads.append(threading.Thread(target=math_worker, args=(i,), daemon=True))
    for i in range(config.CODING_WORKERS):
        threads.append(threading.Thread(target=coding_worker, args=(i,), daemon=True))
    for i in range(config.HINDI_WORKERS):
        threads.append(threading.Thread(target=hindi_worker, args=(i,), daemon=True))
    for i in range(config.EXAM_WORKERS):
        threads.append(threading.Thread(target=exam_worker, args=(i,), daemon=True))
    threads.append(threading.Thread(target=reporter, daemon=True))

    for t in threads:
        t.start()

    try:
        while not _STOP.is_set():
            time.sleep(2)
            if config.MAX_RUNTIME_MIN > 0 and \
                    (time.time() - _START) > config.MAX_RUNTIME_MIN * 60:
                log("MAX_RUNTIME_MIN reached -> stopping")
                _STOP.set()
    except KeyboardInterrupt:
        _STOP.set()

    log("shutting down... final total=%d" % STATS.get_total())
    time.sleep(2)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
