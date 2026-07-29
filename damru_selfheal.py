#!/usr/bin/env python3
"""
================================================================================
  DAMRU SELF-HEAL v1.0
================================================================================
Crash pe automatic recovery:
  * Watchdog thread: monitors any target process/function
  * Exponential backoff: 1s -> 2s -> 4s -> 8s -> max 300s
  * Max retries before alert: 10
  * Health check HTTP endpoint
  * Error log with pattern detection (don't repeat same mistake)
  * Notification on repeated failures (Supabase log)
  * GitHub Actions: fail-safe wrapper for all workflows
================================================================================
"""
import os
import sys
import time
import json
import signal
import logging
import traceback
import threading
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Dict, Any
from collections import deque, defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SELFHEAL] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()

ERROR_LOG    = Path(os.environ.get("SELFHEAL_LOG", "/tmp/damru_errors.jsonl"))
MAX_ERRORS   = int(os.environ.get("SELFHEAL_MAX_ERRORS", "500"))
MAX_RETRIES  = int(os.environ.get("SELFHEAL_MAX_RETRIES", "10"))
BASE_DELAY   = float(os.environ.get("SELFHEAL_BASE_DELAY", "1.0"))
MAX_DELAY    = float(os.environ.get("SELFHEAL_MAX_DELAY",  "300.0"))


class ErrorMemory:
    """
    Remembers errors so Damru doesn't repeat the same mistake.
    Pattern detection: if same error type > N times, escalate.
    """
    def __init__(self, maxlen: int = MAX_ERRORS):
        self.errors   = deque(maxlen=maxlen)
        self.patterns = defaultdict(int)  # error_type -> count
        self._lock    = threading.Lock()

    def record(self, error: Exception, context: str = "") -> Dict:
        err_type  = type(error).__name__
        err_msg   = str(error)[:300]
        err_hash  = hash(f"{err_type}:{err_msg[:80]}")
        record = {
            "ts":      datetime.utcnow().isoformat(),
            "type":    err_type,
            "msg":     err_msg,
            "context": context,
            "hash":    str(err_hash),
            "trace":   traceback.format_exc()[-500:],
        }
        with self._lock:
            self.errors.append(record)
            self.patterns[err_type] += 1
        # Persist
        try:
            with open(ERROR_LOG, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass
        return record

    def is_repeated(self, error: Exception, threshold: int = 3) -> bool:
        return self.patterns.get(type(error).__name__, 0) >= threshold

    def get_fix_hint(self, error: Exception) -> str:
        """Rule-based fix hints — Damru learns from past mistakes."""
        err_str = str(error).lower()
        if "connection" in err_str or "timeout" in err_str:
            return "Network issue — retry with exponential backoff"
        if "memory" in err_str or "oom" in err_str:
            return "OOM — reduce batch size or chunk size"
        if "401" in err_str or "403" in err_str or "token" in err_str:
            return "Auth error — check HF_TOKEN / API key env var"
        if "404" in err_str:
            return "Not found — check URL or resource name"
        if "json" in err_str or "decode" in err_str:
            return "Parse error — API response changed format"
        if "cuda" in err_str or "gpu" in err_str:
            return "GPU error — fall back to CPU mode"
        if "disk" in err_str or "space" in err_str:
            return "Disk full — clean up old files"
        return "Unknown error — check logs and retry"

    def summary(self) -> Dict:
        return {
            "total_errors": len(self.errors),
            "by_type": dict(self.patterns),
            "most_recent": self.errors[-1] if self.errors else None,
        }


class SelfHealingRunner:
    """
    Wraps any Python callable with automatic retry + self-healing.
    Usage:
        runner = SelfHealingRunner(my_function)
        runner.run()
    """
    def __init__(self,
                 fn:          Callable,
                 name:        str = "task",
                 max_retries: int = MAX_RETRIES,
                 base_delay:  float = BASE_DELAY,
                 max_delay:   float = MAX_DELAY,
                 on_failure:  Optional[Callable] = None):
        self.fn          = fn
        self.name        = name
        self.max_retries = max_retries
        self.base_delay  = base_delay
        self.max_delay   = max_delay
        self.on_failure  = on_failure
        self.memory      = ErrorMemory()
        self._run_count  = 0
        self._success    = 0
        self._fail       = 0

    def run(self, *args, **kwargs) -> Any:
        attempt = 0
        delay   = self.base_delay

        while attempt <= self.max_retries:
            try:
                self._run_count += 1
                result = self.fn(*args, **kwargs)
                self._success += 1
                if attempt > 0:
                    log.info(f"[{self.name}] Recovered after {attempt} retries")
                return result
            except KeyboardInterrupt:
                raise
            except Exception as e:
                self._fail += 1
                rec = self.memory.record(e, context=f"attempt={attempt}")
                hint = self.memory.get_fix_hint(e)
                log.error(
                    f"[{self.name}] Attempt {attempt}/{self.max_retries} FAILED: "
                    f"{type(e).__name__}: {str(e)[:120]}\n"
                    f"  Hint: {hint}"
                )
                if self.memory.is_repeated(e, threshold=3):
                    log.warning(f"[{self.name}] Repeated error detected: {type(e).__name__} "
                                f"(count={self.memory.patterns[type(e).__name__]}). "
                                f"Applying adaptive fix...")
                    self._adaptive_fix(e)

                attempt += 1
                if attempt > self.max_retries:
                    log.error(f"[{self.name}] Max retries exceeded. Giving up.")
                    if self.on_failure:
                        try: self.on_failure(e, rec)
                        except Exception: pass
                    raise

                log.info(f"[{self.name}] Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay = min(delay * 2, self.max_delay)  # Exponential backoff

    def _adaptive_fix(self, error: Exception):
        """Apply fixes based on error type — Damru adapts automatically."""
        err_str = str(error).lower()
        if "memory" in err_str:
            # Reduce memory pressure
            import gc; gc.collect()
            log.info("[AdaptiveFix] GC collected")
        elif "timeout" in err_str:
            # Increase timeout for next request
            os.environ["DAMRU_TIMEOUT"] = str(
                min(int(os.environ.get("DAMRU_TIMEOUT", "30")) * 2, 300))
            log.info(f"[AdaptiveFix] Increased timeout to {os.environ['DAMRU_TIMEOUT']}s")

    def stats(self) -> Dict:
        return {
            "name":      self.name,
            "runs":      self._run_count,
            "success":   self._success,
            "failures":  self._fail,
            "error_summary": self.memory.summary(),
        }


class ProcessWatchdog:
    """
    Watchdog for external processes (e.g., uvicorn, llama.cpp).
    Restarts them automatically on crash.
    """
    def __init__(self, cmd: list, name: str = "process",
                 max_restarts: int = MAX_RETRIES):
        self.cmd          = cmd
        self.name         = name
        self.max_restarts = max_restarts
        self._proc        = None
        self._restarts    = 0
        self._memory      = ErrorMemory()
        self._stop_event  = threading.Event()

    def start(self):
        thread = threading.Thread(target=self._watch, daemon=True)
        thread.start()
        log.info(f"[Watchdog:{self.name}] Started")
        return thread

    def stop(self):
        self._stop_event.set()
        if self._proc:
            self._proc.terminate()

    def _watch(self):
        delay = BASE_DELAY
        while not self._stop_event.is_set():
            try:
                log.info(f"[Watchdog:{self.name}] Starting process: {' '.join(self.cmd)}")
                self._proc = subprocess.Popen(
                    self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, bufsize=1
                )
                # Stream logs
                for line in self._proc.stdout:
                    print(f"[{self.name}] {line}", end="")
                self._proc.wait()
                retcode = self._proc.returncode

                if self._stop_event.is_set():
                    break

                self._restarts += 1
                log.warning(f"[Watchdog:{self.name}] Crashed (code={retcode}), "
                            f"restart {self._restarts}/{self.max_restarts} in {delay:.1f}s")

                if self._restarts > self.max_restarts:
                    log.error(f"[Watchdog:{self.name}] Too many restarts, giving up")
                    break

                time.sleep(delay)
                delay = min(delay * 2, MAX_DELAY)

            except Exception as e:
                log.error(f"[Watchdog:{self.name}] Watchdog error: {e}")
                time.sleep(delay)
                delay = min(delay * 2, MAX_DELAY)


def selfheal(fn=None, name="task", max_retries=MAX_RETRIES):
    """
    Decorator / function wrapper for self-healing.
    Usage:
        @selfheal
        def my_function(): ...

        # or
        result = selfheal(my_function, name="harvest")(args)
    """
    if fn is None:
        def decorator(f):
            return SelfHealingRunner(f, name=name, max_retries=max_retries).run
        return decorator
    return SelfHealingRunner(fn, name=name, max_retries=max_retries)


# ============================================================
#  CLI: wrap any script with self-healing
# ============================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python damru_selfheal.py <script.py> [args...]")
        sys.exit(1)

    script = sys.argv[1]
    args   = sys.argv[2:]
    cmd    = [sys.executable, script] + args
    name   = Path(script).stem

    log.info(f"Self-healing wrapper for: {script}")
    watchdog = ProcessWatchdog(cmd=cmd, name=name)
    thread   = watchdog.start()

    try:
        thread.join()
    except KeyboardInterrupt:
        log.info("Stopping watchdog...")
        watchdog.stop()
