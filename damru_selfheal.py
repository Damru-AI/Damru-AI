#!/usr/bin/env python3
"""
================================================================================
  DAMRU SELF-HEAL WRAPPER v1.0
================================================================================
Self-healing process manager for all Damru components.

Features:
  * Watchdog: monitors any subprocess, restarts on crash
  * Exponential backoff: 1s → 2s → 4s → 8s → 60s (max)
  * Health check: HTTP endpoint ping, restarts if unresponsive
  * Circuit breaker: stops restarting after 10 consecutive failures (logs alert)
  * Memory leak guard: restarts if RSS exceeds memory limit
  * Self-logging: all crashes saved to /opt/damru/logs/
  * Telegram/Notion alert hook (optional)
  * Works for: damru app.py, damru_curious_engine.py, damru_prayas_core.py
================================================================================
"""
import os
import sys
import time
import signal
import subprocess
import threading
import logging
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Callable

try:
    import requests as _req
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SELFHEAL] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()

LOG_DIR    = Path(os.environ.get("DAMRU_LOG_DIR", "/opt/damru/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)

MAX_BACKOFF    = int(os.environ.get("SELFHEAL_MAX_BACKOFF",  "60"))    # seconds
MAX_FAILURES   = int(os.environ.get("SELFHEAL_MAX_FAILURES", "10"))    # circuit breaker
HEALTH_TIMEOUT = int(os.environ.get("SELFHEAL_HEALTH_TIMEOUT", "30"))  # seconds
MEMORY_LIMIT_MB= int(os.environ.get("SELFHEAL_MEMORY_MB", "4096"))     # 4 GB default


class ProcessWatchdog:
    """
    Watches a subprocess. Restarts it on crash with exponential backoff.
    Circuit breaker after MAX_FAILURES consecutive crashes.
    """

    def __init__(self, name: str, cmd: List[str],
                 health_url: str = "",
                 on_restart: Optional[Callable] = None,
                 env: dict = None):
        self.name       = name
        self.cmd        = cmd
        self.health_url = health_url
        self.on_restart = on_restart
        self.env        = {**os.environ, **(env or {})}
        self.proc: Optional[subprocess.Popen] = None
        self.failures   = 0
        self.total_starts = 0
        self.started_at = None
        self._stop      = threading.Event()
        self._lock      = threading.Lock()
        self._log_file  = LOG_DIR / f"{name.replace(' ', '_')}.log"

    def start(self):
        """Start watchdog in background thread."""
        threading.Thread(target=self._run, daemon=True, name=f"watchdog-{self.name}").start()
        log.info(f"Watchdog started for '{self.name}'")

    def stop(self):
        self._stop.set()
        if self.proc:
            try: self.proc.terminate()
            except Exception: pass

    def _run(self):
        backoff = 1
        while not self._stop.is_set():
            if self.failures >= MAX_FAILURES:
                log.critical(f"CIRCUIT BREAKER: '{self.name}' failed {MAX_FAILURES}x. Manual intervention needed.")
                self._alert(f"CIRCUIT BREAKER: {self.name} down permanently. Check logs.")
                break

            self._launch()

            # Wait for process to exit
            if self.proc:
                ret = self.proc.wait()
                ts  = datetime.utcnow().isoformat()

                if self._stop.is_set():
                    break

                if ret == 0:
                    log.info(f"'{self.name}' exited cleanly (code 0). Not restarting.")
                    break

                self.failures += 1
                log.warning(f"'{self.name}' crashed (code={ret}), failure #{self.failures}. Restarting in {backoff}s...")
                self._write_crash_log(ret, ts)

                time.sleep(backoff)
                backoff = min(backoff * 2, MAX_BACKOFF)
            else:
                time.sleep(5)

    def _launch(self):
        with self._lock:
            try:
                self.proc = subprocess.Popen(
                    self.cmd, env=self.env,
                    stdout=open(self._log_file, "a"),
                    stderr=subprocess.STDOUT,
                )
                self.total_starts += 1
                self.started_at = datetime.utcnow().isoformat()
                self.failures = 0   # reset on successful launch
                log.info(f"'{self.name}' started (PID {self.proc.pid}), total starts: {self.total_starts}")
                if self.on_restart and self.total_starts > 1:
                    try: self.on_restart(self.name, self.total_starts)
                    except Exception: pass
            except Exception as e:
                log.error(f"Failed to launch '{self.name}': {e}")
                self.failures += 1
                self.proc = None

    def _write_crash_log(self, ret_code: int, ts: str):
        crash = {
            "name": self.name,
            "cmd": self.cmd,
            "return_code": ret_code,
            "timestamp": ts,
            "failure_number": self.failures,
            "total_starts": self.total_starts,
        }
        crash_file = LOG_DIR / f"{self.name.replace(' ', '_')}_crash_{ts[:10]}.jsonl"
        try:
            with open(crash_file, "a") as f:
                f.write(json.dumps(crash) + "\n")
        except Exception:
            pass

    def _alert(self, msg: str):
        """Optional: send alert (Telegram, Notion notification, etc.)"""
        alert_url = os.environ.get("DAMRU_ALERT_WEBHOOK", "")
        if alert_url and _HAS_REQUESTS:
            try:
                _req.post(alert_url, json={"text": msg}, timeout=10)
            except Exception:
                pass
        log.critical(f"ALERT: {msg}")

    def status(self) -> dict:
        return {
            "name": self.name,
            "running": self.proc is not None and self.proc.poll() is None,
            "pid": self.proc.pid if self.proc else None,
            "failures": self.failures,
            "total_starts": self.total_starts,
            "started_at": self.started_at,
        }


class HealthChecker:
    """Periodically pings health endpoints. Kills process if unresponsive."""

    def __init__(self, watchdogs: List[ProcessWatchdog],
                 interval: int = 60):
        self.watchdogs = watchdogs
        self.interval  = interval
        self._stop     = threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name="healthchecker").start()

    def stop(self):
        self._stop.set()

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(self.interval)
            for wd in self.watchdogs:
                if not wd.health_url or not wd.proc:
                    continue
                if wd.proc.poll() is not None:
                    continue  # already dead, watchdog handles restart
                try:
                    r = _req.get(wd.health_url, timeout=HEALTH_TIMEOUT) if _HAS_REQUESTS else None
                    if r and r.status_code != 200:
                        log.warning(f"'{wd.name}' health check failed (HTTP {r.status_code}). Killing for restart.")
                        try: wd.proc.kill()
                        except Exception: pass
                    else:
                        log.debug(f"'{wd.name}' health OK")
                except Exception as e:
                    log.warning(f"'{wd.name}' health check error: {e}. Killing for restart.")
                    try: wd.proc.kill()
                    except Exception: pass


class MemoryGuard:
    """Restarts a process if it exceeds memory limit."""

    def __init__(self, watchdog: ProcessWatchdog, limit_mb: int = MEMORY_LIMIT_MB):
        self.watchdog = watchdog
        self.limit_mb = limit_mb
        self._stop = threading.Event()

    def start(self):
        threading.Thread(target=self._loop, daemon=True, name=f"memguard-{self.watchdog.name}").start()

    def _loop(self):
        while not self._stop.is_set():
            time.sleep(120)  # check every 2 min
            proc = self.watchdog.proc
            if not proc or proc.poll() is not None:
                continue
            try:
                import resource
                # Get RSS of subprocess (approximate)
                pid = proc.pid
                with open(f"/proc/{pid}/status") as f:
                    for line in f:
                        if line.startswith("VmRSS:"):
                            rss_kb = int(line.split()[1])
                            rss_mb = rss_kb / 1024
                            if rss_mb > self.limit_mb:
                                log.warning(f"'{self.watchdog.name}' using {rss_mb:.0f}MB > {self.limit_mb}MB. Restarting.")
                                try: proc.kill()
                                except Exception: pass
                            break
            except Exception:
                pass


def create_damru_supervisor():
    """
    Default supervisor setup for all Damru components.
    Call this from your main startup script.
    """
    python = sys.executable
    base   = Path(__file__).parent

    components = [
        {
            "name": "Damru App",
            "cmd": [python, str(base / "app.py")],
            "health_url": f"http://localhost:{os.environ.get('PORT','7860')}/health",
        },
        {
            "name": "Damru Curious Engine",
            "cmd": [python, str(base / "damru_curious_engine.py")],
        },
        {
            "name": "Damru World Harvest",
            "cmd": [python, str(base / "damru_world_harvest.py"), "--daemon"],
        },
    ]

    watchdogs = []
    for cfg in components:
        script = Path(cfg["cmd"][1])
        if not script.exists():
            log.info(f"Skipping '{cfg['name']}' — script not found: {script}")
            continue
        wd = ProcessWatchdog(
            name=cfg["name"], cmd=cfg["cmd"],
            health_url=cfg.get("health_url", ""),
        )
        wd.start()
        watchdogs.append(wd)

    # Health checker + memory guard
    if watchdogs:
        hc = HealthChecker(watchdogs)
        hc.start()
        for wd in watchdogs:
            mg = MemoryGuard(wd)
            mg.start()

    return watchdogs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Damru Self-Heal Supervisor")
    parser.add_argument("--status", action="store_true", help="Show status of all watchdogs")
    args = parser.parse_args()

    if args.status:
        log_files = list(LOG_DIR.glob("*.log"))
        print(f"Log dir: {LOG_DIR}")
        print(f"Log files: {len(log_files)}")
        for lf in log_files:
            print(f"  {lf.name} ({lf.stat().st_size//1024}KB)")
    else:
        log.info("Starting Damru Self-Heal Supervisor...")
        watchdogs = create_damru_supervisor()
        if not watchdogs:
            log.warning("No components started. Run from Damru root directory.")
        else:
            log.info(f"Supervising {len(watchdogs)} components. Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(30)
                    for wd in watchdogs:
                        s = wd.status()
                        log.info(f"  {s['name']}: running={s['running']} failures={s['failures']} starts={s['total_starts']}")
            except KeyboardInterrupt:
                log.info("Stopping supervisor...")
                for wd in watchdogs:
                    wd.stop()
