#!/usr/bin/env python3
"""Damru Open Brain router.

Primary: Hugging Face OpenAI-compatible router with gpt-oss-120b providers.
Fallback: direct Groq API. Caller can retain local GGUF as final fallback.
No fine-tuning or local 120B inference is attempted.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import requests

HF_ROUTER_URL = os.getenv("HF_ROUTER_URL", "https://router.huggingface.co/v1/chat/completions")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
HF_TOKEN = os.getenv("HF_TOKEN", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
HF_MODELS = [x.strip() for x in os.getenv(
    "OPEN_BRAIN_HF_MODELS",
    "openai/gpt-oss-120b:groq,openai/gpt-oss-120b:cerebras,openai/gpt-oss-120b:fireworks-ai",
).split(",") if x.strip()]
GROQ_MODELS = [x.strip() for x in os.getenv(
    "OPEN_BRAIN_GROQ_MODELS", "openai/gpt-oss-120b,openai/gpt-oss-20b"
).split(",") if x.strip()]
TIMEOUT = int(os.getenv("OPEN_BRAIN_TIMEOUT", "120"))

HIGH_HINTS = (
    "prove", "derive", "debug", "architecture", "design a system", "research",
    "analyze", "compare", "algorithm", "complexity", "theorem", "diagnose",
    "medical", "legal", "security", "step by step", "deep think", "strategy",
)
LOW_HINTS = ("hello", "hi", "thanks", "thank you", "define", "meaning", "translate")


def reasoning_effort(query: str) -> str:
    q = (query or "").lower().strip()
    if len(q) > 500 or any(x in q for x in HIGH_HINTS) or "```" in q:
        return "high"
    if len(q) < 80 and any(q == x or q.startswith(x + " ") for x in LOW_HINTS):
        return "low"
    return "medium"


def wants_json(messages: list[dict[str, Any]]) -> bool:
    text = "\n".join(str(x.get("content") or "") for x in messages).lower()
    return "reply only as json" in text or "valid json" in text or "json object" in text


def last_query(messages: list[dict[str, Any]]) -> str:
    for item in reversed(messages):
        if item.get("role") == "user":
            return str(item.get("content") or "")
    return ""


def safe_error(exc: Exception) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return re.sub(r"(hf_|gsk_|sk-)[A-Za-z0-9._-]+", "[REDACTED]", text)[:500]


class OpenBrain:
    def __init__(self):
        self.session = requests.Session()
        self.last_meta: dict[str, Any] = {}

    @property
    def available(self) -> bool:
        return bool(HF_TOKEN or GROQ_API_KEY)

    def _payload(self, model: str, messages: list[dict[str, Any]], max_tokens: int,
                 temperature: float, effort: str, structured: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "reasoning_effort": effort,
        }
        if structured:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _post(self, url: str, key: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        variants = [dict(payload)]
        # Provider compatibility fallbacks: remove optional controls on HTTP 400.
        no_reason = dict(payload); no_reason.pop("reasoning_effort", None)
        variants.append(no_reason)
        minimal = dict(no_reason); minimal.pop("response_format", None); minimal.pop("temperature", None)
        variants.append(minimal)
        last = None
        for variant in variants:
            for attempt in range(3):
                try:
                    r = self.session.post(url, headers=headers, json=variant, timeout=TIMEOUT)
                    if r.ok:
                        return r.json()
                    last = RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                    if r.status_code == 400:
                        break
                    if r.status_code in {408, 409, 429, 500, 502, 503, 504}:
                        time.sleep(min(8, 1.5 * (attempt + 1)))
                        continue
                    break
                except Exception as exc:
                    last = exc
                    time.sleep(min(8, 1.5 * (attempt + 1)))
            if last and "HTTP 400" not in str(last):
                break
        raise RuntimeError(safe_error(last or RuntimeError("empty provider response")))

    @staticmethod
    def _content(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(str(x.get("text") or "") if isinstance(x, dict) else str(x) for x in content)
        return str(content or "").strip()

    def complete(self, messages: list[dict[str, Any]], max_tokens: int = 1200,
                 temperature: float = 0.4, query: str | None = None,
                 effort: str | None = None) -> dict[str, Any]:
        query = query if query is not None else last_query(messages)
        effort = effort or reasoning_effort(query)
        structured = wants_json(messages)
        attempts = []

        if HF_TOKEN:
            for model in HF_MODELS:
                t0 = time.time()
                try:
                    data = self._post(
                        HF_ROUTER_URL, HF_TOKEN,
                        self._payload(model, messages, max_tokens, temperature, effort, structured),
                    )
                    text = self._content(data)
                    if text:
                        meta = {"provider": "hf-router", "model": model, "reasoning_effort": effort,
                                "latency_sec": round(time.time() - t0, 2)}
                        self.last_meta = meta
                        return {"content": text, **meta}
                    attempts.append({"provider": "hf-router", "model": model, "error": "empty"})
                except Exception as exc:
                    attempts.append({"provider": "hf-router", "model": model, "error": safe_error(exc)})

        if GROQ_API_KEY:
            for model in GROQ_MODELS:
                t0 = time.time()
                try:
                    data = self._post(
                        GROQ_URL, GROQ_API_KEY,
                        self._payload(model, messages, max_tokens, temperature, effort, structured),
                    )
                    text = self._content(data)
                    if text:
                        meta = {"provider": "groq-direct", "model": model, "reasoning_effort": effort,
                                "latency_sec": round(time.time() - t0, 2)}
                        self.last_meta = meta
                        return {"content": text, **meta}
                    attempts.append({"provider": "groq-direct", "model": model, "error": "empty"})
                except Exception as exc:
                    attempts.append({"provider": "groq-direct", "model": model, "error": safe_error(exc)})

        raise RuntimeError("Open Brain exhausted providers: " + json.dumps(attempts)[-1800:])


def self_test() -> None:
    assert reasoning_effort("hi") == "low"
    assert reasoning_effort("Design a system architecture and diagnose security issues") == "high"
    assert reasoning_effort("Explain photosynthesis") == "medium"
    assert wants_json([{"role": "user", "content": "Reply ONLY as JSON"}])
    sample = {"choices": [{"message": {"content": "OK"}}]}
    assert OpenBrain._content(sample) == "OK"
    print("Open Brain self-test PASS")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-probe", action="store_true")
    args = parser.parse_args()
    self_test()
    if args.live_probe:
        brain = OpenBrain()
        if not brain.available:
            raise SystemExit("No HF_TOKEN/GROQ_API_KEY configured")
        result = brain.complete(
            [{"role": "system", "content": "Reply concisely."},
             {"role": "user", "content": "Return exactly: DAMRU_OPEN_BRAIN_OK"}],
            max_tokens=40, temperature=0.0, effort="low")
        print(json.dumps({k: result.get(k) for k in
              ("content", "provider", "model", "reasoning_effort", "latency_sec")}, indent=2))
