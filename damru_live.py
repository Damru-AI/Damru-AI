"""
damru_live.py
=============
Step 3 / 5 -- Damru Live : realistic AI video chat (look + conversation + behaviour)

Turn pipeline (every stage optional + graceful):
  audio --STT--> text --BRAIN--> reply --TTS--> speech --HEAD--> video
  (text-in is also fully supported and works on plain CPU)

  STT      faster-whisper (lazy)                  audio -> text
  BRAIN    injected (Damru cortex_answer / llm)    text  -> reply
  EMOTION  injected engine OR keyword fallback     drives voice + face
  TTS      edge-tts -> gTTS -> pyttsx3 -> XTTS      text  -> speech
  HEAD     Wav2Lip / MuseTalk / SadTalker (lazy)   speech-> video
           (if no head backend, frontend animates avatar via audio lip-sync)

Mount in app.py:
    from damru_live import build_live_router
    api.include_router(build_live_router(brain=_brain, llm_complete=_llm, emotion=EMO))

Endpoints: GET /live/health  GET /live/persona  POST /live/text  POST /live/turn
           WS  /live/ws   (realtime conversational loop)

ZERO hard deps: everything lazy-imported; `import damru_live` never breaks the Space.
Built by Shiva AI for Damru.
"""

from __future__ import annotations

import os
import io
import json
import base64
import tempfile
import logging
import threading
from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, List, Optional

__version__ = "1.0.0"
__all__ = ["LiveConfig", "LiveSession", "get_live", "build_live_router"]

log = logging.getLogger("damru.live")
if not logging.getLogger().handlers:
    logging.basicConfig(level=os.getenv("DAMRU_LOG_LEVEL", "INFO"))

# --------------------------------------------------------------------------- #
# Emotion keyword fallback (Hindi + English) -> drives face + voice style
# --------------------------------------------------------------------------- #
_EMO_MAP = {
    "happy":   ("grinning", ("great", "thanks", "awesome", "love", "glad",
                             "haan", "badhiya", "mast", "khush", "shukriya")),
    "sad":     ("pensive",  ("sad", "sorry", "unfortunately", "dukh", "udaas", "bura")),
    "excited": ("star",     ("wow", "amazing", "incredible", "launch", "chalo",
                             "zabardast", "lets go", "let's go")),
    "angry":   ("angry",    ("angry", "stop", "hate", "gussa", "bakwaas")),
    "curious": ("thinking", ("why", "how", "what", "kaise", "kyun", "kya", "?")),
}


def _keyword_emotion(text: str) -> str:
    t = (text or "").lower()
    for emo, (_, kws) in _EMO_MAP.items():
        if any(k in t for k in kws):
            return emo
    return "neutral"


@dataclass
class LiveConfig:
    persona: str = os.getenv(
        "LIVE_PERSONA",
        "You are Damru -- a warm, witty, human-like Indian AI companion. "
        "Speak naturally and briefly, like a close friend; code-switch Hindi/English "
        "when it feels natural. Show real emotion and curiosity.")
    lang: str = os.getenv("LIVE_LANG", "en")
    voice: str = os.getenv("LIVE_VOICE", "en-IN-NeerjaNeural")
    voice_hi: str = os.getenv("LIVE_VOICE_HI", "hi-IN-SwaraNeural")
    stt_model: str = os.getenv("LIVE_STT_MODEL", "base")
    tts_engine: str = os.getenv("LIVE_TTS", "auto")          # auto|edge|gtts|pyttsx3|xtts|off
    talking_head: str = os.getenv("LIVE_TALKINGHEAD", "none")  # none|wav2lip|musetalk|sadtalker
    avatar_image: str = os.getenv("LIVE_AVATAR", "")
    max_reply_chars: int = int(os.getenv("LIVE_MAX_REPLY", "600"))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Live session
# --------------------------------------------------------------------------- #
class LiveSession:
    def __init__(self, brain: Optional[Callable] = None,
                 llm_complete: Optional[Callable] = None,
                 emotion: Optional[Any] = None,
                 cfg: Optional[LiveConfig] = None) -> None:
        self.cfg = cfg or LiveConfig()
        self.brain = brain            # (text, user_id=, history=) -> reply | (reply, meta)
        self.llm = llm_complete       # (messages) -> str
        self.emotion = emotion        # engine with build_emotional_context(text, intent)

    # ----------------------------- STT ----------------------------- #
    def stt(self, audio_bytes: bytes, mime: str = "audio/webm",
            lang: Optional[str] = None) -> str:
        if not audio_bytes:
            return ""
        try:
            from faster_whisper import WhisperModel  # lazy
            ext = (mime.split("/")[-1].split(";")[0] or "webm")
            with tempfile.NamedTemporaryFile(suffix="." + ext, delete=False) as f:
                f.write(audio_bytes)
                path = f.name
            model = WhisperModel(self.cfg.stt_model, device="cpu", compute_type="int8")
            segs, _ = model.transcribe(path, language=(lang or self.cfg.lang) or None)
            return " ".join(s.text.strip() for s in segs).strip()
        except Exception as e:
            log.info("stt failed: %s", e)
            return ""

    # --------------------------- EMOTION --------------------------- #
    def emotion_of(self, text: str, intent: str = "general") -> str:
        if self.emotion is not None:
            try:
                ctx = self.emotion.build_emotional_context(text, intent)
                return ctx.get("user_emotion") or _keyword_emotion(text)
            except Exception as e:
                log.info("emotion engine failed: %s", e)
        return _keyword_emotion(text)

    # ---------------------------- THINK ---------------------------- #
    def think(self, text: str, history: Optional[List] = None,
              user_id: str = "anonymous") -> tuple:
        text = (text or "").strip()
        if not text:
            return "", {"path": "empty"}
        if self.brain is not None:
            try:
                out = self.brain(text, user_id=user_id, history=history)
                if isinstance(out, tuple):
                    out = out[0]
                if out:
                    return str(out)[: self.cfg.max_reply_chars], {"path": "brain"}
            except Exception as e:
                log.info("brain failed: %s", e)
        if self.llm is not None:
            try:
                msgs = [{"role": "system", "content": self.cfg.persona},
                        {"role": "user", "content": text}]
                return str(self.llm(msgs))[: self.cfg.max_reply_chars], {"path": "llm"}
            except Exception as e:
                log.info("llm failed: %s", e)
        return "(Damru brain offline)", {"path": "none"}

    # ----------------------------- TTS ----------------------------- #
    def tts(self, text: str, emotion: str = "neutral",
            lang: Optional[str] = None):
        text = (text or "").strip()
        if not text or self.cfg.tts_engine == "off":
            return None, None
        lang = lang or self.cfg.lang
        order = (["edge", "gtts", "pyttsx3", "xtts"]
                 if self.cfg.tts_engine == "auto" else [self.cfg.tts_engine])
        for eng in order:
            fn = getattr(self, f"_tts_{eng}", None)
            if fn is None:
                continue
            try:
                data, mime = fn(text, lang)
                if data:
                    log.info("tts via %s -> %d bytes", eng, len(data))
                    return data, mime
            except Exception as e:
                log.info("tts %s failed: %s", eng, e)
        return None, None

    def _tts_edge(self, text, lang):
        import asyncio
        import edge_tts  # lazy
        voice = self.cfg.voice_hi if str(lang).startswith("hi") else self.cfg.voice

        async def _go():
            buf = io.BytesIO()
            comm = edge_tts.Communicate(text, voice)
            async for ch in comm.stream():
                if ch["type"] == "audio":
                    buf.write(ch["data"])
            return buf.getvalue()
        return asyncio.run(_go()), "audio/mpeg"

    def _tts_gtts(self, text, lang):
        from gtts import gTTS  # lazy
        buf = io.BytesIO()
        gTTS(text=text, lang=("hi" if str(lang).startswith("hi") else "en")).write_to_fp(buf)
        return buf.getvalue(), "audio/mpeg"

    def _tts_pyttsx3(self, text, lang):
        import pyttsx3  # lazy
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        eng = pyttsx3.init()
        eng.save_to_file(text, path)
        eng.runAndWait()
        with open(path, "rb") as fh:
            return fh.read(), "audio/wav"

    def _tts_xtts(self, text, lang):
        from TTS.api import TTS  # lazy (Coqui XTTS-v2)
        model = os.getenv("XTTS_MODEL", "tts_models/multilingual/multi-dataset/xtts_v2")
        tts = TTS(model)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        tts.tts_to_file(text=text, file_path=path,
                        language=("hi" if str(lang).startswith("hi") else "en"))
        with open(path, "rb") as fh:
            return fh.read(), "audio/wav"

    # ------------------------- TALKING HEAD ------------------------ #
    def animate(self, audio_bytes: bytes, image: Optional[str] = None):
        mode = self.cfg.talking_head
        if not audio_bytes or mode in ("none", "", "off"):
            return None, None
        fn = getattr(self, f"_head_{mode}", None)
        if fn is None:
            return None, None
        try:
            return fn(audio_bytes, image or self.cfg.avatar_image)
        except Exception as e:
            log.info("talking-head %s failed: %s", mode, e)
            return None, None

    def _head_wav2lip(self, audio_bytes, image):
        raise NotImplementedError("wav2lip weights not configured on this Space")

    def _head_musetalk(self, audio_bytes, image):
        raise NotImplementedError("musetalk not configured on this Space")

    def _head_sadtalker(self, audio_bytes, image):
        raise NotImplementedError("sadtalker not configured on this Space")

    # --------------------------- FULL TURN ------------------------- #
    def turn(self, audio_bytes: Optional[bytes] = None, mime: str = "audio/webm",
             text: Optional[str] = None, history: Optional[List] = None,
             user_id: str = "anonymous", want_video: bool = False,
             lang: Optional[str] = None) -> Dict[str, Any]:
        transcript = (text or "").strip()
        if not transcript and audio_bytes:
            transcript = self.stt(audio_bytes, mime, lang)
        user_emotion = self.emotion_of(transcript)
        reply, meta = self.think(transcript, history=history, user_id=user_id)
        reply_emotion = self.emotion_of(reply)
        audio_b64 = video_b64 = audio_mime = None
        speech, audio_mime = self.tts(reply, reply_emotion, lang)
        if speech:
            audio_b64 = base64.b64encode(speech).decode()
            if want_video:
                vid, _ = self.animate(speech)
                if vid:
                    video_b64 = base64.b64encode(vid).decode()
        return {"ok": True, "transcript": transcript, "reply": reply,
                "emotion": reply_emotion, "user_emotion": user_emotion,
                "audio_b64": audio_b64, "audio_mime": audio_mime,
                "video_b64": video_b64, "meta": meta}

    # ---------------------------- health --------------------------- #
    def health(self) -> Dict[str, Any]:
        def _imp(m: str) -> bool:
            try:
                __import__(m)
                return True
            except Exception:
                return False
        return {
            "stt": {"faster_whisper": _imp("faster_whisper")},
            "tts": {"edge_tts": _imp("edge_tts"), "gtts": _imp("gtts"),
                    "pyttsx3": _imp("pyttsx3"), "xtts": _imp("TTS"),
                    "engine": self.cfg.tts_engine},
            "talking_head": {"mode": self.cfg.talking_head, "cv2": _imp("cv2")},
            "brain": bool(self.brain), "llm": bool(self.llm),
            "emotion_engine": bool(self.emotion),
            "persona": self.cfg.persona[:90],
        }


# --------------------------------------------------------------------------- #
# Singleton + FastAPI router
# --------------------------------------------------------------------------- #
_LIVE: Optional[LiveSession] = None
_LIVE_LOCK = threading.Lock()


def get_live(brain: Optional[Callable] = None,
             llm_complete: Optional[Callable] = None,
             emotion: Optional[Any] = None,
             cfg: Optional[LiveConfig] = None) -> LiveSession:
    global _LIVE
    if _LIVE is None or brain is not None or llm_complete is not None:
        with _LIVE_LOCK:
            _LIVE = LiveSession(brain=brain, llm_complete=llm_complete,
                                emotion=emotion, cfg=cfg)
    return _LIVE


def build_live_router(brain: Optional[Callable] = None,
                      llm_complete: Optional[Callable] = None,
                      emotion: Optional[Any] = None):
    """Return a FastAPI APIRouter exposing /live/*. fastapi imported lazily."""
    from fastapi import APIRouter, Body, WebSocket, WebSocketDisconnect  # type: ignore
    live = get_live(brain=brain, llm_complete=llm_complete, emotion=emotion)
    r = APIRouter(prefix="/live", tags=["live"])

    @r.get("/health")
    def _health():
        return live.health()

    @r.get("/persona")
    def _persona():
        c = live.cfg
        return {"persona": c.persona, "avatar": c.avatar_image, "voice": c.voice,
                "lang": c.lang, "talking_head": c.talking_head}

    @r.post("/text")
    def _text(body: dict = Body(...)):
        return live.turn(text=body.get("text") or body.get("message", ""),
                         history=body.get("history"),
                         user_id=body.get("user_id", "anonymous"),
                         want_video=bool(body.get("want_video", False)),
                         lang=body.get("lang"))

    @r.post("/turn")
    def _turn(body: dict = Body(...)):
        ab = body.get("audio_b64", "")
        audio = base64.b64decode(ab) if ab else None
        return live.turn(audio_bytes=audio, mime=body.get("mime", "audio/webm"),
                         text=body.get("text"), history=body.get("history"),
                         user_id=body.get("user_id", "anonymous"),
                         want_video=bool(body.get("want_video", False)),
                         lang=body.get("lang"))

    @r.websocket("/ws")
    async def _ws(ws: WebSocket):
        await ws.accept()
        hist: List[Dict[str, str]] = []
        try:
            while True:
                msg = await ws.receive_json()
                if msg.get("type") == "bye":
                    break
                await ws.send_json({"type": "stage", "stage": "thinking"})
                audio = base64.b64decode(msg["audio_b64"]) if msg.get("audio_b64") else None
                res = live.turn(audio_bytes=audio, mime=msg.get("mime", "audio/webm"),
                                text=msg.get("text"), history=hist,
                                user_id=msg.get("user_id", "anonymous"),
                                want_video=bool(msg.get("want_video", False)),
                                lang=msg.get("lang"))
                hist.append({"role": "user", "content": res["transcript"]})
                hist.append({"role": "assistant", "content": res["reply"]})
                await ws.send_json({"type": "reply", **res})
        except WebSocketDisconnect:
            return
        except Exception as e:
            try:
                await ws.send_json({"type": "error", "error": str(e)[:200]})
            except Exception:
                pass
    return r


# --------------------------------------------------------------------------- #
# Self-test (offline-safe: stub brain, no TTS/STT backend needed)
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    live = get_live(
        brain=lambda m, user_id="anon", history=None: f"Arre {m.split()[0] if m else 'dost'}! "
              f"Main Damru hoon, bilkul yahin hoon tere saath.",
        emotion=None)

    print("=== HEALTH ===")
    print(json.dumps(live.health(), indent=2))

    print("\n=== EMOTION FALLBACK ===")
    for t in ["yeh toh zabardast hai bhai!", "mujhe bahut dukh hai",
              "yeh kaise hota hai?", "theek hai"]:
        print("  ", repr(t), "->", live.emotion_of(t))

    print("\n=== TURN (text-in) ===")
    res = live.turn(text="Damru kaise ho bhai?")
    show = {k: v for k, v in res.items() if k != "audio_b64"}
    print(json.dumps(show, indent=2, ensure_ascii=False))
    print("audio present:", bool(res["audio_b64"]))

    print("\n=== ROUTER ===")
    try:
        rt = build_live_router(brain=lambda m, **k: "hi")
        print("routes:", sorted({getattr(x, "path", "?") for x in rt.routes}))
    except Exception as e:
        print("router build skipped (fastapi not in sandbox):", e)

    print("\nOK damru_live v" + __version__)
