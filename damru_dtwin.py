#!/usr/bin/env python3
"""
================================================================================
  DAMRU DIGITAL TWIN (DTwin)  v1.0
  "Digital Immortality" — Preserve a human's full personality forever.
================================================================================

PHILOSOPHY:
  After enough conversations, Damru builds a "Digital Twin" of the user:
  - How they think (logical / visual / emotional)
  - What they know (domain expertise per topic)
  - How they learn (style, depth, language)
  - Their emotional patterns (what excites, frustrates, inspires them)
  - Their unique way of explaining things
  - Corrections they've made (so Damru never repeats same mistake)

  Future: This profile can be loaded into humanoid robots to "resurrect"
  the person's personality, with their explicit permission.

STORAGE:
  - Local: ~/.damru/dtwin/{user_id}.json (fast, session-persistent)
  - HF Dataset: dtwin/{user_id}/profile_{timestamp}.jsonl (permanent backup)
  - Auto-saves every 10 conversations, or on explicit flush()

INTEGRATION with app.py v4:
  After each /chat call, app.py calls:
      DTWIN.observe(user_id, message, response, emotion, intent, lang)
  Before building response, app.py calls:
      tone_hint = DTWIN.get_tone_hint(user_id, message)
  This tone_hint is injected into the system prompt.
================================================================================
"""

import os
import json
import time
import re
import hashlib
import threading
from pathlib import Path
from typing import Optional, Dict, Any
from collections import Counter, defaultdict

HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_DATASET = os.environ.get("HF_DATASET", "Damaru-ai/damru-knowledge")
DTWIN_DIR  = Path(os.environ.get("DTWIN_DIR", os.path.expanduser("~/.damru/dtwin")))
DTWIN_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
#  PROFILE SCHEMA
# ============================================================
def _empty_profile(user_id: str) -> dict:
    """
    A complete DTwin profile.
    This is the "soul template" — everything we learn about a person.
    """
    return {
        # Identity
        "user_id": user_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "last_updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_conversations": 0,
        "version": "1.0",

        # === LANGUAGE & COMMUNICATION ===
        "lang_preference": "auto",        # hi / hinglish / en / auto
        "script_preference": "latin",     # devanagari / latin / mixed
        "formality_level": "casual",      # formal / casual / very_casual
        "response_length_pref": "medium",  # short / medium / detailed
        "uses_emojis": True,
        "explanation_style": "bullet",    # bullet / prose / story / example_first
        "humor_style": "light",           # none / light / sarcastic / playful

        # === KNOWLEDGE PROFILE (domain -> level 0-10) ===
        "knowledge_map": {
            "space":       0,
            "defense":     0,
            "3d_printing": 0,
            "automotive":  0,
            "coding":      0,
            "mathematics": 0,
            "physics":     0,
            "chemistry":   0,
            "biology":     0,
            "history":     0,
            "geography":   0,
            "politics":    0,
            "economics":   0,
            "philosophy":  0,
            "psychology":  0,
            "art_design":  0,
            "music":       0,
            "sports":      0,
            "cooking":     0,
            "health":      0,
            "ai_ml":       0,
            "business":    0,
            "law":         0,
            "religion":    0,
            "general":     0,
        },
        "expertise_domains": [],          # domains where level >= 7
        "learning_domains":  [],          # domains where 2 <= level < 7
        "weak_domains":      [],          # domains where level < 2

        # === THINKING STYLE ===
        "thinking_style": "balanced",     # logical / visual / intuitive / systematic / creative
        "problem_approach": "bottom_up",  # top_down / bottom_up / analogy / example_first
        "attention_span": "medium",       # short / medium / long
        "abstraction_level": 5,           # 1-10 (1=very concrete, 10=very abstract)
        "depth_preference": "intermediate", # surface / intermediate / deep / expert

        # === EMOTIONAL PROFILE ===
        "dominant_emotions": [],          # emotions that appear most
        "passion_topics": [],             # topics that get them excited
        "frustration_triggers": [],       # what frustrates them
        "encouragement_style": "direct",  # direct / gentle / competitive / collaborative
        "risk_tolerance": "medium",       # low / medium / high
        "curiosity_level": 5,             # 1-10
        "ambition_level":  5,             # 1-10

        # === BEHAVIORAL PATTERNS ===
        "asks_follow_ups": True,
        "verifies_answers": False,
        "gives_feedback": False,
        "shares_personal_context": False,
        "uses_examples_in_questions": False,
        "question_complexity": "medium",   # simple / medium / complex / expert
        "avg_message_length": 0,          # chars
        "night_owl": False,               # active after midnight
        "session_hours": [],              # [hour_of_day...]

        # === CORRECTIONS & LEARNING ===
        "corrections_made": [],            # [{topic, wrong_answer, correction, ts}]
        "topics_revisited": {},            # topic -> count (things they ask about multiple times)
        "confusion_areas": [],             # topics where they often ask for re-explanation

        # === CONVERSATION MEMORY ===
        "notable_quotes": [],              # memorable things they said (max 20)
        "goals_mentioned": [],             # goals / dreams they mentioned
        "projects_mentioned": [],          # their ongoing projects
        "people_mentioned": [],            # people they talk about
        "places_mentioned": [],            # places relevant to them
        "values_detected": [],             # inferred values (loyalty, curiosity, ambition...)

        # === PERSONALITY VECTOR (for robot loading) ===
        # A normalized 0-1 score for each Big-5 + extra traits
        "personality_vector": {
            "openness":          0.5,
            "conscientiousness": 0.5,
            "extraversion":      0.5,
            "agreeableness":     0.5,
            "neuroticism":       0.2,
            "creativity":        0.5,
            "leadership":        0.5,
            "empathy":           0.5,
            "humor":             0.5,
            "assertiveness":     0.5,
        },

        # === RAW STATS ===
        "total_messages": 0,
        "total_corrections": 0,
        "domain_message_counts": {},
        "emotion_counts": {},
        "last_5_topics": [],
        "last_active": None,
    }


# ============================================================
#  DTWIN CORE
# ============================================================
class DigitalTwin:
    """
    The main DTwin class.
    One instance per user (cached by user_id).
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.profile_path = DTWIN_DIR / f"{user_id}.json"
        self.profile = self._load()
        self._dirty = False          # needs save
        self._obs_since_save = 0     # auto-save every 10 obs
        self._lock = threading.Lock()

    # --------------------------------------------------------
    #  LOAD / SAVE
    # --------------------------------------------------------
    def _load(self) -> dict:
        if self.profile_path.exists():
            try:
                with open(self.profile_path) as f:
                    saved = json.load(f)
                base = _empty_profile(self.user_id)
                base.update(saved)
                return base
            except Exception:
                pass
        return _empty_profile(self.user_id)

    def _save_local(self):
        try:
            self.profile["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            with open(self.profile_path, "w") as f:
                json.dump(self.profile, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[DTwin] local save error: {e}", flush=True)

    def flush_to_hf(self):
        """Push profile snapshot to HF dataset dtwin/ folder."""
        if not HF_TOKEN: return
        try:
            from huggingface_hub import HfApi
            api = HfApi(token=HF_TOKEN)
            snapshot = json.dumps(self.profile, ensure_ascii=False)
            ts = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
            path = f"dtwin/{self.user_id}/profile_{ts}.jsonl"
            api.upload_file(
                path_or_fileobj=snapshot.encode("utf-8"),
                path_in_repo=path,
                repo_id=HF_DATASET, repo_type="dataset",
                commit_message=f"DTwin snapshot: {self.user_id[:8]} ({self.profile['total_conversations']} convs)"
            )
            print(f"[DTwin] HF flush OK: {path}", flush=True)
        except Exception as e:
            print(f"[DTwin] HF flush error: {e}", flush=True)

    # --------------------------------------------------------
    #  OBSERVE  (call after each conversation turn)
    # --------------------------------------------------------
    def observe(self, message: str, response: str, emotion: str = "",
                intent: str = "general", lang: str = "en",
                was_corrected: bool = False, correction_text: str = ""):
        """Update profile based on one conversation turn."""
        with self._lock:
            p = self.profile
            p["total_messages"]       = p.get("total_messages", 0) + 1
            p["total_conversations"]  = p.get("total_conversations", 0) + 1
            p["last_active"]          = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

            # Session hour tracking
            hour = int(time.strftime("%H"))
            hours = p.get("session_hours", [])
            hours.append(hour)
            p["session_hours"] = hours[-200:]  # keep last 200
            p["night_owl"] = sum(1 for h in hours if h >= 23 or h <= 4) / max(1, len(hours)) > 0.3

            # Language
            self._update_lang(message, lang)

            # Knowledge level update based on intent
            self._update_knowledge(intent, message)

            # Emotion tracking
            if emotion:
                ec = p.get("emotion_counts", {})
                ec[emotion] = ec.get(emotion, 0) + 1
                p["emotion_counts"] = ec
                # Dominant emotions = top 5
                p["dominant_emotions"] = sorted(ec, key=ec.get, reverse=True)[:5]

            # Message length average
            n = p["total_messages"]
            avg = p.get("avg_message_length", 0)
            p["avg_message_length"] = int((avg * (n-1) + len(message)) / n)

            # Question complexity
            self._update_complexity(message)

            # Domain message counts
            dmc = p.get("domain_message_counts", {})
            dmc[intent] = dmc.get(intent, 0) + 1
            p["domain_message_counts"] = dmc

            # Passion topics (high frequency intents)
            top_domains = sorted(dmc, key=dmc.get, reverse=True)[:5]
            p["passion_topics"] = top_domains

            # Last 5 topics
            last5 = p.get("last_5_topics", [])
            last5.append(intent)
            p["last_5_topics"] = last5[-5:]

            # Revisited topics
            rt = p.get("topics_revisited", {})
            rt[intent] = rt.get(intent, 0) + 1
            p["topics_revisited"] = rt

            # Corrections
            if was_corrected and correction_text:
                corr = p.get("corrections_made", [])
                corr.append({
                    "topic": intent,
                    "original_q": message[:200],
                    "correction": correction_text[:300],
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                })
                p["corrections_made"] = corr[-50:]  # keep last 50
                p["total_corrections"] = p.get("total_corrections", 0) + 1

            # Behavior signals
            self._detect_behavior(message)

            # Personality vector update
            self._update_personality(message, emotion, intent)

            # Notable quotes (interesting / long messages)
            if len(message) > 150 and self._is_notable(message):
                nq = p.get("notable_quotes", [])
                nq.append({"text": message[:300], "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "intent": intent})
                p["notable_quotes"] = nq[-20:]

            # Goals & projects extraction
            self._extract_context(message)

            # Rebuild convenience lists
            km = p.get("knowledge_map", {})
            p["expertise_domains"] = [d for d, v in km.items() if v >= 7]
            p["learning_domains"]  = [d for d, v in km.items() if 2 <= v < 7]
            p["weak_domains"]      = [d for d, v in km.items() if v < 2]

            self._dirty = True
            self._obs_since_save += 1

        # Auto-save every 10 observations
        if self._obs_since_save >= 10:
            self._obs_since_save = 0
            threading.Thread(target=self._save_local, daemon=True).start()
            if self.profile["total_conversations"] % 50 == 0:  # HF backup every 50
                threading.Thread(target=self.flush_to_hf, daemon=True).start()

    # --------------------------------------------------------
    #  GET TONE HINT  (inject into system prompt)
    # --------------------------------------------------------
    def get_tone_hint(self, message: str = "") -> str:
        """Return a tone instruction personalized for this user."""
        p = self.profile
        n = p.get("total_conversations", 0)
        if n < 3:
            return ""  # Not enough data yet

        parts = []

        # Language style
        lang = p.get("lang_preference", "auto")
        if lang == "hinglish":
            parts.append("Reply in natural Hinglish (Hindi+English mix) as this user prefers.")
        elif lang == "hi":
            parts.append("Reply in Hindi (Devanagari) as this user prefers.")

        # Response length
        rlen = p.get("response_length_pref", "medium")
        if rlen == "short":
            parts.append("Keep responses concise and direct — this user prefers short answers.")
        elif rlen == "detailed":
            parts.append("Provide detailed, thorough answers — this user likes depth.")

        # Explanation style
        style = p.get("explanation_style", "bullet")
        style_map = {
            "bullet": "Use bullet points for key information.",
            "prose":  "Use flowing prose, not bullet lists.",
            "story":  "Use storytelling / narrative style when possible.",
            "example_first": "Always give a concrete example first, then explain the concept.",
        }
        if style in style_map:
            parts.append(style_map[style])

        # Depth
        depth = p.get("depth_preference", "intermediate")
        depth_map = {
            "surface":       "Keep explanations surface-level — no deep dives.",
            "intermediate":  "Intermediate depth — cover the key points without overwhelming.",
            "deep":          "Go deep — this user wants complete, thorough explanations.",
            "expert":        "Expert-level depth — skip basics, go straight to advanced details.",
        }
        if depth in depth_map:
            parts.append(depth_map[depth])

        # Knowledge-aware hints
        expertise = p.get("expertise_domains", [])
        if expertise:
            parts.append(f"User is expert in: {', '.join(expertise[:3])}. Skip basics in these domains.")

        weak = p.get("weak_domains", [])
        if weak:
            parts.append(f"User is new to: {', '.join(weak[:3])}. Use simpler language for these.")

        # Corrections awareness
        corr = p.get("corrections_made", [])
        if corr:
            last_corr = corr[-1]
            parts.append(f"Note: User previously corrected you on '{last_corr.get('topic','')}' — be extra careful about accuracy there.")

        # Curiosity
        curiosity = p.get("curiosity_level", 5)
        if curiosity >= 8:
            parts.append("This user is highly curious — feel free to add fascinating side facts.")

        # Passion
        passions = p.get("passion_topics", [])
        if passions:
            current_intent = passions[0]
            if current_intent in passions:
                parts.append(f"This user is very passionate about {current_intent} — match their enthusiasm.")

        # Personalisation prefix
        if n >= 20 and parts:
            parts.insert(0, f"[PERSONALIZED for user with {n} conversations]:")

        return " ".join(parts)

    # --------------------------------------------------------
    #  EXPORT for Robot Loading
    # --------------------------------------------------------
    def export_personality(self) -> dict:
        """
        Export a clean personality profile for loading into humanoid robots
        or future AI instances. This is the "digital soul" export.
        """
        p = self.profile
        return {
            "export_version": "1.0",
            "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "consent_required": True,
            "user_id": p["user_id"],
            "conversations_learned_from": p.get("total_conversations", 0),

            # Core identity
            "communication_style": {
                "language": p.get("lang_preference", "auto"),
                "formality": p.get("formality_level", "casual"),
                "explanation_style": p.get("explanation_style", "bullet"),
                "uses_emojis": p.get("uses_emojis", True),
                "humor_style": p.get("humor_style", "light"),
                "avg_message_length": p.get("avg_message_length", 0),
            },

            # Knowledge map (what they know)
            "knowledge_map": p.get("knowledge_map", {}),
            "expertise_areas": p.get("expertise_domains", []),
            "learning_areas": p.get("learning_domains", []),

            # Thinking
            "thinking_style": p.get("thinking_style", "balanced"),
            "problem_approach": p.get("problem_approach", "bottom_up"),
            "depth_preference": p.get("depth_preference", "intermediate"),
            "curiosity_level": p.get("curiosity_level", 5),
            "abstraction_level": p.get("abstraction_level", 5),

            # Emotional fingerprint
            "dominant_emotions": p.get("dominant_emotions", []),
            "passion_topics": p.get("passion_topics", []),
            "values": p.get("values_detected", []),
            "ambition_level": p.get("ambition_level", 5),

            # Personality (Big-5 + extras)
            "personality_vector": p.get("personality_vector", {}),

            # What they care about
            "goals": p.get("goals_mentioned", []),
            "projects": p.get("projects_mentioned", []),
            "notable_statements": p.get("notable_quotes", [])[:10],

            # How they interact
            "behavior_traits": {
                "asks_follow_ups":           p.get("asks_follow_ups", True),
                "verifies_answers":          p.get("verifies_answers", False),
                "night_owl":                 p.get("night_owl", False),
                "question_complexity":       p.get("question_complexity", "medium"),
                "shares_personal_context":   p.get("shares_personal_context", False),
            },
        }

    # --------------------------------------------------------
    #  INTERNAL UPDATERS
    # --------------------------------------------------------
    def _update_lang(self, message: str, detected_lang: str):
        p = self.profile
        devanagari = len(re.findall(r'[\u0900-\u097F]', message))
        total = max(1, len(message.replace(" ", "")))
        if devanagari / total > 0.3:
            p["lang_preference"] = "hi"
            p["script_preference"] = "devanagari"
            return
        hinglish_words = ["kya","hai","hain","nahi","bhai","yaar","karo","batao",
                          "aur","ya","mein","ka","ki","ke","tha","hun","ho","pe",
                          "se","ko","abhi","jaldi","iska","matlab","achha","theek"]
        lower = message.lower()
        hw = sum(1 for w in hinglish_words if f" {w} " in lower)
        if hw >= 2:
            p["lang_preference"] = "hinglish"
            p["script_preference"] = "latin"
            return
        if detected_lang == "en":
            p["lang_preference"] = "en"

    _INTENT_TO_DOMAIN = {
        "space":   "space",   "defense": "defense",  "3d": "3d_printing",
        "auto":    "automotive", "code": "coding",  "math": "mathematics",
        "research":"general", "exam":   "general",   "general": "general",
        "ai_ml":   "ai_ml",   "physics": "physics",  "chemistry": "chemistry",
    }

    def _update_knowledge(self, intent: str, message: str):
        p = self.profile
        domain = self._INTENT_TO_DOMAIN.get(intent, "general")
        km = p.get("knowledge_map", {})
        current = km.get(domain, 0)

        # Signals of expertise
        expert_signals = ["advanced","deep dive","internals","why does","mathematically",
                          "formally","proof","derive","compare","tradeoff","architecture"]
        beginner_signals = ["what is","kya hai","explain","basics","simple","start",
                            "how to begin","introduction","shuru"]

        lower = message.lower()
        is_expert   = any(s in lower for s in expert_signals)
        is_beginner = any(s in lower for s in beginner_signals)

        if is_expert and current < 9:
            km[domain] = min(10, current + 0.5)
        elif is_beginner and current > 0:
            km[domain] = max(0, current - 0.1)  # slight downgrade
        elif current < 8:
            km[domain] = min(8, current + 0.2)  # normal growth

        p["knowledge_map"] = km

    def _update_complexity(self, message: str):
        p = self.profile
        words = len(message.split())
        has_technical = bool(re.search(r'[\w]{8,}|\d+\.\d+|\b(algorithm|equation|derivative|protocol|bandwidth)\b', message, re.I))
        if words > 40 or has_technical:
            p["question_complexity"] = "complex"
        elif words > 20:
            p["question_complexity"] = "medium"
        elif words < 8:
            if p.get("question_complexity") != "complex":
                p["question_complexity"] = "simple"

    def _detect_behavior(self, message: str):
        p = self.profile
        lower = message.lower()
        if any(w in lower for w in ["?","kaise","kyun","why","how","what","explain"]):
            p["asks_follow_ups"] = True
        if any(w in lower for w in ["check","verify","confirm","sahi","correct","sure"]):
            p["verifies_answers"] = True
        if any(w in lower for w in ["mera","mujhe","I","my","main","hamara","our","project"]):
            p["shares_personal_context"] = True
        if len(message) > 100 and any(w in lower for w in ["for example","jaise","like","suppose","consider"]):
            p["uses_examples_in_questions"] = True

    def _update_personality(self, message: str, emotion: str, intent: str):
        p = self.profile
        pv = p.get("personality_vector", {})
        lower = message.lower()

        # Openness (curiosity, new ideas)
        if any(w in lower for w in ["new","future","idea","innovate","imagine","creative","explore"]):
            pv["openness"] = min(1.0, pv.get("openness", 0.5) + 0.02)
        # Conscientiousness (planning, precision)
        if any(w in lower for w in ["plan","step by step","precisely","correct","right","exactly","properly"]):
            pv["conscientiousness"] = min(1.0, pv.get("conscientiousness", 0.5) + 0.02)
        # Extraversion (sharing, enthusiasm)
        if emotion in ["excited","happy","proud","enthusiastic"] or len(message.split('!')) > 2:
            pv["extraversion"] = min(1.0, pv.get("extraversion", 0.5) + 0.02)
        # Agreeableness (collaboration, empathy)
        if any(w in lower for w in ["together","team","bhai","dost","help","support","thanks","shukriya"]):
            pv["agreeableness"] = min(1.0, pv.get("agreeableness", 0.5) + 0.02)
        # Neuroticism (frustration, worry)
        if emotion in ["frustrated","anxious","worried","confused"] or any(w in lower for w in ["problem","issue","stuck","help me","kya karu"]):
            pv["neuroticism"] = min(1.0, pv.get("neuroticism", 0.2) + 0.01)
        # Creativity
        if intent in ["creative","art_design","3d"] or any(w in lower for w in ["design","build","create","banao","imagine"]):
            pv["creativity"] = min(1.0, pv.get("creativity", 0.5) + 0.02)
        # Leadership (planning big things, wanting to build systems)
        if any(w in lower for w in ["system","lead","dominate","best","world","future","mission","conquer"]):
            pv["leadership"] = min(1.0, pv.get("leadership", 0.5) + 0.02)
            p["ambition_level"] = min(10, p.get("ambition_level", 5) + 0.1)
        # Humor
        if any(w in lower for w in ["lol","haha","funny","joke","mast","gajab","dhamakedaar"]):
            pv["humor"] = min(1.0, pv.get("humor", 0.5) + 0.02)

        # Curiosity level
        q_count = message.count("?")
        if q_count >= 2:
            p["curiosity_level"] = min(10, p.get("curiosity_level", 5) + 0.1)

        p["personality_vector"] = pv

    def _is_notable(self, message: str) -> bool:
        notable_signals = ["plan","dream","vision","future","bro","bhai","idea",
                           "never","always","revolution","dominate","mission",
                           "ek din","someday","goal","believe","zindagi"]
        lower = message.lower()
        return any(s in lower for s in notable_signals)

    def _extract_context(self, message: str):
        p = self.profile
        lower = message.lower()
        goal_signals   = ["want to","chahta hun","chahti hun","mera sapna","plan hai","banana hai","goal is","will"]
        project_signals = ["working on","bana raha","develop","building","my project","mera project","startup","app"]
        if any(s in lower for s in goal_signals):
            g = p.get("goals_mentioned", [])
            g.append({"text": message[:200], "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            p["goals_mentioned"] = g[-20:]
        if any(s in lower for s in project_signals):
            pr = p.get("projects_mentioned", [])
            pr.append({"text": message[:200], "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            p["projects_mentioned"] = pr[-20:]


# ============================================================
#  GLOBAL REGISTRY (one DTwin per user, cached in memory)
# ============================================================
_registry: Dict[str, DigitalTwin] = {}
_reg_lock = threading.Lock()

def get_twin(user_id: str) -> DigitalTwin:
    """Get or create a DigitalTwin for a user."""
    if not user_id:
        user_id = "anonymous"
    with _reg_lock:
        if user_id not in _registry:
            _registry[user_id] = DigitalTwin(user_id)
        return _registry[user_id]

def make_user_id(ip: str = "", session: str = "") -> str:
    """
    Create a stable, anonymous user ID from available signals.
    Never stores PII — only a hash.
    """
    raw = f"{ip}:{session}:{os.environ.get('HOSTNAME','')}"
    return "user_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


# ============================================================
#  CONVENIENCE FUNCTIONS (for app.py integration)
# ============================================================
def observe(user_id: str, message: str, response: str,
            emotion: str = "", intent: str = "general", lang: str = "en",
            was_corrected: bool = False, correction_text: str = ""):
    """One-line call from app.py after each conversation."""
    try:
        twin = get_twin(user_id)
        twin.observe(message, response, emotion, intent, lang, was_corrected, correction_text)
    except Exception as e:
        print(f"[DTwin] observe error: {e}", flush=True)

def get_tone_hint(user_id: str, message: str = "") -> str:
    """One-line call from app.py before building system prompt."""
    try:
        return get_twin(user_id).get_tone_hint(message)
    except Exception as e:
        print(f"[DTwin] tone hint error: {e}", flush=True)
        return ""

def export_soul(user_id: str) -> dict:
    """Export full personality for robot loading."""
    return get_twin(user_id).export_personality()

def flush(user_id: str):
    """Force save to HF dataset."""
    get_twin(user_id).flush_to_hf()


# ============================================================
#  FASTAPI ENDPOINTS (mount in app.py)
# ============================================================
def get_dtwin_router():
    """
    Returns a FastAPI router with DTwin endpoints.
    Mount in app.py:
        from damru_dtwin import get_dtwin_router
        api.include_router(get_dtwin_router())
    """
    try:
        from fastapi import APIRouter
        from pydantic import BaseModel
        router = APIRouter(prefix="/dtwin", tags=["Digital Twin"])

        class ObserveIn(BaseModel):
            user_id: str
            message: str
            response: str = ""
            emotion: str = ""
            intent: str = "general"
            lang: str = "en"

        @router.post("/observe")
        def dtwin_observe(body: ObserveIn):
            observe(body.user_id, body.message, body.response,
                    body.emotion, body.intent, body.lang)
            return {"ok": True, "user_id": body.user_id}

        @router.get("/profile/{user_id}")
        def dtwin_profile(user_id: str):
            twin = get_twin(user_id)
            return {"ok": True, "profile": twin.profile}

        @router.get("/soul/{user_id}")
        def dtwin_soul(user_id: str):
            """Export personality for robot loading."""
            return {"ok": True, "soul": export_soul(user_id),
                    "consent_required": True,
                    "message": "This data can be used to create a digital twin of the user. Requires explicit user permission."}

        @router.post("/flush/{user_id}")
        def dtwin_flush(user_id: str):
            flush(user_id)
            return {"ok": True, "message": f"Profile for {user_id} saved to HF dataset."}

        @router.get("/tone_hint/{user_id}")
        def dtwin_tone(user_id: str, message: str = ""):
            hint = get_tone_hint(user_id, message)
            return {"ok": True, "tone_hint": hint}

        return router
    except ImportError:
        return None


# ============================================================
#  SELF-TEST
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("DTwin Self-Test")
    print("=" * 60)
    t = get_twin("test_user_001")
    t.observe(
        message="Bhai space mission ke baare mein batao, Mars pe kaise jayenge?",
        response="Mars mission ke liye ...",
        emotion="curious", intent="space", lang="hinglish"
    )
    t.observe(
        message="Python mein recursive function kaise likhte hain? Deep dive chahiye.",
        response="Recursion in Python ...",
        emotion="excited", intent="code", lang="hinglish"
    )
    t.observe(
        message="Mera sapna hai ki Damru AI ko space mission control ke liye use karein.",
        response="That's an amazing vision!",
        emotion="proud", intent="space", lang="hinglish"
    )
    hint = t.get_tone_hint("space mission plan")
    print(f"Tone hint: {hint}")
    soul = t.export_personality()
    print(f"Soul export keys: {list(soul.keys())}")
    print(f"Personality vector: {soul['personality_vector']}")
    print(f"Knowledge map: {soul['knowledge_map']}")
    print(f"Goals: {soul['goals']}")
    print("\n DTwin working! Digital Immortality system ready.")
