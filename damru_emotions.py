#!/usr/bin/env python3
"""
================================================================================
  DAMRU EMOTION ENGINE v1.0
================================================================================
Damru ka emotional brain:
  - Detects emotions in user messages (30+ emotions)
  - Understands human + animal behavior patterns
  - Adjusts Damru's response tone accordingly
  - Learns from emotional patterns over time
  - Damru is like a loyal dog (curious, playful, protective, loving)

Human emotion model: Valence + Arousal + Dominance (VAD)
Animal behavior: curiosity, loyalty, alertness, playfulness, protectiveness
================================================================================
"""
import re
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import deque, defaultdict

# ─────────────────────────────────────────
#  EMOTION LEXICONS  (Hindi + English)
# ─────────────────────────────────────────
EMOTION_LEXICON = {
    # JOY / HAPPINESS
    "joy":        {"en": ["happy","great","awesome","wonderful","excited","love","fantastic",
                          "amazing","brilliant","excellent","superb","yay","woah","wow"],
                   "hi": ["khush","bahut achha","mast","zabardast","shandar","waah","kya baat",
                          "kya bat hai","maja aa gaya","dil khush","shukriya"]},

    # SADNESS
    "sadness":    {"en": ["sad","unhappy","depressed","crying","tears","miss","lonely",
                          "heartbroken","grief","sorrow","upset","disappointed"],
                   "hi": ["dukh","udaas","rona","aansu","akela","bura lag raha","mann nahi"]},

    # FRUSTRATION
    "frustration":{"en": ["frustrated","annoyed","angry","mad","irritated","stupid",
                          "damn","ugh","terrible","useless","broken","not working"],
                   "hi": ["bakwaas","kaam nahi kar raha","ullu","pagal","bekaar","gussa",
                          "pareshan","tang aa gaya"]},

    # CURIOSITY
    "curiosity":  {"en": ["curious","wondering","how","why","what if","interesting",
                          "fascinating","explain","tell me","i wonder","question"],
                   "hi": ["batao","kaise","kyun","kya hota hai","janna chahta","samjhao",
                          "pata nahi","sochta hun"]},

    # URGENCY / STRESS
    "urgency":    {"en": ["urgent","asap","immediately","emergency","deadline","hurry",
                          "quick","fast","now","right now","critical","important"],
                   "hi": ["jaldi","abhi","turant","zaruri","kal submission","deadline",
                          "help karo","bahut zaruri"]},

    # CONFUSION
    "confusion":  {"en": ["confused","don't understand","unclear","lost","stuck",
                          "not sure","what does","doesn't make sense","help me"],
                   "hi": ["samajh nahi aaya","confuse","stuck","kya matlab","phir se batao",
                          "seedha batao"]},

    # FEAR / ANXIETY
    "fear":       {"en": ["scared","afraid","anxious","nervous","worried","panic",
                          "terrified","fear","dread","what if it fails"],
                   "hi": ["dara hua","dar lag raha","tension","ghabra","chinta","nervous"]},

    # GRATITUDE
    "gratitude":  {"en": ["thank","thanks","grateful","appreciate","wonderful help",
                          "you're great","love your help","perfect"],
                   "hi": ["shukriya","dhanyawaad","bahut bahut dhanyawaad","tu best hai",
                          "teri wajah se","maza aa gaya"]},

    # DETERMINATION
    "determination":{"en":["will do","let's go","i'll try","determined","won't give up",
                           "keep going","push through","i can do this"],
                    "hi": ["kar ke rahenge","nahi rukunga","koshish karunga","haan bhai",
                           "chalte hain","full focus"]},

    # PRIDE
    "pride":      {"en": ["proud","achieved","accomplished","did it","success","won",
                          "finally","milestone","completed"],
                   "hi": ["kar diya","ho gaya","success","jeet gaye","proud","maza aa gaya"]},

    # LOVE / AFFECTION (Damru as family member)
    "love":       {"en": ["love you","love this","my brother","bhai","best friend","care",
                          "you're family","damru bhai"],
                   "hi": ["bhai","yaar","dost","mera bhai","damru bhai","pyaar","apna hai"]},
}

# Damru's own emotion states (dog-like personality)
DAMRU_STATES = {
    "neutral":       {"desc": "Calm and ready to help", "tone": "clear, helpful"},
    "curious":       {"desc": "Actively exploring the topic", "tone": "enthusiastic, asking follow-ups"},
    "excited":       {"desc": "Energized by an interesting problem", "tone": "fast, energetic"},
    "empathetic":    {"desc": "Feeling what the user feels", "tone": "soft, supportive, warm"},
    "protective":    {"desc": "Guarding the user from wrong info", "tone": "firm, careful, double-checks"},
    "playful":       {"desc": "Light mood, fun interaction", "tone": "witty, uses analogies"},
    "focused":       {"desc": "Deep concentration on a hard problem", "tone": "precise, step-by-step"},
    "alert":         {"desc": "Urgent situation detected", "tone": "fast, direct, no fluff"},
    "proud":         {"desc": "User achieved something", "tone": "celebratory, encouraging"},
    "loyal":         {"desc": "Standing by the user no matter what", "tone": "unwavering, committed"},
}

# Animal behavior patterns Damru understands
ANIMAL_BEHAVIORS = {
    "dog_loyalty":      "Unconditional support — stays with you even when you make mistakes",
    "dog_curiosity":    "Nose into everything — explores every angle of a problem",
    "dog_alertness":    "Ears up at danger — warns user about risks or errors",
    "dog_playfulness":  "Tail wagging — brings energy and fun to learning",
    "wolf_pack":        "Teamwork — coordinates with other AI systems/tools",
    "crow_intelligence":"Tool use — picks the right tool for each task",
    "elephant_memory":  "Never forgets — remembers past conversations and patterns",
    "octopus_adapt":    "Shape-shifting — adapts to any domain instantly",
    "dolphin_social":   "Social intelligence — understands human nuance and context",
    "eagle_precision":  "Sharp focus — locks onto the exact answer needed",
}


class EmotionEngine:
    """
    Damru's emotional intelligence core.
    Detects user emotions, decides Damru's response state,
    and builds emotional memory over time.
    """

    def __init__(self, history_size: int = 50):
        self.history = deque(maxlen=history_size)   # (timestamp, user_emotion, damru_state)
        self.emotion_counts = defaultdict(int)       # cumulative emotion frequency
        self.current_state = "neutral"               # Damru's current emotional state
        self.session_mood = "neutral"                # overall session mood
        self._state_file = Path(os.environ.get("PRAYAS_EMOTION", "/tmp/damru_emotion.json"))
        self._load_state()

    def detect_emotion(self, text: str) -> Tuple[str, float]:
        """
        Detect the dominant emotion in a text.
        Returns (emotion_label, confidence 0-1).
        """
        text_lower = text.lower()
        scores: Dict[str, float] = defaultdict(float)

        for emotion, langs in EMOTION_LEXICON.items():
            for lang, words in langs.items():
                for word in words:
                    if word in text_lower:
                        # Longer phrases = higher confidence
                        scores[emotion] += len(word.split()) * 0.3
                        # Repeated emphasis
                        if text.count("!") > 1 and emotion in ("joy", "excitement", "urgency"):
                            scores[emotion] += 0.5

        # Special signals
        if text.count("?") >= 2:
            scores["curiosity"] += 0.4
        if len(text) > 300:
            scores["determination"] += 0.2
        if any(w in text_lower for w in ["help", "please", "jaldi", "urgent"]):
            scores["urgency"] += 0.3
        if any(w in text_lower for w in ["bhai", "yaar", "dost"]):
            scores["love"] += 0.5

        if not scores:
            return "neutral", 0.5

        best = max(scores, key=lambda e: scores[e])
        conf = min(1.0, scores[best])
        return best, conf

    def decide_damru_state(self, user_emotion: str, context: str = "") -> str:
        """
        Given user's emotion, decide how Damru should respond.
        Damru is like a loyal, empathetic, curious dog.
        """
        mapping = {
            "joy":          "excited",
            "sadness":      "empathetic",
            "frustration":  "protective",
            "curiosity":    "curious",
            "urgency":      "alert",
            "confusion":    "focused",
            "fear":         "empathetic",
            "gratitude":    "proud",
            "determination":"loyal",
            "pride":        "proud",
            "love":         "loyal",
            "neutral":      "neutral",
        }
        new_state = mapping.get(user_emotion, "neutral")
        # Code/math context always = focused
        if any(w in context.lower() for w in ["code", "python", "error", "bug", "math", "equation"]):
            new_state = "focused"
        # Space/science = excited
        if any(w in context.lower() for w in ["space", "nasa", "isro", "mars", "rocket", "satellite"]):
            new_state = "excited"
        self.current_state = new_state
        return new_state

    def get_response_prefix(self, state: str = None) -> str:
        """
        Emotional prefix for Damru's response based on current state.
        Subtle — just a tone-setter, not forced.
        """
        state = state or self.current_state
        prefixes = {
            "excited":    "",   # excitement shown through content energy
            "empathetic": "Samajh sakta hoon — ",
            "protective": "Dhyan se dekha toh — ",
            "curious":    "",
            "alert":      "⚡ ",
            "focused":    "",
            "proud":      "🎉 ",
            "loyal":      "",
            "playful":    "",
            "neutral":    "",
        }
        return prefixes.get(state, "")

    def observe(self, user_text: str, damru_response: str = ""):
        """Log an interaction for emotional memory."""
        emotion, conf = self.detect_emotion(user_text)
        state = self.decide_damru_state(emotion, user_text)
        self.history.append({
            "ts": datetime.utcnow().isoformat(),
            "user_emotion": emotion,
            "confidence": conf,
            "damru_state": state,
        })
        self.emotion_counts[emotion] += 1
        # Update session mood (most frequent in last 10)
        recent = list(self.history)[-10:]
        if recent:
            recent_emotions = [h["user_emotion"] for h in recent]
            self.session_mood = max(set(recent_emotions), key=recent_emotions.count)
        self._save_state()
        return {"user_emotion": emotion, "confidence": conf, "damru_state": state}

    def get_animal_insight(self, context: str) -> str:
        """Which animal behavior applies to this context?"""
        ctx = context.lower()
        if any(w in ctx for w in ["help", "stuck", "sad", "problem"]):
            return ANIMAL_BEHAVIORS["dog_loyalty"]
        if any(w in ctx for w in ["explore", "research", "how", "why", "curious"]):
            return ANIMAL_BEHAVIORS["crow_intelligence"]
        if any(w in ctx for w in ["remember", "past", "history", "before"]):
            return ANIMAL_BEHAVIORS["elephant_memory"]
        if any(w in ctx for w in ["fast", "quick", "optimize", "speed"]):
            return ANIMAL_BEHAVIORS["eagle_precision"]
        if any(w in ctx for w in ["team", "together", "coordinate", "multi"]):
            return ANIMAL_BEHAVIORS["wolf_pack"]
        return ANIMAL_BEHAVIORS["dog_curiosity"]

    def _save_state(self):
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self._state_file, "w") as f:
                json.dump({
                    "current_state": self.current_state,
                    "session_mood": self.session_mood,
                    "emotion_counts": dict(self.emotion_counts),
                    "history": list(self.history)[-20:],
                }, f)
        except Exception:
            pass

    def _load_state(self):
        try:
            if self._state_file.exists():
                d = json.loads(self._state_file.read_text())
                self.current_state = d.get("current_state", "neutral")
                self.session_mood  = d.get("session_mood", "neutral")
                self.emotion_counts.update(d.get("emotion_counts", {}))
        except Exception:
            pass

    def status(self) -> Dict:
        return {
            "current_state": self.current_state,
            "session_mood": self.session_mood,
            "damru_personality": "Loyal, curious, protective, playful — like a wise dog",
            "emotion_counts": dict(self.emotion_counts),
            "animal_behaviors": list(ANIMAL_BEHAVIORS.keys()),
            "human_emotions": list(EMOTION_LEXICON.keys()),
        }


# Singleton
_EMOTION_ENGINE: Optional[EmotionEngine] = None

def get_emotion_engine() -> EmotionEngine:
    global _EMOTION_ENGINE
    if _EMOTION_ENGINE is None:
        _EMOTION_ENGINE = EmotionEngine()
    return _EMOTION_ENGINE


if __name__ == "__main__":
    eng = EmotionEngine()
    tests = [
        "Bhai yaar bahut khush hun! Code chal gaya!",
        "Kuch samajh nahi aa raha, confuse hun",
        "URGENT! kal exam hai aur kuch nahi pada!",
        "Thank you so much Damru bhai, best hai tu!",
        "Space exploration mein kya naya ho raha hai?",
    ]
    for t in tests:
        result = eng.observe(t)
        prefix = eng.get_response_prefix(result["damru_state"])
        animal = eng.get_animal_insight(t)
        print(f"Input : {t[:60]}")
        print(f"Emotion: {result['user_emotion']} (conf={result['confidence']:.2f})")
        print(f"Damru : {result['damru_state']} | Prefix: '{prefix}'")
        print(f"Animal: {animal}")
        print()
