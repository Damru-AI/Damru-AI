#!/usr/bin/env python3
"""
================================================================================
  PRAYAS CORE — Parallel Reasoning And Yielding Advanced Synthesis
================================================================================
Revolutionary architecture: MEMORY and REASONING are COMPLETELY SEPARATE.

  Memory Layer  : Knowledge Tiles (JSONL, BM25 + Bloom Filter, no GPU needed)
  Reasoning Layer: Tiny composer model — NEVER stores facts, only synthesizes
  Skill Layer   : Pre-compiled code modules (math, code, web, space, etc.)

This means:
  * New knowledge = append a tile (milliseconds, zero retraining)
  * Model stays small (1-3B) because it only does COMPOSITION
  * Runs fully on CPU — no API, no GPU, no cloud dependency
  * Self-healing: every component restarts independently
================================================================================
"""
import os
import re
import json
import time
import math
import hashlib
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRAYAS] %(levelname)s %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
TILE_DIR      = Path(os.environ.get("PRAYAS_TILE_DIR",  "/opt/damru/tiles"))
INDEX_FILE    = Path(os.environ.get("PRAYAS_INDEX",     "/opt/damru/prayas_index.json"))
EMOTION_FILE  = Path(os.environ.get("PRAYAS_EMOTION",  "/opt/damru/emotion_state.json"))
MAX_TILES     = int(os.environ.get("PRAYAS_MAX_TILES",  "500000"))
TOP_K         = int(os.environ.get("PRAYAS_TOP_K",      "8"))
CHUNK_SIZE    = int(os.environ.get("PRAYAS_CHUNK",      "120"))  # words per tile

TILE_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────
#  KNOWLEDGE TILE
# ─────────────────────────────────────────
class KnowledgeTile:
    """
    The atomic unit of Damru's memory.
    100-120 words, self-contained, with rich metadata.
    Stored as JSONL — readable, appendable, no database.
    """
    __slots__ = ("id", "text", "topic", "domain", "source", "lang",
                 "keywords", "timestamp", "confidence", "emotion_tags")

    def __init__(self, text: str, topic: str = "", domain: str = "general",
                 source: str = "unknown", lang: str = "en",
                 keywords: List[str] = None, confidence: float = 1.0,
                 emotion_tags: List[str] = None):
        self.id         = hashlib.md5(text.encode()).hexdigest()[:16]
        self.text       = text.strip()
        self.topic      = topic
        self.domain     = domain  # science, space, tech, history, math, emotion, etc.
        self.source     = source
        self.lang       = lang
        self.keywords   = keywords or _extract_keywords(text)
        self.timestamp  = datetime.utcnow().isoformat()
        self.confidence = confidence
        self.emotion_tags = emotion_tags or []

    def to_dict(self) -> dict:
        return {s: getattr(self, s) for s in self.__slots__}

    @classmethod
    def from_dict(cls, d: dict) -> "KnowledgeTile":
        t = cls.__new__(cls)
        for s in cls.__slots__:
            setattr(t, s, d.get(s, ""))
        return t


# ─────────────────────────────────────────
#  BM25 RETRIEVER  (no GPU, pure CPU)
# ─────────────────────────────────────────
class BM25Retriever:
    """
    Classic BM25 over in-memory inverted index.
    Handles 500k tiles comfortably on CPU.
    No vector embedding needed — keyword overlap is fast and strong.
    """
    k1 = 1.5
    b  = 0.75

    def __init__(self):
        self.tiles: List[KnowledgeTile] = []
        self.inv_idx: Dict[str, List[Tuple[int, int]]] = defaultdict(list)  # term -> [(tile_idx, freq)]
        self.avg_dl  = 0.0
        self._lock   = threading.RLock()

    def add(self, tile: KnowledgeTile):
        with self._lock:
            idx = len(self.tiles)
            self.tiles.append(tile)
            words = _tokenize(tile.text + " " + tile.topic + " " + " ".join(tile.keywords))
            freq = defaultdict(int)
            for w in words: freq[w] += 1
            for w, c in freq.items():
                self.inv_idx[w].append((idx, c))
            # Running avg doc length
            n = len(self.tiles)
            self.avg_dl = (self.avg_dl * (n - 1) + len(words)) / n

    def retrieve(self, query: str, k: int = TOP_K) -> List[Tuple[float, KnowledgeTile]]:
        with self._lock:
            if not self.tiles:
                return []
            qterms = _tokenize(query)
            scores: Dict[int, float] = defaultdict(float)
            N = len(self.tiles)
            for term in qterms:
                if term not in self.inv_idx: continue
                df = len(self.inv_idx[term])
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                for idx, tf in self.inv_idx[term]:
                    dl = len(_tokenize(self.tiles[idx].text))
                    norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / max(1, self.avg_dl)))
                    scores[idx] += idf * norm
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [(s, self.tiles[i]) for i, s in ranked[:k]]

    def size(self) -> int:
        return len(self.tiles)


# ─────────────────────────────────────────
#  BLOOM FILTER  (ultra-fast dedup)
# ─────────────────────────────────────────
class BloomFilter:
    """Simple bit-array bloom filter for tile deduplication."""
    def __init__(self, capacity: int = MAX_TILES, error_rate: float = 0.01):
        self.size = int(-capacity * math.log(error_rate) / (math.log(2) ** 2))
        self.bits = bytearray(self.size // 8 + 1)
        self.num_hashes = int(self.size / capacity * math.log(2))

    def _hashes(self, item: str):
        for i in range(self.num_hashes):
            h = int(hashlib.md5(f"{i}:{item}".encode()).hexdigest(), 16)
            yield h % self.size

    def add(self, item: str):
        for pos in self._hashes(item):
            self.bits[pos // 8] |= (1 << (pos % 8))

    def __contains__(self, item: str) -> bool:
        return all(self.bits[p // 8] & (1 << (p % 8)) for p in self._hashes(item))


# ─────────────────────────────────────────
#  PRAYAS ENGINE
# ─────────────────────────────────────────
class PrayasEngine:
    """
    The main PRAYAS engine.
    Stores knowledge as tiles, retrieves with BM25,
    composes answers with a tiny local model or rule-based synthesis.
    """
    def __init__(self):
        self.retriever = BM25Retriever()
        self.bloom     = BloomFilter()
        self._loaded   = 0
        self._lock     = threading.RLock()
        log.info("PRAYAS Engine initialized (BM25 + BloomFilter)")
        threading.Thread(target=self._load_tiles, daemon=True).start()

    def _load_tiles(self):
        """Load all JSONL tile files from TILE_DIR into the retriever."""
        count = 0
        for f in sorted(TILE_DIR.glob("**/*.jsonl")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line: continue
                        try:
                            d = json.loads(line)
                            tile = KnowledgeTile.from_dict(d)
                            if tile.id not in self.bloom:
                                self.bloom.add(tile.id)
                                self.retriever.add(tile)
                                count += 1
                        except Exception:
                            pass
            except Exception as e:
                log.warning(f"Tile load error {f}: {e}")
        self._loaded = count
        log.info(f"Loaded {count} knowledge tiles from {TILE_DIR}")

    def add_tile(self, tile: KnowledgeTile) -> bool:
        """Add a tile, returns False if duplicate."""
        if tile.id in self.bloom:
            return False
        with self._lock:
            self.bloom.add(tile.id)
            self.retriever.add(tile)
            self._loaded += 1
            # Persist
            domain_dir = TILE_DIR / tile.domain
            domain_dir.mkdir(exist_ok=True)
            with open(domain_dir / f"{datetime.utcnow().strftime('%Y%m%d')}.jsonl",
                      "a", encoding="utf-8") as fh:
                fh.write(json.dumps(tile.to_dict(), ensure_ascii=False) + "\n")
            return True

    def ingest_text(self, text: str, topic: str = "", domain: str = "general",
                    source: str = "unknown", lang: str = "en") -> int:
        """Chunk text into tiles and ingest. Returns number of new tiles added."""
        words = text.split()
        chunks = [words[i:i+CHUNK_SIZE] for i in range(0, len(words), CHUNK_SIZE)]
        added = 0
        for chunk in chunks:
            if len(chunk) < 20: continue
            tile = KnowledgeTile(
                text=" ".join(chunk), topic=topic, domain=domain,
                source=source, lang=lang
            )
            if self.add_tile(tile):
                added += 1
        return added

    def search(self, query: str, k: int = TOP_K) -> List[Dict]:
        """Search tiles, return list of dicts with score and text."""
        results = self.retriever.retrieve(query, k)
        return [
            {"score": round(s, 3), "text": t.text, "topic": t.topic,
             "domain": t.domain, "source": t.source}
            for s, t in results
        ]

    def compose_answer(self, query: str, context_tiles: List[Dict]) -> str:
        """
        Rule-based answer composition from retrieved tiles.
        No LLM needed for factual questions — tiles already contain the answer.
        """
        if not context_tiles:
            return ""
        # Rank by score and pick best 3
        top = context_tiles[:3]
        # Simple extractive composition
        texts = [t["text"] for t in top]
        combined = " ".join(texts)
        # Trim to ~400 words
        words = combined.split()
        if len(words) > 400:
            combined = " ".join(words[:400]) + "..."
        return combined

    def stats(self) -> Dict:
        return {
            "total_tiles": self._loaded,
            "retriever_size": self.retriever.size(),
            "tile_dir": str(TILE_DIR),
            "domains": [d.name for d in TILE_DIR.iterdir() if d.is_dir()]
                       if TILE_DIR.exists() else [],
        }


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
_STOP = {"the","a","an","is","was","are","were","in","on","at","to","of",
         "and","or","but","it","its","this","that","for","with","from",
         "ka","ki","ke","hai","hain","ho","tha","thi","the","aur","ya"}

def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in re.findall(r'[\w\u0900-\u097F]+', text)
            if len(w) > 2 and w.lower() not in _STOP]

def _extract_keywords(text: str, n: int = 10) -> List[str]:
    words = _tokenize(text)
    freq = defaultdict(int)
    for w in words: freq[w] += 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]]


# Singleton
_ENGINE: Optional[PrayasEngine] = None
_ENGINE_LOCK = threading.Lock()

def get_engine() -> PrayasEngine:
    global _ENGINE
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = PrayasEngine()
    return _ENGINE


if __name__ == "__main__":
    e = get_engine()
    time.sleep(2)  # let tiles load
    print(e.stats())
    # Quick test
    added = e.ingest_text(
        "The Ashoka Chakra is the emblem of India, featuring 24 spokes representing "
        "the 24 hours of the day. It is derived from the dharma chakra of Ashoka the Great.",
        topic="India", domain="history", source="test"
    )
    print(f"Added {added} tiles")
    results = e.search("Ashoka Chakra India")
    for r in results:
        print(f"  [{r['score']:.2f}] {r['text'][:80]}...")
