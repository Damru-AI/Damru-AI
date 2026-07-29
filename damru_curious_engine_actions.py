#!/usr/bin/env python3
"""
================================================================================
 DAMRU CURIOUS ENGINE -- GITHUB ACTIONS VERSION
================================================================================
Yeh GitHub Actions pe chalti hai:
  - Koi server nahi chahiye
  - Koi credit card nahi chahiye
  - Public repo pe UNLIMITED free runs
  - Har 6 ghante automatically chalti hai
  - Wikipedia + RSS se data collect karti hai
  - HuggingFace pe dataset push karti hai
  - Damru LLM nahi hai yahan -- smart text processing se Q&A banata hai

Special Features vs original:
  - No llama.cpp (no GPU) -- rule-based Q&A extraction
  - No SearXNG -- direct web scraping
  - MAX_CYCLES se limited run -- Actions timeout se pehle khatam hoga
  - Stats file /tmp/damru_stats.txt mein
================================================================================
"""
import os
import re
import json
import time
import random
import hashlib
import logging
import requests
import feedparser
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False

try:
    from huggingface_hub import HfApi
    _HF = True
except ImportError:
    _HF = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CURIOUS-ACTIONS] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()

# Config
HF_TOKEN   = os.environ.get("HF_TOKEN", "")
HF_DATASET = os.environ.get("HF_DATASET", "Damaru-ai/damru-knowledge")
MAX_CYCLES = int(os.environ.get("MAX_CYCLES", "20"))
STATS_FILE = Path("/tmp/damru_stats.txt")

# Topics Damru ko sikhne chahiye
TOPICS = [
    # Science
    "quantum entanglement", "CRISPR applications medicine", "nuclear fusion ITER 2024",
    "transformer neural network architecture", "reinforcement learning AlphaGo",
    "GPT large language model training", "diffusion model image generation",
    "protein folding AlphaFold", "climate change renewable energy",
    # India
    "ISRO Chandrayaan mission", "Indian Space Research Organisation history",
    "IIT research artificial intelligence", "India GDP economy growth 2024",
    "Bharat startup unicorn", "India renewable energy solar",
    "Sanskrit mathematics ancient India", "Aryabhata mathematician",
    # Math/Reasoning
    "Fermat Last Theorem proof", "Riemann hypothesis mathematics",
    "graph theory shortest path algorithm", "probability Bayes theorem",
    "linear algebra eigenvalues", "calculus gradient descent",
    "combinatorics counting problems", "number theory prime numbers",
    # Technology
    "blockchain decentralized technology", "quantum computing qubit",
    "edge computing IoT devices", "5G network technology India",
    "cybersecurity encryption RSA", "operating system kernel Linux",
    "compiler design programming language", "database indexing B-tree",
    # General Knowledge
    "World War history causes effects", "French Revolution causes",
    "economics supply demand market", "psychology cognitive biases",
    "philosophy Socrates Plato", "biology cell division mitosis",
    "chemistry periodic table elements", "physics relativity Einstein",
]

RSS_FEEDS = [
    "https://arxiv.org/rss/cs.AI",
    "https://arxiv.org/rss/cs.LG",
    "https://arxiv.org/rss/cs.CL",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://www.thehindu.com/sci-tech/feeder/default.rss",
]

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
HEADERS  = {"User-Agent": "DamruBot/1.0 (educational AI; contact: damruai@gmail.com)"}


# ---- Smart Q&A extraction (no LLM needed!) -----------------------------------
def extract_qa_from_text(text: str, topic: str, n: int = 5) -> List[Dict]:
    """
    Rule-based Q&A extraction -- no LLM, no API.
    Turns any text into training pairs using patterns.
    """
    if len(text) < 150:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 50]
    if len(sentences) < 3:
        return []

    pairs = []

    # Pattern 1: Definition questions
    for sent in sentences[:20]:
        # "X is a Y" -> "What is X?"
        m = re.match(r'^([A-Z][\w\s]{2,40}) (?:is|are|was|were) ([^.]{20,})\.?$', sent)
        if m:
            subj = m.group(1).strip()
            defn = sent
            if len(subj.split()) <= 5:
                pairs.append({
                    "instruction": f"What is {subj}?",
                    "output": defn,
                })
                if len(pairs) >= n: break

    # Pattern 2: First sentence = topic definition
    if sentences and len(pairs) < n:
        intro = " ".join(sentences[:2])
        pairs.append({
            "instruction": f"Explain {topic} in simple terms.",
            "output": intro,
        })

    # Pattern 3: Key facts
    fact_patterns = [
        r'(\d{4})[,\s].*?(discovered|invented|founded|established|launched)',
        r'(first|largest|smallest|fastest|oldest|newest)[\s\w]+',
        r'(approximately|about|around|nearly)[\s\d,]+',
    ]
    for pat in fact_patterns:
        for sent in sentences:
            if re.search(pat, sent, re.I) and len(sent) > 60:
                words = sent.split()[:6]
                q_start = " ".join(words)
                pairs.append({
                    "instruction": f"Tell me about: {q_start}...",
                    "output": sent,
                })
                if len(pairs) >= n: break
        if len(pairs) >= n: break

    # Pattern 4: Hinglish version (Damru is Bhartiya AI!)
    if sentences and len(pairs) < n:
        pairs.append({
            "instruction": f"{topic} ke baare mein batao.",
            "output": f"{topic} ke baare mein: {sentences[0]}",
        })

    # Add metadata
    final = []
    for p in pairs[:n]:
        if p.get("instruction") and p.get("output") and len(p["output"]) > 30:
            final.append({
                "instruction": p["instruction"],
                "output": p["output"],
                "topic": topic,
                "source": "github_actions_curious",
                "timestamp": datetime.utcnow().isoformat(),
            })
    return final


# ---- Data Sources ------------------------------------------------------------
def fetch_wikipedia(topic: str) -> Optional[Dict]:
    try:
        r = requests.get(
            f"{WIKI_API}{requests.utils.quote(topic.replace(' ', '_'))}",
            headers=HEADERS, timeout=15
        )
        if r.status_code == 200:
            d = r.json()
            text = d.get("extract", "")
            if text and len(text) > 200:
                return {"title": d.get("title", topic), "text": text, "source": "wikipedia"}
    except Exception as e:
        log.debug(f"Wikipedia {topic}: {e}")
    return None


def fetch_rss() -> List[Dict]:
    articles = []
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                title   = entry.get("title", "")
                summary = entry.get("summary", "")
                if title and len(summary) > 100:
                    articles.append({
                        "title": title,
                        "text": f"{title}. {summary}",
                        "source": "rss",
                    })
        except Exception:
            pass
    return articles


def fetch_wikipedia_sections(topic: str) -> Optional[str]:
    """Get longer Wikipedia article text via HTML scraping."""
    if not _BS4:
        return None
    try:
        url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(topic.replace(' ', '_'))}"
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code == 200:
            soup = BeautifulSoup(r.content, "html.parser")
            # Remove nav, infoboxes, etc.
            for tag in soup.select(".navbox, .infobox, .reflist, #References, .mw-editsection"):
                tag.decompose()
            # Get first 3 paragraphs
            paras = [p.get_text(strip=True) for p in soup.select("#mw-content-text p") if len(p.get_text(strip=True)) > 100]
            return " ".join(paras[:5])[:4000] if paras else None
    except Exception:
        pass
    return None


# ---- HF Pusher ---------------------------------------------------------------
class HFPusher:
    def __init__(self):
        self.buffer = []

    def add(self, records: List[Dict]):
        self.buffer.extend(records)

    def push(self) -> bool:
        if not self.buffer or not HF_TOKEN or not _HF:
            log.info(f"Buffer: {len(self.buffer)} records (not pushed: token={bool(HF_TOKEN)} hf={_HF})")
            return False
        try:
            api = HfApi(token=HF_TOKEN)
            fname = f"curious/actions_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
            content = "\n".join(json.dumps(r, ensure_ascii=False) for r in self.buffer)
            api.upload_file(
                path_or_fileobj=content.encode("utf-8"),
                path_in_repo=fname,
                repo_id=HF_DATASET,
                repo_type="dataset",
                commit_message=f"GitHub Actions curious: {len(self.buffer)} Q&A pairs",
            )
            log.info(f"Pushed {len(self.buffer)} records to HF: {fname}")
            return True
        except Exception as e:
            log.error(f"HF push failed: {e}")
        return False


# ---- Main --------------------------------------------------------------------
def main():
    start = time.time()
    log.info("=" * 55)
    log.info(" DAMRU CURIOUS ENGINE -- GitHub Actions Mode")
    log.info(f" Max cycles: {MAX_CYCLES} | Dataset: {HF_DATASET}")
    log.info("=" * 55)

    pusher  = HFPusher()
    topics  = list(TOPICS)
    random.shuffle(topics)

    total_qa = 0
    topic_idx = 0

    for cycle in range(1, MAX_CYCLES + 1):
        log.info(f"--- Cycle {cycle}/{MAX_CYCLES} ---")
        new_qa = []

        # Wikipedia (main source)
        topic = topics[topic_idx % len(topics)]
        topic_idx += 1

        wiki = fetch_wikipedia(topic)
        if wiki:
            # Try to get fuller text
            full_text = fetch_wikipedia_sections(topic) or wiki["text"]
            pairs = extract_qa_from_text(full_text, topic, n=6)
            new_qa.extend(pairs)
            log.info(f"  Wiki '{topic}': {len(pairs)} Q&A")
        else:
            log.info(f"  Wiki '{topic}': not found")

        # RSS (every 3 cycles)
        if cycle % 3 == 0:
            articles = fetch_rss()
            for art in articles[:2]:
                pairs = extract_qa_from_text(art["text"], art["title"], n=3)
                new_qa.extend(pairs)
            if articles:
                log.info(f"  RSS: {len(articles)} articles -> Q&A")

        if new_qa:
            pusher.add(new_qa)
            total_qa += len(new_qa)
            log.info(f"  Total Q&A this run: {total_qa}")

        time.sleep(2)  # polite delay

    # Final push
    log.info("\nFinal push to HuggingFace...")
    success = pusher.push()

    elapsed = time.time() - start
    stats = (
        f"Run complete at {datetime.utcnow().isoformat()}\n"
        f"Cycles: {MAX_CYCLES}\n"
        f"Total Q&A pairs: {total_qa}\n"
        f"Pushed to HF: {success}\n"
        f"Elapsed: {elapsed/60:.1f} min\n"
    )
    log.info("\n" + stats)
    STATS_FILE.write_text(stats)

    log.info("=" * 55)
    log.info(f" DONE! {total_qa} Q&A pairs -> {HF_DATASET}")
    log.info("=" * 55)


if __name__ == "__main__":
    main()
