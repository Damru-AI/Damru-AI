#!/usr/bin/env python3
"""
================================================================================
 DAMRU CURIOUS ENGINE  --  API-FREE, AUTONOMOUS SELF-LEARNING
================================================================================
Yeh engine:
  1. Khud topics choose karta hai (curious exploration)
  2. SearXNG (self-hosted) se web search karta hai -- NO API KEY NEEDED
  3. Web pages + RSS + Wikipedia padhta hai -- NO API NEEDED
  4. Local llama.cpp se Q&A generate karta hai -- NO API NEEDED
  5. Dataset HF pe push karta hai
  6. Kaggle training trigger karta hai
  7. Loop chalata rehta hai 24/7

WHY THIS IS UNLIMITED:
  - llama.cpp = local model, koi API limit nahi
  - SearXNG = self-hosted search, Google/Bing API nahi chahiye
  - Wikipedia dumps = petabytes of free data
  - RSS feeds = real-time news, no API
  - Common Crawl = entire internet, free
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
from typing import List, Dict, Optional
from datetime import datetime, timedelta

# Try imports gracefully
try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False

try:
    from huggingface_hub import HfApi, CommitOperationAdd
    _HF = True
except ImportError:
    _HF = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CURIOUS] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("curious")

# ---- Config ------------------------------------------------------------------
LLAMA_URL     = os.environ.get("LLAMA_URL", "http://localhost:8080")
SEARXNG_URL   = os.environ.get("SEARXNG_URL", "http://localhost:8888")
HF_TOKEN      = os.environ.get("HF_TOKEN", "")
HF_DATASET    = os.environ.get("HF_DATASET", "Damaru-ai/damru-knowledge")
KAGGLE_TRIGGER_INTERVAL = int(os.environ.get("KAGGLE_TRIGGER_H", "12")) * 3600

DATA_DIR  = Path("/opt/damru/data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
SEEN_FILE = DATA_DIR / "seen_urls.json"
STATE_FILE = DATA_DIR / "engine_state.json"

# ---- Damru's Curiosity Curriculum (no API needed) ----------------------------
# Ye topics hain jo Damru khud explore karta hai -- automatically expand hote hain
BASE_TOPICS = [
    # Science & Tech
    "quantum computing", "CRISPR gene editing", "nuclear fusion reactor 2024",
    "large language model training", "reinforcement learning from human feedback",
    "transformer architecture explained", "diffusion models stable diffusion",
    # India-specific (Damru is Bhartiya AI!)
    "ISRO space mission 2024", "Indian startup ecosystem", "Bharat AI research",
    "IIT research papers 2024", "DRDO technology defense", "India manufacturing",
    # Math & Reasoning
    "competitive math olympiad problems", "graph theory algorithms",
    "probability statistics puzzles", "linear algebra deep learning",
    # General Knowledge
    "world history major events", "economics monetary policy",
    "climate change solutions", "renewable energy solar wind",
    # Current Affairs (via RSS)
    "latest AI research arxiv", "technology news 2024",
]

# Free RSS feeds (no API, no limit)
RSS_FEEDS = [
    "https://arxiv.org/rss/cs.AI",
    "https://arxiv.org/rss/cs.LG",
    "https://feeds.feedburner.com/TechCrunch",
    "https://www.theverge.com/rss/index.xml",
    "https://rss.nytimes.com/services/xml/rss/nyt/Technology.xml",
    "https://timesofindia.indiatimes.com/rssfeedstopstories.cms",
    "https://www.thehindu.com/sci-tech/feeder/default.rss",
    "https://economictimes.indiatimes.com/tech/rssfeeds/13357270.cms",
]

# Free Wikipedia entry points
WIKI_TOPICS_URL = "https://en.wikipedia.org/wiki/Special:Random"
WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"

# ---- Seen URL tracker --------------------------------------------------------
class SeenTracker:
    def __init__(self):
        self.seen = set()
        if SEEN_FILE.exists():
            try:
                data = json.loads(SEEN_FILE.read_text())
                self.seen = set(data)
            except Exception:
                pass

    def add(self, url: str):
        h = hashlib.md5(url.encode()).hexdigest()
        self.seen.add(h)
        if len(self.seen) % 100 == 0:
            SEEN_FILE.write_text(json.dumps(list(self.seen)))

    def has(self, url: str) -> bool:
        h = hashlib.md5(url.encode()).hexdigest()
        return h in self.seen


SEEN = SeenTracker()


# ---- Local llama.cpp inference (NO API KEY) ----------------------------------
class LocalLLM:
    """
    Calls our own llama.cpp server.
    Cost: ZERO. Limit: NONE. Speed: Fast on Oracle ARM.
    """
    def __init__(self, base_url: str = LLAMA_URL):
        self.url = base_url
        self._available = None

    def is_available(self) -> bool:
        try:
            r = requests.get(f"{self.url}/health", timeout=5)
            self._available = r.status_code == 200
        except Exception:
            self._available = False
        return self._available

    def chat(self, messages: List[Dict], max_tokens: int = 1024,
             temperature: float = 0.7) -> Optional[str]:
        """
        OpenAI-compatible chat endpoint from llama.cpp.
        Works 100% locally -- no internet, no API, no limit.
        """
        try:
            r = requests.post(
                f"{self.url}/v1/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": False,
                },
                timeout=120,
            )
            if r.status_code == 200:
                return (r.json()["choices"][0]["message"]["content"] or "").strip()
        except Exception as e:
            log.warning(f"llama.cpp call failed: {e}")
        return None

    def generate_qa_from_text(self, text: str, topic: str,
                               n_pairs: int = 5) -> List[Dict]:
        """
        Damru reads text -> generates its own Q&A training data.
        This is self-distillation: the model teaches itself!
        NO external API needed.
        """
        if len(text) < 200:
            return []
        # Trim text to avoid context overflow
        text_chunk = text[:3000]
        prompt = f"""You are an expert teacher. Read this text about {topic} and generate {n_pairs} high-quality question-answer pairs for training an AI.

TEXT:
{text_chunk}

Generate exactly {n_pairs} Q&A pairs. Format:
Q: <question>
A: <detailed answer>

Make questions diverse: factual, conceptual, analytical, application-based."""

        response = self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=1500,
            temperature=0.8,
        )
        if not response:
            return []

        pairs = []
        blocks = re.split(r'\n(?=Q:)', response.strip())
        for block in blocks:
            lines = block.strip().splitlines()
            q_lines, a_lines = [], []
            in_a = False
            for line in lines:
                if line.startswith("Q:"):
                    q_lines.append(line[2:].strip())
                elif line.startswith("A:"):
                    in_a = True
                    a_lines.append(line[2:].strip())
                elif in_a:
                    a_lines.append(line.strip())
            if q_lines and a_lines:
                q = " ".join(q_lines).strip()
                a = " ".join(a_lines).strip()
                if q and a and len(a) > 20:
                    pairs.append({
                        "instruction": q,
                        "output": a,
                        "topic": topic,
                        "source": "self_generated",
                        "timestamp": datetime.utcnow().isoformat(),
                    })
        return pairs[:n_pairs]

    def expand_topics(self, current_topics: List[str]) -> List[str]:
        """
        Damru khud naye topics sochta hai -- self-directed curiosity!
        """
        sample = random.sample(current_topics, min(5, len(current_topics)))
        prompt = f"""Based on these topics I've been learning: {', '.join(sample)}

Suggest 10 NEW related topics I should explore next. Be creative and specific.
Return only a JSON list of strings."""

        response = self.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=300, temperature=1.0,
        )
        if not response:
            return []
        try:
            match = re.search(r'\[.*?\]', response, re.DOTALL)
            if match:
                new_topics = json.loads(match.group())
                log.info(f"Damru discovered {len(new_topics)} new topics!")
                return [str(t) for t in new_topics]
        except Exception:
            pass
        return []


# ---- Free Data Sources (no API key) ------------------------------------------
class FreeWebScraper:
    """
    Uses only FREE, unlimited data sources:
    - RSS feeds (no API)
    - Wikipedia (no API)
    - SearXNG self-hosted search (no API)
    - Direct web scraping (wget style)
    """
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (compatible; DamruBot/1.0; +https://damru.ai)"
    }

    def fetch_rss(self) -> List[Dict]:
        """RSS feeds -- completely free, real-time, no API."""
        articles = []
        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]:
                    url = entry.get("link", "")
                    if not url or SEEN.has(url):
                        continue
                    title = entry.get("title", "")
                    summary = entry.get("summary", "")
                    if title and len(summary) > 100:
                        articles.append({
                            "url": url, "title": title,
                            "text": f"{title}. {summary}",
                            "source": "rss",
                        })
                        SEEN.add(url)
            except Exception as e:
                log.debug(f"RSS failed {feed_url}: {e}")
        log.info(f"RSS: got {len(articles)} new articles")
        return articles

    def fetch_wikipedia(self, topic: str) -> Optional[Dict]:
        """Wikipedia API -- free, no key."""
        try:
            topic_url = topic.replace(" ", "_")
            r = requests.get(
                f"{WIKI_API}{requests.utils.quote(topic_url)}",
                headers=self.HEADERS, timeout=15,
            )
            if r.status_code == 200:
                data = r.json()
                text = data.get("extract", "")
                url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
                if text and len(text) > 200 and not SEEN.has(url):
                    SEEN.add(url)
                    return {
                        "url": url,
                        "title": data.get("title", topic),
                        "text": text,
                        "source": "wikipedia",
                    }
        except Exception as e:
            log.debug(f"Wikipedia failed {topic}: {e}")
        return None

    def search_searxng(self, query: str, n: int = 5) -> List[Dict]:
        """SearXNG self-hosted -- replaces Google API, completely free."""
        try:
            r = requests.get(
                f"{SEARXNG_URL}/search",
                params={"q": query, "format": "json", "categories": "general"},
                timeout=15,
            )
            if r.status_code != 200:
                return []
            results = r.json().get("results", [])[:n]
            out = []
            for res in results:
                url = res.get("url", "")
                if SEEN.has(url):
                    continue
                content = res.get("content", "") or res.get("title", "")
                if len(content) > 80:
                    out.append({
                        "url": url,
                        "title": res.get("title", query),
                        "text": content,
                        "source": "searxng",
                    })
                    SEEN.add(url)
            return out
        except Exception as e:
            log.debug(f"SearXNG failed: {e}")
        return []

    def scrape_url(self, url: str) -> Optional[str]:
        """Direct scrape -- no API needed."""
        if not _BS4:
            return None
        try:
            r = requests.get(url, headers=self.HEADERS, timeout=20)
            if r.status_code == 200:
                soup = BeautifulSoup(r.content, "html.parser")
                # Remove nav, ads, scripts
                for tag in soup(["script", "style", "nav", "header",
                                  "footer", "aside", "ads"]):
                    tag.decompose()
                text = soup.get_text(separator=" ", strip=True)
                # Trim whitespace
                text = re.sub(r'\s+', ' ', text).strip()
                return text[:5000] if len(text) > 200 else None
        except Exception:
            pass
        return None


# ---- HF Dataset Pusher -------------------------------------------------------
class HFPusher:
    def __init__(self):
        self.buffer = []
        self.buffer_file = DATA_DIR / "pending_data.jsonl"
        self.pushed_count = 0
        # Load existing buffer
        if self.buffer_file.exists():
            try:
                for line in self.buffer_file.read_text().splitlines():
                    if line.strip():
                        self.buffer.append(json.loads(line))
            except Exception:
                pass

    def add(self, records: List[Dict]):
        self.buffer.extend(records)
        # Append to file
        with open(self.buffer_file, "a") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.info(f"Buffer: {len(self.buffer)} records")

    def push_if_ready(self, min_records: int = 500) -> bool:
        """Push to HF when enough data collected."""
        if len(self.buffer) < min_records or not HF_TOKEN or not _HF:
            return False
        try:
            api = HfApi(token=HF_TOKEN)
            fname = f"curious/auto_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.jsonl"
            content = "\n".join(json.dumps(r, ensure_ascii=False)
                                 for r in self.buffer)
            api.upload_file(
                path_or_fileobj=content.encode(),
                path_in_repo=fname,
                repo_id=HF_DATASET,
                repo_type="dataset",
                commit_message=f"curious-engine: {len(self.buffer)} records",
            )
            log.info(f"Pushed {len(self.buffer)} records to HF: {fname}")
            self.pushed_count += len(self.buffer)
            self.buffer = []
            self.buffer_file.write_text("")  # clear
            return True
        except Exception as e:
            log.error(f"HF push failed: {e}")
        return False


# ---- Engine State ------------------------------------------------------------
class EngineState:
    def __init__(self):
        self.topics = list(BASE_TOPICS)
        self.cycle = 0
        self.total_qa = 0
        self.last_kaggle_trigger = 0
        self.load()

    def load(self):
        if STATE_FILE.exists():
            try:
                d = json.loads(STATE_FILE.read_text())
                self.topics  = d.get("topics", self.topics)
                self.cycle   = d.get("cycle", 0)
                self.total_qa= d.get("total_qa", 0)
                self.last_kaggle_trigger = d.get("last_kaggle_trigger", 0)
            except Exception:
                pass

    def save(self):
        STATE_FILE.write_text(json.dumps({
            "topics": self.topics,
            "cycle": self.cycle,
            "total_qa": self.total_qa,
            "last_kaggle_trigger": self.last_kaggle_trigger,
        }, indent=2))


# ---- Main Loop ---------------------------------------------------------------
def main():
    log.info("=" * 60)
    log.info(" DAMRU CURIOUS ENGINE -- API-FREE AUTONOMOUS LEARNING")
    log.info("=" * 60)
    log.info(f"LLaMA server: {LLAMA_URL}")
    log.info(f"SearXNG:      {SEARXNG_URL}")
    log.info(f"HF Dataset:   {HF_DATASET}")

    llm     = LocalLLM(LLAMA_URL)
    scraper = FreeWebScraper()
    pusher  = HFPusher()
    state   = EngineState()

    # Wait for llama.cpp to be ready
    log.info("Waiting for local LLM to be ready...")
    for attempt in range(30):
        if llm.is_available():
            log.info("Local LLM ready! Starting curious learning loop.")
            break
        time.sleep(10)
    else:
        log.warning("LLM not ready after 5 min. Running in data-collection-only mode.")

    CYCLE_SLEEP = 300  # 5 min between cycles

    while True:
        state.cycle += 1
        cycle_start = time.time()
        log.info(f"\n{'='*50}")
        log.info(f"CYCLE {state.cycle} | Topics: {len(state.topics)} | "
                 f"Total QA: {state.total_qa}")
        log.info(f"{'='*50}")

        new_qa = []

        # --- Step 1: RSS (always fresh, no API) ---
        articles = scraper.fetch_rss()
        for article in articles[:3]:
            if llm.is_available():
                pairs = llm.generate_qa_from_text(article["text"], article["title"])
                new_qa.extend(pairs)
                log.info(f"  RSS: '{article['title'][:50]}' -> {len(pairs)} Q&A")

        # --- Step 2: Wikipedia (free, infinite) ---
        topic = random.choice(state.topics)
        wiki = scraper.fetch_wikipedia(topic)
        if wiki:
            # Try to get full article
            full_text = scraper.scrape_url(wiki["url"]) or wiki["text"]
            if llm.is_available():
                pairs = llm.generate_qa_from_text(full_text, topic, n_pairs=8)
                new_qa.extend(pairs)
                log.info(f"  Wiki: '{topic}' -> {len(pairs)} Q&A")

        # --- Step 3: SearXNG search (self-hosted, no API) ---
        search_topic = random.choice(state.topics)
        results = scraper.search_searxng(search_topic, n=3)
        for res in results:
            full_text = scraper.scrape_url(res["url"]) or res["text"]
            if full_text and llm.is_available():
                pairs = llm.generate_qa_from_text(full_text, search_topic, n_pairs=5)
                new_qa.extend(pairs)
                log.info(f"  Search: '{search_topic}' -> {len(pairs)} Q&A")

        # --- Step 4: Add to buffer & push ---
        if new_qa:
            pusher.add(new_qa)
            state.total_qa += len(new_qa)
            state.save()
            log.info(f"  Cycle {state.cycle}: +{len(new_qa)} Q&A | "
                     f"Total: {state.total_qa} | "
                     f"Buffer: {len(pusher.buffer)}")

        # Push to HF when 500+ records ready
        if pusher.push_if_ready(500):
            log.info(f"  Pushed to HF! Total pushed: {pusher.pushed_count}")

        # --- Step 5: Expand topics every 10 cycles ---
        if state.cycle % 10 == 0 and llm.is_available():
            new_topics = llm.expand_topics(state.topics)
            if new_topics:
                # Add only truly new topics
                existing = set(state.topics)
                added = [t for t in new_topics if t not in existing]
                state.topics.extend(added)
                # Cap to 500 topics
                if len(state.topics) > 500:
                    state.topics = state.topics[-500:]
                state.save()
                log.info(f"  Topic expansion: +{len(added)} new topics! "
                         f"Total: {len(state.topics)}")

        # --- Step 6: Kaggle trigger (every 12h) ---
        if (time.time() - state.last_kaggle_trigger) > KAGGLE_TRIGGER_INTERVAL:
            try:
                import subprocess
                log.info("  Triggering Kaggle training run...")
                subprocess.Popen([
                    "/opt/damru/venv/bin/python3",
                    "/opt/damru/kaggle_trigger.py",
                ])
                state.last_kaggle_trigger = time.time()
                state.save()
            except Exception as e:
                log.warning(f"  Kaggle trigger failed: {e}")

        # Sleep until next cycle
        elapsed = time.time() - cycle_start
        sleep_time = max(0, CYCLE_SLEEP - elapsed)
        log.info(f"  Cycle done in {elapsed:.0f}s. Next in {sleep_time:.0f}s...")
        time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log.info("Curious Engine stopped.")
