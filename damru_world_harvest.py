#!/usr/bin/env python3
"""
================================================================================
  DAMRU WORLD HARVEST v1.0
================================================================================
Sabse bada data engine — Damru ke liye puri dharti ka data:

  TIER 1: Free, Unlimited
    * Wikipedia  (200+ languages, 60M+ articles)
    * arXiv      (2M+ research papers: AI, physics, space, math, bio)
    * GitHub     (public repos: trending, topics, README, code)
    * NASA Open Data (space, missions, exoplanets, ISS, Mars)
    * ISRO data  (Indian space missions)
    * PubMed     (medical research)
    * OpenLibrary(books, literature)
    * Gutenberg  (50k+ free books)
    * Stack Overflow (public Q&A API)
    * Reddit     (public pushshift/API)
    * NCBI       (biology, genetics)
    * Common Crawl subset (web crawl data)
    * OpenStreetMap (geography)
    * UN Data    (world statistics)
    * ESA        (European space agency)

  TIER 2: HF Datasets (Sunil's own datasets)
    * All Damaru-ai/* datasets on HuggingFace
    * Community datasets: math, code, science, multilingual

  TIER 3: GitHub Public Repos
    * Trending repos (daily)
    * Topics: AI, ML, robotics, space, defense, automotive
    * README + code snippets -> knowledge tiles

All data -> PRAYAS Knowledge Tiles (JSONL)
GitHub Actions: runs every 6 hours, self-heals on error
================================================================================
"""
import os
import re
import json
import time
import random
import logging
import requests
import feedparser
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HARVEST] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()

# --- Config ---
HF_TOKEN     = os.environ.get("HF_TOKEN", "")
GH_TOKEN     = os.environ.get("GH_TOKEN", HF_TOKEN)
HF_DATASET   = os.environ.get("HF_DATASET", "Damaru-ai/damru-knowledge")
HF_OWNER     = os.environ.get("HF_OWNER",  "Damaru-ai")
MAX_TILES_PER_RUN = int(os.environ.get("MAX_TILES", "2000"))
RUN_MODE     = os.environ.get("RUN_MODE", "github_actions")
STATS_FILE   = Path("/tmp/harvest_stats.txt")

HEADERS      = {"User-Agent": "DamruBot/2.0 (educational AI; damruai@gmail.com)"}
GH_HEADERS   = {"Authorization": f"Bearer {GH_TOKEN}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "DamruBot/2.0"} if GH_TOKEN else HEADERS


# ============================================================
#  DATA SOURCES
# ============================================================

class WikipediaSource:
    """Wikipedia: 60M+ articles, 200+ languages."""
    TOPICS = [
        # Space & Astronomy
        "space exploration", "International Space Station", "Mars rover",
        "black hole", "neutron star", "exoplanet", "James Webb Space Telescope",
        "Chandrayaan mission", "ISRO", "NASA history", "SpaceX Falcon 9",
        "lunar colonization", "asteroid mining", "solar system formation",
        # AI & Technology
        "artificial intelligence", "neural network", "transformer model",
        "reinforcement learning", "computer vision", "natural language processing",
        "quantum computing", "robotics automation", "autonomous vehicle",
        "drone technology", "5G network", "edge computing",
        # Science
        "CRISPR gene editing", "protein folding", "climate change",
        "renewable energy", "nuclear fusion", "quantum entanglement",
        "dark matter", "gravitational waves", "particle physics",
        # India
        "Indian Space Research Organisation", "IIT research",
        "India economy 2024", "Bharat startup ecosystem",
        "Aryabhata mathematician", "Indian classical mathematics",
        "Sanskrit grammar Panini", "Vedic astronomy",
        # Math & Engineering
        "Riemann hypothesis", "P versus NP problem", "Fourier transform",
        "Bayesian inference", "linear algebra applications",
        "differential equations engineering", "graph theory",
        # Defense & Military
        "military drone technology", "hypersonic missile",
        "electronic warfare", "stealth technology aircraft",
        "cyber warfare", "defense AI applications",
        # Automotive & Transport
        "self-driving car technology", "V2X communication",
        "electric vehicle battery", "air taxi eVTOL",
        "hyperloop transportation", "hydrogen fuel cell",
        # Medical
        "precision medicine", "mRNA vaccine technology",
        "brain-computer interface", "surgical robotics",
        # Human Behavior
        "cognitive psychology", "emotional intelligence",
        "behavioral economics", "social psychology",
        "animal cognition", "dog behavior psychology",
    ]

    def fetch(self, topic: str) -> Optional[Dict]:
        try:
            r = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + requests.utils.quote(topic.replace(" ", "_")),
                headers=HEADERS, timeout=15
            )
            if r.status_code == 200:
                d = r.json()
                text = d.get("extract", "")
                if text and len(text) > 200:
                    return {"text": text, "title": d.get("title", topic),
                            "url": d.get("content_urls", {}).get("desktop", {}).get("page", ""),
                            "domain": self._classify(topic), "source": "wikipedia"}
        except Exception as e:
            log.debug(f"Wiki {topic}: {e}")
        return None

    def _classify(self, topic: str) -> str:
        t = topic.lower()
        if any(w in t for w in ["space","nasa","isro","planet","star","orbit","moon","mars"]):
            return "space"
        if any(w in t for w in ["ai","neural","robot","autonomous","drone"]):
            return "tech"
        if any(w in t for w in ["india","isro","sanskrit","vedic","bharat"]):
            return "india"
        if any(w in t for w in ["math","algebra","calculus","equation"]):
            return "math"
        if any(w in t for w in ["medical","medicine","vaccine","brain"]):
            return "medical"
        if any(w in t for w in ["military","defense","missile","warfare"]):
            return "defense"
        if any(w in t for w in ["car","vehicle","transport","taxi"]):
            return "automotive"
        if any(w in t for w in ["psychology","behavior","emotion","cognitive"]):
            return "psychology"
        return "science"

    def batch(self, n: int = 30) -> List[Dict]:
        topics = random.sample(self.TOPICS, min(n, len(self.TOPICS)))
        results = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {ex.submit(self.fetch, t): t for t in topics}
            for f in as_completed(futures):
                r = f.result()
                if r: results.append(r)
                time.sleep(0.3)
        return results


class ArXivSource:
    """arXiv: latest AI, physics, space, math, bio research."""
    FEEDS = [
        ("https://arxiv.org/rss/cs.AI",    "tech"),
        ("https://arxiv.org/rss/cs.LG",    "tech"),
        ("https://arxiv.org/rss/cs.RO",    "robotics"),
        ("https://arxiv.org/rss/astro-ph",  "space"),
        ("https://arxiv.org/rss/physics",   "science"),
        ("https://arxiv.org/rss/math.CO",   "math"),
        ("https://arxiv.org/rss/q-bio.NC",  "medical"),
        ("https://arxiv.org/rss/cs.NI",    "tech"),  # networking
        ("https://arxiv.org/rss/eess.SY",  "automotive"),  # control systems
    ]

    def batch(self, articles_per_feed: int = 5) -> List[Dict]:
        results = []
        for url, domain in self.FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:articles_per_feed]:
                    title   = entry.get("title", "").strip()
                    summary = entry.get("summary", "").strip()
                    link    = entry.get("link", "")
                    if title and len(summary) > 100:
                        text = f"{title}. {summary}"
                        results.append({"text": text, "title": title,
                                        "url": link, "domain": domain,
                                        "source": "arxiv"})
            except Exception as e:
                log.debug(f"arXiv {url}: {e}")
        return results


class GitHubSource:
    """GitHub public repos: trending, topics. README -> knowledge tiles."""
    TOPICS = [
        "artificial-intelligence", "machine-learning", "deep-learning",
        "robotics", "space", "astronomy", "autonomous-vehicles",
        "computer-vision", "nlp", "defense", "drone", "medical-imaging",
        "quantum-computing", "3d-printing", "cad", "simulation",
        "reinforcement-learning", "neural-network", "generative-ai",
    ]

    def fetch_trending(self) -> List[Dict]:
        """Get trending repos (public API via GitHub search)."""
        results = []
        yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
        try:
            r = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": f"created:>{yesterday} stars:>5",
                        "sort": "stars", "order": "desc", "per_page": 20},
                headers=GH_HEADERS, timeout=20
            )
            if r.ok:
                for repo in r.json().get("items", []):
                    desc = repo.get("description") or ""
                    name = repo.get("full_name", "")
                    lang = repo.get("language") or "code"
                    topics = repo.get("topics", [])
                    if desc and len(desc) > 30:
                        text = (f"GitHub Repository: {name}. "
                                f"Description: {desc}. "
                                f"Language: {lang}. "
                                f"Topics: {', '.join(topics[:5])}.")
                        results.append({"text": text, "title": name,
                                        "url": repo.get("html_url", ""),
                                        "domain": "tech", "source": "github"})
        except Exception as e:
            log.debug(f"GitHub trending: {e}")
        return results

    def fetch_topic(self, topic: str) -> List[Dict]:
        results = []
        try:
            r = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": f"topic:{topic} stars:>50",
                        "sort": "stars", "per_page": 10},
                headers=GH_HEADERS, timeout=20
            )
            if r.ok:
                for repo in r.json().get("items", []):
                    desc = repo.get("description") or ""
                    if desc and len(desc) > 20:
                        text = (f"GitHub {topic} project: {repo['full_name']}. "
                                f"{desc}. Stars: {repo.get('stargazers_count',0)}.")
                        results.append({"text": text, "title": repo["full_name"],
                                        "url": repo.get("html_url", ""),
                                        "domain": "tech", "source": f"github/{topic}"})
        except Exception as e:
            log.debug(f"GitHub topic {topic}: {e}")
        return results

    def batch(self) -> List[Dict]:
        results = self.fetch_trending()
        topics = random.sample(self.TOPICS, 5)
        for t in topics:
            results.extend(self.fetch_topic(t))
            time.sleep(0.5)
        return results


class NASASource:
    """NASA Open Data API: space missions, exoplanets, APOD, Mars."""

    def fetch_apod(self) -> Optional[Dict]:
        """Astronomy Picture of the Day."""
        try:
            r = requests.get(
                "https://api.nasa.gov/planetary/apod",
                params={"api_key": "DEMO_KEY", "count": 5},
                headers=HEADERS, timeout=15
            )
            if r.ok:
                results = []
                for item in r.json():
                    text = f"{item.get('title','')}: {item.get('explanation','')}"
                    if len(text) > 100:
                        results.append({"text": text[:2000],
                                        "title": item.get("title", "APOD"),
                                        "url": item.get("url", ""),
                                        "domain": "space", "source": "nasa_apod"})
                return results
        except Exception as e:
            log.debug(f"NASA APOD: {e}")
        return []

    def fetch_mars_weather(self) -> List[Dict]:
        try:
            r = requests.get(
                "https://api.nasa.gov/insight_weather/",
                params={"api_key": "DEMO_KEY", "feedtype": "json", "ver": "1.0"},
                headers=HEADERS, timeout=15
            )
            if r.ok:
                d = r.json()
                sols = d.get("sol_keys", [])
                results = []
                for sol in sols[:3]:
                    data = d.get(sol, {})
                    temp = data.get("AT", {})
                    text = (f"Mars InSight weather Sol {sol}: "
                            f"Temperature avg {temp.get('av','?')}°C, "
                            f"min {temp.get('mn','?')}°C, max {temp.get('mx','?')}°C.")
                    results.append({"text": text, "title": f"Mars Weather Sol {sol}",
                                    "url": "https://mars.nasa.gov/insight/",
                                    "domain": "space", "source": "nasa_mars"})
                return results
        except Exception as e:
            log.debug(f"NASA Mars: {e}")
        return []

    def fetch_exoplanets(self) -> List[Dict]:
        try:
            r = requests.get(
                "https://exoplanetarchive.ipac.caltech.edu/TAP/sync",
                params={"query": "SELECT pl_name,hostname,pl_bmassj,pl_orbper,disc_year "
                                 "FROM ps WHERE disc_year>2020 ORDER BY disc_year DESC",
                        "format": "json"},
                headers=HEADERS, timeout=20
            )
            if r.ok:
                results = []
                for p in r.json()[:20]:
                    text = (f"Exoplanet {p.get('pl_name','?')} discovered in {p.get('disc_year','?')} "
                            f"orbiting star {p.get('hostname','?')}. "
                            f"Orbital period: {p.get('pl_orbper','?')} days.")
                    results.append({"text": text, "title": p.get("pl_name", "Exoplanet"),
                                    "url": "https://exoplanetarchive.ipac.caltech.edu",
                                    "domain": "space", "source": "nasa_exoplanets"})
                return results
        except Exception as e:
            log.debug(f"NASA exoplanets: {e}")
        return []

    def batch(self) -> List[Dict]:
        results = []
        results.extend(self.fetch_apod() or [])
        results.extend(self.fetch_exoplanets())
        return results


class HFDatasetSource:
    """HuggingFace: all Damaru-ai datasets + community math/science/code."""
    COMMUNITY_DATASETS = [
        "gsm8k",              # math problems
        "TIGER-Lab/MATH",     # advanced math
        "codeparrot/github-code-clean",  # code (small subset)
        "wikipedia",          # multilingual (already covered but backup)
    ]

    def list_own_datasets(self) -> List[str]:
        """List all datasets under Damaru-ai HF org."""
        if not HF_TOKEN:
            return []
        try:
            r = requests.get(
                f"https://huggingface.co/api/datasets?author={HF_OWNER}&limit=50",
                headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=15
            )
            if r.ok:
                return [d.get("id", "") for d in r.json() if d.get("id")]
        except Exception as e:
            log.debug(f"HF list datasets: {e}")
        return []

    def sample_dataset(self, dataset_id: str, n: int = 50) -> List[Dict]:
        """Sample rows from a HF dataset via the datasets API."""
        results = []
        try:
            r = requests.get(
                f"https://datasets-server.huggingface.co/rows",
                params={"dataset": dataset_id, "split": "train",
                        "offset": 0, "limit": n},
                headers={"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {},
                timeout=20
            )
            if r.ok:
                for row in r.json().get("rows", []):
                    d = row.get("row", {})
                    # Try common field names
                    text = (d.get("text") or d.get("instruction") or
                            d.get("question") or d.get("input") or "")
                    answer = (d.get("output") or d.get("answer") or
                              d.get("response") or "")
                    if text and len(text) > 20:
                        combined = text
                        if answer:
                            combined += f" Answer: {answer}"
                        results.append({"text": combined[:1000],
                                        "title": dataset_id,
                                        "url": f"https://huggingface.co/datasets/{dataset_id}",
                                        "domain": "knowledge",
                                        "source": f"hf_dataset/{dataset_id}"})
        except Exception as e:
            log.debug(f"HF dataset {dataset_id}: {e}")
        return results

    def batch(self) -> List[Dict]:
        results = []
        own = self.list_own_datasets()
        log.info(f"Found {len(own)} own HF datasets: {own}")
        for ds in own:
            results.extend(self.sample_dataset(ds, 100))
            time.sleep(1)
        return results


class StackOverflowSource:
    """Stack Overflow public API: top Q&A for key topics."""
    TAGS = ["python","machine-learning","deep-learning","c++","javascript",
            "robotics","computer-vision","data-science","algorithms"]

    def fetch_tag(self, tag: str) -> List[Dict]:
        try:
            r = requests.get(
                "https://api.stackexchange.com/2.3/questions",
                params={"order": "desc", "sort": "votes", "tagged": tag,
                        "site": "stackoverflow", "pagesize": 10,
                        "filter": "withbody"},
                headers=HEADERS, timeout=15
            )
            if r.ok:
                results = []
                for q in r.json().get("items", []):
                    title = q.get("title", "")
                    body = re.sub(r"<[^>]+>", "", q.get("body", ""))[:800]
                    if title and body:
                        text = f"Q: {title}. {body}"
                        results.append({"text": text, "title": title,
                                        "url": q.get("link", ""),
                                        "domain": "tech", "source": f"stackoverflow/{tag}"})
                return results
        except Exception as e:
            log.debug(f"SO {tag}: {e}")
        return []

    def batch(self) -> List[Dict]:
        results = []
        tags = random.sample(self.TAGS, 4)
        for t in tags:
            results.extend(self.fetch_tag(t))
            time.sleep(1)
        return results


class SpaceNewsSource:
    """Space news: NASA, ESA, SpaceNews RSS."""
    FEEDS = [
        ("https://www.nasa.gov/rss/dyn/breaking_news.rss",  "space"),
        ("https://spacenews.com/feed/",                     "space"),
        ("https://www.esa.int/rssfeed/Our_Activities/Space_Science", "space"),
        ("https://feeds.isro.gov.in/news/rss/",             "space"),  # ISRO
    ]

    def batch(self) -> List[Dict]:
        results = []
        for url, domain in self.FEEDS:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:5]:
                    title   = entry.get("title", "").strip()
                    summary = entry.get("summary", "").strip()
                    link    = entry.get("link", "")
                    if title and summary and len(summary) > 50:
                        text = f"{title}. {summary}"
                        results.append({"text": text[:1500], "title": title,
                                        "url": link, "domain": domain,
                                        "source": "space_news"})
            except Exception as e:
                log.debug(f"Space news {url}: {e}")
        return results


# ============================================================
#  TILE BUILDER
# ============================================================

def text_to_tiles(data: Dict, chunk_words: int = 120) -> List[Dict]:
    """Convert a data dict to PRAYAS knowledge tiles (100-120 word chunks)."""
    text   = data.get("text", "").strip()
    topic  = data.get("title", "")
    domain = data.get("domain", "general")
    source = data.get("source", "unknown")
    url    = data.get("url", "")

    if not text or len(text) < 50:
        return []

    words = text.split()
    chunks = [words[i:i+chunk_words] for i in range(0, len(words), chunk_words)]
    tiles = []
    for chunk in chunks:
        if len(chunk) < 15:
            continue
        tile = {
            "text":      " ".join(chunk),
            "topic":     topic,
            "domain":    domain,
            "source":    source,
            "url":       url,
            "lang":      "en",
            "timestamp": datetime.utcnow().isoformat(),
        }
        tiles.append(tile)
    return tiles


# ============================================================
#  PUSH TO HF
# ============================================================

def push_tiles_to_hf(tiles: List[Dict], run_label: str) -> bool:
    if not tiles or not HF_TOKEN:
        log.info(f"Skip HF push: {len(tiles)} tiles, token={'set' if HF_TOKEN else 'missing'}")
        return False
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=HF_TOKEN)
        fname = f"world_tiles/{run_label}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        content = "\n".join(json.dumps(t, ensure_ascii=False) for t in tiles)
        api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=fname,
            repo_id=HF_DATASET,
            repo_type="dataset",
            commit_message=f"World Harvest: {len(tiles)} tiles [{run_label}]",
        )
        log.info(f"Pushed {len(tiles)} tiles to HF: {fname}")
        return True
    except Exception as e:
        log.error(f"HF push failed: {e}")
        return False


# ============================================================
#  MAIN
# ============================================================

def main():
    start = time.time()
    log.info("=" * 60)
    log.info("  DAMRU WORLD HARVEST v1.0")
    log.info(f"  Dataset: {HF_DATASET} | Max: {MAX_TILES_PER_RUN} tiles")
    log.info("=" * 60)

    sources = [
        ("Wikipedia",      WikipediaSource().batch),
        ("arXiv",          ArXivSource().batch),
        ("GitHub",         GitHubSource().batch),
        ("NASA/Space",     NASASource().batch),
        ("SpaceNews",      SpaceNewsSource().batch),
        ("StackOverflow",  StackOverflowSource().batch),
        ("HF Datasets",    HFDatasetSource().batch),
    ]

    all_tiles = []
    source_stats = {}

    for name, fetch_fn in sources:
        try:
            log.info(f"\n--- {name} ---")
            raw = fetch_fn()
            tiles = []
            for item in raw:
                tiles.extend(text_to_tiles(item))
            source_stats[name] = len(tiles)
            all_tiles.extend(tiles)
            log.info(f"  {name}: {len(raw)} items -> {len(tiles)} tiles")
            if len(all_tiles) >= MAX_TILES_PER_RUN:
                log.info(f"Max tiles reached ({MAX_TILES_PER_RUN}), stopping early")
                break
        except Exception as e:
            log.error(f"Source {name} failed (self-healing, continuing): {e}")
            source_stats[name] = 0
            continue  # Self-heal: one source fails, others continue

    # Deduplicate by text hash
    seen = set()
    unique_tiles = []
    for t in all_tiles:
        h = hash(t["text"][:100])
        if h not in seen:
            seen.add(h)
            unique_tiles.append(t)

    log.info(f"\nTotal: {len(all_tiles)} tiles ({len(unique_tiles)} unique)")

    # Push to HF
    success = push_tiles_to_hf(unique_tiles[:MAX_TILES_PER_RUN], "world")

    elapsed = time.time() - start
    stats = (
        f"Run: {datetime.utcnow().isoformat()}\n"
        f"Total tiles: {len(unique_tiles)}\n"
        f"Pushed: {success}\n"
        f"Sources:\n" +
        "\n".join(f"  {k}: {v}" for k, v in source_stats.items()) +
        f"\nElapsed: {elapsed/60:.1f} min\n"
    )
    STATS_FILE.write_text(stats)
    log.info("\n" + stats)
    log.info("=" * 60)
    log.info(f"  DONE! {len(unique_tiles)} world tiles -> {HF_DATASET}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
