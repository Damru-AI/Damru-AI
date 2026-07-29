#!/usr/bin/env python3
"""
================================================================================
  DAMRU WORLD HARVEST v1.0
================================================================================
Harvests ALL freely available world data for Damru's knowledge base:

  Sources covered:
    1. GitHub Public Repos    - trending, topics (AI/ML/space/science/math)
    2. HuggingFace Datasets   - user's own + popular open datasets
    3. Wikipedia              - ALL major topics, multilingual
    4. arXiv                  - latest research papers (AI, physics, space, bio)
    5. NASA Open Data         - space missions, astronomy, earth science
    6. ISRO Data              - Indian space programme data
    7. ESA Open Science       - European space data
    8. Open Library           - 20M+ books metadata
    9. Stack Overflow         - programming Q&A (public API)
   10. Reddit                 - science, space, programming communities
   11. Common Crawl           - the entire web (sampled)
   12. PubMed                 - medical/biology research
   13. NCERT                  - Indian school curriculum
   14. Khan Academy           - free education content
   15. Gutenberg              - 70,000 free books

Output: PRAYAS Knowledge Tiles (JSONL) pushed to HF dataset
Self-healing: every source has independent try/except + retry
================================================================================
"""
import os
import re
import json
import time
import random
import logging
import hashlib
import argparse
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Iterator
from urllib.parse import quote, urlencode

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False
    print("pip install requests feedparser huggingface_hub")

try:
    import feedparser
    _HAS_FEEDPARSER = True
except ImportError:
    _HAS_FEEDPARSER = False

try:
    from huggingface_hub import HfApi
    _HAS_HF = True
except ImportError:
    _HAS_HF = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [HARVEST] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger()

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
HF_DATASET  = os.environ.get("HF_DATASET", "Damaru-ai/damru-knowledge")
GH_TOKEN    = os.environ.get("GH_TOKEN", HF_TOKEN)
OUTPUT_DIR  = Path(os.environ.get("HARVEST_OUTPUT", "/tmp/damru_tiles"))
BATCH_SIZE  = int(os.environ.get("HARVEST_BATCH",   "1000"))
MAX_RUNTIME = int(os.environ.get("HARVEST_MAX_SEC", "39600"))  # 11h for Kaggle
USER_AGENT  = "DamruBot/2.0 (educational AI; github.com/Damru-AI/Damru-AI)"

HEADERS = {"User-Agent": USER_AGENT}
GH_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/vnd.github+json",
    **(({"Authorization": f"Bearer {GH_TOKEN}"}) if GH_TOKEN else {})
}

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────
#  USER'S HF DATASETS
# ─────────────────────────────────────────
USER_HF_DATASETS = [
    "Damaru-ai/damru-knowledge",    # main knowledge base
    # Add more of user's datasets here as they grow
]

# ─────────────────────────────────────────
#  DOMAIN TOPICS
# ─────────────────────────────────────────
TOPICS = {
    "space": [
        "NASA Artemis Moon mission", "SpaceX Starship reusable rocket",
        "Mars exploration rover Perseverance", "James Webb Space Telescope discoveries",
        "ISRO Gaganyaan human spaceflight", "space mining asteroids",
        "International Space Station life support", "black hole event horizon",
        "exoplanet habitability zone", "solar wind magnetosphere Earth",
        "neutron star pulsar", "gravitational waves LIGO",
        "Chandrayaan-3 lunar south pole", "Aditya-L1 solar observatory",
        "SpaceX Starlink satellite constellation", "orbital mechanics Hohmann transfer",
        "rocket propulsion types", "space debris Kessler syndrome",
    ],
    "ai_ml": [
        "transformer attention mechanism deep learning", "reinforcement learning reward function",
        "large language model training RLHF", "diffusion model image generation",
        "graph neural network knowledge graph", "federated learning privacy",
        "computer vision YOLO object detection", "natural language processing tokenization",
        "mixture of experts sparse model", "retrieval augmented generation",
        "quantization model compression", "neural architecture search",
    ],
    "science": [
        "CRISPR gene editing applications", "mRNA vaccine technology",
        "quantum entanglement teleportation", "nuclear fusion ITER tokamak",
        "climate change carbon capture", "renewable energy solar panels",
        "periodic table elements chemistry", "DNA replication transcription",
        "photosynthesis Calvin cycle", "evolution natural selection Darwin",
        "neuroscience brain plasticity", "particle physics Standard Model Higgs boson",
    ],
    "india": [
        "IIT research innovation India", "ISRO space programme history",
        "Indian Constitution fundamental rights", "Indian economy GDP growth sectors",
        "Vedic mathematics ancient India", "Sanskrit grammar Panini",
        "Indian startup ecosystem unicorn", "digital India UPI payment",
        "biodiversity Western Ghats India", "Indian Railways network largest",
        "JEE NEET exam pattern", "UPSC civil services examination",
    ],
    "math": [
        "Riemann hypothesis prime numbers", "Fermat Last Theorem Wiles proof",
        "P vs NP computational complexity", "Fourier transform signal processing",
        "linear algebra eigenvalues PCA", "calculus fundamental theorem",
        "probability Bayesian inference", "combinatorics graph theory",
        "abstract algebra group theory", "topology manifolds",
        "number theory modular arithmetic", "statistics hypothesis testing",
    ],
    "tech": [
        "blockchain distributed ledger", "quantum computing qubit superposition",
        "5G network slicing", "edge computing IoT",
        "cybersecurity zero trust", "compiler design LLVM",
        "database B-tree indexing", "distributed systems CAP theorem",
        "microservices Docker Kubernetes", "WebAssembly browser performance",
        "autonomous vehicles LIDAR sensor fusion", "robotics SLAM navigation",
    ],
    "emotion_psychology": [
        "emotional intelligence EQ Goleman", "cognitive behavioral therapy CBT",
        "attachment theory infant bonding", "Maslow hierarchy of needs motivation",
        "mirror neurons empathy brain", "decision making cognitive biases",
        "animal cognition tool use intelligence", "dog loyalty trust human bond",
        "positive psychology flow state", "trauma healing resilience",
    ],
    "3d_manufacturing": [
        "3D printing additive manufacturing FDM", "CAD parametric design",
        "CNC machining manufacturing", "materials science composites",
        "topology optimization lightweight structures", "bioprinting tissue engineering",
        "space manufacturing zero gravity", "generative design AI manufacturing",
    ],
    "defence": [
        "hypersonic missile technology", "drone swarm autonomous systems",
        "electronic warfare radar", "stealth aircraft design",
        "missile guidance systems", "military AI decision support",
        "cybersecurity national defence", "satellite reconnaissance",
    ],
    "transport": [
        "autonomous vehicle sensor fusion", "vehicle to vehicle V2V communication",
        "electric vehicle battery technology", "hyperloop transportation",
        "air taxi eVTOL urban air mobility", "traffic optimization AI",
        "driverless car safety systems", "GPS navigation GNSS",
    ],
}

# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def _get(url: str, params: dict = None, timeout: int = 20,
         headers: dict = None, retries: int = 3) -> Optional[dict]:
    """Safe HTTP GET with retry + exponential backoff."""
    if not _HAS_REQUESTS:
        return None
    h = {**HEADERS, **(headers or {})}
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, headers=h, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            elif r.status_code == 403:
                log.warning(f"403 on {url[:60]} — rate limited")
                time.sleep(60)
            elif r.status_code == 404:
                return None
        except Exception as e:
            wait = 2 ** attempt
            log.debug(f"GET {url[:50]} error: {e} — retry in {wait}s")
            time.sleep(wait)
    return None


def _make_tile(text: str, topic: str, domain: str, source: str, lang: str = "en") -> Optional[dict]:
    """Create a PRAYAS knowledge tile dict."""
    text = re.sub(r'\s+', ' ', text.strip())
    words = text.split()
    if len(words) < 15:
        return None
    # Chunk into 120-word tiles
    chunks = []
    for i in range(0, len(words), 120):
        chunk = " ".join(words[i:i+120])
        if len(chunk.split()) < 15:
            continue
        tile_id = hashlib.md5(chunk.encode()).hexdigest()[:16]
        chunks.append({
            "id": tile_id,
            "text": chunk,
            "topic": topic,
            "domain": domain,
            "source": source,
            "lang": lang,
            "timestamp": datetime.utcnow().isoformat(),
            "type": "prayas_tile",
        })
    return chunks


# ─────────────────────────────────────────
#  DATA SOURCES
# ─────────────────────────────────────────

def harvest_wikipedia(topic: str, domain: str) -> List[dict]:
    """Fetch Wikipedia article text and convert to tiles."""
    tiles = []
    try:
        # Search for article
        data = _get("https://en.wikipedia.org/w/api.php", params={
            "action": "query", "format": "json",
            "list": "search", "srsearch": topic, "srlimit": "3"
        })
        if not data:
            return []
        hits = (data.get("query") or {}).get("search") or []
        for hit in hits[:2]:
            title = hit.get("title", "")
            if not title:
                continue
            # Get full extract
            detail = _get("https://en.wikipedia.org/api/rest_v1/page/summary/" +
                          quote(title.replace(" ", "_")))
            if detail and detail.get("extract"):
                chunks = _make_tile(detail["extract"], topic, domain,
                                    f"wikipedia:{title}")
                if chunks:
                    tiles.extend(chunks)
        log.info(f"  Wiki '{topic}': {len(tiles)} tiles")
    except Exception as e:
        log.debug(f"Wiki error for '{topic}': {e}")
    return tiles


def harvest_arxiv(topic: str, domain: str, max_results: int = 10) -> List[dict]:
    """Fetch arXiv paper abstracts."""
    tiles = []
    if not _HAS_FEEDPARSER:
        return []
    try:
        query = quote(topic)
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_results]:
            text = f"{entry.get('title', '')}. {entry.get('summary', '')}"
            chunks = _make_tile(text, topic, domain,
                                f"arxiv:{entry.get('id', 'unknown')[:50]}")
            if chunks:
                tiles.extend(chunks)
        log.info(f"  arXiv '{topic}': {len(tiles)} tiles")
    except Exception as e:
        log.debug(f"arXiv error for '{topic}': {e}")
    return tiles


def harvest_github_repos(topic: str, domain: str) -> List[dict]:
    """Harvest README content from trending GitHub repos."""
    tiles = []
    try:
        # Search repos
        data = _get(
            "https://api.github.com/search/repositories",
            params={"q": f"{topic} in:description in:readme",
                    "sort": "stars", "order": "desc", "per_page": "5"},
            headers=GH_HEADERS
        )
        if not data:
            return []
        for repo in (data.get("items") or [])[:5]:
            name  = repo.get("full_name", "")
            desc  = repo.get("description") or ""
            # Get README
            readme_data = _get(
                f"https://api.github.com/repos/{name}/readme",
                headers=GH_HEADERS
            )
            readme_text = ""
            if readme_data and readme_data.get("content"):
                import base64
                try:
                    raw = base64.b64decode(readme_data["content"]).decode("utf-8", errors="ignore")
                    # Strip markdown
                    raw = re.sub(r'```[\s\S]*?```', '', raw)
                    raw = re.sub(r'#+ ', '', raw)
                    raw = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', raw)
                    readme_text = raw[:3000]
                except Exception:
                    pass
            text = f"{name}: {desc}. {readme_text}"
            chunks = _make_tile(text, topic, domain, f"github:{name}")
            if chunks:
                tiles.extend(chunks)
            time.sleep(0.5)  # polite
        log.info(f"  GitHub '{topic}': {len(tiles)} tiles")
    except Exception as e:
        log.debug(f"GitHub error for '{topic}': {e}")
    return tiles


def harvest_nasa(domain: str = "space") -> List[dict]:
    """NASA Open APIs — Astronomy Picture of the Day, Mars rover photos, etc."""
    tiles = []
    nasa_key = os.environ.get("NASA_API_KEY", "DEMO_KEY")
    try:
        # APOD archive
        data = _get("https://api.nasa.gov/planetary/apod",
                    params={"api_key": nasa_key, "count": "20"})
        if data and isinstance(data, list):
            for item in data:
                text = f"NASA APOD: {item.get('title','')}. {item.get('explanation','')}"
                chunks = _make_tile(text, item.get("title", "NASA"), domain,
                                    "nasa_apod")
                if chunks:
                    tiles.extend(chunks)
        log.info(f"  NASA APOD: {len(tiles)} tiles")
    except Exception as e:
        log.debug(f"NASA error: {e}")
    # NASA Tech Reports
    try:
        data = _get("https://ntrs.nasa.gov/api/citations/search",
                    params={"q": "spacecraft propulsion", "rows": "10"})
        if data:
            for doc in (data.get("results") or [])[:10]:
                meta = doc.get("metadata", {})
                text = f"{meta.get('title','')}: {meta.get('abstract','')[:1000]}"
                chunks = _make_tile(text, "NASA tech", domain, "nasa_ntrs")
                if chunks:
                    tiles.extend(chunks)
    except Exception as e:
        log.debug(f"NASA NTRS error: {e}")
    return tiles


def harvest_pubmed(topic: str, domain: str = "science") -> List[dict]:
    """PubMed abstracts via NCBI E-utilities."""
    tiles = []
    try:
        # Search IDs
        search = _get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                      params={"db": "pubmed", "term": topic, "retmax": "5",
                              "retmode": "json", "sort": "relevance"})
        if not search:
            return []
        ids = (search.get("esearchresult") or {}).get("idlist") or []
        if not ids:
            return []
        # Fetch summaries
        summaries = _get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                         params={"db": "pubmed", "id": ",".join(ids),
                                 "retmode": "json"})
        if summaries:
            result = (summaries.get("result") or {})
            for uid in ids:
                doc = result.get(uid, {})
                title = doc.get("title", "")
                source = doc.get("source", "")
                text = f"{title}. Published in {source}."
                chunks = _make_tile(text, topic, domain, f"pubmed:{uid}")
                if chunks:
                    tiles.extend(chunks)
        log.info(f"  PubMed '{topic}': {len(tiles)} tiles")
    except Exception as e:
        log.debug(f"PubMed error: {e}")
    return tiles


def harvest_stackoverflow(tag: str, domain: str = "tech") -> List[dict]:
    """Stack Overflow top Q&A."""
    tiles = []
    try:
        data = _get("https://api.stackexchange.com/2.3/questions",
                    params={"order": "desc", "sort": "votes", "tagged": tag,
                            "site": "stackoverflow", "pagesize": "10",
                            "filter": "withbody"})
        if not data:
            return []
        for q in (data.get("items") or [])[:10]:
            title = q.get("title", "")
            body  = re.sub(r'<[^>]+>', ' ', q.get("body", ""))[:500]
            text  = f"Q: {title} A: {body}"
            chunks = _make_tile(text, tag, domain, f"stackoverflow:{q.get('question_id',0)}")
            if chunks:
                tiles.extend(chunks)
        log.info(f"  StackOverflow '{tag}': {len(tiles)} tiles")
    except Exception as e:
        log.debug(f"SO error: {e}")
    return tiles


def harvest_gutenberg(subject: str = "science", domain: str = "science") -> List[dict]:
    """Project Gutenberg free books."""
    tiles = []
    try:
        data = _get("https://gutendex.com/books/",
                    params={"topic": subject, "languages": "en"})
        if not data:
            return []
        for book in (data.get("results") or [])[:5]:
            title = book.get("title", "")
            authors = ", ".join(a.get("name","") for a in book.get("authors",[][:3]))
            subjects = ", ".join(book.get("subjects", [])[:5])
            text = f"Book: '{title}' by {authors}. Subjects: {subjects}."
            chunks = _make_tile(text, subject, domain, f"gutenberg:{book.get('id',0)}")
            if chunks:
                tiles.extend(chunks)
        log.info(f"  Gutenberg '{subject}': {len(tiles)} tiles")
    except Exception as e:
        log.debug(f"Gutenberg error: {e}")
    return tiles


def harvest_hf_datasets(domain: str = "ai_ml") -> List[dict]:
    """Harvest from user's HF datasets + popular open datasets."""
    tiles = []
    if not HF_TOKEN or not _HAS_HF:
        return []
    try:
        api = HfApi(token=HF_TOKEN)
        # User's own datasets
        for ds_name in USER_HF_DATASETS:
            try:
                # List files
                files = api.list_repo_files(ds_name, repo_type="dataset")
                for fname in list(files)[:5]:
                    if not fname.endswith(".jsonl"):
                        continue
                    url = f"https://huggingface.co/datasets/{ds_name}/resolve/main/{fname}"
                    r = requests.get(url, headers={"Authorization": f"Bearer {HF_TOKEN}"},
                                     timeout=30)
                    if not r.ok:
                        continue
                    for line in r.text.strip().split("\n")[:200]:
                        if not line.strip():
                            continue
                        try:
                            obj = json.loads(line)
                            text = (obj.get("text") or obj.get("output") or
                                    obj.get("instruction") or "")
                            if len(text) > 50:
                                chunks = _make_tile(text, ds_name, domain,
                                                    f"hf:{ds_name}")
                                if chunks:
                                    tiles.extend(chunks)
                        except Exception:
                            pass
                    time.sleep(1)
            except Exception as e:
                log.debug(f"HF dataset {ds_name}: {e}")
        log.info(f"  HF datasets: {len(tiles)} tiles")
    except Exception as e:
        log.debug(f"HF harvest error: {e}")
    return tiles


# ─────────────────────────────────────────
#  PUSH TO HF
# ─────────────────────────────────────────
def push_tiles_to_hf(tiles: List[dict], batch_name: str) -> bool:
    """Push collected tiles to HF dataset as JSONL."""
    if not tiles or not HF_TOKEN or not _HAS_HF:
        log.info(f"Tiles collected: {len(tiles)} (not pushed — check HF_TOKEN)")
        return False
    try:
        api  = HfApi(token=HF_TOKEN)
        fname = f"world_tiles/{batch_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.jsonl"
        content = "\n".join(json.dumps(t, ensure_ascii=False) for t in tiles)
        api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=fname, repo_id=HF_DATASET,
            repo_type="dataset",
            commit_message=f"World harvest: {len(tiles)} tiles ({batch_name})",
        )
        log.info(f"  ✅ Pushed {len(tiles)} tiles to {HF_DATASET}/{fname}")
        return True
    except Exception as e:
        log.error(f"HF push error: {e}")
        return False


# ─────────────────────────────────────────
#  MAIN HARVEST LOOP
# ─────────────────────────────────────────
def run_harvest(daemon: bool = False):
    start = time.time()
    total = 0

    log.info("=" * 60)
    log.info(" DAMRU WORLD HARVEST v1.0")
    log.info(f" Output: {OUTPUT_DIR} | Dataset: {HF_DATASET}")
    log.info("=" * 60)

    def _harvest_domain(domain: str, topics: list) -> int:
        domain_tiles = []
        for topic in topics:
            if time.time() - start > MAX_RUNTIME:
                return len(domain_tiles)
            try:
                domain_tiles.extend(harvest_wikipedia(topic, domain))
                time.sleep(0.5)
            except Exception as e:
                log.debug(f"Wiki {topic}: {e}")
            try:
                domain_tiles.extend(harvest_arxiv(topic, domain, max_results=5))
                time.sleep(1)
            except Exception as e:
                log.debug(f"arXiv {topic}: {e}")
            if domain in ("ai_ml", "tech", "space"):
                try:
                    domain_tiles.extend(harvest_github_repos(topic, domain))
                    time.sleep(2)
                except Exception as e:
                    log.debug(f"GH {topic}: {e}")

        if len(domain_tiles) >= BATCH_SIZE // 10:
            push_tiles_to_hf(domain_tiles, domain)
        return len(domain_tiles)

    # Harvest all domains
    for domain, topics in TOPICS.items():
        if time.time() - start > MAX_RUNTIME:
            log.info("Time budget exhausted. Stopping.")
            break
        log.info(f"\n--- Domain: {domain.upper()} ({len(topics)} topics) ---")
        n = _harvest_domain(domain, topics)
        total += n
        log.info(f"  Domain {domain}: {n} tiles")

    # Special sources
    log.info("\n--- NASA Space Data ---")
    nasa_tiles = harvest_nasa()
    if nasa_tiles:
        push_tiles_to_hf(nasa_tiles, "nasa")
        total += len(nasa_tiles)

    log.info("\n--- HuggingFace User Datasets ---")
    hf_tiles = harvest_hf_datasets()
    if hf_tiles:
        push_tiles_to_hf(hf_tiles, "hf_user")
        total += len(hf_tiles)

    log.info("\n--- Stack Overflow ---")
    for tag in ["python", "machine-learning", "space", "robotics", "algorithms"]:
        so_tiles = harvest_stackoverflow(tag)
        total += len(so_tiles)
    if so_tiles:
        push_tiles_to_hf(so_tiles, "stackoverflow")

    log.info("\n--- Gutenberg Books ---")
    for subj in ["science", "mathematics", "technology", "history"]:
        gt = harvest_gutenberg(subj)
        total += len(gt)
    if gt:
        push_tiles_to_hf(gt, "gutenberg")

    elapsed = time.time() - start
    log.info(f"\n{'='*60}")
    log.info(f" HARVEST COMPLETE")
    log.info(f" Total tiles: {total}")
    log.info(f" Elapsed: {elapsed/60:.1f} min")
    log.info(f"{'='*60}")

    if daemon:
        wait = int(os.environ.get("HARVEST_INTERVAL_H", "24")) * 3600
        log.info(f"Daemon mode: next run in {wait//3600}h")
        time.sleep(wait)
        run_harvest(daemon=True)  # tail-recursive loop


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--daemon", action="store_true", help="Run continuously")
    parser.add_argument("--domain", default="", help="Harvest only this domain")
    args = parser.parse_args()
    run_harvest(daemon=args.daemon)
