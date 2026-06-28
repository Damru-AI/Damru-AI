"""Central config for the Damru learning engine. All via env vars (with safe defaults)."""
import os

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "") or os.environ.get("OPENROUTER_KEY", "")
STACKEXCHANGE_KEY = os.environ.get("STACKEXCHANGE_KEY", "")

# --- Multi-provider LLM (rate-limit buster). Each is OPTIONAL; the brain uses
# whatever keys exist and rotates across all of them to beat 429 limits. ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GEMINI_KEY", "")

TABLE = os.environ.get("DAMRU_TABLE", "damru_knowledge")

# Volume targets
DAILY_TARGET = int(os.environ.get("DAILY_TARGET", "90000"))

# Worker pool (coding is 4X by design -> see run_engine)
GENERAL_WORKERS = int(os.environ.get("GENERAL_WORKERS", "6"))
ANALYSIS_WORKERS = int(os.environ.get("ANALYSIS_WORKERS", "2"))
MATH_WORKERS = int(os.environ.get("MATH_WORKERS", "2"))
CODING_WORKERS = int(os.environ.get("CODING_WORKERS", "8"))  # 4X of math

# Hindi / regional-language track + exam-syllabus track
HINDI_WORKERS = int(os.environ.get("HINDI_WORKERS", "3"))
EXAM_WORKERS = int(os.environ.get("EXAM_WORKERS", "3"))
# Languages the Hindi track produces (comma sep). hi=Devanagari, hinglish=Roman Hindi.
REGIONAL_LANGS = [
    x.strip()
    for x in os.environ.get("REGIONAL_LANGS", "hi,hinglish").split(",")
    if x.strip()
]
# Share of exam questions generated in Hindi (0..1)
EXAM_HINDI_RATIO = float(os.environ.get("EXAM_HINDI_RATIO", "0.25"))

# --- Parallel sharding: run several engines at once, each covering a DISJOINT
# slice of subjects so they don't duplicate each other. SHARD_TOTAL=1 -> full. ---
SHARD_ID = int(os.environ.get("SHARD_ID", "0"))
SHARD_TOTAL = max(1, int(os.environ.get("SHARD_TOTAL", "1")))

# --- Eval-driven self-improvement: with this probability a worker focuses on a
# subject Damru is currently WEAK at (from eval/weak_topics.json). ---
FOCUS_RATIO = float(os.environ.get("FOCUS_RATIO", "0.30"))
WEAK_TOPICS_FILE = os.environ.get(
    "WEAK_TOPICS_FILE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval", "weak_topics.json"),
)

# Fine-tune readiness floor (rows). The ready-check workflow alerts at this count.
READY_TARGET = int(os.environ.get("READY_TARGET", "2000000"))

# 0 = run forever; otherwise stop after N minutes (use for cron-chunked runs)
MAX_RUNTIME_MIN = int(os.environ.get("MAX_RUNTIME_MIN", "0"))

# Local state dir (sqlite dedup + curriculum + checkpoints)
DATA_DIR = os.environ.get(
    "DAMRU_DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "_state")
)
os.makedirs(DATA_DIR, exist_ok=True)

# Free LLMs via OpenRouter (fallback order). Override with LLM_MODELS env (comma sep).
LLM_MODELS = [
    m.strip()
    for m in os.environ.get(
        "LLM_MODELS",
        "deepseek/deepseek-chat-v3-0324:free,"
        "meta-llama/llama-3.3-70b-instruct:free,"
        "qwen/qwen-2.5-72b-instruct:free,"
        "google/gemini-2.0-flash-exp:free",
    ).split(",")
    if m.strip()
]

# Groq free models (OpenAI-compatible API; very fast, generous free tier).
GROQ_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GROQ_MODELS",
        "llama-3.3-70b-versatile,llama-3.1-8b-instant,gemma2-9b-it",
    ).split(",")
    if m.strip()
]

# Google Gemini free models (native API).
GEMINI_MODELS = [
    m.strip()
    for m in os.environ.get(
        "GEMINI_MODELS",
        "gemini-2.0-flash,gemini-1.5-flash",
    ).split(",")
    if m.strip()
]

# Self-evaluation: quality bar that RISES as the dataset grows (gets stricter over time)
START_QUALITY = float(os.environ.get("START_QUALITY", "0.55"))
MAX_QUALITY = float(os.environ.get("MAX_QUALITY", "0.88"))
QUALITY_RISE_PER_10K = float(os.environ.get("QUALITY_RISE_PER_10K", "0.015"))

# Per-subject mastery before moving to the next subject
MASTERY_TARGET = int(os.environ.get("MASTERY_TARGET", "5000"))

APP_REFERER = os.environ.get("APP_REFERER", "https://damru-ai.vercel.app")
