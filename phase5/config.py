"""Central config for the Damru learning engine. All via env vars (with safe defaults)."""
import os

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
STACKEXCHANGE_KEY = os.environ.get("STACKEXCHANGE_KEY", "")

TABLE = os.environ.get("DAMRU_TABLE", "damru_knowledge")

# Volume targets
DAILY_TARGET = int(os.environ.get("DAILY_TARGET", "90000"))

# Worker pool (coding is 4X by design -> see run_engine)
GENERAL_WORKERS = int(os.environ.get("GENERAL_WORKERS", "6"))
ANALYSIS_WORKERS = int(os.environ.get("ANALYSIS_WORKERS", "2"))
MATH_WORKERS = int(os.environ.get("MATH_WORKERS", "2"))
CODING_WORKERS = int(os.environ.get("CODING_WORKERS", "8"))  # 4X of math

# NEW: Hindi / regional-language track + exam-syllabus track
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

# Self-evaluation: quality bar that RISES as the dataset grows (gets stricter over time)
START_QUALITY = float(os.environ.get("START_QUALITY", "0.55"))
MAX_QUALITY = float(os.environ.get("MAX_QUALITY", "0.88"))
QUALITY_RISE_PER_10K = float(os.environ.get("QUALITY_RISE_PER_10K", "0.015"))

# Per-subject mastery before moving to the next subject
MASTERY_TARGET = int(os.environ.get("MASTERY_TARGET", "5000"))

APP_REFERER = os.environ.get("APP_REFERER", "https://damru-ai.vercel.app")
