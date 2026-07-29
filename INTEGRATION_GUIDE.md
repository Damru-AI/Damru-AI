# 🛠️ DAMRU AI — INTEGRATION GUIDE

How all new systems connect to existing `app.py` without overlap.

## Existing Systems (DO NOT MODIFY)
- `app.py` — main FastAPI + Gradio server
- `rag.py` — existing RAG engine
- `damru_brain_patch.py` — existing brain patch
- `damru_chetna.py` — consciousness gate
- `damru_cortex.py` / `damru_reflex.py` — existing cognition

## New Systems (ADDITIVE ONLY — no overlap)

### `damru_prayas_core.py`
Plug-in replacement for RAG retrieval.
In `app.py`, when `RAG is None`, try PRAYAS:
```python
try:
    from damru_prayas_core import get_engine as get_prayas
    PRAYAS = get_prayas()
except Exception:
    PRAYAS = None
```

### `damru_emotions.py`
Called in `/chat` endpoint BEFORE building messages:
```python
from damru_emotions import get_emotion_engine
EMO = get_emotion_engine()

# In /chat:
emo_ctx = EMO.build_emotional_context(body.message, intent or 'general')
sys_prompt = SYS + '\n' + emo_ctx['tone_instruction']
```

### `damru_selfheal.py`
Wrap any critical function:
```python
from damru_selfheal import SelfHealingRunner
health_runner = SelfHealingRunner(fn=load_llm, name='llm_loader')
```

### `damru_world_harvest.py`
Runs ONLY in GitHub Actions (separate from HF Space).
Outputs tiles to HF dataset `Damaru-ai/damru-knowledge/world_tiles/`
Existing `app.py` can read these via PRAYAS tile loader.

## Data Flow
```
GitHub Actions (every 6h)
    damru_curious_engine_actions.py  → curious/ JSONL tiles
    damru_world_harvest.py           → world_tiles/ JSONL tiles
            ↓
    HF Dataset: Damaru-ai/damru-knowledge
            ↓
    PRAYAS Engine (damru_prayas_core.py) loads tiles on startup
            ↓
    app.py /chat endpoint uses PRAYAS for retrieval
            ↓
    Emotion Engine adjusts response tone
            ↓
    Self-Heal monitors everything
```

## Environment Variables to Add
```env
PRAYAS_TILE_DIR=/opt/damru/tiles
EMOTION_STATE=/opt/damru/emotion.json
SELFHEAL_MAX_RETRIES=10
HF_LOG_CHATS=1
```
