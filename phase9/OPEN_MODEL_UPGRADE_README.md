# Damru Open Model + BGE-M3 Upgrade

## Decision

Do not train a foundation model from scratch. Use open-weight frontier models as inference/teacher muscle; invest Damru effort in retrieval, verified data, tools, coding execution, science, multilingual UX and evaluation.

## Architecture

```text
Query
→ BGE-M3 hot Supabase (latest 10k) + cold HF FAISS (curated 200k)
→ context + query
→ gpt-oss-120b via HF Router/Groq
→ low/medium/high reasoning chosen automatically
→ local trained GGUF fallback
```

## Why not 10M vectors in Supabase?

10M × 1024 × float32 is ~40GB before indexes. Even halfvec is ~20GB. Supabase free tier is the hot cache only. HF Parquet remains the source of truth; FAISS is the cold retrieval index.

## Files and destinations

### HF Space root (`Damaru-ai/Damru`)

```text
app.py
open_brain.py
rag.py
```

`requirements.txt` does not need a new dependency; `requests` already exists.

### GitHub

```text
phase5/brain.py                    <- use phase5_brain.py file provided
phase9/open_brain.py
phase9/hot_embed_sync.py
phase9/BGE_M3_HOT_MEMORY.sql
phase9/bge_m3_colab_backfill.py
phase9/build_bge_m3_index.py
.github/workflows/open-brain-probe.yml
.github/workflows/sync-hf.yml
.github/workflows/controlled-learning.yml
.github/workflows/rejection-diagnostics.yml
```

## Safe rollout order

1. Run `BGE_M3_HOT_MEMORY.sql` in Supabase.
2. Upload GitHub files, but leave old learners disabled.
3. Add/fix `GROQ_API_KEY`; ensure `HF_TOKEN` has Inference Providers permission.
4. Run **Damru Open Brain Probe**. Require `DAMRU_OPEN_BRAIN_OK` plus provider/model metadata.
5. Run controlled-learning pilot and diagnostics again; coding/Hindi/exam should produce parseable samples.
6. Run `bge_m3_colab_backfill.py` once in Colab (default 10k hot rows).
7. Run `build_bge_m3_index.py` in Colab for curated cold index (default 200k).
8. Upload new `rag.py`, `open_brain.py`, `app.py` to HF Space.
9. Keep `USE_OPEN_BRAIN=0` for first rebuild; verify health/local fallback.
10. Set `USE_OPEN_BRAIN=1`; restart; `/health` should show `open_brain:true`.

## Environment

### HF Space

```text
USE_OPEN_BRAIN=1
HF_TOKEN=<token with inference permission>
GROQ_API_KEY=<optional direct Groq fallback>
OPEN_BRAIN_HF_MODELS=openai/gpt-oss-120b:groq,openai/gpt-oss-120b:cerebras,openai/gpt-oss-120b:fireworks-ai
USE_HOT_RAG=1
SUPABASE_SERVICE_KEY=<server-side only>
```

### GitHub

Existing secrets plus:

```text
GROQ_API_KEY
SUPABASE_SERVICE_KEY
HF_TOKEN
```

## Notes

- gpt-oss-120b is not fine-tuned by Damru.
- Provider quotas still exist; router fallback + local GGUF prevent downtime.
- Do not expose service/Groq/HF tokens in `index.html`.
- Do not replace the old RAG index until the BGE-M3 index build succeeds.
