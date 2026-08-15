# Damru Daily Learning Pipeline - 1M Lines/Day Scaling Guide

## Overview
The Daily Learning Pipeline now generates **1M lines of high-quality training data per day** using Groq API with parallel batch processing.

## Architecture

### Three Layers:

1. **Seed Packs Layer** (Static)
   - Curated hand-written Q&A in `.jsonl` files
   - Auto-loaded from `learn/` directory
   - Always included in every daily run

2. **Parallel LLM Generation** (Dynamic)
   - 20 concurrent Groq API workers
   - 50 Q&A pairs per batch
   - ~1000 lines per batch (~13 lines/QA)
   - Auto-retry with exponential backoff
   - Rate-limit aware (429 handling)

3. **Knowledge Ingestion** (RAG)
   - Daily corpus committed to git
   - Pushed to HuggingFace dataset: `Damaru-ai/damru-knowledge`
   - Integrated via `damru_wire.py` → RAG brain

---

## Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Target Lines/Day | 1,000,000 | Configurable via `DAILY_TARGET_LINES` |
| Parallel Workers | 20 | Configurable via `PARALLEL_WORKERS` |
| Questions/Batch | 50 | Tuned for Groq API token limits |
| Lines/QA Pair | ~13 | Includes Q, A, metadata |
| Total Batches/Day | ~1500 | For 1M lines target |
| Runtime | ~60-90 min | Depends on Groq API speed |

---

## Environment Variables

Set these in GitHub Actions secrets:

```bash
GROQ_API_KEY              # Your Groq API key (required)
HF_TOKEN                  # HuggingFace token for dataset push (optional)
DAILY_TARGET_LINES        # Default: 1000000
GROQ_MODEL                # Default: llama-3.3-70b-versatile
PARALLEL_WORKERS          # Default: 20 (tune based on rate limits)
HF_REPO                   # Default: Damaru-ai/damru-knowledge
```

---

## Daily Subject Rotation

The pipeline rotates through 10 subjects daily:

1. **human_behaviour** - Emotions, empathy, conversation
2. **psychology** - Motivation, habits, cognitive biases
3. **conversation** - Small talk, listening, de-escalation
4. **coding** - Production code, algorithms, design
5. **mathematics** - Arithmetic to calculus
6. **science** - Physics, chemistry, biology
7. **india_gk** - History, polity, geography
8. **life_skills** - Decision-making, productivity, health
9. **language** - English, Hindi, translation
10. **reasoning** - Logic puzzles, problem-solving

**Pattern**: `subject = SUBJECTS[day_of_year % 10]`

---

## Quality Control

### Filtering:
- Minimum question length: 5 characters
- Minimum answer length: 20 characters  
- Deduplication by first 10 words (case-insensitive)
- Language detection: Auto-tags Hindi (Devanagari ॰०-९)

### Retry Logic:
- HTTP 429 (rate limit): Automatic backoff with `Retry-After` header
- HTTP 5xx: Exponential retry up to 3 times
- Timeout: 120 seconds per request
- Failed batches logged but don't block pipeline

---

## Output Structure

```
learn/
├── daily/
│   ├── 2024-08-15.jsonl      # Today's corpus (1M lines)
│   ├── 2024-08-14.jsonl
│   └── ...
├── daily_manifest.json        # Index of all runs
├── human_behaviour.jsonl      # Seed pack (always included)
└── daily_teacher.py           # Generator script
```

### Manifest Format:
```json
{
  "2024-08-15": {
    "subject": "human_behaviour",
    "records": 76923,
    "approx_lines": 1000145,
    "timestamp": "2024-08-15T18:45:32.123456Z"
  }
}
```

### Daily JSONL Format:
```jsonl
{"question": "What is empathy?", "answer": "Empathy is...", "domain": "human_behaviour", "source": "damru-daily-teach", "intent": "qa", "lang": "en"}
{"question": "क्या है सहानुभूति?", "answer": "सहानुभूति है...", "domain": "human_behaviour", "source": "damru-daily-teach", "intent": "qa", "lang": "hi"}
```

---

## Scaling Considerations

### To increase throughput:

1. **Increase `PARALLEL_WORKERS`** (20 → 50)
   - Monitor for Groq rate limits
   - May need API plan upgrade

2. **Increase `DAILY_TARGET_LINES`** (1M → 10M)
   - Scales linearly with batches
   - Requires proportional API quota

3. **Add more subjects**
   - Edit `SUBJECTS` list in `daily_teacher.py`
   - Enables different domain expertise

4. **Optimize batch size**
   - Current: 50 Q&A/batch
   - Max Groq token limit: ~16K per request

### Cost Estimation (Groq):
- ~1.5K batches/day
- ~2K tokens per batch average
- ~3M tokens/day
- **Free tier**: Should cover this volume
- **Paid tiers**: Extremely cost-effective

---

## Monitoring

### GitHub Actions Logs:
- `[daily-teacher]` prefix for all messages
- Progress: "Progress: X/N batches, Y records"
- Stats: success/failed batch count
- Timing: Start-to-finish elapsed time

### HuggingFace Dataset:
- Auto-synced daily to `Damaru-ai/damru-knowledge`
- Visible at: https://huggingface.co/datasets/Damaru-ai/damru-knowledge

### Local Testing:
```bash
export GROQ_API_KEY="your-key"
export DAILY_TARGET_LINES=5000  # Small test run
python learn/daily_teacher.py
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `GROQ_API_KEY not set` | Add to GitHub secrets or `.env` |
| `rate limit 429` | Reduce `PARALLEL_WORKERS` or wait for backoff |
| `JSON parse error` | Groq returned malformed JSON; auto-retry |
| `Low record count` | Check Groq API status or API plan limits |
| `HF push failed` | Verify `HF_TOKEN` and dataset permissions |

---

## Future Enhancements

- [ ] Adaptive batch sizing based on Groq latency
- [ ] Multi-provider support (Gemini, HuggingFace Inference API)
- [ ] Quality scoring via embedding similarity
- [ ] Streaming to database during generation
- [ ] Distributed worker orchestration
- [ ] A/B testing of LLM prompts

---

**Built by Shiva AI for Damru**
**Last updated**: 2024-08-15
