# Damru Rejection Diagnostics v1 🔬

## Purpose

Specialist workers ke `+0` ka exact reason nikaalna without changing any database/model:

```text
provider health
→ generation/JSON parsing
→ coding execution stderr
→ heuristic score + bonus
→ required threshold
→ exact rejection reason
```

## Files

```text
phase9/rejection_diagnostics.py
phase9/REJECTION_DIAGNOSTICS_README.md
.github/workflows/rejection-diagnostics.yml
```

## What it probes

- first configured OpenRouter model
- first configured Groq model
- first configured Gemini model
- one coding generation + real Python execution/tests
- one analysis item
- one Hinglish item
- one exam item
- exact evaluator score vs threshold

## Safety

- manual workflow only
- no Supabase reads/writes
- no HF writes
- no training/model change
- provider errors are redacted
- only small diagnostic API calls

## Run

1. Upload files preserving folders.
2. GitHub Actions → **Damru Rejection Diagnostics v1**.
3. Run workflow manually.
4. Open log or download artifacts:
   - `rejection_diagnostics.json`
   - `rejection_diagnostics.md`
5. Share Recommendations + Worker probes result.

## Existing secrets used

```text
OPENROUTER_KEY
GROQ_API_KEY
GEMINI_KEY
```

If Groq is missing, the report will explicitly recommend adding/fixing `GROQ_API_KEY`.
