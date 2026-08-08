# Damru Direct-Learning (learn/)

Daily direct-teaching pipeline that grows Damru knowledge every day and wires it into the RAG brain.

## Files
- daily_teacher.py: builds the daily corpus (schema: question, answer, domain, source, intent, lang) and can push it to the HF dataset Damaru-ai/damru-knowledge.
- human_behaviour.jsonl: curated seed pack on human behaviour, emotions, social cues and conversation. Always included when the daily subject rotates to human_behaviour.
- daily/YYYY-MM-DD.jsonl: the dated learning log committed each day by the cron.
- daily_manifest.json: index of daily runs (subject, record count, approx lines).

## Scale and honesty
The daily GitHub Action (.github/workflows/daily-learn.yml) targets about 50000 teaching lines per day. That volume is produced by an LLM API through the GROQ_API_KEY secret, quality gated, across a rotating subject list. Without the key it writes only the curated seed packs. The daily file is committed to git and, when HF_TOKEN is set, uploaded to the knowledge dataset so the brain ingests it.

## Add a new seed pack
Drop any .jsonl file in this folder with objects that have question and answer fields. It is picked up automatically on the next run.
