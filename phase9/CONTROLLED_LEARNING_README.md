# Damru Controlled Learning Pilot v1 🧠

## Purpose

Damru learning ko safely restart karna without old high-load recursive loops.

```text
manual 30-minute pilot
→ 10 bounded workers
→ rising quality gate
→ Supabase hot buffer
→ HF sync
→ Storage Guardian at 400 MB
```

## Differences from old learners

- manual only; no cron in pilot
- no self-redispatch
- one engine, no parallel shards
- 30-minute runtime
- 10 total workers instead of ~30 per shard
- coding remains 4× math
- answer target 20–60 useful lines, not 100–150 storage-heavy lines
- stricter starting quality threshold 0.70
- existing dedup, curriculum, weak-topic focus and execution checks preserved

## File

```text
.github/workflows/controlled-learning.yml
```

## Safe rollout

1. Keep old `learn-py.yml`, `learn-py-2.yml`, `learn-py-3.yml` and Distillation disabled.
2. Upload `controlled-learning.yml` to `.github/workflows/`.
3. Run **Damru Controlled Learning Pilot v1** manually once.
4. Check logs for active providers and accepted-row totals.
5. Check Supabase row count and storage after the pilot.
6. HF Sync and Storage Guardian stay active.
7. Only after review, create a scheduled v2 workflow.

## Expected log signals

```text
LLM providers active: ...
workers: general=2 analysis=1 math=1 coding=4 hindi=1 exam=1
TOTAL ...
projected rows/day ...
```

## Required existing secrets

```text
SUPABASE_URL
SUPABASE_KEY
OPENROUTER_KEY
GROQ_API_KEY
GEMINI_KEY
STACKEXCHANGE_KEY
```

Missing teacher keys reduce capabilities but the general public-source harvester can still run.
