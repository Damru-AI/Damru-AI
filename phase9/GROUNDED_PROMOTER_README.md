# Damru Grounded Promoter v1 📚

## Purpose

Only verified live observations become deterministic, source-cited Q&A rows.

```text
verified observation
→ fixed factual template (no LLM)
→ source URL + timestamp + verification score
→ damru_knowledge
→ observation status = promoted
```

Pending and rejected observations are untouched.

## Files

```text
phase9/grounded_promoter.py
phase9/GROUNDED_PROMOTER_README.md
.github/workflows/grounded-promoter.yml
```

## Safety rollout

1. Upload files preserving folders.
2. Leave repository variable `PROMOTE_ENABLED` unset or `0`.
3. Run **Damru Grounded Promoter v1** manually.
4. Review dry-run counts and sample Q&A. No rows change.
5. Set `PROMOTE_ENABLED=1` only after approval.
6. Run once; verified observations become grounded knowledge.
7. Hourly schedule promotes newly verified observations.

## Rules

- verification status must be `verified`
- verification score must be at least `0.70`
- observation must be no older than 30 days
- answer always includes source URL, UTC timestamp and verification evidence
- no LLM is called, so promotion cannot invent extra facts
- duplicate knowledge questions are ignored safely through `qnorm`
- observation is marked `promoted` only after successful knowledge insert

## Required secrets

```text
SUPABASE_URL
SUPABASE_SERVICE_KEY
```

## Important

Storage Guardian may temporarily lock `damru_knowledge` during archiving. In that case promotion fails safely and retries on the next scheduled run; an observation is not marked promoted unless insertion succeeds.
