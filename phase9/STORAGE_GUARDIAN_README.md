# Damru Storage Guardian v1 🛡️

## Goal

When `damru_knowledge` reaches **400 MB**, make sure all rows exist on Hugging Face, then instantly reclaim Supabase storage by truncating the hot table.

```text
Supabase reaches 400 MB
→ maintenance lock
→ compare HF _sync_state.json
→ upload only unsynced IDs
→ verify HF shard + state + manifest
→ guarded TRUNCATE (continue identity)
→ Supabase becomes small
→ new learning continues with higher IDs
```

## Why TRUNCATE, not DELETE?

PostgreSQL `DELETE` creates dead tuples and may not reduce billed database size immediately. `TRUNCATE` releases table/index storage immediately. The Guardian uses it only after HF proof and an exact count/max-ID match.

## Safety

- existing HF files are never deleted or overwritten
- automatic mode is OFF by default
- missing HF state = abort
- upload verification failure = abort
- snapshot count/max changed = abort
- maintenance lock expires automatically
- exception path calls unlock RPC
- HF pre-archive tag and manifest are created
- PostgreSQL identity sequence continues after truncate

## Files

```text
phase9/storage_guardian.py
phase9/STORAGE_GUARDIAN.sql
phase9/STORAGE_GUARDIAN_README.md
.github/workflows/storage-guardian.yml
```

## Setup sequence

1. Keep learning/distillation workflows disabled during first rescue.
2. Supabase SQL Editor: run `phase9/STORAGE_GUARDIAN.sql` once.
3. Upload all files preserving folders.
4. GitHub repository variable:
   - `ARCHIVE_ENABLED` — leave unset or `0` for dry run.
5. Required GitHub Secrets:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY`
   - `HF_TOKEN`
6. Run **Damru Storage Guardian v1** manually.
7. Confirm log says `WOULD ARCHIVE` and shows current size around 578 MB.
8. Only after reviewing dry-run, set repository variable `ARCHIVE_ENABLED=1`.
9. Run manually once. Expected final log: `ARCHIVE COMPLETE`.
10. Supabase usage dashboard can take up to one hour to refresh.

## After rescue

- keep `sync-hf.yml` enabled with `SUPABASE_CLEANUP: 'false'`
- enable learning workflows gradually, not all at once
- Guardian runs at minutes 15 and 45 and archives again whenever table reaches 400 MB
- HF Parquet remains the long-term knowledge source; Supabase is the hot buffer

## Never do

- never run old `reset-sync.py`
- never use a public/anon key for Guardian
- never put `SUPABASE_SERVICE_KEY` in `index.html`
- never set `ARCHIVE_ENABLED=1` before a successful dry run
