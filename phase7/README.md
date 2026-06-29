# Damru phase7 — Bulk Harvester (direct-to-HuggingFace)

Pull millions of **genuine** open Q&A rows → dedup → write straight to the HF
dataset. **Supabase is bypassed** for bulk, so the 500MB NANO buffer is no
longer the bottleneck. The live engine (phase5) keeps adding fresh
Indian-exam / Hindi / self-checked rows on top.

## Files to upload to GitHub
- `phase7/dedup_bloom.py` — persistent bloom dedup filter (no dupes ever)
- `phase7/bulk_harvest.py` — the harvester
- `.github/workflows/bulk-harvest.yml` — scheduled + manual, auto-resumes

## Secrets needed (already in repo)
- `HF_TOKEN` — that's it. (No Supabase needed for this track.)

## How it works
1. Loads `_dedup.bloom.gz` + `_bulk_state.json` from the HF dataset repo.
2. Streams each dataset; maps to `{question, answer, intent, lang, upvotes, created_at}`.
3. Bloom filter drops any duplicate/copied question (across ALL runs + the live track).
4. Writes kept rows as `data/bulk-*.parquet` shards into the HF repo.
5. Saves bloom + state after every shard → safe to stop/resume any time.
6. `concurrency: 1` so only one run touches the bloom at a time.

`load_dataset("Damaru-ai/damru-knowledge")` reads live + bulk shards together.

## Run it
- **Auto:** every 3 hours via cron — just upload and forget. It resumes until
  every dataset is done.
- **Manual:** Actions → "Damru Bulk Harvest" → Run workflow.
  - `only` (optional): e.g. `math,coding` to run a subset first.
  - `per_dataset`: max kept rows per dataset per pass.

### First-time test (recommended)
Run manually with `only = camel,sciq` and `per_dataset = 50000` to confirm
shards + bloom + state appear on HF, then let the schedule take over.

## Knobs (env in the workflow)
| var | default | meaning |
|-----|---------|---------|
| PER_DATASET | 1200000 | max kept rows per dataset per pass |
| SHARD_SIZE | 100000 | rows per parquet shard |
| RUN_BUDGET_MIN | 320 | soft time budget per run (job timeout 350) |
| ONLY | (empty) | comma-substring filter of dataset ids |
| MIN_Q / MIN_A | 8 / 40 | min question / answer length |
| BLOOM_CAPACITY | 60000000 | expected unique items (sizes the filter) |
| BLOOM_ERROR | 0.01 | bloom false-positive rate (~72MB file) |

## Sources currently wired (graceful-skip if a field guess is wrong)
Math: OpenMathInstruct-2, MetaMathQA, MathInstruct, GSM8K ·
Reasoning: WebInstructSub, OpenThoughts-114k, OpenOrca, Open-Platypus, OpenHermes-2.5 ·
Coding: OpenCodeInstruct, Magicoder, glaive-code-assistant ·
Science: CAMEL physics/chemistry/biology/math, SciQ ·
Medical/Nursing: MedMCQA, MedNurse-QA · Indian exams: ExamBench.

> Add more by appending to `DATASETS` in `bulk_harvest.py`.
> ⚠️ Licenses vary (some gated / non-commercial). Fine for training/experiments;
> check the license before any commercial release.
