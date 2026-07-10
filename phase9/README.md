# Damru World Observer v1 🌍

## Purpose

Damru ko real world ke near-real-time signals dena—without illegal scraping, backdoors, login bypass or direct pollution of the trusted training dataset.

```text
Public live sources
→ World Observer
→ damru_observations (pending quarantine)
→ verification / corroboration
→ approved knowledge only
→ RAG immediately, training later
```

## Sources in v1

| Source | Signal | Key |
|---|---|---|
| USGS | live earthquakes | none |
| NASA EONET | fires, storms, volcanoes, natural events | none |
| GDELT | global latest news index | none |
| Hacker News | technology/community signals | none |
| RSS/Atom | configured official/news feeds | none |
| Bluesky public search | public social signals | none; optional query |
| YouTube Data API | latest videos from configured topics | `YOUTUBE_KEY`, optional |

Trust tiers:
- **A:** official authority/sensor source
- **B:** news index or configured RSS feed
- **C:** social/community signal; never accept without corroboration

## Files

```text
phase9/world_observer.py
phase9/WORLD_OBSERVER.sql
phase9/README.md
.github/workflows/world-observer.yml
```

## Non-coder setup

1. Supabase → **SQL Editor** → paste and run `WORLD_OBSERVER.sql` once.
2. GitHub repository → **Settings → Secrets and variables → Actions**.
3. Add secrets by name only:
   - `SUPABASE_URL`
   - `SUPABASE_SERVICE_KEY` (server-side only; never put in `index.html`)
   - optional `YOUTUBE_KEY`
4. Upload files preserving their folders.
5. GitHub → **Actions → Damru World Observer v1 → Run workflow**.
6. Check workflow artifact `damru-world-observations` and Supabase table `damru_observations`.

## Configuration variables

- `WORLD_QUERIES`: comma-separated global news topics
- `RSS_URLS`: comma-separated RSS/Atom feeds
- `BLUESKY_QUERIES`: comma-separated public social searches
- `YOUTUBE_QUERIES`: comma-separated YouTube searches
- `MAX_PER_SOURCE`: rows per source/run (default 20)

## Safety boundary

World Observer does **not** write into `damru_knowledge`. This is intentional.

Raw news/social signals can be false, duplicated or manipulated. A separate verifier must require:
- official source or two-source corroboration,
- freshness and URL validation,
- deduplication,
- contradiction check,
- quality score,
- then promotion to trusted memory.

## What this is not

- no private-account access
- no authentication bypass
- no paywall bypass
- no covert scraping
- no unauthorized satellite/control-system access

The beast route is official/public APIs + authorized connectors + strong verification.
