# Damru Source Verifier v1 ✅

## Purpose

`damru_observations` ke raw live signals ko evidence ke hisaab se label karta hai:

```text
pending observation
→ URL/timestamp/injection safety check
→ official authority OR independent corroboration
→ verified / rejected / pending
```

V1 trusted `damru_knowledge` me kuch insert nahi karta. Promotion separate controlled stage hoga.

## Rules

- **Trust A:** USGS/NASA/NOAA/ESA/ISRO/WHO/CDC official domain → verified
- **Trust B:** GDELT/RSS/news → at least 2 independent domains with similar event → verified
- **Trust C:** social/community/YouTube/HN → official match or 2 independent news domains required
- insufficient evidence → pending
- invalid URL, future timestamp or prompt-injection pattern → rejected

## Files

```text
phase9/source_verifier.py
phase9/SOURCE_VERIFIER_README.md
.github/workflows/source-verifier.yml
```

## Safe rollout

1. Upload files preserving folders.
2. Keep repository variable `VERIFY_ENABLED` unset or `0`.
3. Run **Damru Source Verifier v1** manually.
4. Dry-run must print proposed counts and `No rows changed`.
5. Review results.
6. Set repository variable `VERIFY_ENABLED=1` only after dry-run approval.
7. Run manually once; then hourly schedule keeps verifying new pending signals.

## Required GitHub Secrets

```text
SUPABASE_URL
SUPABASE_SERVICE_KEY
```

## Important

Verification is not training. Only a future promotion layer may convert verified observations into grounded Q&A with citations.
