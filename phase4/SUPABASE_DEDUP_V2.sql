-- =====================================================================
-- DAMRU DEDUP V2  (run ONCE in Supabase -> SQL Editor)
-- WHY: the v1 plain UNIQUE index REJECTED duplicates with an ERROR (409),
-- so every duplicate insert attempt spammed Postgres with errors and made the
-- instance 'Unhealthy' (35k+ errors / 34% success rate).
-- FIX: switch to a generated column `qnorm` + UNIQUE CONSTRAINT, so the new
-- phase5/store.py can insert with ON CONFLICT DO NOTHING (silently ignore dupes,
-- no errors). Dedup stays just as strong.
-- Safe to re-run. Run the steps top to bottom.
-- =====================================================================

-- STEP 1: add a generated normalized-question column (DB computes it automatically)
ALTER TABLE damru_knowledge
    ADD COLUMN IF NOT EXISTS qnorm text
    GENERATED ALWAYS AS (md5(lower(btrim(question)))) STORED;

-- STEP 2: drop the old v1 expression index (we move to a constraint on qnorm)
DROP INDEX IF EXISTS uq_damru_qnorm;

-- STEP 3: remove any duplicates that arrived since v1
--         (keep best copy: highest upvotes, then lowest id)
DELETE FROM damru_knowledge d
WHERE d.id NOT IN (
    SELECT DISTINCT ON (qnorm) id
    FROM damru_knowledge
    WHERE qnorm IS NOT NULL
    ORDER BY qnorm, upvotes DESC, id ASC
);

-- STEP 4: add the UNIQUE CONSTRAINT on qnorm (required for ON CONFLICT ignore)
--         If it already exists this will error -- that's fine, skip to STEP 5.
ALTER TABLE damru_knowledge
    ADD CONSTRAINT uq_damru_qnorm UNIQUE (qnorm);

-- STEP 5: verify -> duplicates should be 0
SELECT count(*) AS total_rows,
       count(DISTINCT qnorm) AS unique_questions,
       count(*) - count(DISTINCT qnorm) AS duplicates
FROM damru_knowledge;

-- DONE. After uploading the new phase5/store.py and re-running the workflows,
-- duplicate inserts are silently ignored (no more Postgres error spam),
-- the instance goes back to Healthy, and the table never stores a dup again.
