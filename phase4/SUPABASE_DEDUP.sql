-- =====================================================================
-- DAMRU PERMANENT DEDUP  (run ONCE in Supabase -> SQL Editor)
-- Goal: never store the same question twice again, at the DATABASE level.
-- Safe to re-run. Works together with phase5/store.py (handles 409 skips).
-- =====================================================================

-- STEP 1 -------------------------------------------------------------
-- See how many duplicates exist right now (normalized = lower + trim + collapse spaces).
SELECT count(*) AS total_rows,
       count(DISTINCT md5(lower(btrim(question)))) AS unique_questions,
       count(*) - count(DISTINCT md5(lower(btrim(question)))) AS duplicates
FROM damru_knowledge;

-- STEP 2 -------------------------------------------------------------
-- DELETE duplicate rows, KEEPING the best copy of each question
-- (best = highest upvotes, then the oldest/lowest id).
-- NOTE: this permanently removes rows. The kept copy stays.
DELETE FROM damru_knowledge d
WHERE d.id NOT IN (
    SELECT DISTINCT ON (md5(lower(btrim(question)))) id
    FROM damru_knowledge
    ORDER BY md5(lower(btrim(question))), upvotes DESC, id ASC
);

-- STEP 3 -------------------------------------------------------------
-- Create the UNIQUE index on the normalized question.
-- After this, any INSERT of a question that already exists is rejected (HTTP 409),
-- and phase5/store.py will silently skip it.
CREATE UNIQUE INDEX IF NOT EXISTS uq_damru_qnorm
    ON damru_knowledge (md5(lower(btrim(question))));

-- STEP 4 -------------------------------------------------------------
-- Verify: duplicates should now be 0.
SELECT count(*) AS total_rows,
       count(DISTINCT md5(lower(btrim(question)))) AS unique_questions,
       count(*) - count(DISTINCT md5(lower(btrim(question)))) AS duplicates
FROM damru_knowledge;

-- DONE. From now on duplicates can never enter the table again.
-- (If STEP 3 ever errors with 'could not create unique index ... duplicate key',
--  it means STEP 2 was skipped or new dups arrived -> just run STEP 2 again, then STEP 3.)
