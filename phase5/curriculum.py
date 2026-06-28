"""
Curriculum = the world's subjects, learned in a ROTATION.

Rotation rule (user-requested):
  - Stay on the current subject until ROTATE_EVERY (default 500) rows are accepted
    for it, then move to the NEXT subject.
  - When a full round over all subjects completes, the pointer wraps back to the
    first subject and each subject CONTINUES with its next batch -> because its
    cumulative count is higher, brain.generate_qa gets a higher `depth` and asks
    progressively more advanced questions ('aage ka content').

Progress persisted in sqlite so it resumes after restarts.

Sharding: when SHARD_TOTAL > 1, each parallel engine covers a DISJOINT strided
slice of subjects (shard 0 -> subjects 0,3,6..; shard 1 -> 1,4,7..; etc.) so
multiple engines running at once never duplicate each other's coverage.
"""
import os
import sqlite3

import config

SUBJECTS = [
    # Mathematics (deep)
    "Mathematics", "Algebra", "Calculus", "Linear Algebra", "Probability",
    "Statistics", "Number Theory", "Geometry", "Topology", "Differential Equations",
    "Discrete Mathematics", "Mathematical Olympiad Problems", "Real Analysis",
    # Physics
    "Physics", "Classical Mechanics", "Electromagnetism", "Quantum Mechanics",
    "Thermodynamics", "Optics", "Astrophysics", "Relativity",
    # Chemistry
    "Chemistry", "Organic Chemistry", "Inorganic Chemistry", "Physical Chemistry",
    "Biochemistry",
    # Biology / Medicine
    "Biology", "Genetics", "Molecular Biology", "Ecology", "Human Anatomy",
    "Neuroscience", "Medicine", "Pharmacology", "Public Health", "Immunology",
    # CS / AI
    "Computer Science", "Algorithms", "Data Structures", "Operating Systems",
    "Databases", "Computer Networks", "Machine Learning", "Deep Learning",
    "Artificial Intelligence", "Cryptography", "Distributed Systems", "Compilers",
    # Engineering
    "Electrical Engineering", "Mechanical Engineering", "Civil Engineering",
    "Electronics", "Robotics", "Control Systems", "Aerospace Engineering",
    # Social / humanities
    "Economics", "Microeconomics", "Macroeconomics", "Finance", "Accounting",
    "Business Management", "History", "World History", "Geography",
    "Political Science", "Law", "Philosophy", "Psychology", "Sociology",
    "Ethics", "Logic", "Linguistics", "Literature",
    # Earth / space / data
    "Environmental Science", "Climate Science", "Astronomy", "Geology",
    "Data Science", "Information Theory",
    # Applied
    "Real World Problem Solving", "Critical Thinking",
]

_DB = os.path.join(config.DATA_DIR, "curriculum.db")


def _pool():
    """Subjects this shard is responsible for."""
    if config.SHARD_TOTAL > 1:
        sub = SUBJECTS[config.SHARD_ID % config.SHARD_TOTAL::config.SHARD_TOTAL]
        return sub or SUBJECTS
    return SUBJECTS


def _ptr_key():
    return "ptr_%d_%d" % (config.SHARD_ID, config.SHARD_TOTAL) if config.SHARD_TOTAL > 1 else "ptr"


def _batch_key():
    return "batch_%d_%d" % (config.SHARD_ID, config.SHARD_TOTAL) if config.SHARD_TOTAL > 1 else "batch"


def _conn():
    c = sqlite3.connect(_DB, timeout=30)
    c.execute(
        "CREATE TABLE IF NOT EXISTS progress ("
        "subject TEXT PRIMARY KEY, count INTEGER DEFAULT 0, "
        "avg_quality REAL DEFAULT 0, mastered INTEGER DEFAULT 0)"
    )
    c.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    return c


def _get_meta(c, k, default=0):
    row = c.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone()
    try:
        return int(row[0]) if row else default
    except Exception:
        return default


def _set_meta(c, k, v):
    c.execute(
        "INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
        (k, str(v)),
    )


def current_subject():
    pool = _pool()
    c = _conn()
    try:
        ptr = _get_meta(c, _ptr_key(), 0)
        return pool[ptr % len(pool)]
    finally:
        c.close()


def depth_of(subject):
    """How many full ROTATE_EVERY batches this subject has already completed.
    Pass 0 = first time; higher = ask deeper/more advanced questions."""
    c = _conn()
    try:
        row = c.execute("SELECT count FROM progress WHERE subject=?", (subject,)).fetchone()
        cnt = int(row[0]) if row else 0
        return cnt // max(1, config.ROTATE_EVERY)
    finally:
        c.close()


def record(subject, n, quality):
    """Add n accepted rows to `subject`. Advance the rotation pointer once the
    CURRENT subject accumulates ROTATE_EVERY rows in this visit.
    Returns True if we just rotated to the next subject."""
    if n <= 0:
        return False
    pool = _pool()
    c = _conn()
    try:
        # cumulative per-subject stats (used for depth + mastered flag)
        c.execute(
            "INSERT INTO progress(subject,count,avg_quality) VALUES(?,?,?) "
            "ON CONFLICT(subject) DO UPDATE SET count=count+?, "
            "avg_quality=(avg_quality+?)/2",
            (subject, n, quality, n, quality),
        )
        cnt = c.execute("SELECT count FROM progress WHERE subject=?", (subject,)).fetchone()[0]
        if cnt >= config.MASTERY_TARGET:
            c.execute("UPDATE progress SET mastered=1 WHERE subject=?", (subject,))
        c.commit()

        # Rotation is driven only by rows on the CURRENT pointer subject. Rows that
        # a weak-subject override sent elsewhere still count for that subject's depth
        # but do not skip the rotation pointer.
        cur = pool[_get_meta(c, _ptr_key(), 0) % len(pool)]
        if subject != cur:
            return False
        batch = _get_meta(c, _batch_key(), 0) + n
        if batch >= config.ROTATE_EVERY:
            ptr = _get_meta(c, _ptr_key(), 0) + 1
            _set_meta(c, _ptr_key(), ptr)
            _set_meta(c, _batch_key(), batch - config.ROTATE_EVERY)  # carry remainder
            c.commit()
            return True
        _set_meta(c, _batch_key(), batch)
        c.commit()
        return False
    finally:
        c.close()


def snapshot():
    c = _conn()
    try:
        rows = c.execute(
            "SELECT subject,count,mastered FROM progress ORDER BY count DESC LIMIT 12"
        ).fetchall()
        return rows
    finally:
        c.close()
