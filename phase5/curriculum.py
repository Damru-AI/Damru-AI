"""
Curriculum = the world's subjects, learned ONE AT A TIME until mastery, then next.
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


def _conn():
    c = sqlite3.connect(_DB, timeout=30)
    c.execute(
        "CREATE TABLE IF NOT EXISTS progress ("
        "subject TEXT PRIMARY KEY, count INTEGER DEFAULT 0, "
        "avg_quality REAL DEFAULT 0, mastered INTEGER DEFAULT 0)"
    )
    c.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT)")
    return c


def current_subject():
    pool = _pool()
    c = _conn()
    try:
        row = c.execute("SELECT v FROM meta WHERE k=?", (_ptr_key(),)).fetchone()
        ptr = int(row[0]) if row else 0
        if ptr >= len(pool):
            ptr = ptr % len(pool)  # wrap -> deeper passes
        return pool[ptr]
    finally:
        c.close()


def record(subject, n, quality):
    """Add n mastered items to subject; advance pointer when mastery reached.
    Returns True if the subject just got mastered."""
    if n <= 0:
        return False
    c = _conn()
    try:
        c.execute(
            "INSERT INTO progress(subject,count,avg_quality) VALUES(?,?,?) "
            "ON CONFLICT(subject) DO UPDATE SET count=count+?, "
            "avg_quality=(avg_quality+?)/2",
            (subject, n, quality, n, quality),
        )
        c.commit()
        cnt = c.execute("SELECT count FROM progress WHERE subject=?", (subject,)).fetchone()[0]
        if cnt >= config.MASTERY_TARGET:
            c.execute("UPDATE progress SET mastered=1 WHERE subject=?", (subject,))
            row = c.execute("SELECT v FROM meta WHERE k=?", (_ptr_key(),)).fetchone()
            ptr = (int(row[0]) if row else 0) + 1
            c.execute(
                "INSERT INTO meta(k,v) VALUES(?,?) "
                "ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (_ptr_key(), str(ptr)),
            )
            c.commit()
            return True
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
