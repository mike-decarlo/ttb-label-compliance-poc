"""
Persists batch/single-run results to SQLite, so a run's output survives
after the terminal closes -- the gap flagged early in the project and
left open until now.

Deliberately a SEPARATE database from rules/label_rules.db: that file is
regenerated config (init_db.py clears and rebuilds it on every run) and
results are an accumulating history that must never be touched by a
rules-regeneration step -- keeping them in different files makes that
impossible to get wrong by accident, rather than relying on remembering
which table a wipe applies to.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

DEFAULT_RESULTS_DB_PATH = "results/results.db"


def init_results_db(db_path: str = DEFAULT_RESULTS_DB_PATH) -> None:
    """
    One-time setup: create the results table if it doesn't exist. Unlike
    build_rules_db(), this does NOT clear existing rows -- results
    accumulate, they aren't regenerated config.
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            label_path TEXT NOT NULL,
            route TEXT,
            overall TEXT NOT NULL,
            field_results TEXT,
            processed_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_result(result: dict, db_path: str = DEFAULT_RESULTS_DB_PATH) -> None:
    """
    Persist one process_single_application()-shaped result dict.
    Self-initializes the table if needed, rather than requiring a
    separate setup step that's easy to forget -- the same lesson the
    rules-db staleness bug taught earlier this session, applied here
    from the start instead of retrofitted after a bug.
    """
    init_results_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO results (label_path, route, overall, field_results, processed_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            result.get("label"),
            result.get("route"),
            result.get("overall"),
            json.dumps(result.get("fields", [])),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()
    conn.close()


def load_results(db_path: str = DEFAULT_RESULTS_DB_PATH, limit: int = 50) -> list[dict]:
    """
    Query helper: most recent results first, ordered by id (not
    timestamp) -- id is guaranteed monotonic with insertion order, while
    two results processed within the same batch could share a timestamp
    at typical precision.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM results ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    results = []
    for row in rows:
        d = dict(row)
        d["fields"] = json.loads(d.pop("field_results") or "[]")
        results.append(d)
    return results