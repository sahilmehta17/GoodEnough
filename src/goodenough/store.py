"""
SQLite store. One row per (item, model_role).

The schema is the data model from CLAUDE.md. Everything is append-only: we
never overwrite a result, so a re-run adds rows rather than mutating history.
This is what makes the whole run reproducible and auditable after the fact.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset             TEXT NOT NULL,
    dataset_version     TEXT,
    split               TEXT NOT NULL,
    item_id             TEXT NOT NULL,
    model_role          TEXT NOT NULL,      -- local | hosted
    model_id_requested  TEXT,
    model_id_returned   TEXT,
    semantic_prompt     TEXT,               -- prompt without deployment controls
    rendered_input      TEXT,               -- exact content sent to the model
    raw_response        TEXT,               -- verbatim, never trimmed
    normalized_answer   TEXT,
    parser_version      TEXT,
    parse_status        TEXT,
    correct             INTEGER,            -- 0 | 1
    error               TEXT,
    retries             INTEGER,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    latency_ms_uncached REAL,
    cache_hit           INTEGER,            -- 0 | 1
    finish_reason       TEXT,
    run_date            TEXT,
    seed                INTEGER
);
CREATE INDEX IF NOT EXISTS idx_item ON results (dataset, split, item_id, model_role);
"""


@dataclass
class ResultRow:
    dataset: str
    split: str
    item_id: str
    model_role: str
    model_id_requested: str | None
    model_id_returned: str | None
    semantic_prompt: str | None
    rendered_input: str | None
    raw_response: str | None
    normalized_answer: str | None
    parser_version: str | None
    parse_status: str | None
    correct: bool | None
    error: str | None
    retries: int | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms_uncached: float | None
    cache_hit: bool
    finish_reason: str | None
    run_date: str
    seed: int | None
    dataset_version: str | None = None


def connect(db_path: str) -> sqlite3.Connection:
    # timeout + WAL let a local and a hosted runner write concurrently without
    # tripping "database is locked". WAL allows one writer plus readers and
    # serializes the two writers with short waits rather than errors.
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.executescript(SCHEMA)
    conn.commit()
    return conn


def insert_row(conn: sqlite3.Connection, row: ResultRow) -> int:
    cur = conn.execute(
        """
        INSERT INTO results (
            dataset, dataset_version, split, item_id, model_role,
            model_id_requested, model_id_returned, semantic_prompt, rendered_input,
            raw_response, normalized_answer, parser_version, parse_status, correct,
            error, retries, input_tokens, output_tokens, latency_ms_uncached,
            cache_hit, finish_reason, run_date, seed
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        (
            row.dataset, row.dataset_version, row.split, row.item_id, row.model_role,
            row.model_id_requested, row.model_id_returned, row.semantic_prompt,
            row.rendered_input, row.raw_response, row.normalized_answer,
            row.parser_version, row.parse_status,
            None if row.correct is None else int(row.correct),
            row.error, row.retries, row.input_tokens, row.output_tokens,
            row.latency_ms_uncached, int(row.cache_hit), row.finish_reason,
            row.run_date, row.seed,
        ),
    )
    conn.commit()
    return cur.lastrowid
