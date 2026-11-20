"""SQLite connection handling and schema management.

Concurrency model
-----------------
SQLite in WAL mode allows multiple readers concurrently with one writer.
Every state-changing request runs inside an explicit ``BEGIN IMMEDIATE``
transaction, which acquires the reserved (write) lock up front. Combined
with the busy timeout, uploads and withdrawals are serialized against one
another and each sees a fully committed state. Reads use their own
snapshot transaction, so a query never observes a half-written path:
paths are *derived* from committed event rows at read time, there is no
separate path table that could be updated partially.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    event_id      TEXT    NOT NULL PRIMARY KEY,
    visitor_id    TEXT    NOT NULL,
    page_view_id  TEXT    NOT NULL,
    exhibit_id    TEXT    NOT NULL,
    event_type    TEXT    NOT NULL CHECK (event_type IN ('open', 'close')),
    event_time    TEXT    NOT NULL,
    received_at   TEXT    NOT NULL
);

-- Hot lookup paths: pairing by page view, path ordering by open time,
-- and per-visitor scans / consent checks.
CREATE INDEX IF NOT EXISTS idx_events_visitor_time
    ON events (visitor_id, event_time);
CREATE INDEX IF NOT EXISTS idx_events_visitor_page
    ON events (visitor_id, page_view_id);

CREATE TABLE IF NOT EXISTS withdrawals (
    visitor_id  TEXT    NOT NULL PRIMARY KEY,
    withdrawn_at TEXT   NOT NULL
);

-- Derived "next exhibit" statistics. One row per visitor per day per
-- direction collapses repeat walks by the same person on the same day.
CREATE TABLE IF NOT EXISTS visitor_daily_transitions (
    day          TEXT NOT NULL,
    from_exhibit TEXT NOT NULL,
    to_exhibit   TEXT NOT NULL,
    visitor_id   TEXT NOT NULL,
    PRIMARY KEY (day, from_exhibit, to_exhibit, visitor_id)
);
-- Recomputation and withdrawal look up / delete a single visitor's rows.
CREATE INDEX IF NOT EXISTS idx_visitor_daily_transitions_visitor
    ON visitor_daily_transitions (visitor_id);

-- Saved visitor counts read by the daily query API.
CREATE TABLE IF NOT EXISTS daily_transitions (
    day          TEXT    NOT NULL,
    from_exhibit TEXT    NOT NULL,
    to_exhibit   TEXT    NOT NULL,
    visitors     INTEGER NOT NULL CHECK (visitors >= 0),
    PRIMARY KEY (day, from_exhibit, to_exhibit)
);
CREATE INDEX IF NOT EXISTS idx_daily_transitions_from
    ON daily_transitions (day, from_exhibit);

-- Startup markers (e.g. which derivation version the stats were built with).
CREATE TABLE IF NOT EXISTS stats_meta (
    key   TEXT NOT NULL PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def connect(db_path: str | None = None) -> sqlite3.Connection:
    """Open a connection with the pragmas required for safe concurrency."""
    conn = sqlite3.connect(
        db_path or config.DB_PATH,
        timeout=config.BUSY_TIMEOUT_MS / 1000.0,
        isolation_level=None,  # manual transaction control
        # Sync FastAPI endpoints run in uvicorn's worker threadpool, so the
        # connection created by the dependency may be touched from a pool
        # thread. Each request gets its own connection (never shared), which
        # makes cross-thread use safe; SQLite file locking + BEGIN IMMEDIATE
        # still serialize the actual writers.
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout = {config.BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db(db_path: str | None = None) -> None:
    """Create tables and indexes if they do not yet exist.

    Also upgrades a legacy database on its first startup: statistics tables
    are created empty and then backfilled from the stored event history, so
    the daily query has data without replaying old reports.
    """
    from .transitions import ensure_stats_ready

    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        ensure_stats_ready(conn)
    finally:
        conn.close()


def get_connection() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency: one connection per request, closed afterwards.

    FastAPI consumes this generator directly: the yielded connection is
    injected into the endpoint, and the ``finally`` block runs when the
    request finishes. Do not wrap it in ``@contextmanager`` — doing so hides
    the generator signature from FastAPI's dependency analysis.
    """
    conn = connect()
    try:
        yield conn
    finally:
        conn.close()


@contextmanager
def write_transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Acquire the write lock immediately and commit/rollback atomically."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
