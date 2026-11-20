"""Domain logic: event ingestion, path reconstruction, consent withdrawal.

All functions take an open :class:`sqlite3.Connection`; transaction
boundaries are controlled by the caller (see :mod:`app.db`) so that an
API request is always one atomic unit of work.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

from .models import (
    EventIn,
    IngestResult,
    VisitPath,
    WithdrawResult,
)
from .path import pair_visits
from . import transitions


class BadEventTime(ValueError):
    """Raised when event_time is not a timezone-aware ISO-8601 timestamp."""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_utc(value: str) -> datetime:
    """Parse an ISO-8601 timestamp and normalize it to UTC.

    Naive timestamps are rejected: the API contract requires explicit UTC
    so that open/close durations are never computed on ambiguous local time.
    """
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise BadEventTime(f"event_time 不是合法 ISO-8601 时间: {value!r}") from exc
    if dt.tzinfo is None:
        raise BadEventTime(f"event_time 必须带时区（UTC），收到无时区时间: {value!r}")
    return dt.astimezone(timezone.utc)


def is_withdrawn(conn: sqlite3.Connection, visitor_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM withdrawals WHERE visitor_id = ? LIMIT 1",
        (visitor_id,),
    ).fetchone()
    return row is not None


def ingest_event(
    conn: sqlite3.Connection, event: EventIn, *, recompute_stats: bool = True
) -> IngestResult:
    """Store one event. Must be called inside a write transaction.

    Returns ``duplicate`` when ``event_id`` already exists (idempotent
    replay, payload differences are ignored), and ``rejected_withdrawn``
    when the visitor has withdrawn consent — such events are never stored
    and therefore can never resurrect a deleted path.

    ``recompute_stats`` lets a batch caller defer the per-visitor statistics
    refresh until all of the batch's events are inserted, so one visitor's
    path is scanned once per batch rather than once per event.
    """
    if is_withdrawn(conn, event.visitor_id):
        return IngestResult(event_id=event.event_id, status="rejected_withdrawn")

    existing = conn.execute(
        "SELECT 1 FROM events WHERE event_id = ? LIMIT 1",
        (event.event_id,),
    ).fetchone()
    if existing is not None:
        return IngestResult(event_id=event.event_id, status="duplicate")

    # Validated before INSERT so a bad timestamp fails the whole transaction
    # instead of leaving a partial write behind.
    normalized = parse_utc(event.event_time).isoformat()
    conn.execute(
        """
        INSERT INTO events
            (event_id, visitor_id, page_view_id, exhibit_id,
             event_type, event_time, received_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            event.event_id,
            event.visitor_id,
            event.page_view_id,
            event.exhibit_id,
            event.event_type.value,
            normalized,
            utc_now_iso(),
        ),
    )
    # Same transaction as the insert: the visitor's saved transition counts
    # either reflect this event or the event is rolled back with them.
    if recompute_stats:
        transitions.recompute_visitor(conn, event.visitor_id)
    return IngestResult(event_id=event.event_id, status="stored")


def get_visitor_path(conn: sqlite3.Connection, visitor_id: str) -> VisitPath:
    """Reconstruct a visitor's path from committed events.

    Runs in a single read transaction so the withdrawal check and the event
    scan observe one consistent snapshot.
    """
    conn.execute("BEGIN DEFERRED")
    try:
        withdrawn = is_withdrawn(conn, visitor_id)
        if withdrawn:
            # Consent withdrawn: the path must disappear and stay empty.
            conn.rollback()
            return VisitPath(
                visitor_id=visitor_id,
                consent_withdrawn=True,
                visits=[],
                total_visits=0,
            )

        rows = conn.execute(
            """
            SELECT page_view_id, exhibit_id, event_type, event_time
            FROM events
            WHERE visitor_id = ?
            """,
            (visitor_id,),
        ).fetchall()
        visits = pair_visits(rows)
        conn.rollback()  # read-only snapshot; nothing to commit
        return VisitPath(
            visitor_id=visitor_id,
            consent_withdrawn=False,
            visits=visits,
            total_visits=len(visits),
        )
    except Exception:
        conn.rollback()
        raise


def withdraw_consent(conn: sqlite3.Connection, visitor_id: str) -> WithdrawResult:
    """Record withdrawal and delete the visitor's events atomically.

    Must be called inside a write transaction. The withdrawal marker
    persists, so any late upload or replayed old event is rejected
    afterwards and the path can never reappear.
    """
    already = is_withdrawn(conn, visitor_id)
    deleted = conn.execute(
        "DELETE FROM events WHERE visitor_id = ?", (visitor_id,)
    ).rowcount
    conn.execute(
        """
        INSERT INTO withdrawals (visitor_id, withdrawn_at)
        VALUES (?, ?)
        ON CONFLICT(visitor_id) DO UPDATE SET withdrawn_at = withdrawn_at
        """,
        (visitor_id, utc_now_iso()),
    )
    # Remove the visitor's contribution from every day in the same
    # transaction as the event deletion.
    transitions.remove_visitor(conn, visitor_id)
    return WithdrawResult(
        visitor_id=visitor_id,
        consent_withdrawn=True,
        already_withdrawn=already,
        events_deleted=deleted,
    )
