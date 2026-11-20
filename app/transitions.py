"""Materialized daily exhibit-to-exhibit visitor counts ("next exhibit").

The source of truth remains the ``events`` table; the daily counts are a
derived cache so that the exhibition page can answer "where do visitors go
after opening exhibit X on day D?" with a single indexed lookup instead of
reconstructing every visitor's path on each request.

Grain
-----
* ``visitor_daily_transitions`` stores one row per (day, from, to, visitor).
  The primary key collapses repeated traversals of the same direction by the
  same visitor on the same day, so "one person per direction per day" holds
  even after a late event splits one transition into two.
* ``daily_transitions`` stores the saved visitor count per (day, from, to),
  recomputed incrementally from the per-visitor rows. Queries read this
  table directly.

Day key
-------
A transition is attributed to the UTC open date of its *destination* visit
(``B.opened_at`` for a path segment A -> B). The path itself is the same one
returned by ``GET /visitors/{id}/path``: adjacent stops with different
exhibits, ordered by open time.

Update strategy
---------------
Ingestion recomputes only the reporting visitor's contributions from that
visitor's full path, inside the same write transaction as the event insert.
That keeps a late report that splits A -> C into A -> B and B -> C correct
without rescanning all visitors, and the per-visitor primary key keeps a
repeated direction at one person for the day. Withdrawal deletes the
visitor's rows for every day and refreshes the affected saved counts, also
inside the withdrawal transaction.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from typing import Iterable

from .models import DailyTransitions, TransitionCount
from .path import pair_visits

# Bump when the derivation rules change; startup will rebuild from events.
STATS_VERSION_KEY = "daily_transitions_version"
STATS_VERSION = "1"

TransitionKey = tuple[str, str, str]  # (day, from_exhibit, to_exhibit)


def utc_day(timestamp: str) -> str:
    """UTC calendar date (YYYY-MM-DD) of a normalized event timestamp."""
    return datetime.fromisoformat(timestamp).date().isoformat()


def contributions(visits) -> set[TransitionKey]:
    """Distinct (day, from, to) keys contributed by one visitor's path.

    Adjacent stops with the same exhibit are not a transition. The result is
    a set because a visitor walking the same direction several times on one
    day still counts as one person.
    """
    keys: set[TransitionKey] = set()
    for previous, following in zip(visits, visits[1:]):
        if previous.exhibit_id == following.exhibit_id:
            continue
        keys.add(
            (
                utc_day(following.opened_at),
                previous.exhibit_id,
                following.exhibit_id,
            )
        )
    return keys


def _visitor_rows(conn: sqlite3.Connection, visitor_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT page_view_id, exhibit_id, event_type, event_time
        FROM events
        WHERE visitor_id = ?
        """,
        (visitor_id,),
    ).fetchall()


def _visitor_keys(conn: sqlite3.Connection, visitor_id: str) -> set[TransitionKey]:
    return contributions(pair_visits(_visitor_rows(conn, visitor_id)))


def _refresh_counts(
    conn: sqlite3.Connection, keys: Iterable[TransitionKey]
) -> None:
    """Recompute saved counts for the given keys from per-visitor rows."""
    for day, from_exhibit, to_exhibit in set(keys):
        count = conn.execute(
            """
            SELECT COUNT(*) FROM visitor_daily_transitions
            WHERE day = ? AND from_exhibit = ? AND to_exhibit = ?
            """,
            (day, from_exhibit, to_exhibit),
        ).fetchone()[0]
        if count:
            conn.execute(
                """
                INSERT INTO daily_transitions
                    (day, from_exhibit, to_exhibit, visitors)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(day, from_exhibit, to_exhibit)
                DO UPDATE SET visitors = excluded.visitors
                """,
                (day, from_exhibit, to_exhibit, count),
            )
        else:
            conn.execute(
                """
                DELETE FROM daily_transitions
                WHERE day = ? AND from_exhibit = ? AND to_exhibit = ?
                """,
                (day, from_exhibit, to_exhibit),
            )


def recompute_visitor(conn: sqlite3.Connection, visitor_id: str) -> None:
    """Recompute one visitor's contributions after their events changed.

    Must run inside the same write transaction as the event insert. A late
    event may reorder the visitor's path or split one segment into two, so
    the visitor's full path is rebuilt. Contributions are stored as a set:
    if the visitor still walked the same direction that day through another
    pair of stops, the direction remains present and is still one person.
    """
    old = {
        (row["day"], row["from_exhibit"], row["to_exhibit"])
        for row in conn.execute(
            """
            SELECT day, from_exhibit, to_exhibit
            FROM visitor_daily_transitions
            WHERE visitor_id = ?
            """,
            (visitor_id,),
        ).fetchall()
    }
    new = _visitor_keys(conn, visitor_id)
    if old == new:
        return

    conn.execute(
        "DELETE FROM visitor_daily_transitions WHERE visitor_id = ?",
        (visitor_id,),
    )
    if new:
        conn.executemany(
            """
            INSERT INTO visitor_daily_transitions
                (day, from_exhibit, to_exhibit, visitor_id)
            VALUES (?, ?, ?, ?)
            """,
            [(day, src, dst, visitor_id) for day, src, dst in new],
        )
    # Only pairs whose contributor set changed need their saved count refreshed.
    _refresh_counts(conn, old | new)


def remove_visitor(conn: sqlite3.Connection, visitor_id: str) -> None:
    """Remove all of a visitor's saved contributions (consent withdrawal)."""
    old = {
        (row["day"], row["from_exhibit"], row["to_exhibit"])
        for row in conn.execute(
            """
            SELECT day, from_exhibit, to_exhibit
            FROM visitor_daily_transitions
            WHERE visitor_id = ?
            """,
            (visitor_id,),
        ).fetchall()
    }
    if not old:
        return
    conn.execute(
        "DELETE FROM visitor_daily_transitions WHERE visitor_id = ?",
        (visitor_id,),
    )
    _refresh_counts(conn, old)


def get_daily(
    conn: sqlite3.Connection, day: str, from_exhibit: str | None = None
) -> DailyTransitions:
    """Read saved counts for one UTC day; never recomputes on the read path."""
    sql = """
        SELECT from_exhibit, to_exhibit, visitors
        FROM daily_transitions
        WHERE day = ?
    """
    params: list[object] = [day]
    if from_exhibit is not None:
        sql += " AND from_exhibit = ?"
        params.append(from_exhibit)
    sql += " ORDER BY visitors DESC, from_exhibit ASC, to_exhibit ASC"
    rows = conn.execute(sql, params).fetchall()
    return DailyTransitions(
        date=datetime.fromisoformat(day).date(),
        from_exhibit=from_exhibit,
        transitions=[
            TransitionCount(
                from_exhibit=row["from_exhibit"],
                to_exhibit=row["to_exhibit"],
                visitors=row["visitors"],
            )
            for row in rows
        ],
    )


def backfill_all(conn: sqlite3.Connection) -> None:
    """Rebuild both stats tables from the event history (legacy DB upgrade)."""
    conn.execute("DELETE FROM visitor_daily_transitions")
    conn.execute("DELETE FROM daily_transitions")
    visitor_ids = [
        row[0]
        for row in conn.execute(
            """
            SELECT DISTINCT e.visitor_id
            FROM events AS e
            WHERE NOT EXISTS (
                SELECT 1 FROM withdrawals AS w WHERE w.visitor_id = e.visitor_id
            )
            ORDER BY e.visitor_id
            """
        ).fetchall()
    ]
    for visitor_id in visitor_ids:
        keys = _visitor_keys(conn, visitor_id)
        if keys:
            conn.executemany(
                """
                INSERT INTO visitor_daily_transitions
                    (day, from_exhibit, to_exhibit, visitor_id)
                VALUES (?, ?, ?, ?)
                """,
                [(day, src, dst, visitor_id) for day, src, dst in keys],
            )
    conn.execute(
        """
        INSERT INTO daily_transitions
            (day, from_exhibit, to_exhibit, visitors)
        SELECT day, from_exhibit, to_exhibit, COUNT(*)
        FROM visitor_daily_transitions
        GROUP BY day, from_exhibit, to_exhibit
        """
    )


def ensure_stats_ready(conn: sqlite3.Connection) -> None:
    """Backfill history exactly once, on the first startup of a legacy DB.

    Runs under ``BEGIN IMMEDIATE``: with several uvicorn workers starting at
    once, the first one backfills and records the marker, late ones re-check
    the marker after acquiring the lock and do nothing.
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        row = conn.execute(
            "SELECT value FROM stats_meta WHERE key = ?",
            (STATS_VERSION_KEY,),
        ).fetchone()
        if row is not None and row["value"] == STATS_VERSION:
            conn.rollback()
            return
        backfill_all(conn)
        conn.execute(
            """
            INSERT INTO stats_meta (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            (STATS_VERSION_KEY, STATS_VERSION),
        )
    except Exception:
        conn.rollback()
        raise
    else:
        conn.commit()
