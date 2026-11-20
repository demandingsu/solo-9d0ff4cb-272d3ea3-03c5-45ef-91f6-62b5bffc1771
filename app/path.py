"""Visit-path reconstruction shared by path queries and transition stats.

The path is always derived from committed event rows; nothing about the
order is materialized. Both the ``/visitors/{id}/path`` endpoint and the
daily "next exhibit" statistics use this exact pairing so the statistics
always describe the same visit path the API exposes.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime

from .models import Visit


def pair_visits(rows: list[sqlite3.Row]) -> list[Visit]:
    """Pair open/close rows by page_view_id and order the resulting path."""
    opens: dict[str, sqlite3.Row] = {}
    closes: dict[str, list[sqlite3.Row]] = {}

    for row in rows:
        bucket = opens if row["event_type"] == "open" else closes
        if row["event_type"] == "open":
            # Keep the earliest open if a page view ever reports more than one.
            if row["page_view_id"] not in bucket or row["event_time"] < bucket[row["page_view_id"]]["event_time"]:
                bucket[row["page_view_id"]] = row
        else:
            closes.setdefault(row["page_view_id"], []).append(row)

    visits: list[Visit] = []
    for page_view_id, open_row in opens.items():
        opened_at = open_row["event_time"]
        close_rows = closes.get(page_view_id, [])
        # Earliest close for this page view.
        close_row = min(close_rows, key=lambda r: r["event_time"]) if close_rows else None

        closed_at = close_row["event_time"] if close_row is not None else None
        dwell_seconds: float | None = None
        dwell_known = False

        if closed_at is not None:
            open_dt = datetime.fromisoformat(opened_at)
            close_dt = datetime.fromisoformat(closed_at)
            delta = (close_dt - open_dt).total_seconds()
            if delta >= 0:
                dwell_seconds = round(delta, 3)
                dwell_known = True
            # delta < 0: close earlier than its own open — anomalous input,
            # closed_at is returned but stay duration is treated as unknown.

        visits.append(
            Visit(
                page_view_id=page_view_id,
                exhibit_id=open_row["exhibit_id"],
                opened_at=opened_at,
                closed_at=closed_at,
                dwell_seconds=dwell_seconds,
                dwell_known=dwell_known,
                sequence=0,  # assigned after sorting
            )
        )

    # Path order = open time. A late-arriving earlier open therefore simply
    # re-sorts the path on the next query; nothing about the order is stored.
    visits.sort(key=lambda v: (v.opened_at, v.page_view_id))
    for index, visit in enumerate(visits, start=1):
        visit.sequence = index
    return visits
