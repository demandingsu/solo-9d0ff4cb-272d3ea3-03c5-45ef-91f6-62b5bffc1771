"""Tests for the daily "next exhibit" statistics endpoint.

Rules under test:
  * adjacent visits with different exhibits, destination's UTC open date
  * same visitor, same direction, same day -> one person
  * late report that splits A -> C into A -> B, B -> C recomputes only that
    visitor's contribution; if the visitor still walked A -> C that day the
    direction stays at one person
  * withdrawals subtract the visitor from every day
  * single / batch ingestion and withdrawal update counts atomically
  * a legacy database is backfilled once on first startup
  * the query reads saved counts and never recomputes on the read path
"""
from __future__ import annotations

import os
import sqlite3

from fastapi.testclient import TestClient

from app import config
from app.db import init_db


def _event(event_id, visitor, page_view, exhibit, event_type, event_time):
    return {
        "event_id": event_id,
        "visitor_id": visitor,
        "page_view_id": page_view,
        "exhibit_id": exhibit,
        "event_type": event_type,
        "event_time": event_time,
    }


def _visit(client, visitor, page_view, exhibit, opened_at, closed_at=None):
    _post(client, event_id=f"{page_view}-o", visitor=visitor,
          page_view=page_view, exhibit=exhibit, event_type="open",
          event_time=opened_at)
    if closed_at is not None:
        _post(client, event_id=f"{page_view}-c", visitor=visitor,
              page_view=page_view, exhibit=exhibit, event_type="close",
              event_time=closed_at)


def _post(client, **kwargs):
    return client.post("/events", json=_event(**kwargs))


def _stats(client, day, **params):
    resp = client.get("/stats/transitions", params={"date": day, **params})
    assert resp.status_code == 200, resp.text
    return resp.json()


def _pairs(body):
    return {(t["from_exhibit"], t["to_exhibit"]): t["visitors"]
            for t in body["transitions"]}


# ---------------------------------------------------------------------------
# Basic shape and attribution rules.
# ---------------------------------------------------------------------------
def test_adjacent_different_exhibits_counted_by_destination_open_date(client):
    v = "v-adjacent"
    # A -> A (same exhibit, not a transition), A -> B, B -> C.
    _visit(client, v, "pv1", "A", "2026-11-18T07:00:00Z")
    _visit(client, v, "pv2", "A", "2026-11-18T07:10:00Z")
    # B opens just before midnight; C opens after midnight -> different days.
    _visit(client, v, "pv3", "B", "2026-11-18T23:50:00Z")
    _visit(client, v, "pv4", "C", "2026-11-19T00:10:00Z")

    day18 = _stats(client, "2026-11-18")
    assert day18["date"] == "2026-11-18"
    assert day18["from_exhibit"] is None
    assert _pairs(day18) == {("A", "B"): 1}
    day19 = _stats(client, "2026-11-19")
    assert _pairs(day19) == {("B", "C"): 1}


def test_same_visitor_same_direction_same_day_counts_once(client):
    v = "v-repeat"
    # Walks A -> B twice, then B -> A on the same UTC day.
    _visit(client, v, "pv1", "A", "2026-11-18T08:00:00Z")
    _visit(client, v, "pv2", "B", "2026-11-18T08:10:00Z")
    _visit(client, v, "pv3", "A", "2026-11-18T09:00:00Z")
    _visit(client, v, "pv4", "B", "2026-11-18T09:10:00Z")

    # A -> B is collapsed; B -> A is the reverse direction and counts.
    assert _pairs(_stats(client, "2026-11-18")) == {("A", "B"): 1, ("B", "A"): 1}

    # A different visitor walking the same direction adds one person.
    _visit(client, "v-repeat-2", "pv5", "A", "2026-11-18T10:00:00Z")
    _visit(client, "v-repeat-2", "pv6", "B", "2026-11-18T10:10:00Z")
    assert _pairs(_stats(client, "2026-11-18")) == {("A", "B"): 2, ("B", "A"): 1}


def test_direction_is_ordered(client):
    v = "v-direction"
    _visit(client, v, "pv1", "A", "2026-11-18T08:00:00Z")
    _visit(client, v, "pv2", "B", "2026-11-18T08:10:00Z")
    _visit(client, v, "pv3", "A", "2026-11-18T08:20:00Z")
    assert _pairs(_stats(client, "2026-11-18")) == {("A", "B"): 1, ("B", "A"): 1}


def test_from_exhibit_filter(client):
    v = "v-filter"
    _visit(client, v, "pv1", "A", "2026-11-18T08:00:00Z")
    _visit(client, v, "pv2", "B", "2026-11-18T08:10:00Z")
    _visit(client, v, "pv3", "C", "2026-11-18T08:20:00Z")

    filtered = _stats(client, "2026-11-18", from_exhibit="B")
    assert filtered["from_exhibit"] == "B"
    assert _pairs(filtered) == {("B", "C"): 1}

    empty = _stats(client, "2026-11-18", from_exhibit="Z")
    assert empty["transitions"] == []


def test_invalid_date_is_rejected(client):
    assert client.get("/stats/transitions",
                      params={"date": "18-11-2026"}).status_code == 422


# ---------------------------------------------------------------------------
# Late reports: only the reporting visitor is recomputed.
# ---------------------------------------------------------------------------
def test_late_report_splits_segment_and_removes_direct_pair(client):
    v = "v-split"
    _visit(client, v, "pv1", "A", "2026-11-18T08:00:00Z")
    # C is reported first, so the path is A -> C.
    _visit(client, v, "pv3", "C", "2026-11-18T09:00:00Z")
    assert _pairs(_stats(client, "2026-11-18")) == {("A", "C"): 1}

    # Backfill of B (opened between A and C) splits A -> C into A -> B, B -> C.
    _visit(client, v, "pv2", "B", "2026-11-18T08:30:00Z")
    assert _pairs(_stats(client, "2026-11-18")) == {
        ("A", "B"): 1,
        ("B", "C"): 1,
    }


def test_split_keeps_direction_at_one_person_if_still_walked_that_day(client):
    v = "v-split-repeat"
    _visit(client, v, "pv1", "A", "2026-11-18T08:00:00Z")
    _visit(client, v, "pv2", "C", "2026-11-18T08:30:00Z")
    # Same direction A -> C was also walked later that day.
    _visit(client, v, "pv3", "A", "2026-11-18T10:00:00Z")
    _visit(client, v, "pv4", "C", "2026-11-18T10:30:00Z")
    assert _pairs(_stats(client, "2026-11-18")) == {("A", "C"): 1, ("C", "A"): 1}

    # Backfilled B only splits the first A -> C; the second remains direct.
    _visit(client, v, "pv1b", "B", "2026-11-18T08:15:00Z")
    # C -> A comes from the second walk's return leg, not from the split.
    assert _pairs(_stats(client, "2026-11-18")) == {
        ("A", "B"): 1,
        ("B", "C"): 1,
        ("C", "A"): 1,
        ("A", "C"): 1,
    }


def test_late_report_recomputes_only_that_visitor(client):
    _visit(client, "v-other", "po1", "X", "2026-11-18T08:00:00Z")
    _visit(client, "v-other", "po2", "Y", "2026-11-18T08:30:00Z")
    v = "v-late"
    _visit(client, v, "pv1", "A", "2026-11-18T08:00:00Z")
    _visit(client, v, "pv2", "B", "2026-11-18T08:30:00Z")
    assert _pairs(_stats(client, "2026-11-18"))[("X", "Y")] == 1

    # A late earlier open for v reorders only v's path.
    _visit(client, v, "pv0", "Z", "2026-11-18T07:00:00Z")
    pairs = _pairs(_stats(client, "2026-11-18"))
    assert pairs[("Z", "A")] == 1
    assert pairs[("A", "B")] == 1
    # Other visitor's saved contribution is untouched.
    assert pairs[("X", "Y")] == 1


# ---------------------------------------------------------------------------
# Withdrawal and atomicity.
# ---------------------------------------------------------------------------
def test_withdrawal_subtracts_all_dates_for_that_visitor(client):
    v = "v-withdraw"
    _visit(client, v, "pv1", "A", "2026-11-18T08:00:00Z")
    _visit(client, v, "pv2", "B", "2026-11-18T08:10:00Z")
    _visit(client, v, "pv3", "B", "2026-11-19T08:00:00Z")
    _visit(client, v, "pv4", "C", "2026-11-19T08:10:00Z")
    _visit(client, "v-keep", "pk1", "A", "2026-11-18T09:00:00Z")
    _visit(client, "v-keep", "pk2", "B", "2026-11-18T09:10:00Z")

    assert _pairs(_stats(client, "2026-11-18")) == {("A", "B"): 2}
    assert _pairs(_stats(client, "2026-11-19")) == {("B", "C"): 1}

    assert client.post(f"/visitors/{v}/withdraw").status_code == 200

    assert _pairs(_stats(client, "2026-11-18")) == {("A", "B"): 1}
    assert _stats(client, "2026-11-19")["transitions"] == []


def test_duplicate_event_does_not_change_counts(client):
    v = "v-dup"
    payload = _event("d1", v, "pv1", "A", "open", "2026-11-18T08:00:00Z")
    client.post("/events", json=payload)
    _visit(client, v, "pv2", "B", "2026-11-18T08:10:00Z")
    before = _pairs(_stats(client, "2026-11-18"))
    assert client.post("/events", json=payload).json()["status"] == "duplicate"
    assert _pairs(_stats(client, "2026-11-18")) == before


def test_failed_batch_rolls_back_path_and_counts(client):
    v = "v-bad-batch"
    _visit(client, v, "pv1", "A", "2026-11-18T08:00:00Z")
    _visit(client, v, "pv2", "B", "2026-11-18T08:30:00Z")
    before = _pairs(_stats(client, "2026-11-18"))

    resp = client.post("/events/batch", json={"events": [
        _event("g1", v, "pv3", "C", "open", "2026-11-18T09:00:00Z"),
        _event("g2", v, "pv3", "C", "close", "not-a-timestamp"),
        # Another visitor's otherwise valid event must roll back with it.
        _event("g3", "v-bad-batch-2", "pv4", "P", "open",
               "2026-11-18T09:00:00Z"),
        _event("g4", "v-bad-batch-2", "pv5", "Q", "open",
               "2026-11-18T09:10:00Z"),
    ]})
    assert resp.status_code == 400

    # Neither the path nor the saved counts contain the rolled-back batch.
    path = client.get(f"/visitors/{v}/path").json()
    assert [x["exhibit_id"] for x in path["visits"]] == ["A", "B"]
    assert _pairs(_stats(client, "2026-11-18")) == before
    assert client.get("/visitors/v-bad-batch-2/path").json()["total_visits"] == 0


def test_failed_single_event_leaves_counts_untouched(client):
    v = "v-bad-single"
    _visit(client, v, "pv1", "A", "2026-11-18T08:00:00Z")
    _visit(client, v, "pv2", "B", "2026-11-18T08:10:00Z")
    before = _pairs(_stats(client, "2026-11-18"))

    resp = _post(client, event_id="bad", visitor=v, page_view="pv3",
                 exhibit="C", event_type="open",
                 event_time="2026-11-18T09:00:00")  # naive -> 400
    assert resp.status_code == 400
    assert _pairs(_stats(client, "2026-11-18")) == before


def test_batch_of_only_rejected_events_leaves_counts_alone(client):
    _visit(client, "v-batch-rej", "po1", "X", "2026-11-18T08:00:00Z")
    _visit(client, "v-batch-rej", "po2", "Y", "2026-11-18T08:10:00Z")
    assert client.post("/visitors/v-batch-rej/withdraw").status_code == 200
    before = _pairs(_stats(client, "2026-11-18"))

    resp = client.post("/events/batch", json={"events": [
        _event("r1", "v-batch-rej", "pv1", "A", "open",
               "2026-11-18T09:00:00Z"),
        _event("r2", "v-batch-rej", "pv2", "B", "open",
               "2026-11-18T09:10:00Z"),
    ]})
    assert resp.status_code == 201
    assert resp.json()["rejected_withdrawn"] == 2
    assert _pairs(_stats(client, "2026-11-18")) == before


def test_batch_insert_updates_counts_in_same_transaction(client):
    v = "v-batch"
    resp = client.post("/events/batch", json={"events": [
        _event("b1", v, "pv1", "A", "open", "2026-11-18T08:00:00Z"),
        _event("b2", v, "pv1", "A", "close", "2026-11-18T08:05:00Z"),
        _event("b3", v, "pv2", "B", "open", "2026-11-18T08:10:00Z"),
    ]})
    assert resp.status_code == 201
    assert _pairs(_stats(client, "2026-11-18")) == {("A", "B"): 1}


# ---------------------------------------------------------------------------
# Query reads saved counts; legacy databases are backfilled at startup.
# ---------------------------------------------------------------------------
def test_query_reads_saved_counts_without_recomputing(client):
    # Insert event rows behind the API's back: the read path must not notice
    # them, because it reads the saved stats table rather than event history.
    conn = sqlite3.connect(config.DB_PATH, isolation_level=None)
    try:
        conn.execute(
            """
            INSERT INTO events
                (event_id, visitor_id, page_view_id, exhibit_id,
                 event_type, event_time, received_at)
            VALUES ('raw1', 'v-raw', 'pv1', 'A', 'open',
                    '2026-11-18T08:00:00+00:00', '2026-11-18T08:00:00+00:00')
            """
        )
        conn.execute(
            """
            INSERT INTO events
                (event_id, visitor_id, page_view_id, exhibit_id,
                 event_type, event_time, received_at)
            VALUES ('raw2', 'v-raw', 'pv2', 'B', 'open',
                    '2026-11-18T08:10:00+00:00', '2026-11-18T08:10:00+00:00')
            """
        )
    finally:
        conn.close()
    assert _stats(client, "2026-11-18")["transitions"] == []

    # Next startup's backfill sees them (marker forces a rebuild here).
    conn = sqlite3.connect(config.DB_PATH, isolation_level=None)
    try:
        conn.execute("DELETE FROM stats_meta")
    finally:
        conn.close()
    init_db(config.DB_PATH)
    assert _pairs(_stats(client, "2026-11-18")) == {("A", "B"): 1}


def test_legacy_database_is_backfilled_on_first_startup(tmp_path):
    db_path = str(tmp_path / "legacy.db")
    # Simulate a pre-upgrade database: events only, no stats tables.
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute(
            """
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY, visitor_id TEXT NOT NULL,
                page_view_id TEXT NOT NULL, exhibit_id TEXT NOT NULL,
                event_type TEXT NOT NULL, event_time TEXT NOT NULL,
                received_at TEXT NOT NULL
            )
            """
        )
        rows = [
            ("l1", "v1", "p1", "A", "open", "2026-11-18T08:00:00+00:00"),
            ("l2", "v1", "p2", "B", "open", "2026-11-18T08:10:00+00:00"),
            ("l3", "v2", "p3", "A", "open", "2026-11-18T09:00:00+00:00"),
            ("l4", "v2", "p4", "B", "open", "2026-11-18T09:10:00+00:00"),
        ]
        conn.executemany(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
            [row + ("2026-11-18T00:00:00+00:00",) for row in rows],
        )
    finally:
        conn.close()

    init_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        pairs = conn.execute(
            "SELECT from_exhibit, to_exhibit, visitors FROM daily_transitions"
        ).fetchall()
        assert pairs == [("A", "B", 2)]
        # Running init again (e.g. another worker / restart) must not double.
        init_db(db_path)
        again = conn.execute(
            "SELECT SUM(visitors) FROM daily_transitions"
        ).fetchone()[0]
        assert again == 2
    finally:
        conn.close()


def test_backfill_skips_withdrawn_visitors(tmp_path):
    db_path = str(tmp_path / "legacy-withdrawn.db")
    init_db(db_path)
    conn = sqlite3.connect(db_path, isolation_level=None)
    try:
        conn.execute(
            "INSERT INTO withdrawals VALUES ('v-gone', '2026-11-18T00:00:00+00:00')"
        )
        conn.execute(
            """
            INSERT INTO events VALUES
            ('x1', 'v-gone', 'p1', 'A', 'open',
             '2026-11-18T08:00:00+00:00', '2026-11-18T00:00:00+00:00'),
            ('x2', 'v-gone', 'p2', 'B', 'open',
             '2026-11-18T08:10:00+00:00', '2026-11-18T00:00:00+00:00')
            """
        )
        conn.execute("DELETE FROM stats_meta")
    finally:
        conn.close()

    init_db(db_path)

    conn = sqlite3.connect(db_path)
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM daily_transitions"
        ).fetchone()[0] == 0
    finally:
        conn.close()
