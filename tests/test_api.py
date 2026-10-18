"""Tests for the HTTP layer.

The endpoints must not change the answers the library already gives, so the
central check is that /stats matches stats() called directly. The rest covers
the response shapes, the 400 cases, and that NaN never reaches the wire.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from backend.api import MAX_WIDTH, MIN_WIDTH, app, get_state
from backend.events import EVENTS_PATH, Event
from backend.query import index_to_ms, stats

DAY_MS = 86_400_000


@pytest.fixture(scope="module")
def client():
    # As a context manager so lifespan runs and the tree is loaded.
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def tree(client):
    return get_state().tree


@pytest.fixture(scope="module")
def meta(client):
    return client.get("/meta").json()


# --- /meta ----------------------------------------------------------------


def test_meta_shape_and_extent(meta):
    assert set(meta) == {"start", "end", "n_points", "n_missing", "levels", "band"}
    assert meta["n_points"] == 2_678_400
    assert meta["levels"] == 23
    assert meta["n_missing"] == 0  # this month has no gaps
    assert meta["band"] == {"low": 49.5, "high": 50.5}


def test_meta_end_is_exclusive(meta):
    """end - start must be exactly n_points seconds, so a full-range query is clean."""
    assert meta["end"] - meta["start"] == meta["n_points"] * 1000


def test_meta_start_is_the_first_second(meta, tree):
    assert meta["start"] == tree.t0_ms
    assert meta["end"] == index_to_ms(tree, tree.n)


# --- /query ---------------------------------------------------------------


@pytest.mark.parametrize("width", [1, 10, 800, 5000])
def test_query_respects_width(client, meta, width):
    r = client.get("/query", params={"start": meta["start"], "end": meta["end"],
                                     "width": width})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"level", "window_seconds", "points"}
    assert body["window_seconds"] == 2 ** body["level"]
    assert len(body["points"]) <= width + 1
    for point in body["points"]:
        assert set(point) == {"t", "min", "max", "mean", "count"}


def test_query_of_the_whole_month_still_shows_the_9_august_dip(client, meta):
    """Downsampling to 800 points must not hide the event."""
    r = client.get("/query", params={"start": meta["start"], "end": meta["end"],
                                     "width": 800})
    points = r.json()["points"]
    assert min(p["min"] for p in points) == pytest.approx(48.787, abs=1e-3)


def test_query_matches_the_library(client, tree):
    from backend.query import query as query_fn

    a = index_to_ms(tree, 1000)
    b = index_to_ms(tree, 100_000)
    r = client.get("/query", params={"start": a, "end": b, "width": 500})
    assert r.json() == query_fn(tree, a, b, 500)


def test_query_second_level_detail(client, tree):
    """Zoomed all the way in, the API returns raw seconds (level 0)."""
    a = index_to_ms(tree, 0)
    r = client.get("/query", params={"start": a, "end": a + 60_000, "width": 800})
    body = r.json()
    assert body["level"] == 0 and body["window_seconds"] == 1
    assert len(body["points"]) == 60
    assert all(p["count"] == 1 for p in body["points"])


# --- /stats ---------------------------------------------------------------


def test_stats_matches_the_library(client, tree, meta):
    """The whole point: the HTTP layer must not change the numbers."""
    event_ms = tree.t0_ms + 8 * DAY_MS + (15 * 3600 + 52 * 60) * 1000
    ranges = [
        (meta["start"], meta["end"]),                     # full month
        (event_ms - 600_000, event_ms + 600_000),         # the 9 August event
        (tree.t0_ms, tree.t0_ms + 3_600_000),             # one hour
        (tree.t0_ms + 17 * DAY_MS, tree.t0_ms + 24 * DAY_MS),  # a week
    ]
    for a, b in ranges:
        body = client.get("/stats", params={"start": a, "end": b}).json()
        elapsed = body.pop("elapsed_ms")
        assert body == stats(tree, a, b), f"range [{a}, {b})"
        assert isinstance(elapsed, float) and elapsed >= 0.0


def test_stats_of_the_event_window(client, tree):
    event_ms = tree.t0_ms + 8 * DAY_MS + (15 * 3600 + 52 * 60) * 1000
    body = client.get("/stats", params={"start": event_ms - 600_000,
                                        "end": event_ms + 600_000}).json()
    assert body["min"] == pytest.approx(48.787, abs=1e-3)
    assert body["count"] == 1200
    assert body["nodes_touched"] <= 46  # ~2 * log2(N)


def test_stats_stays_cheap_for_the_full_month(client, meta):
    body = client.get("/stats", params={"start": meta["start"],
                                        "end": meta["end"]}).json()
    assert body["count"] == meta["n_points"]
    assert body["nodes_touched"] <= 46


# --- /events --------------------------------------------------------------


def test_events_returns_the_9_august_event(client):
    events = client.get("/events").json()
    assert isinstance(events, list) and len(events) == 1
    event = events[0]
    assert set(event) == set(Event.__dataclass_fields__)
    assert event["direction"] == "under"
    assert event["extreme_hz"] == pytest.approx(48.787, abs=1e-3)
    assert event["end"] - event["start"] == event["duration_s"] * 1000


def test_events_match_the_file_on_disk(client):
    assert client.get("/events").json() == json.loads(EVENTS_PATH.read_text())["events"]


# --- validation -----------------------------------------------------------


@pytest.mark.parametrize("endpoint", ["/query", "/stats"])
@pytest.mark.parametrize("start,end", [(1000, 1000), (2000, 1000)])
def test_400_when_start_is_not_before_end(client, endpoint, start, end):
    params = {"start": start, "end": end}
    if endpoint == "/query":
        params["width"] = 100
    r = client.get(endpoint, params=params)
    assert r.status_code == 400
    assert "start must be before end" in r.json()["detail"]


@pytest.mark.parametrize("width", [0, -1, MAX_WIDTH + 1, 10_000])
def test_400_for_width_outside_the_allowed_range(client, meta, width):
    r = client.get("/query", params={"start": meta["start"],
                                     "end": meta["start"] + 60_000, "width": width})
    assert r.status_code == 400
    assert str(width) in r.json()["detail"]


@pytest.mark.parametrize("width", [MIN_WIDTH, MAX_WIDTH])
def test_width_boundaries_are_accepted(client, meta, width):
    r = client.get("/query", params={"start": meta["start"],
                                     "end": meta["start"] + 60_000, "width": width})
    assert r.status_code == 200


def test_missing_parameters_are_rejected(client):
    # FastAPI's own validation; 422 is standard for a missing/unparseable param.
    assert client.get("/stats").status_code == 422
    assert client.get("/query", params={"start": 1, "end": 2}).status_code == 422
    assert client.get("/stats", params={"start": "noon", "end": 2}).status_code == 422


# --- NaN never reaches the wire -------------------------------------------


def test_responses_contain_no_nan_literals(client, meta):
    """Bare NaN is invalid JSON; empty windows must serialize as null."""
    for r in [
        client.get("/query", params={"start": meta["start"], "end": meta["end"],
                                     "width": 800}),
        client.get("/stats", params={"start": meta["start"], "end": meta["end"]}),
        client.get("/events"),
        client.get("/meta"),
    ]:
        assert "NaN" not in r.text
        json.loads(r.text)  # strict parse: would raise on a bare NaN


def test_empty_windows_serialize_as_null(tree):
    """The real month has no gaps, so check the shape the API would emit for one."""
    import numpy as np

    from backend.query import query as query_fn
    from backend.tree import build_tree

    values = np.full(120, 50.0, dtype=np.float32)
    values[60:120] = np.nan
    gapped = build_tree(tree.t0_ms, values)
    body = query_fn(gapped, gapped.t0_ms, gapped.t0_ms + 120_000, 2)
    text = json.dumps(body)
    assert "NaN" not in text
    assert '"min": null' in text and '"mean": null' in text


# --- CORS -----------------------------------------------------------------


def test_cors_allows_the_vite_dev_server(client, meta):
    r = client.get("/meta", headers={"Origin": "http://localhost:5173"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:5173"


# --- startup ---------------------------------------------------------------


def test_missing_events_file_triggers_a_search_and_says_so(monkeypatch, caplog, tmp_path):
    """A missing events.json must be rebuilt, loudly -- not silently served empty."""
    import backend.api as api

    written = {}
    monkeypatch.setattr(api, "load_events", lambda: [])
    monkeypatch.setattr(api, "write_events", lambda result: written.update(result))

    with caplog.at_level("WARNING", logger="gridfreq.api"):
        state = api.load_state()

    assert len(state.events) == 1  # the 9 August event, found on the spot
    assert state.band == {"low": 49.5, "high": 50.5}
    assert written["n_events"] == 1  # it also persisted the result
    messages = " ".join(r.getMessage() for r in caplog.records)
    assert "running the event search now" in messages
    assert "found 1 event(s)" in messages


def test_startup_failure_explains_how_to_get_the_data(monkeypatch):
    """A fresh clone has no data/, so the error must name the fix, not just the cause."""
    import backend.api as api

    def boom():
        raise FileNotFoundError("no .csv or .zip files in data/raw")

    monkeypatch.setattr(api, "load_tree", boom)
    with pytest.raises(RuntimeError) as excinfo:
        api.load_state()

    message = str(excinfo.value)
    assert "no .csv or .zip files" in message          # the cause
    assert api.NESO_URL in message                     # where to get it
    assert "data/raw/" in message                      # where to put it
    assert "python -m backend.loader" in message       # what to run
