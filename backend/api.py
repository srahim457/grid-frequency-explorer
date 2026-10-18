"""HTTP access to the series, the tree and the event search.

Every endpoint is a thin wrapper over a function that already exists and is
already tested; nothing here computes anything new. The server's real job is
holding the tree in memory so the dashboard can ask thousands of range
questions without rebuilding or rescanning anything.

Run it with:
    uvicorn backend.api:app --reload --port 8000

All times in and out are epoch milliseconds UTC, and all ranges are half-open
[start, end) -- the same convention as the rest of the project.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.events import (
    EVENTS_PATH,
    HIGH_HZ,
    LOW_HZ,
    find_events,
    load_events,
    write_events,
)
from backend.query import index_to_ms, query, stats
from backend.tree import SegmentTree, load_tree

logger = logging.getLogger("gridfreq.api")

FRONTEND_ORIGIN = "http://localhost:5173"
MIN_WIDTH, MAX_WIDTH = 1, 5000
NESO_URL = "https://www.neso.energy/data-portal/system-frequency-data"


@dataclass
class ServerState:
    """Everything loaded once at startup and reused by every request."""

    tree: SegmentTree
    events: list[dict[str, Any]]
    band: dict[str, float]


_state: ServerState | None = None


def get_state() -> ServerState:
    if _state is None:  # pragma: no cover - only reachable outside the lifespan
        raise RuntimeError("server state not loaded; is the app running?")
    return _state


def load_state() -> ServerState:
    """Load the series, tree and events, building whatever is missing.

    Each stage is announced *before* it runs so a slow or failing step is
    attributable. `data/` is gitignored, so on a fresh clone this will fail --
    the error has to say exactly what to do about it rather than leaving a
    server whose every endpoint 500s.
    """
    try:
        logger.info("loading series + tree (building caches if missing)...")
        t0, tree = load_tree()
        logger.info("series: %s points from %s", f"{tree.n:,}", t0.isoformat())
        logger.info("tree:   %d levels, %.1f MB", tree.n_levels, tree.nbytes / 1e6)

        events = load_events()
        band = dict(low=LOW_HZ, high=HIGH_HZ)
        if events:
            logger.info("events: %d loaded from %s", len(events), EVENTS_PATH)
            import json

            stored = json.loads(EVENTS_PATH.read_text())
            band = stored.get("band", band)
        else:
            logger.warning("%s missing or empty -> running the event search now",
                           EVENTS_PATH)
            clock = time.perf_counter()
            result = find_events(tree)
            write_events(result)
            events, band = result["events"], result["band"]
            logger.warning("event search found %d event(s) in %.1f ms -> wrote %s",
                           len(events), (time.perf_counter() - clock) * 1000, EVENTS_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"could not start: {type(exc).__name__}: {exc}\n"
            f"  The data directory is not committed to git, so a fresh clone has to\n"
            f"  fetch it first:\n"
            f"    1. download the August 2019 system frequency file from {NESO_URL}\n"
            f"    2. put the .csv (or .zip) in data/raw/\n"
            f"    3. run: python -m backend.loader\n"
            f"  The tree and results/events.json are then built automatically."
        ) from exc

    logger.info("ready: CORS allowed for %s", FRONTEND_ORIGIN)
    return ServerState(tree=tree, events=events, band=band)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _state
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    _state = load_state()
    yield
    _state = None


app = FastAPI(
    title="Grid Frequency Explorer",
    description="Range queries over one month of GB grid frequency, via a segment tree.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["GET"],
    allow_headers=["*"],
)


def _validate_range(start: int, end: int) -> None:
    """FastAPI's own validators answer 422; the spec asks for 400 here."""
    if start >= end:
        raise HTTPException(
            status_code=400,
            detail=f"start must be before end (got start={start}, end={end})",
        )


def _validate_width(width: int) -> None:
    if not MIN_WIDTH <= width <= MAX_WIDTH:
        raise HTTPException(
            status_code=400,
            detail=f"width must be between {MIN_WIDTH} and {MAX_WIDTH} (got {width})",
        )


@app.get("/meta")
def get_meta() -> dict[str, Any]:
    """Extent and shape of the dataset, for the frontend's initial view."""
    state = get_state()
    tree = state.tree
    # The root's count is the number of valid points, so n_missing is free
    # rather than a 2.7M-element scan.
    valid = int(tree.levels[-1].count[0]) if tree.levels else int(tree.n)
    return {
        "start": tree.t0_ms,
        "end": index_to_ms(tree, tree.n),  # exclusive, so end - start == n * 1000
        "n_points": tree.n,
        "n_missing": tree.n - valid,
        "levels": tree.n_levels,
        "band": state.band,
    }


@app.get("/query")
def get_query(start: int, end: int, width: int) -> dict[str, Any]:
    """Downsampled points for plotting [start, end) in at most `width` windows."""
    _validate_range(start, end)
    _validate_width(width)
    return query(get_state().tree, start, end, width)


@app.get("/stats")
def get_stats(start: int, end: int) -> dict[str, Any]:
    """Exact min/max/mean/count over [start, end), plus how long it took."""
    _validate_range(start, end)
    clock = time.perf_counter()
    result = stats(get_state().tree, start, end)
    result["elapsed_ms"] = (time.perf_counter() - clock) * 1000
    return result


@app.get("/events")
def get_events() -> list[dict[str, Any]]:
    """Every excursion outside the statutory band, from the step-3 search."""
    return get_state().events
