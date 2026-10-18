"""Find and measure every excursion outside the statutory 49.5-50.5 Hz band.

Why the tree helps here: almost all of August 2019 sits well inside the band, so
almost every node in the tree can be rejected whole. A node whose [min, max]
fits inside the band cannot contain an excursion, and a node with count 0 holds
no data at all -- both are skipped without looking at a single raw second. Only
the handful of subtrees that actually straddle a limit are descended to level 0.
That turns "scan 2.7M points" into "visit a few hundred nodes".

Conventions: epoch milliseconds UTC, and `end` is exclusive, so
`end - start == duration_s * 1000` and an event's (start, end) can be handed
straight to query()/stats().
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.query import index_to_ms
from backend.tree import SegmentTree

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
EVENTS_PATH = RESULTS_DIR / "events.json"

LOW_HZ = 49.5
HIGH_HZ = 50.5
NOMINAL_HZ = 50.0
MERGE_GAP_S = 30
ROCOF_WINDOW_S = 60


@dataclass
class Event:
    """One merged excursion outside the band.

    `start`/`end` are half-open; `duration_s` counts the first to the last
    out-of-band second inclusive, which is the same number.
    """

    id: int
    start: int
    end: int
    duration_s: int
    direction: str  # "under" | "over" | "both"
    extreme_hz: float
    time_of_extreme: int
    recovery_s: int | None
    max_rocof_hz_per_s: float | None


def out_of_band_mask(values: np.ndarray, low: float = LOW_HZ, high: float = HIGH_HZ) -> np.ndarray:
    """Brute-force reference mask. NaN compares False, so gaps are never events."""
    return (values < low) | (values > high)


def find_out_of_band_indices(
    tree: SegmentTree,
    low: float = LOW_HZ,
    high: float = HIGH_HZ,
) -> tuple[np.ndarray, int]:
    """Descend the tree, pruning in-band and empty nodes. Returns (indices, nodes_visited).

    Children are pushed right-first so the stack pops left-first and the
    recorded indices come out already sorted.
    """
    indices: list[int] = []
    nodes_visited = 0
    stack: list[tuple[int, int]] = [(tree.n_levels - 1, 0)]

    while stack:
        k, i = stack.pop()
        nodes_visited += 1
        node = tree.nodes(k, i, i + 1)
        if int(node.count[0]) == 0:
            continue  # nothing but gaps under here
        if node.min[0] >= low and node.max[0] <= high:
            continue  # the whole subtree is inside the band
        if k == 0:
            indices.append(i)
            continue
        child = 2 * i
        n_children = tree.windows_at(k - 1)
        if child + 1 < n_children:
            stack.append((k - 1, child + 1))
        if child < n_children:
            stack.append((k - 1, child))

    return np.array(indices, dtype=np.int64), nodes_visited


def _merge(indices: np.ndarray, merge_gap_s: int) -> list[tuple[int, int]]:
    """Group sorted indices into (first, last) spans, joining gaps <= merge_gap_s."""
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) > merge_gap_s)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [indices.size - 1]))
    return [(int(indices[a]), int(indices[b])) for a, b in zip(starts, ends)]


def _recovery_s(tree: SegmentTree, i_extreme: int, i_last: int,
                low: float, high: float) -> int | None:
    """Seconds from the extreme until frequency is back in band *and stays there*.

    By construction the second after a merged event is in band (otherwise it
    would belong to the event), but it can be NaN or past the end of the data,
    so step forward to the first valid in-band second and report None if the
    recovery is never observed.
    """
    values = tree.values
    i = i_last + 1
    while i < tree.n and np.isnan(values[i]):
        i += 1
    if i >= tree.n:
        return None
    return int(i - i_extreme)


def _max_rocof(tree: SegmentTree, i_first: int, i_last: int,
               window_s: int = ROCOF_WINDOW_S) -> float | None:
    """Largest absolute second-to-second change within +/- window_s of the event.

    1-second samples make this a coarse lower bound on the true RoCoF: a real
    frequency transient moves faster than our sampling interval can resolve.
    """
    a = max(i_first - window_s, 0)
    b = min(i_last + window_s + 1, tree.n)
    diffs = np.abs(np.diff(tree.values[a:b].astype(np.float64)))
    if diffs.size == 0 or np.all(np.isnan(diffs)):
        return None
    return float(np.nanmax(diffs))


def _measure(tree: SegmentTree, event_id: int, i_first: int, i_last: int,
             low: float, high: float) -> Event:
    """Turn a span of out-of-band seconds into a measured Event."""
    window = tree.values[i_first : i_last + 1]
    oob = out_of_band_mask(window, low, high)

    went_under = bool(np.any(window[oob] < low))
    went_over = bool(np.any(window[oob] > high))
    direction = "both" if went_under and went_over else ("under" if went_under else "over")

    # Pick the extreme by largest deviation from nominal, which reduces to the
    # min for an "under" event and the max for an "over" one.
    dev = np.where(np.isnan(window), -np.inf, np.abs(window.astype(np.float64) - NOMINAL_HZ))
    offset = int(np.argmax(dev))
    i_extreme = i_first + offset

    return Event(
        id=event_id,
        start=index_to_ms(tree, i_first),
        end=index_to_ms(tree, i_last + 1),
        duration_s=i_last - i_first + 1,
        direction=direction,
        extreme_hz=round(float(window[offset]), 4),
        time_of_extreme=index_to_ms(tree, i_extreme),
        recovery_s=_recovery_s(tree, i_extreme, i_last, low, high),
        max_rocof_hz_per_s=(
            None if (r := _max_rocof(tree, i_first, i_last)) is None else round(r, 4)
        ),
    )


def find_events(
    tree: SegmentTree,
    low: float = LOW_HZ,
    high: float = HIGH_HZ,
    merge_gap_s: int = MERGE_GAP_S,
) -> dict[str, Any]:
    """Full search: prune the tree, merge the hits, measure each event."""
    indices, nodes_visited = find_out_of_band_indices(tree, low, high)
    spans = _merge(indices, merge_gap_s)
    events = [_measure(tree, n, a, b, low, high) for n, (a, b) in enumerate(spans)]
    return {
        "band": {"low": low, "high": high},
        "merge_gap_s": merge_gap_s,
        "n_points": tree.n,
        "n_out_of_band_seconds": int(indices.size),
        "nodes_visited": nodes_visited,
        "n_events": len(events),
        "events": [asdict(e) for e in events],
    }


def write_events(result: dict[str, Any], path: Path = EVENTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")


def load_events(path: Path = EVENTS_PATH) -> list[dict[str, Any]]:
    """The event list alone, for the API. Returns [] if the search has not run."""
    if not path.exists():
        return []
    return json.loads(path.read_text())["events"]


def _utc(t_ms: int) -> str:
    import pandas as pd

    return pd.Timestamp(t_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M:%S")


def summary_table(result: dict[str, Any]) -> str:
    rows = [
        f"{'id':>3} {'start (UTC)':>19} {'dur_s':>6} {'dir':>5} "
        f"{'extreme':>8} {'at (UTC)':>19} {'rec_s':>6} {'rocof':>7}",
        "-" * 80,
    ]
    for e in result["events"]:
        rec = "-" if e["recovery_s"] is None else f"{e['recovery_s']:,}"
        roc = "-" if e["max_rocof_hz_per_s"] is None else f"{e['max_rocof_hz_per_s']:.4f}"
        rows.append(
            f"{e['id']:>3} {_utc(e['start']):>19} {e['duration_s']:>6,} "
            f"{e['direction']:>5} {e['extreme_hz']:>8.3f} {_utc(e['time_of_extreme']):>19} "
            f"{rec:>6} {roc:>7}"
        )
    if not result["events"]:
        rows.append("  (no events)")
    return "\n".join(rows)


def main() -> None:
    import time

    from backend.tree import load_tree

    _, tree = load_tree()

    clock = time.perf_counter()
    result = find_events(tree)
    elapsed = (time.perf_counter() - clock) * 1000

    print(f"band {result['band']['low']}-{result['band']['high']} Hz, "
          f"merge_gap_s={result['merge_gap_s']}\n")
    print(summary_table(result))

    nodes, n = result["nodes_visited"], result["n_points"]
    print(f"\nevents:                {result['n_events']}")
    print(f"out-of-band seconds:   {result['n_out_of_band_seconds']:,}")
    print(f"nodes visited:         {nodes:,} of {n:,} points "
          f"({100 * nodes / n:.4f}% -- {n / nodes:.0f}x fewer)")
    print(f"search time:           {elapsed:.1f} ms")

    # Cross-check the pruned search against a plain numpy mask over everything.
    mask_indices = np.flatnonzero(out_of_band_mask(tree.values))
    tree_indices, _ = find_out_of_band_indices(tree)
    same = np.array_equal(tree_indices, mask_indices)
    print(f"matches numpy mask:    {same} ({mask_indices.size:,} seconds)")

    write_events(result)
    print(f"\nwrote -> {EVENTS_PATH}")


if __name__ == "__main__":
    main()
