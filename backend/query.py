"""Range queries over the segment tree: one for drawing, one for exact numbers.

Two different jobs, so two different algorithms:

`query` answers "give me at most W points covering this range" -- a chart needs
a fixed number of pixels' worth of data, not the raw seconds. It picks the one
level whose window size makes the range fit in W points and reads a contiguous
slice of it, so the cost depends on W, not on the width of the range.

`stats` answers "exactly what are the min/max/mean here" -- it walks the tree
bottom-up, taking O(log N) whole nodes that tile the range precisely, so the
answer is exact rather than snapped to window boundaries.

Times in and out are epoch milliseconds UTC. Ranges are half-open [start, end).
Empty windows and empty ranges report None rather than NaN, so results are
directly JSON-serializable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from backend.tree import SegmentTree


def ms_to_index(tree: SegmentTree, t_ms: int) -> int:
    """Epoch ms -> raw index, clamped to [0, N]. One sample per second."""
    idx = (int(t_ms) - tree.t0_ms) // 1000
    return int(min(max(idx, 0), tree.n))


def index_to_ms(tree: SegmentTree, index: int) -> int:
    """Raw index -> epoch ms of that second."""
    return tree.t0_ms + int(index) * 1000


def choose_level(span: int, max_points: int, n_levels: int) -> int:
    """Smallest level k >= 0 whose windows cover `span` seconds in <= max_points.

    Doubling the window each level means this loop runs at most log2(N) times.
    """
    if max_points < 1:
        raise ValueError("max_points must be >= 1")
    k = 0
    while k < n_levels - 1 and -(-span // (1 << k)) > max_points:
        k += 1
    return k


def query(
    tree: SegmentTree,
    start: int,
    end: int,
    max_points: int = 1000,
) -> dict[str, Any]:
    """Downsampled points for plotting [start, end) in at most max_points+1 windows.

    Returns {level, window_seconds, points: [{t, min, max, mean, count}]}.
    Edge windows may overhang the requested range; for a chart that is harmless
    and it keeps every point aligned to a real tree node.
    """
    i0, i1 = ms_to_index(tree, start), ms_to_index(tree, end)
    if i1 <= i0:
        return {"level": 0, "window_seconds": 1, "points": []}

    k = choose_level(i1 - i0, max_points, tree.n_levels)
    lo, hi = i0 >> k, ((i1 - 1) >> k) + 1
    nodes = tree.nodes(k, lo, hi)

    step_ms = (1 << k) * 1000
    t_first = index_to_ms(tree, lo << k)
    counts = nodes.count
    # Divide once, vectorized; count 0 would be a divide-by-zero so guard it.
    means = np.where(counts > 0, nodes.sum / np.maximum(counts, 1), np.nan)

    points = [
        {
            "t": t_first + i * step_ms,
            "min": float(nodes.min[i]) if c else None,
            "max": float(nodes.max[i]) if c else None,
            "mean": float(means[i]) if c else None,
            "count": int(c),
        }
        for i, c in enumerate(counts.tolist())
    ]
    return {"level": k, "window_seconds": 1 << k, "points": points}


def stats(tree: SegmentTree, start: int, end: int) -> dict[str, Any]:
    """Exact min/max/mean/count over [start, end), plus how many nodes it took.

    The standard iterative bottom-up range query: at each level, an odd left
    boundary means that one node is inside the range and the rest of its parent
    is not, so take it and step right; same for an odd right boundary. Then
    halve both boundaries and go up. Both boundaries stay exact multiples of the
    current window size, which is why this still works with ceil-sized levels
    and a partial last window -- a taken node never reaches past N.
    """
    lo, hi = ms_to_index(tree, start), ms_to_index(tree, end)

    acc_min = np.float32(np.nan)
    acc_max = np.float32(np.nan)
    acc_sum = 0.0
    acc_count = 0
    nodes_touched = 0
    k = 0

    while lo < hi:
        if lo & 1:
            node = tree.nodes(k, lo, lo + 1)
            acc_min = np.fmin(acc_min, node.min[0])
            acc_max = np.fmax(acc_max, node.max[0])
            acc_sum += float(node.sum[0])
            acc_count += int(node.count[0])
            nodes_touched += 1
            lo += 1
        if hi & 1:
            hi -= 1
            node = tree.nodes(k, hi, hi + 1)
            acc_min = np.fmin(acc_min, node.min[0])
            acc_max = np.fmax(acc_max, node.max[0])
            acc_sum += float(node.sum[0])
            acc_count += int(node.count[0])
            nodes_touched += 1
        lo >>= 1
        hi >>= 1
        k += 1

    return {
        "min": float(acc_min) if acc_count else None,
        "max": float(acc_max) if acc_count else None,
        "mean": acc_sum / acc_count if acc_count else None,
        "count": acc_count,
        "nodes_touched": nodes_touched,
    }


def main() -> None:
    import time

    from backend.tree import load_tree

    t0, tree = load_tree()
    t0_ms = tree.t0_ms
    end_ms = index_to_ms(tree, tree.n)
    day = 86_400_000

    print(f"series: {tree.n:,} points, {tree.n_levels} levels\n")

    print("query() level selection for max_points=800:")
    for label, span in [("1 min", 60), ("1 hour", 3600), ("1 day", 86_400),
                        ("1 week", 604_800), ("full month", tree.n)]:
        r = query(tree, t0_ms, t0_ms + span * 1000, 800)
        print(f"  {label:>11}: level {r['level']:>2}  window {r['window_seconds']:>7,} s  "
              f"{len(r['points']):>4} points")

    print("\nstats() over a few ranges:")
    # 9 Aug 2019 15:52 UTC, the power cut, +/- 10 minutes.
    event_ms = t0_ms + (8 * day) + (15 * 3600 + 52 * 60) * 1000
    for label, a, b in [
        ("full month", t0_ms, end_ms),
        ("9 Aug event +/-10m", event_ms - 600_000, event_ms + 600_000),
        ("1 Aug 00:00-01:00", t0_ms, t0_ms + 3_600_000),
    ]:
        clock = time.perf_counter()
        s = stats(tree, a, b)
        elapsed = (time.perf_counter() - clock) * 1000
        print(f"  {label:>19}: min={s['min']:.3f} max={s['max']:.3f} "
              f"mean={s['mean']:.4f} count={s['count']:>9,} "
              f"nodes={s['nodes_touched']:>3} ({elapsed:.2f} ms)")


if __name__ == "__main__":
    main()
