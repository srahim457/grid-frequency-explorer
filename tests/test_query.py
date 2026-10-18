"""Tests for query() and stats().

The contract that matters: stats() must be *exact*. Every assertion compares it
against numpy run directly on the raw slice, over many seeded random ranges plus
the edge cases that the ceil-sized levels and partial last window make risky.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.query import choose_level, index_to_ms, ms_to_index, query, stats
from backend.tree import build_tree

T0_MS = 1_564_617_600_000  # 2019-08-01T00:00:00Z, so index<->ms mirrors the real data
NAN = float("nan")


def _tree(n: int, seed: int, gap: tuple[int, int] | None = None):
    """A tree over n seconds of plausible frequency data, optionally with a NaN gap."""
    rng = np.random.default_rng(seed)
    values = (50.0 + rng.normal(0, 0.08, n)).astype(np.float32)
    if gap is not None:
        values[gap[0] : gap[1]] = np.nan
    return build_tree(T0_MS, values)


def _ref(values: np.ndarray, i0: int, i1: int):
    """Ground truth over raw[i0:i1]: min, max, mean, count, NaN-aware."""
    sl = values[i0:i1]
    valid = sl[~np.isnan(sl)]
    if valid.size == 0:
        return None, None, None, 0
    # float64 for the mean reference: np.nanmean on a float32 array accumulates
    # in float32 and drifts well past 1e-9 over millions of points, so the
    # float64 result is the more accurate ground truth, not a loosened one.
    return (
        float(valid.min()),
        float(valid.max()),
        float(valid.astype(np.float64).mean()),
        int(valid.size),
    )


def _assert_stats_exact(tree, i0: int, i1: int):
    got = stats(tree, index_to_ms(tree, i0), index_to_ms(tree, i1))
    e_min, e_max, e_mean, e_count = _ref(tree.values, i0, i1)
    where = f"range [{i0}, {i1}) of N={tree.n}"

    assert got["count"] == e_count, where
    if e_count == 0:
        assert got["min"] is None and got["max"] is None and got["mean"] is None, where
    else:
        # min/max come straight from stored float32s, so they match exactly.
        assert got["min"] == e_min, where
        assert got["max"] == e_max, where
        assert got["mean"] == pytest.approx(e_mean, rel=1e-9), where
    return got


# --- stats: random ranges ------------------------------------------------


@pytest.mark.parametrize("n", [10_000, 10_001])  # even and odd lengths
def test_stats_matches_numpy_over_1000_random_ranges(n):
    tree = _tree(n, seed=n)
    rng = np.random.default_rng(7)
    for _ in range(1000):
        a, b = sorted(rng.integers(0, n + 1, size=2).tolist())
        if a == b:
            b = min(b + 1, n)
        _assert_stats_exact(tree, a, b)


@pytest.mark.parametrize("n", [10_000, 10_001])
def test_stats_matches_numpy_with_nan_gaps(n):
    tree = _tree(n, seed=n + 1, gap=(4_000, 4_500))
    rng = np.random.default_rng(11)
    for _ in range(500):
        a, b = sorted(rng.integers(0, n + 1, size=2).tolist())
        if a == b:
            b = min(b + 1, n)
        _assert_stats_exact(tree, a, b)


# --- stats: edge cases ---------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 7, 13, 16, 100, 1023, 1024, 1025])
def test_stats_edge_ranges(n):
    tree = _tree(n, seed=n, gap=(n // 2, min(n // 2 + 3, n)) if n > 8 else None)
    cases = [
        (0, n),              # full range
        (0, 1),              # length 1 at the start
        (n - 1, n),          # length 1 at the end
        (0, max(n // 2, 1)),  # touching the start
        (n // 2, n),         # touching the end, ends in the partial last window
        (max(n - 3, 0), n),  # inside the partial last window
    ]
    for i0, i1 in cases:
        if i1 > i0:
            _assert_stats_exact(tree, i0, i1)


def test_stats_every_range_of_a_small_odd_series():
    """Exhaustive: all 105 ranges of a 13-point series with a NaN gap and a dip."""
    values = np.array(
        [50.0, 49.9, NAN, NAN, 50.2, 48.8, 50.1, 49.95, NAN, 50.3, 49.8, 50.0, 50.05],
        dtype=np.float32,
    )
    tree = build_tree(T0_MS, values)
    for i0 in range(14):
        for i1 in range(i0 + 1, 14):
            _assert_stats_exact(tree, i0, i1)


def test_stats_entirely_inside_a_nan_gap():
    tree = _tree(1000, seed=3, gap=(200, 400))
    got = _assert_stats_exact(tree, 250, 350)
    assert got["count"] == 0 and got["mean"] is None


def test_stats_empty_and_inverted_ranges():
    tree = _tree(100, seed=4)
    for a, b in [(0, 0), (50, 50), (60, 40)]:
        got = stats(tree, index_to_ms(tree, a), index_to_ms(tree, b))
        assert got == {"min": None, "max": None, "mean": None,
                       "count": 0, "nodes_touched": 0}


def test_stats_clamps_times_outside_the_data():
    tree = _tree(100, seed=5)
    day = 86_400_000
    # Asking for a week around the data must return exactly the data.
    got = stats(tree, T0_MS - 7 * day, T0_MS + 7 * day)
    full = stats(tree, index_to_ms(tree, 0), index_to_ms(tree, 100))
    assert got == full
    # Entirely before / after the data is empty, not an error.
    assert stats(tree, T0_MS - 2 * day, T0_MS - day)["count"] == 0
    assert stats(tree, T0_MS + day, T0_MS + 2 * day)["count"] == 0


def test_stats_sub_second_times_floor_to_the_second():
    tree = _tree(100, seed=6)
    assert ms_to_index(tree, T0_MS + 1500) == 1
    a = stats(tree, T0_MS + 1500, T0_MS + 3999)
    b = stats(tree, T0_MS + 1000, T0_MS + 3000)
    assert a == b


# --- stats: node budget --------------------------------------------------


@pytest.mark.parametrize("n", [1023, 1024, 100_000])
def test_nodes_touched_stays_logarithmic(n):
    tree = _tree(n, seed=n)
    budget = 2 * math.ceil(math.log2(n)) + 2
    rng = np.random.default_rng(13)
    worst = 0
    for _ in range(400):
        a, b = sorted(rng.integers(0, n + 1, size=2).tolist())
        if a == b:
            b = min(b + 1, n)
        got = stats(tree, index_to_ms(tree, a), index_to_ms(tree, b))
        worst = max(worst, got["nodes_touched"])
        assert got["nodes_touched"] <= budget
    assert worst <= budget
    # The full range is cheap: it decomposes into a handful of big nodes.
    assert stats(tree, index_to_ms(tree, 0), index_to_ms(tree, n))["nodes_touched"] <= budget


# --- query ---------------------------------------------------------------


@pytest.mark.parametrize("n", [10_000, 10_001])
@pytest.mark.parametrize("max_points", [1, 10, 100, 800, 5000])
def test_query_window_count_and_values(n, max_points):
    tree = _tree(n, seed=n, gap=(3000, 3100))
    rng = np.random.default_rng(max_points)
    for _ in range(40):
        a, b = sorted(rng.integers(0, n + 1, size=2).tolist())
        if a == b:
            b = min(b + 1, n)
        res = query(tree, index_to_ms(tree, a), index_to_ms(tree, b), max_points)
        k, pts = res["level"], res["points"]
        assert res["window_seconds"] == 1 << k
        assert len(pts) <= max_points + 1, f"[{a},{b}) max_points={max_points}"

        # Every point must summarise its own raw slice, partial last window
        # included, and the windows must tile the range contiguously.
        first_index = (a >> k) << k
        for i, p in enumerate(pts):
            i0 = first_index + i * (1 << k)
            assert p["t"] == index_to_ms(tree, i0)
            e_min, e_max, e_mean, e_count = _ref(tree.values, i0, min(i0 + (1 << k), n))
            assert p["count"] == e_count
            if e_count == 0:
                assert p["min"] is None and p["max"] is None and p["mean"] is None
            else:
                assert p["min"] == e_min and p["max"] == e_max
                assert p["mean"] == pytest.approx(e_mean, rel=1e-9)

        # The returned windows must cover the whole requested range.
        assert first_index <= a
        assert first_index + len(pts) * (1 << k) >= min(b, n)


def test_query_picks_the_smallest_sufficient_level():
    tree = _tree(100_000, seed=1)
    # 1024 seconds into <= 64 points needs 16-second windows: level 4 exactly.
    res = query(tree, index_to_ms(tree, 0), index_to_ms(tree, 1024), 64)
    assert res["level"] == 4 and res["window_seconds"] == 16
    assert len(res["points"]) == 64
    # A narrow range fits at level 0, i.e. raw seconds.
    res = query(tree, index_to_ms(tree, 0), index_to_ms(tree, 60), 800)
    assert res["level"] == 0 and len(res["points"]) == 60


def test_query_level_is_capped_at_the_top_of_the_tree():
    tree = _tree(1000, seed=2)
    res = query(tree, index_to_ms(tree, 0), index_to_ms(tree, 1000), 1)
    assert res["level"] == tree.n_levels - 1
    assert len(res["points"]) == 1
    assert res["points"][0]["count"] == 1000


def test_query_empty_range_and_bad_max_points():
    tree = _tree(100, seed=8)
    assert query(tree, T0_MS, T0_MS, 800) == {
        "level": 0, "window_seconds": 1, "points": []
    }
    assert query(tree, T0_MS - 86_400_000, T0_MS - 1000, 800)["points"] == []
    with pytest.raises(ValueError):
        query(tree, T0_MS, T0_MS + 60_000, 0)


def test_choose_level_arithmetic():
    # 1000 points at <= 1000 per level fits raw; 1001 needs one halving.
    assert choose_level(1000, 1000, 23) == 0
    assert choose_level(1001, 1000, 23) == 1
    assert choose_level(2_678_400, 800, 23) == 12
    # Never asks for a level the tree does not have.
    assert choose_level(10**9, 1, 5) == 4


# --- the real month ------------------------------------------------------


def test_real_series_stats_and_query_agree_with_numpy():
    """Spot-check the actual August data, including the 9 August event."""
    from backend.tree import load_tree

    _, tree = load_tree()
    day = 86_400_000
    event_ms = tree.t0_ms + 8 * day + (15 * 3600 + 52 * 60) * 1000

    for a_ms, b_ms in [
        (tree.t0_ms, index_to_ms(tree, tree.n)),              # full month
        (event_ms - 600_000, event_ms + 600_000),             # event +/- 10 min
        (tree.t0_ms + 17 * day, tree.t0_ms + 24 * day),       # a week
    ]:
        i0, i1 = ms_to_index(tree, a_ms), ms_to_index(tree, b_ms)
        got = stats(tree, a_ms, b_ms)
        e_min, e_max, e_mean, e_count = _ref(tree.values, i0, i1)
        assert got["min"] == e_min and got["max"] == e_max
        assert got["mean"] == pytest.approx(e_mean, rel=1e-9)
        assert got["count"] == e_count
        assert got["nodes_touched"] <= 2 * math.ceil(math.log2(tree.n)) + 2

    # The dip is visible in a month-wide chart query, not just in raw data.
    res = query(tree, tree.t0_ms, index_to_ms(tree, tree.n), 800)
    assert min(p["min"] for p in res["points"]) == pytest.approx(48.787, abs=1e-3)
