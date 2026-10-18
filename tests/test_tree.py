"""Tests for the segment tree.

The core property: every window's summary must equal numpy computed directly on
that window's raw slice -- including partial last windows, NaN gaps and
all-NaN windows -- at every level, for even and odd lengths.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from backend.tree import (
    Level,
    SegmentTree,
    TREE_CACHE_PATH,
    _cache_is_fresh,
    build_tree,
    load_tree,
    save_tree,
)

NAN = float("nan")


def _expected(slice_: np.ndarray) -> tuple[float, float, float, int]:
    """min, max, sum, count over a raw slice, NaN-aware, without warnings."""
    valid = slice_[~np.isnan(slice_)]
    if valid.size == 0:
        return NAN, NAN, 0.0, 0
    return float(valid.min()), float(valid.max()), float(valid.astype(np.float64).sum()), valid.size


def _check_every_window(raw: list[float]) -> SegmentTree:
    values = np.array(raw, dtype=np.float32)
    n = len(values)
    tree = build_tree(0, values)

    assert tree.n == n
    if n > 1:
        assert len(tree.levels[-1]) == 1, "build must stop at a single window"
    assert tree.windows_at(tree.n_levels - 1) == 1

    for k in range(tree.n_levels):
        windows = tree.windows_at(k)
        assert windows == math.ceil(n / 2**k)
        lv = tree.nodes(k, 0, windows)
        assert len(lv) == windows
        assert lv.min.dtype == np.float32 and lv.max.dtype == np.float32
        assert lv.sum.dtype == np.float64 and lv.count.dtype == np.int32

        for i in range(windows):
            sl = values[i << k : min((i + 1) << k, n)]
            e_min, e_max, e_sum, e_count = _expected(sl)
            where = f"level {k} window {i} (raw {i << k}..{min((i + 1) << k, n)})"
            assert int(lv.count[i]) == e_count, where
            assert float(lv.sum[i]) == pytest.approx(e_sum, rel=1e-6), where
            if e_count == 0:
                assert np.isnan(lv.min[i]) and np.isnan(lv.max[i]), where
            else:
                assert float(lv.min[i]) == pytest.approx(e_min, rel=1e-6), where
                assert float(lv.max[i]) == pytest.approx(e_max, rel=1e-6), where
    return tree


# --- shapes and lengths ---------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 8, 13, 16, 17, 31, 33, 100])
def test_all_windows_match_numpy_for_many_lengths(n):
    rng = np.random.default_rng(n)
    _check_every_window(list(49.5 + rng.random(n)))


def test_length_one():
    tree = _check_every_window([50.0])
    assert tree.n_levels == 1  # level 0 already has a single window
    assert tree.levels == []


def test_odd_lengths_7_and_13():
    # The build appends at most one empty slot per level; the partial last
    # window must still summarise only the real raw values it covers.
    t7 = _check_every_window([50.0, 49.0, 51.0, 48.0, 52.0, 47.0, 53.0])
    assert [tree_len for tree_len in map(len, t7.levels)] == [4, 2, 1]
    t13 = _check_every_window([float(i) for i in range(13)])
    assert [len(lv) for lv in t13.levels] == [7, 4, 2, 1]


# --- NaN handling --------------------------------------------------------


def test_nan_gap_is_excluded_not_counted():
    _check_every_window([50.0, NAN, NAN, 49.0, 50.5, NAN, 51.0])


def test_all_nan_window_has_nan_minmax_and_zero_count():
    # values[2:4] is an entirely missing level-1 window.
    tree = _check_every_window([50.0, 50.1, NAN, NAN, 49.9, 50.2])
    lv1 = tree.nodes(1, 0, 3)
    assert np.isnan(lv1.min[1]) and np.isnan(lv1.max[1])
    assert int(lv1.count[1]) == 0 and float(lv1.sum[1]) == 0.0
    # The level above still recovers the valid half.
    lv2 = tree.nodes(2, 0, 2)
    assert float(lv2.min[0]) == pytest.approx(50.0)
    assert int(lv2.count[0]) == 2


def test_entirely_nan_series():
    tree = _check_every_window([NAN] * 9)
    top = tree.levels[-1]
    assert int(top.count[0]) == 0 and np.isnan(top.min[0])


def test_half_empty_pair_keeps_valid_side():
    tree = build_tree(0, np.array([NAN, 49.0], dtype=np.float32))
    lv = tree.nodes(1, 0, 1)
    assert float(lv.min[0]) == pytest.approx(49.0)
    assert float(lv.max[0]) == pytest.approx(49.0)
    assert int(lv.count[0]) == 1


# --- accessor behaviour --------------------------------------------------


def test_nodes_slices_and_clamps():
    values = np.arange(10, dtype=np.float32)
    tree = build_tree(0, values)
    assert len(tree.nodes(1, 1, 3)) == 2
    np.testing.assert_allclose(tree.nodes(1, 1, 3).min, [2.0, 4.0])
    # Out-of-range bounds clamp rather than raise, so query code can be simple.
    assert len(tree.nodes(1, 3, 99)) == 2
    assert len(tree.nodes(0, -5, 2)) == 2
    with pytest.raises(IndexError):
        tree.nodes(tree.n_levels, 0, 1)


def test_level_zero_is_derived_not_stored():
    values = np.array([50.0, NAN, 49.0, 51.0], dtype=np.float32)
    tree = build_tree(0, values)
    lv0 = tree.nodes(0, 0, 4)
    assert lv0.min is tree.values or np.shares_memory(lv0.min, tree.values)
    np.testing.assert_array_equal(lv0.count, [1, 0, 1, 1])
    np.testing.assert_allclose(lv0.sum, [50.0, 0.0, 49.0, 51.0])


def test_empty_series_rejected():
    with pytest.raises(ValueError):
        build_tree(0, np.array([], dtype=np.float32))


def test_padding_adds_at_most_one_slot_per_level():
    """The odd-length fix must cost one slot per level, not a power-of-two pad."""
    n = 100_000
    tree = build_tree(0, np.full(n, 50.0, dtype=np.float32))
    assert tree.n_levels == math.ceil(math.log2(n)) + 1
    exact = sum(math.ceil(n / 2**k) for k in range(1, tree.n_levels))
    stored = sum(len(lv) for lv in tree.levels)
    assert exact <= stored <= exact + tree.n_levels
    # A power-of-two pad would round N up to 131,072 and nearly double the tree.
    assert stored < 2**math.ceil(math.log2(n))


# --- cache ---------------------------------------------------------------


def test_cache_roundtrip_and_staleness(tmp_path):
    series = tmp_path / "series.npz"
    series.write_bytes(b"stand-in for the series cache")
    tree_path = tmp_path / "tree.npz"

    values = np.array([50.0, 49.0, NAN, 51.0, 50.5], dtype=np.float32)
    tree = build_tree(123_000, values)
    save_tree(tree, tree_path, series)

    assert _cache_is_fresh(tree_path, series, tree.n)
    assert not _cache_is_fresh(tree_path, series, tree.n + 1)  # different length

    with np.load(tree_path) as z:
        assert int(z["n"]) == 5 and int(z["n_levels"]) == tree.n_levels
        assert "min_1" in z and f"count_{tree.n_levels - 1}" in z
        assert "values" not in z  # raw array lives in series.npz only

    # A series file written after the tree invalidates it.
    import os, time as _t

    future = series.stat().st_mtime + 10
    os.utime(series, (future, future))
    assert not _cache_is_fresh(tree_path, series, tree.n)


@pytest.mark.skipif(not TREE_CACHE_PATH.parent.exists(), reason="no processed data dir")
def test_real_tree_top_level_matches_valid_count():
    """The August 2019 tree: one top window, counting every valid second."""
    t0, tree = load_tree()
    assert tree.n == 2_678_400
    assert tree.windows_at(tree.n_levels - 1) == 1
    top = tree.levels[-1]
    assert len(top) == 1
    valid = int(np.count_nonzero(~np.isnan(tree.values)))
    assert int(top.count[0]) == valid
    assert float(top.min[0]) == pytest.approx(float(np.nanmin(tree.values)), rel=1e-6)
    assert float(top.max[0]) == pytest.approx(float(np.nanmax(tree.values)), rel=1e-6)
    assert float(top.sum[0] / top.count[0]) == pytest.approx(
        float(np.nanmean(tree.values.astype(np.float64))), rel=1e-9
    )
