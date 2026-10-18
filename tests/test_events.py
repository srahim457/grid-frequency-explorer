"""Tests for the event search.

The central property: pruning the tree must find *exactly* the seconds a plain
numpy mask over the raw data finds -- no more, no fewer -- on data with injected
excursions in both directions and NaN gaps.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from backend.events import (
    EVENTS_PATH,
    HIGH_HZ,
    LOW_HZ,
    Event,
    _merge,
    find_events,
    find_out_of_band_indices,
    load_events,
    out_of_band_mask,
    summary_table,
    write_events,
)
from backend.tree import build_tree

T0_MS = 1_564_617_600_000  # 2019-08-01T00:00:00Z
NAN = float("nan")


def _tree(values):
    return build_tree(T0_MS, np.array(values, dtype=np.float32))


def _synthetic(n: int, seed: int, n_excursions: int = 4, with_gaps: bool = True):
    """In-band noise with injected excursions and NaN gaps."""
    rng = np.random.default_rng(seed)
    values = (50.0 + rng.normal(0, 0.08, n)).astype(np.float32)
    for j in range(n_excursions):
        start = int(rng.integers(0, max(n - 200, 1)))
        length = int(rng.integers(1, 120))
        # Alternate under/over so both limits get exercised.
        values[start : start + length] = 48.6 + 0.1 * j if j % 2 == 0 else 50.9 + 0.1 * j
    if with_gaps:
        for _ in range(3):
            start = int(rng.integers(0, max(n - 100, 1)))
            values[start : start + int(rng.integers(1, 80))] = np.nan
    return values


def _assert_search_matches_mask(values: np.ndarray, low=LOW_HZ, high=HIGH_HZ):
    tree = build_tree(T0_MS, values)
    found, nodes_visited = find_out_of_band_indices(tree, low, high)
    expected = np.flatnonzero(out_of_band_mask(values, low, high))
    np.testing.assert_array_equal(found, expected)
    assert nodes_visited >= 1
    return tree, found, nodes_visited


# --- the core equivalence ------------------------------------------------


@pytest.mark.parametrize("n", [1, 2, 7, 13, 1000, 1001, 50_000, 50_001])
@pytest.mark.parametrize("seed", [1, 2, 3])
def test_tree_search_equals_numpy_mask(n, seed):
    _assert_search_matches_mask(_synthetic(n, seed * 100 + n))


def test_tree_search_equals_numpy_mask_over_many_random_datasets():
    for seed in range(40):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 5000))
        _assert_search_matches_mask(_synthetic(n, seed, n_excursions=int(rng.integers(0, 8))))


def test_nan_gaps_are_never_events():
    values = np.full(500, 50.0, dtype=np.float32)
    values[100:200] = np.nan
    _, found, _ = _assert_search_matches_mask(values)
    assert found.size == 0


def test_all_out_of_band():
    values = np.full(100, 48.0, dtype=np.float32)
    _, found, _ = _assert_search_matches_mask(values)
    assert found.size == 100


def test_values_exactly_on_the_limits_are_in_band():
    # The band is inclusive: 49.5 and 50.5 are *not* excursions.
    values = np.array([49.5, 50.5, 49.499, 50.501], dtype=np.float32)
    _, found, _ = _assert_search_matches_mask(values)
    np.testing.assert_array_equal(found, [2, 3])


def test_custom_band_is_respected():
    values = np.array([50.0, 50.2, 49.8, 50.0], dtype=np.float32)
    _assert_search_matches_mask(values, low=49.9, high=50.1)
    tree = build_tree(T0_MS, values)
    found, _ = find_out_of_band_indices(tree, low=49.9, high=50.1)
    np.testing.assert_array_equal(found, [1, 2])


# --- pruning actually prunes --------------------------------------------


def test_flat_in_band_data_visits_only_the_root():
    tree = _tree([50.0] * 100_000)
    found, nodes_visited = find_out_of_band_indices(tree)
    assert found.size == 0
    assert nodes_visited == 1  # the root's [min, max] fits in the band


def test_all_nan_data_visits_only_the_root():
    tree = _tree([NAN] * 1000)
    found, nodes_visited = find_out_of_band_indices(tree)
    assert found.size == 0
    assert nodes_visited == 1  # root count == 0


def test_nodes_visited_is_far_below_the_point_count():
    values = _synthetic(100_000, seed=42, n_excursions=3)
    _, found, nodes_visited = _assert_search_matches_mask(values)
    assert found.size > 0
    assert nodes_visited < len(values) / 10


# --- merging -------------------------------------------------------------


def test_merge_joins_gaps_at_the_threshold_and_splits_beyond_it():
    assert _merge(np.array([]), 30) == []
    assert _merge(np.array([5]), 30) == [(5, 5)]
    # gap of exactly 30 merges; 31 does not.
    assert _merge(np.array([0, 30]), 30) == [(0, 30)]
    assert _merge(np.array([0, 31]), 30) == [(0, 0), (31, 31)]
    assert _merge(np.array([0, 1, 2, 100, 101, 400]), 30) == [(0, 2), (100, 101), (400, 400)]


def test_two_dips_inside_the_merge_gap_become_one_event():
    values = np.full(300, 50.0, dtype=np.float32)
    values[50:55] = 49.0   # dip 1
    values[75:80] = 49.2   # dip 2, 20 s later -> same event
    values[200:205] = 49.3  # 120 s later -> separate event
    result = find_events(_tree(values))
    assert result["n_events"] == 2
    first = result["events"][0]
    assert first["duration_s"] == 30  # index 50..79 inclusive
    assert first["start"] == T0_MS + 50_000
    assert first["end"] == T0_MS + 80_000


# --- measurement ---------------------------------------------------------


def test_event_fields_for_a_single_dip():
    values = np.full(500, 50.0, dtype=np.float32)
    values[100:110] = 49.4
    values[105] = 48.8  # the extreme
    result = find_events(_tree(values))

    assert result["n_events"] == 1
    e = result["events"][0]
    assert e["id"] == 0
    assert e["direction"] == "under"
    assert e["duration_s"] == 10
    assert e["start"] == T0_MS + 100_000
    assert e["end"] == T0_MS + 110_000
    # end is exclusive, so the span equals the duration exactly.
    assert e["end"] - e["start"] == e["duration_s"] * 1000
    assert e["extreme_hz"] == pytest.approx(48.8, abs=1e-3)
    assert e["time_of_extreme"] == T0_MS + 105_000
    # Back in band at index 110, five seconds after the extreme.
    assert e["recovery_s"] == 5
    # The step from 49.4 to 48.8 dominates the +/-60 s window.
    assert e["max_rocof_hz_per_s"] == pytest.approx(0.6, abs=1e-3)


def test_direction_over():
    values = np.full(200, 50.0, dtype=np.float32)
    values[20:25] = 50.8
    e = find_events(_tree(values))["events"][0]
    assert e["direction"] == "over"
    assert e["extreme_hz"] == pytest.approx(50.8, abs=1e-3)


def test_direction_both_picks_the_largest_deviation_from_nominal():
    values = np.full(200, 50.0, dtype=np.float32)
    values[20:25] = 50.7   # +0.7
    values[30:35] = 49.1   # -0.9 -> further from 50, so this is the extreme
    e = find_events(_tree(values))["events"][0]
    assert e["direction"] == "both"
    assert e["extreme_hz"] == pytest.approx(49.1, abs=1e-3)
    assert e["time_of_extreme"] == T0_MS + 30_000


def test_recovery_is_none_when_the_data_ends_out_of_band():
    values = np.full(100, 50.0, dtype=np.float32)
    values[90:] = 49.0
    e = find_events(_tree(values))["events"][0]
    assert e["recovery_s"] is None
    assert e["duration_s"] == 10


def test_recovery_skips_a_nan_gap_after_the_event():
    values = np.full(200, 50.0, dtype=np.float32)
    values[50:60] = 49.0
    values[60:70] = np.nan  # missing seconds right after the dip
    e = find_events(_tree(values))["events"][0]
    # Extreme at 50; first *valid* in-band second is 70.
    assert e["time_of_extreme"] == T0_MS + 50_000
    assert e["recovery_s"] == 20


def test_recovery_measures_to_the_end_of_a_merged_event():
    values = np.full(300, 50.0, dtype=np.float32)
    values[100] = 48.9     # the extreme, at index 100
    values[120] = 49.2     # second dip within the merge gap
    e = find_events(_tree(values))["events"][0]
    assert e["duration_s"] == 21
    # Recovery runs past the second dip to index 121, not to the first in-band
    # second at 101.
    assert e["recovery_s"] == 21


def test_rocof_window_reaches_60s_either_side():
    values = np.full(400, 50.0, dtype=np.float32)
    values[200:210] = 49.0
    # A big jump 50 s before the event is inside the +/-60 s window...
    values[149] = 49.6
    e = find_events(_tree(values))["events"][0]
    assert e["max_rocof_hz_per_s"] == pytest.approx(1.0, abs=1e-3)  # 50.0 -> 49.0 step

    # ...while one 200 s away is not.
    values2 = np.full(400, 50.0, dtype=np.float32)
    values2[200:210] = 49.9  # a gentle excursion with a custom band
    e2 = find_events(_tree(values2), low=49.95, high=50.5)["events"][0]
    assert e2["max_rocof_hz_per_s"] == pytest.approx(0.1, abs=1e-3)


def test_no_events_gives_an_empty_list_and_a_readable_table():
    result = find_events(_tree([50.0] * 1000))
    assert result["n_events"] == 0 and result["events"] == []
    assert "(no events)" in summary_table(result)


def test_result_records_the_parameters_used():
    result = find_events(_tree([50.0] * 100), low=49.0, high=51.0, merge_gap_s=5)
    assert result["band"] == {"low": 49.0, "high": 51.0}
    assert result["merge_gap_s"] == 5
    assert result["n_points"] == 100


# --- json ----------------------------------------------------------------


def test_write_and_load_events(tmp_path):
    values = np.full(200, 50.0, dtype=np.float32)
    values[50:60] = 48.9
    result = find_events(_tree(values))
    path = tmp_path / "events.json"
    write_events(result, path)

    on_disk = json.loads(path.read_text())
    assert on_disk["band"] == {"low": LOW_HZ, "high": HIGH_HZ}
    assert on_disk["n_events"] == 1

    events = load_events(path)
    assert len(events) == 1 and set(events[0]) == set(Event.__dataclass_fields__)
    assert load_events(tmp_path / "missing.json") == []


# --- the real month ------------------------------------------------------


def test_real_month_finds_the_9_august_event():
    """The 9 August 2019 power cut must appear near 15:52 UTC, from the tree alone."""
    import pandas as pd

    from backend.tree import load_tree

    _, tree = load_tree()
    result = find_events(tree)

    assert result["n_events"] >= 1
    # Every August 2019 excursion is downward: the month never reaches 50.5 Hz.
    assert all(e["direction"] == "under" for e in result["events"])

    starts = [pd.Timestamp(e["start"], unit="ms", tz="UTC") for e in result["events"]]
    target = pd.Timestamp("2019-08-09 15:52:00", tz="UTC")
    nearest = min(starts, key=lambda ts: abs(ts - target))
    assert abs(nearest - target) <= pd.Timedelta(minutes=5), f"starts={starts}"

    event = result["events"][starts.index(nearest)]
    assert event["extreme_hz"] == pytest.approx(48.787, abs=1e-3)
    assert event["duration_s"] > 60

    # And the pruned search still agrees with a full numpy scan.
    found, nodes_visited = find_out_of_band_indices(tree)
    np.testing.assert_array_equal(found, np.flatnonzero(out_of_band_mask(tree.values)))
    assert nodes_visited < tree.n / 100


def test_committed_events_json_is_current():
    """results/events.json must match a fresh search, so the API serves real data."""
    from backend.tree import load_tree

    if not EVENTS_PATH.exists():
        pytest.skip("events.json not generated yet")
    _, tree = load_tree()
    fresh = find_events(tree)
    on_disk = json.loads(EVENTS_PATH.read_text())
    assert on_disk["events"] == fresh["events"]
    assert on_disk["band"] == fresh["band"]
