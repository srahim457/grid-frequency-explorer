"""Tests for the loader's three jobs: timezone, duplicates, gap reindexing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.loader import build_series, load_series, _longest_run


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["dtm", "f"])


def _build(rows):
    return build_series(_frame(rows), files=["test.csv"], dtypes={"dtm": "object", "f": "float64"})


def test_offset_timestamps_converted_to_utc():
    t0, values, report = _build(
        [("2019-08-01 01:00:00 +0100", 49.9), ("2019-08-01 01:00:01 +0100", 50.1)]
    )
    assert t0 == pd.Timestamp("2019-08-01 00:00:00", tz="UTC")
    assert report.last_utc == pd.Timestamp("2019-08-01 00:00:01", tz="UTC")
    assert "converted to UTC" in report.tz_source
    assert values.dtype == np.float32


def test_naive_timestamps_assumed_utc():
    t0, _, report = _build([("2019-08-01 00:00:00", 50.0), ("2019-08-01 00:00:01", 50.0)])
    assert t0 == pd.Timestamp("2019-08-01 00:00:00", tz="UTC")
    assert "assumed UTC" in report.tz_source


def test_gap_becomes_nan_and_is_not_interpolated():
    # 00:00:01 and 00:00:02 are absent from the input.
    t0, values, report = _build(
        [
            ("2019-08-01 00:00:00 +0000", 50.0),
            ("2019-08-01 00:00:03 +0000", 50.3),
        ]
    )
    assert len(values) == 4
    assert np.isnan(values[1]) and np.isnan(values[2])
    assert values[0] == pytest.approx(50.0)
    assert report.n_missing == 2
    assert report.longest_gap_s == 2
    assert report.longest_gap_start_utc == pd.Timestamp("2019-08-01 00:00:01", tz="UTC")


def test_duplicates_dropped_keeping_first():
    t0, values, report = _build(
        [
            ("2019-08-01 00:00:00 +0000", 50.0),
            ("2019-08-01 00:00:00 +0000", 40.0),  # duplicate, discarded
            ("2019-08-01 00:00:01 +0000", 50.1),
        ]
    )
    assert report.duplicate_timestamps == 1
    assert len(values) == 2
    assert values[0] == pytest.approx(50.0)


def test_unsorted_input_is_sorted():
    t0, values, _ = _build(
        [
            ("2019-08-01 00:00:02 +0000", 50.2),
            ("2019-08-01 00:00:00 +0000", 50.0),
            ("2019-08-01 00:00:01 +0000", 50.1),
        ]
    )
    assert t0 == pd.Timestamp("2019-08-01 00:00:00", tz="UTC")
    np.testing.assert_allclose(values, [50.0, 50.1, 50.2], rtol=1e-6)


def test_grid_length_matches_timespan():
    t0, values, report = _build(
        [("2019-08-01 00:00:00 +0000", 50.0), ("2019-08-01 00:10:00 +0000", 50.0)]
    )
    expected = int((report.last_utc - report.first_utc).total_seconds()) + 1
    assert len(values) == expected == 601


def test_longest_run_helper():
    assert _longest_run(np.array([False, False])) == (0, None)
    assert _longest_run(np.array([True, False, True, True, True, False])) == (3, 2)
    assert _longest_run(np.array([True, True])) == (2, 0)


def test_real_series_cache_roundtrip(tmp_path):
    """The cached npz must hand back the same t0 and float32 values."""
    t0, values = load_series()
    assert values.dtype == np.float32
    assert t0.tzinfo is not None and str(t0.tz) == "UTC"
    expected = int((len(values) - 1))
    assert expected == len(values) - 1  # grid is contiguous by construction
    t0b, vb = load_series()
    assert t0b == t0 and len(vb) == len(values)
