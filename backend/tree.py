"""Segment tree of precomputed min/max/sum/count summaries over the 1-second series.

Why a tree: answering "min/max/mean over this range" by scanning raw data costs
O(range). With summaries at every power-of-two window size, any range is covered
by O(log N) precomputed nodes, and a chart that wants <= W points can read a
whole level at once instead of downsampling 2.7M values on every zoom.

Layout (no padding to a power of two):
  level 0 = the raw array itself, N values               N = 2,678,400
  level k = ceil(N / 2^k) windows; window i covers raw [i*2^k, min((i+1)*2^k, N))
Only the last window at each level can be partial, so a window's summary is
always exactly numpy over its raw slice.

  level 0   | v0 | v1 | v2 | v3 | v4 |            N = 5
  level 1   |  v0:v2  |  v2:v4  | v4 |            ceil(5/2) = 3  (last partial)
  level 2   |      v0:v4        | v4 |            ceil(5/4) = 2  (last partial)
  level 3   |           v0:v5         |           1 window -> stop
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from backend.loader import CACHE_PATH as SERIES_CACHE_PATH
from backend.loader import PROCESSED_DIR, load_series

TREE_CACHE_PATH = PROCESSED_DIR / "tree.npz"


@dataclass(frozen=True)
class Level:
    """Summaries for a contiguous run of windows at one level.

    A window with no valid points has min/max NaN, sum 0.0 and count 0, which
    makes the empty slot we append during the build (and a real NaN gap in the
    data) behave identically.
    """

    min: np.ndarray  # float32
    max: np.ndarray  # float32
    sum: np.ndarray  # float64 -- sums of 2.7M float32s lose precision otherwise
    count: np.ndarray  # int32

    def __len__(self) -> int:
        return len(self.min)

    @property
    def nbytes(self) -> int:
        return self.min.nbytes + self.max.nbytes + self.sum.nbytes + self.count.nbytes


def _first_level(values: np.ndarray) -> Level:
    """Build level 1 straight from the raw array.

    Level 0's four summary arrays are never materialised (they are derivable:
    min = max = value, sum = value or 0, count = 1 or 0), so we pair up raw
    values directly and skip ~32MB of redundant arrays.
    """
    v = values
    if len(v) % 2:
        v = np.append(v, np.float32(np.nan))  # one empty slot, not millions
    pairs = v.reshape(-1, 2)
    valid = ~np.isnan(pairs)
    return Level(
        min=np.fmin(pairs[:, 0], pairs[:, 1]),
        max=np.fmax(pairs[:, 0], pairs[:, 1]),
        sum=np.where(valid, pairs, 0.0).astype(np.float64).sum(axis=1),
        count=valid.sum(axis=1).astype(np.int32),
    )


def _combine(level: Level) -> Level:
    """Combine adjacent window pairs of `level` into the level above."""
    mn, mx, sm, ct = level.min, level.max, level.sum, level.count
    if len(level) % 2:
        mn = np.append(mn, np.float32(np.nan))
        mx = np.append(mx, np.float32(np.nan))
        sm = np.append(sm, 0.0)
        ct = np.append(ct, np.int32(0))
    mn, mx = mn.reshape(-1, 2), mx.reshape(-1, 2)
    return Level(
        # fmin/fmax ignore NaN unless both sides are NaN, so an all-NaN window
        # stays NaN while a half-empty one keeps its valid side.
        min=np.fmin(mn[:, 0], mn[:, 1]),
        max=np.fmax(mx[:, 0], mx[:, 1]),
        sum=sm.reshape(-1, 2).sum(axis=1),
        count=ct.reshape(-1, 2).sum(axis=1).astype(np.int32),
    )


@dataclass
class SegmentTree:
    """Raw series plus one stored Level per level above 0."""

    t0_ms: int
    values: np.ndarray  # level 0, float32
    levels: list[Level]  # levels[k - 1] holds level k
    build_seconds: float = 0.0

    @property
    def n(self) -> int:
        return len(self.values)

    @property
    def n_levels(self) -> int:
        """Number of levels including level 0."""
        return len(self.levels) + 1

    def windows_at(self, k: int) -> int:
        """Number of windows at level k."""
        return -(-self.n // (1 << k))  # ceil

    def nodes(self, k: int, lo: int, hi: int) -> Level:
        """Summaries for windows [lo, hi) at level k.

        Level 0 is derived here for just the requested slice, so callers can
        treat every level the same way without paying for stored level-0 arrays.
        """
        if not 0 <= k < self.n_levels:
            raise IndexError(f"level {k} out of range (0..{self.n_levels - 1})")
        lo = max(lo, 0)
        hi = min(hi, self.windows_at(k))
        if k > 0:
            lv = self.levels[k - 1]
            return Level(lv.min[lo:hi], lv.max[lo:hi], lv.sum[lo:hi], lv.count[lo:hi])
        v = self.values[lo:hi]
        valid = ~np.isnan(v)
        return Level(
            min=v,
            max=v,
            sum=np.where(valid, v, 0.0).astype(np.float64),
            count=valid.astype(np.int32),
        )

    @property
    def nbytes(self) -> int:
        return sum(lv.nbytes for lv in self.levels)

    def summary(self) -> str:
        rows = [f"{'level':>5} {'windows':>10} {'window_s':>9} {'MB':>7}"]
        rows.append(f"{0:>5} {self.n:>10,} {1:>9} {self.values.nbytes / 1e6:>7.1f}  (raw, derived)")
        for k, lv in enumerate(self.levels, start=1):
            rows.append(f"{k:>5} {len(lv):>10,} {1 << k:>9,} {lv.nbytes / 1e6:>7.1f}")
        rows.append("")
        rows.append(f"n points:        {self.n:,}")
        rows.append(f"levels:          {self.n_levels} (0..{self.n_levels - 1})")
        rows.append(f"top-level count: {int(self.levels[-1].count[0]) if self.levels else int(~np.isnan(self.values[0])):,}")
        rows.append(f"valid points:    {int(np.count_nonzero(~np.isnan(self.values))):,}")
        rows.append(f"tree memory:     {self.nbytes / 1e6:.1f} MB "
                    f"({self.nbytes / max(self.values.nbytes, 1):.2f}x the raw array)")
        rows.append(f"build time:      {self.build_seconds * 1000:.0f} ms")
        return "\n".join(rows)


def build_tree(t0_ms: int, values: np.ndarray) -> SegmentTree:
    """Build every level bottom-up until a level has exactly one window."""
    if len(values) == 0:
        raise ValueError("cannot build a tree over an empty series")
    values = np.ascontiguousarray(values, dtype=np.float32)
    start = time.perf_counter()
    levels: list[Level] = []
    if len(values) > 1:
        level = _first_level(values)
        levels.append(level)
        while len(level) > 1:
            level = _combine(level)
            levels.append(level)
    elapsed = time.perf_counter() - start
    return SegmentTree(t0_ms=t0_ms, values=values, levels=levels, build_seconds=elapsed)


def save_tree(tree: SegmentTree, path: Path = TREE_CACHE_PATH,
              series_path: Path = SERIES_CACHE_PATH) -> None:
    """Cache the stored levels. The raw array stays in series.npz, not duplicated."""
    arrays: dict[str, np.ndarray] = {
        "n": np.int64(tree.n),
        "n_levels": np.int64(tree.n_levels),
        "t0_ms": np.int64(tree.t0_ms),
        # Stamp the source's mtime so a reloaded tree can tell it is stale.
        "src_mtime": np.float64(series_path.stat().st_mtime if series_path.exists() else 0.0),
    }
    for k, lv in enumerate(tree.levels, start=1):
        arrays[f"min_{k}"] = lv.min
        arrays[f"max_{k}"] = lv.max
        arrays[f"sum_{k}"] = lv.sum
        arrays[f"count_{k}"] = lv.count
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


def _cache_is_fresh(path: Path, series_path: Path, n: int) -> bool:
    """A cache is usable only if it matches the current series file and length."""
    if not path.exists():
        return False
    with np.load(path) as z:
        if int(z["n"]) != n:
            return False
        stamped = float(z["src_mtime"])
    actual = series_path.stat().st_mtime if series_path.exists() else 0.0
    return stamped >= actual


def load_tree(
    series_cache: Path = SERIES_CACHE_PATH,
    tree_cache: Path = TREE_CACHE_PATH,
    use_cache: bool = True,
) -> tuple[pd.Timestamp, SegmentTree]:
    """Return (t0 UTC, tree), loading from cache when it is newer than the series."""
    t0, values = load_series(cache_path=series_cache, use_cache=use_cache)
    t0_ms = int(t0.value // 1_000_000)

    if use_cache and _cache_is_fresh(tree_cache, series_cache, len(values)):
        with np.load(tree_cache) as z:
            n_levels = int(z["n_levels"])
            levels = [
                Level(z[f"min_{k}"], z[f"max_{k}"], z[f"sum_{k}"], z[f"count_{k}"])
                for k in range(1, n_levels)
            ]
        return t0, SegmentTree(t0_ms=t0_ms, values=values, levels=levels)

    tree = build_tree(t0_ms, values)
    save_tree(tree, tree_cache, series_cache)
    return t0, tree


def main() -> None:
    import psutil

    proc = psutil.Process()
    t0, values = load_series()
    before = proc.memory_info().rss
    tree = build_tree(int(t0.value // 1_000_000), values)
    after = proc.memory_info().rss

    print(tree.summary())
    print(f"process RSS delta: {(after - before) / 1e6:.1f} MB")

    top = tree.levels[-1]
    valid = int(np.count_nonzero(~np.isnan(values)))
    print(f"\ntop level has {len(top)} window(s); count={int(top.count[0]):,} "
          f"valid={valid:,} ok={len(top) == 1 and int(top.count[0]) == valid}")
    print(f"range check: min={float(top.min[0]):.3f} max={float(top.max[0]):.3f} "
          f"mean={top.sum[0] / top.count[0]:.4f} Hz")

    save_tree(tree)
    print(f"\ncached -> {TREE_CACHE_PATH} ({TREE_CACHE_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
