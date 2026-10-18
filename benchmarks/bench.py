"""Honest measurements of when the segment tree helps and when it does not.

Three ways to get the min/max/mean of a time range:

  brute_force    boolean mask over the whole timestamp array, then nan-reductions
                 -- always touches every point, so cost is O(N) whatever you ask
  binary_search  np.searchsorted for the slice bounds, then nan-reductions on it
                 -- O(log N) to locate, then O(range) to reduce
  tree           stats() -- O(log N) nodes, independent of the range width

The interesting question is not "which is fastest" but "how does each one grow".
numpy's reductions are heavily vectorized, so at a few million points a flat
scan can beat a tree walk outright; the tree's advantage is asymptotic, which is
why the scaling run matters more than the headline numbers.

Fairness rules observed here:
  - the timestamp array the baselines need is int64 epoch ms (the fastest dtype
    available to them) and is built *outside* every timed region
  - all three methods see the same random start positions
  - the methods are cross-checked for identical answers before being timed; a
    benchmark of three functions that disagree measures nothing
"""

from __future__ import annotations

import csv
import gc
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import psutil

from backend.events import find_out_of_band_indices, out_of_band_mask
from backend.loader import load_series
from backend.query import index_to_ms, stats
from backend.tree import SegmentTree, build_tree

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
CSV_PATH = RESULTS_DIR / "benchmarks.csv"
PLOT_PATH = RESULTS_DIR / "benchmark_scaling.png"
RANGE_PANELS_PATH = RESULTS_DIR / "benchmark_ranges_panels.png"
RANGE_LOG_PATH = RESULTS_DIR / "benchmark_ranges_log.png"

# Bytes per raw point needed to build *and* hold the tree, used to decide whether
# a scaling size fits in RAM before we try it. The tree itself is 20 bytes per
# point (min f32 + max f32 + sum f64 + count i32 over N/2 + N/4 + ... windows),
# the raw array is 4, and the build needs transient pair buffers. Recalibrated
# from real peak RSS as soon as the first size has run.
EST_BYTES_PER_POINT = 40
RAM_SAFETY = 0.70  # never plan to use more than this share of available memory

SCALING_EXPONENTS = (22, 24, 26, 27)
SWEEP_REPS = 50
LARGE_N_REPS = 10  # full-range scans at 2^24+ cost ~0.5 s each; 50 reps is wasteful
LARGE_N = 1 << 23


# --------------------------------------------------------------------------
# the three methods
# --------------------------------------------------------------------------


def brute_force(ts: np.ndarray, values: np.ndarray, a_ms: int, b_ms: int):
    """Mask the full timestamp array, then reduce the selected values."""
    mask = (ts >= a_ms) & (ts < b_ms)
    sl = values[mask]
    return float(np.nanmin(sl)), float(np.nanmax(sl)), float(np.nanmean(sl))


def binary_search(ts: np.ndarray, values: np.ndarray, a_ms: int, b_ms: int):
    """Locate the slice with searchsorted, then reduce just that slice."""
    i0, i1 = np.searchsorted(ts, (a_ms, b_ms))
    sl = values[i0:i1]
    return float(np.nanmin(sl)), float(np.nanmax(sl)), float(np.nanmean(sl))


def tree_stats(tree: SegmentTree, a_ms: int, b_ms: int):
    s = stats(tree, a_ms, b_ms)
    return s["min"], s["max"], s["mean"]


def timestamps_ms(t0_ms: int, n: int) -> np.ndarray:
    """The int64 epoch-ms column the baselines index into. Never timed."""
    return t0_ms + np.arange(n, dtype=np.int64) * 1000


# --------------------------------------------------------------------------
# timing
# --------------------------------------------------------------------------


@dataclass
class Row:
    """One measurement; the CSV is a flat list of these."""

    benchmark: str
    method: str
    n_points: int
    range_label: str
    range_seconds: int
    reps: int
    median_ms: float
    p05_ms: float
    p95_ms: float
    items_touched: int | str
    note: str = ""


def time_calls(fn: Callable[..., Any], arg_sets: list[tuple]) -> tuple[float, float, float]:
    """Run fn once per arg set, return (median, p05, p95) in ms."""
    elapsed = np.empty(len(arg_sets), dtype=np.float64)
    for i, args in enumerate(arg_sets):
        clock = time.perf_counter()
        fn(*args)
        elapsed[i] = (time.perf_counter() - clock) * 1000.0
    return tuple(float(x) for x in np.percentile(elapsed, [50, 5, 95]))


def verify_methods_agree(tree: SegmentTree, ts: np.ndarray, n_checks: int = 20) -> bool:
    """All three methods must return the same answer before any of them is timed."""
    rng = np.random.default_rng(0)
    for _ in range(n_checks):
        width = int(rng.integers(1, min(tree.n, 100_000)))
        i0 = int(rng.integers(0, tree.n - width + 1))
        a, b = index_to_ms(tree, i0), index_to_ms(tree, i0 + width)
        want = brute_force(ts, tree.values, a, b)
        for got in (binary_search(ts, tree.values, a, b), tree_stats(tree, a, b)):
            # nanmean over float32 accumulates in float32; the tree sums in
            # float64. 1e-5 covers that gap without hiding a real disagreement.
            if not (got[0] == want[0] and got[1] == want[1]
                    and abs(got[2] - want[2]) <= 1e-5 * abs(want[2])):
                return False
    return True


# --------------------------------------------------------------------------
# benchmark 1: range widths on the real month
# --------------------------------------------------------------------------


def range_sweep(tree: SegmentTree, ts: np.ndarray, reps: int = SWEEP_REPS) -> list[Row]:
    widths = [
        ("1 minute", 60),
        ("1 hour", 3_600),
        ("1 day", 86_400),
        ("1 week", 604_800),
        ("full dataset", tree.n),
    ]
    rng = np.random.default_rng(1234)
    rows: list[Row] = []

    for label, width in widths:
        width = min(width, tree.n)
        starts = rng.integers(0, tree.n - width + 1, size=reps)
        spans = [(index_to_ms(tree, int(i)), index_to_ms(tree, int(i) + width)) for i in starts]

        for method, fn, args, touched in [
            ("brute_force", brute_force, [(ts, tree.values, a, b) for a, b in spans], tree.n),
            ("binary_search", binary_search, [(ts, tree.values, a, b) for a, b in spans], width),
            ("tree", tree_stats, [(tree, a, b) for a, b in spans],
             stats(tree, *spans[0])["nodes_touched"]),
        ]:
            med, p05, p95 = time_calls(fn, args)
            rows.append(Row("range_sweep", method, tree.n, label, width, reps,
                            med, p05, p95, touched))
    return rows


# --------------------------------------------------------------------------
# benchmark 2: event search
# --------------------------------------------------------------------------


def event_search_bench(tree: SegmentTree, reps: int = 20) -> list[Row]:
    """Pruned tree descent vs a numpy mask over every point."""
    def numpy_scan():
        return np.flatnonzero(out_of_band_mask(tree.values))

    def tree_scan():
        return find_out_of_band_indices(tree)

    # Confirm they find the same seconds before timing them.
    assert np.array_equal(tree_scan()[0], numpy_scan()), "event search methods disagree"
    nodes_visited = tree_scan()[1]

    rows = []
    for method, fn, touched in [
        ("numpy_mask", numpy_scan, tree.n),
        ("tree", tree_scan, nodes_visited),
    ]:
        med, p05, p95 = time_calls(fn, [()] * reps)
        rows.append(Row("event_search", method, tree.n, "full dataset", tree.n, reps,
                        med, p05, p95, touched))
    return rows


# --------------------------------------------------------------------------
# benchmark 3: scaling
# --------------------------------------------------------------------------


def tiled(values: np.ndarray, n: int) -> np.ndarray:
    """Repeat the month to reach n points. Timing does not care that it repeats."""
    reps = -(-n // len(values))
    return np.ascontiguousarray(np.tile(values, reps)[:n])


def _fits(n: int, bytes_per_point: float) -> tuple[bool, float, float]:
    """(fits, bytes needed, bytes we are willing to spend)."""
    need = n * bytes_per_point
    budget = psutil.virtual_memory().available * RAM_SAFETY
    return need <= budget, need, budget


def scaling_run(month: np.ndarray, t0_ms: int) -> tuple[list[Row], list[str]]:
    """Tree build time and full-range query time for each method at each size.

    Work is staged -- tree first, then the timestamp array -- and everything is
    freed between sizes, so the peak is per-size rather than cumulative.
    """
    rows: list[Row] = []
    notes: list[str] = []
    proc = psutil.Process()
    bytes_per_point = float(EST_BYTES_PER_POINT)

    sizes = [len(month)] + [1 << p for p in SCALING_EXPONENTS]
    for n in sizes:
        label = f"2^{n.bit_length() - 1}" if n & (n - 1) == 0 else "1 month"
        fits, need, budget = _fits(n, bytes_per_point)
        if not fits:
            msg = (f"{label} ({n:,} points) SKIPPED: needs ~{need / 1e9:.2f} GB "
                   f"at {bytes_per_point:.0f} bytes/point, budget is "
                   f"{budget / 1e9:.2f} GB "
                   f"({RAM_SAFETY:.0%} of {psutil.virtual_memory().available / 1e9:.2f} GB free)")
            print(f"  {msg}")
            notes.append(msg)
            rows.append(Row("scaling", "skipped", n, "full dataset", n, 0,
                            float("nan"), float("nan"), float("nan"), 0, msg))
            continue

        reps = SWEEP_REPS if n < LARGE_N else LARGE_N_REPS
        print(f"  {label} ({n:,} points, {reps} reps)...", flush=True)
        values = tiled(month, n)
        baseline_rss = proc.memory_info().rss

        # Stage A: build the tree, time it, time stats(), then free the tree.
        tree = build_tree(t0_ms, values)
        peak_rss = proc.memory_info().rss
        rows.append(Row("scaling_build", "tree", n, "build", n, 1,
                        tree.build_seconds * 1000, tree.build_seconds * 1000,
                        tree.build_seconds * 1000, tree.nbytes,
                        f"tree {tree.nbytes / 1e6:.0f} MB"))

        a, b = t0_ms, t0_ms + n * 1000
        med, p05, p95 = time_calls(tree_stats, [(tree, a, b)] * reps)
        nodes = stats(tree, a, b)["nodes_touched"]
        rows.append(Row("scaling", "tree", n, "full dataset", n, reps,
                        med, p05, p95, nodes,
                        "single root node" if nodes == 1 else ""))

        # At a power-of-two size the full range IS the root, so the query above
        # touches one node and understates the tree's normal cost. Measure a
        # range offset by one second as well: it spans almost the same data but
        # forces the full O(log N) decomposition.
        ua, ub = t0_ms + 1000, t0_ms + (n - 1) * 1000
        med_u, p05_u, p95_u = time_calls(tree_stats, [(tree, ua, ub)] * reps)
        nodes_u = stats(tree, ua, ub)["nodes_touched"]
        rows.append(Row("scaling", "tree_unaligned", n, "full dataset -1s", n - 2, reps,
                        med_u, p05_u, p95_u, nodes_u, "forces log2(N) decomposition"))
        del tree
        gc.collect()

        # Stage B: the baselines, which need the timestamp column instead.
        ts = timestamps_ms(t0_ms, n)
        for method, fn, touched in [
            ("brute_force", brute_force, n),
            ("binary_search", binary_search, n),
        ]:
            med, p05, p95 = time_calls(fn, [(ts, values, a, b)] * reps)
            rows.append(Row("scaling", method, n, "full dataset", n, reps,
                            med, p05, p95, touched))
        peak_rss = max(peak_rss, proc.memory_info().rss)

        # Recalibrate the RAM estimate from what this size actually cost.
        measured = max(peak_rss - baseline_rss, n * 4) / n
        print(f"    peak ~{(peak_rss - baseline_rss) / 1e9:.2f} GB "
              f"({measured:.0f} bytes/point)")
        bytes_per_point = max(bytes_per_point, measured)

        del ts, values
        gc.collect()

    return rows, notes


# --------------------------------------------------------------------------
# output
# --------------------------------------------------------------------------


def write_csv(rows: list[Row], path: Path = CSV_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(asdict(rows[0])))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


# Categorical slots 1-4 of the validated default palette, in fixed order.
# Checked with the palette validator for the light surface: all gates pass;
# aqua and yellow fall below 3:1 contrast, which obliges the visible direct
# labels and the CSV table view that this script already ships.
SERIES_STYLE = {
    "brute_force": ("#2a78d6", "-", "o", "brute force"),
    "binary_search": ("#eb6834", "-", "s", "binary search"),
    # Both tree rows are the same method under two conditions, so they share a
    # hue and are separated by linestyle rather than by a fifth colour.
    "tree_unaligned": ("#1baf7a", "-", "D", "tree stats()"),
    "tree": ("#1baf7a", ":", "d", "tree stats() (aligned, 1 node)"),
    "tree_build": ("#eda100", "--", "^", "tree build (one-off)"),
}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#d9d8d3"


def plot_scaling(rows: list[Row], path: Path = PLOT_PATH) -> bool:
    """Log-log data size vs median time, one line per method. False if too few points."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    series: dict[str, list[tuple[int, float]]] = {}
    for row in rows:
        if row.benchmark == "scaling" and row.method in SERIES_STYLE:
            series.setdefault(row.method, []).append((row.n_points, row.median_ms))
        elif row.benchmark == "scaling_build":
            series.setdefault("tree_build", []).append((row.n_points, row.median_ms))
    series = {k: sorted(v) for k, v in series.items() if len(v) >= 2}
    if not series:
        return False

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    for key in SERIES_STYLE:  # fixed order, never cycled
        if key not in series:
            continue
        color, dash, marker, label = SERIES_STYLE[key]
        xs, ys = zip(*series[key])
        ax.plot(xs, ys, color=color, linestyle=dash, marker=marker,
                linewidth=2, markersize=7, label=label, zorder=3)
        # Direct label at the line end; text wears ink, the line carries
        # identity. The aligned-tree line is labelled in the legend only, to
        # keep it from colliding with the line it sits beside.
        if key != "tree":
            ax.annotate(label, (xs[-1], ys[-1]), textcoords="offset points",
                        xytext=(10, 0), va="center", fontsize=9, color=INK_SECONDARY)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("data size (points)", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("median time (ms)", fontsize=10, color=INK_SECONDARY)
    ax.set_title("Full-range min/max/mean: how each method grows with data size",
                 fontsize=12, color=INK, pad=14)
    measured_sizes = sorted({x for pts in series.values() for x, _ in pts})
    ax.set_xticks(measured_sizes)
    ax.set_xticks([], minor=True)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:.1f}M"))
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)
    # Below the axes: with five lines spanning four decades there is no empty
    # corner a legend box can sit in without covering data.
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)
    ax.margins(x=0.18)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


# --------------------------------------------------------------------------
# range-sweep plots: cost vs how much of the dataset you ask for
# --------------------------------------------------------------------------

# Categorical slots 1-3, which validate on the all-pairs list in both modes.
SWEEP_STYLE = {
    "brute_force": ("#2a78d6", "o", "brute force"),
    "binary_search": ("#eb6834", "s", "binary search"),
    "tree": ("#1baf7a", "D", "tree stats()"),
}
PANEL_ZOOMS = [(0, 4.7), (0, 0.6), (0, 0.07)]


def _sweep_series(rows: list[Row]) -> tuple[list[str], dict[str, list[float]]]:
    """(x labels in sweep order, {method: median_ms per label})."""
    labels: list[str] = []
    series: dict[str, list[float]] = {}
    for row in rows:
        if row.benchmark != "range_sweep":
            continue
        if row.range_label not in labels:
            labels.append(row.range_label)
        series.setdefault(row.method, []).append(row.median_ms)
    return labels, series


def _style_axis(ax, grid_minor: bool = False) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, which="major", color=GRID, linewidth=0.8, zorder=0)
    if grid_minor:
        ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.5, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9)


def plot_range_sweep_panels(rows: list[Row], path: Path = RANGE_PANELS_PATH) -> bool:
    """Three linear panels, each zooming into the floor of the one above.

    The numbers span 0.013-4.59 ms, so one linear axis cannot show both the
    brute-force scale and the crossover at ~0.05 ms. Stacked zooms keep the
    axis linear (and the top panel at the full 0-4.7 ms) while still letting
    the crossover be seen.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, series = _sweep_series(rows)
    if not series:
        return False
    x = list(range(len(labels)))

    fig, axes = plt.subplots(3, 1, figsize=(9, 9.5), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)

    for panel, (ax, (lo, hi)) in enumerate(zip(axes, PANEL_ZOOMS)):
        _style_axis(ax)
        for method, (color, marker, label) in SWEEP_STYLE.items():
            if method not in series:
                continue
            ax.plot(x, series[method], color=color, marker=marker, linewidth=2,
                    markersize=7, label=label, zorder=3, clip_on=True)
        ax.set_ylim(lo, hi)
        ax.set_ylabel("median time (ms)", fontsize=9, color=INK_SECONDARY)
        # Say what this panel is, and that anything above it is cut off.
        above = [SWEEP_STYLE[m][2] for m in series if max(series[m]) > hi]
        corner = f"y: {lo}-{hi} ms"
        if panel > 0 and above:
            corner += f"   (above range: {', '.join(above)})"
        at_left = panel == 0  # the full-range panel's data climbs at the right
        ax.annotate(corner, (0.005 if at_left else 0.995, 0.93),
                    xycoords="axes fraction", ha="left" if at_left else "right",
                    fontsize=8.5, color=INK_SECONDARY)

    axes[0].set_title("Cost vs how much of the month you ask for (2,678,400 points)",
                      fontsize=12, color=INK, pad=14)
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(labels)
    axes[-1].set_xlabel("range requested", fontsize=10, color=INK_SECONDARY)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, fontsize=9,
               labelcolor=INK_SECONDARY, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.005))

    fig.tight_layout(rect=(0, 0.045, 1, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def plot_range_sweep_log(rows: list[Row], path: Path = RANGE_LOG_PATH) -> bool:
    """One panel, log y: the crossover shows up as an actual line intersection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels, series = _sweep_series(rows)
    if not series:
        return False
    x = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style_axis(ax, grid_minor=True)

    for method, (color, marker, label) in SWEEP_STYLE.items():
        if method not in series:
            continue
        ax.plot(x, series[method], color=color, marker=marker, linewidth=2,
                markersize=7, label=label, zorder=3)
        ax.annotate(label, (x[-1], series[method][-1]), textcoords="offset points",
                    xytext=(10, 0), va="center", fontsize=9, color=INK_SECONDARY)

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("range requested", fontsize=10, color=INK_SECONDARY)
    ax.set_ylabel("median time (ms, log scale)", fontsize=10, color=INK_SECONDARY)
    ax.set_title("Cost vs how much of the month you ask for (2,678,400 points)",
                 fontsize=12, color=INK, pad=14)
    ax.margins(x=0.10)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
              loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3)

    fig.tight_layout()
    fig.subplots_adjust(right=0.84)  # room for the direct labels
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return True


def print_table(rows: list[Row]) -> None:
    """The table view the contrast WARN obliges, and the quickest way to read results."""
    groups: dict[str, list[Row]] = {}
    for row in rows:
        groups.setdefault(row.benchmark, []).append(row)

    for benchmark, group in groups.items():
        print(f"\n--- {benchmark} ---")
        print(f"{'method':>14} {'n_points':>12} {'range':>13} {'reps':>5} "
              f"{'median_ms':>11} {'p05':>10} {'p95':>10} {'touched':>14}")
        for row in group:
            def fmt(x: float) -> str:
                return "-" if np.isnan(x) else f"{x:.4f}"

            touched = f"{row.items_touched:,}" if isinstance(row.items_touched, int) else "-"
            print(f"{row.method:>14} {row.n_points:>12,} {row.range_label:>13} "
                  f"{row.reps:>5} {fmt(row.median_ms):>11} {fmt(row.p05_ms):>10} "
                  f"{fmt(row.p95_ms):>10} {touched:>14}")
            if row.note:
                print(f"{'':>14} note: {row.note}")


def read_csv(path: Path = CSV_PATH) -> list[Row]:
    """Rebuild rows from the CSV so the plot can be redrawn without re-measuring."""
    with path.open(newline="") as fh:
        rows = []
        for raw in csv.DictReader(fh):
            rows.append(Row(
                benchmark=raw["benchmark"], method=raw["method"],
                n_points=int(raw["n_points"]), range_label=raw["range_label"],
                range_seconds=int(raw["range_seconds"]), reps=int(raw["reps"]),
                median_ms=float(raw["median_ms"]), p05_ms=float(raw["p05_ms"]),
                p95_ms=float(raw["p95_ms"]),
                items_touched=int(raw["items_touched"]), note=raw["note"],
            ))
    return rows


def main() -> None:
    import sys

    if "--plot-only" in sys.argv:
        cached = read_csv()
        for drew, where in [(plot_scaling(cached), PLOT_PATH),
                            (plot_range_sweep_panels(cached), RANGE_PANELS_PATH),
                            (plot_range_sweep_log(cached), RANGE_LOG_PATH)]:
            print(f"redrew -> {where}" if drew else f"skipped {where.name}: no data")
        return

    from backend.tree import load_tree

    vm = psutil.virtual_memory()
    print(f"RAM: {vm.available / 1e9:.2f} GB available of {vm.total / 1e9:.2f} GB "
          f"({vm.percent}% in use)\n")

    t0, tree = load_tree()
    ts = timestamps_ms(tree.t0_ms, tree.n)
    print(f"loaded {tree.n:,} points, {tree.n_levels} levels")
    print(f"methods agree: {verify_methods_agree(tree, ts)}\n")

    rows: list[Row] = []
    print("range sweep...")
    rows += range_sweep(tree, ts)
    print("event search...")
    rows += event_search_bench(tree)

    month = tree.values
    del tree, ts
    gc.collect()

    print("scaling...")
    scaling_rows, notes = scaling_run(month, int(t0.value // 1_000_000))
    rows += scaling_rows

    print_table(rows)
    write_csv(rows)
    print(f"\nwrote -> {CSV_PATH}")
    if plot_scaling(rows):
        print(f"wrote -> {PLOT_PATH}")
    else:
        print("scaling plot skipped: fewer than two scaling sizes completed")
    for drew, where in [(plot_range_sweep_panels(rows), RANGE_PANELS_PATH),
                        (plot_range_sweep_log(rows), RANGE_LOG_PATH)]:
        print(f"wrote -> {where}" if drew else f"skipped {where.name}: no sweep data")
    for note in notes:
        print(f"note: {note}")


if __name__ == "__main__":
    main()
