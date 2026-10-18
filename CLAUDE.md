# Grid Frequency Explorer

A small demo of how to store, search and visualize a large time series quickly, using a
segment tree of precomputed summaries. Built on one month of Great Britain grid frequency data
(August 2019, 1 reading per second, ~2.7 million points).

The question this project answers:
**How many times did GB grid frequency leave its statutory limits in August 2019, and what did
the 9 August power cut look like compared with the rest of the month?**

The second goal is to show, with measurements, why a tree of summaries beats scanning raw data
as the data grows.

---

## How to work with me (read first)

- **One step at a time.** Only work on the step I name. Do not start the next step until I say so.
- **Before coding a step:** restate its goal and acceptance criteria in a few lines, and list any
  decision this file doesn't cover. Ask me before making a choice that changes a data format,
  an API response, or the repo structure.
- **After finishing a step:**
  1. Run the tests and show me the results.
  2. Explain what you built in plain language (max ~10 bullets), including the key design
     choices and why. I need to be able to explain this code in an interview.
  3. Update the **Progress** and **Data notes** sections at the bottom of this file.
  4. Suggest a commit message. Only commit after I approve. Never push.
- Keep code simple and readable. Prefer vectorized numpy. No abstractions the current step
  doesn't need. Use type hints and short docstrings that explain *why*, not just what.
- Don't add dependencies beyond the stack below without asking.
- Never modify anything in `data/raw/`.
- If a result looks wrong (for example, the 9 August event doesn't show up), stop and
  investigate. Never tune thresholds or benchmarks to make results look better.

---

## Stack

- Python 3.11+ in a conda environment that I manage. Assume it's already activated.
  **Don't install anything yourself.** If a package is missing, stop and tell me the
  `conda install -c conda-forge ...` command to run.
- Backend: numpy, pandas, fastapi, uvicorn, httpx (needed by FastAPI's test client), pytest,
  matplotlib (benchmark plots only), psutil
- Frontend: Node.js 20+, Vite + React + D3, plain JavaScript (see Decisions). Frontend
  packages are installed with `npm install` inside `frontend/`; ask me before running it.
- Keep `environment.yml` in the repo root listing the Python and Node dependencies above,
  so someone else can recreate the environment.

## Repo structure

```
grid-frequency-explorer/
  CLAUDE.md
  README.md
  environment.yml
  data/
    raw/            # NESO download(s) — not committed
    processed/      # cached arrays and tree — not committed
  backend/
    __init__.py
    loader.py       # step 0
    tree.py         # step 1
    query.py        # step 2
    events.py       # step 3
    api.py          # step 5
  benchmarks/
    bench.py        # step 4
  results/          # events.json, benchmarks.csv, plots
  tests/
    test_loader.py
    test_tree.py
    test_query.py
    test_events.py
    test_api.py
  frontend/         # step 6
```

Add `data/` and `frontend/node_modules/` to `.gitignore`.

---

## Data

- Source: NESO (GB grid operator) historic system frequency data, 1-second resolution:
  https://www.neso.energy/data-portal/system-frequency-data
- File: August 2019, placed in `data/raw/`. It may be a `.csv` or a `.zip` containing a CSV.
- Nominal frequency is 50 Hz. NESO states it must keep frequency within ±0.5 Hz of 50 Hz
  (statutory limits), so the event band is **49.5–50.5 Hz**. Cite the NESO page above in the README.
- The 9 August 2019 GB power cut began around **15:52 UTC (16:52 UK local time)**. Use this as
  a sanity check, not an assumption: confirm it from the data.

---

## Steps

### Step 0: Load and check the data (`backend/loader.py`)

Goal: a clean, regular 1-second series ready for the tree.

- Load the file(s) from `data/raw/` (handle `.csv` and `.zip`).
- Inspect and report: column names, dtypes, first/last timestamps, row count, value range.
- Determine the timezone of the timestamps. Store everything internally as **UTC**.
- Check for duplicate timestamps (report count; keep the first) and missing seconds
  (report count and the longest gap).
- Reindex onto a complete 1-second UTC grid from the first to the last timestamp.
  Missing seconds become `NaN`. Do not interpolate.
- Return `t0` (first timestamp, UTC) and `values` (float32 numpy array, one per second).
- Cache to `data/processed/series.npz`.

Acceptance:
- A printed data report, also written into **Data notes** below.
- `len(values)` equals the number of seconds between the first and last timestamp + 1.
- Tests: reindexing puts NaN in a gap; duplicates are dropped; output is float32 UTC.

### Step 1: Build the segment tree (`backend/tree.py`)

Goal: precomputed min / max / sum / count summaries at every level.

- **No padding to a power of two.** N is the real length (August = 2,678,400).
- **Level 0 is the raw array itself.** Don't store separate min/max/sum/count arrays for it;
  derive them (min = max = sum = value, count = 1, or 0 if NaN).
- Level k has `ceil(N / 2^k)` windows. Window `i` at level k covers raw indices
  `[i * 2^k, min((i + 1) * 2^k, N))`. Windows stay aligned to powers of two; only the last
  window at each level can be partial (shorter).
- Each level stores four arrays: `min` (float32), `max` (float32), `sum` (float64),
  `count` (int32).
- Build bottom-up, vectorized: if the level below has an odd length, append **one** empty
  entry (min/max NaN, sum 0, count 0), then reshape to `(-1, 2)` and combine pairs.
  This adds at most one slot per level (~22 total), not millions.
  Use NaN-aware combining: `np.fmin` / `np.fmax` (ignore NaN), sums treat NaN as 0,
  counts count only valid points. A window with no valid points has min/max NaN, count 0.
- Build until a level has exactly one window.
- Cache the tree to `data/processed/tree.npz`; rebuild if the series cache is newer.

Acceptance:
- Top level has exactly one window, and its count equals the number of valid points.
- Tests: on small hand-made arrays (including NaNs, an all-NaN window, and odd lengths such
  as 1, 7 and 13), every window's min/max/sum/count matches numpy computed directly on that
  window's raw slice, including the partial last window.
- Report build time and memory used.

### Step 2: Query functions (`backend/query.py`)

Two functions. Times in and out are **epoch milliseconds, UTC**; convert to indices with
`index = (t_ms - t0_ms) // 1000`, clamped to `[0, N]`. All ranges are
half-open `[start, end)`.

**`query(start, end, max_points)`**: for drawing the chart.
- Pick the smallest level `k ≥ 0` where `ceil((end - start) / 2^k) ≤ max_points`.
- Return the windows at level k from `start >> k` to `(end - 1) >> k` inclusive.
- Each window: `t` (window start, epoch ms), `min`, `max`, `mean` (= sum / count, null if
  count is 0), `count`.
- Return `{level, window_seconds, points}`. Edge windows may extend slightly past the range;
  that's fine for plotting.

**`stats(start, end)`**: exact summary of a range.
- Standard iterative bottom-up segment tree range query: at each level, if the left boundary
  is odd, take that node and move right; if the right boundary is odd, take the node before it;
  then halve both boundaries and go up a level. Combine everything taken.
- Return `{min, max, mean, count, nodes_touched}`.

Acceptance (tests in `tests/test_query.py`, seeded random):
- `stats`: for 1,000+ random ranges, `min` and `max` exactly equal `np.nanmin` / `np.nanmax`
  on the raw slice; `mean` matches `np.nanmean` to a relative tolerance of 1e-9;
  `count` matches the number of non-NaN values.
- Include edge cases: length-1 range, full range, ranges touching the start or end,
  ranges inside a NaN gap, ranges that end inside the partial last window.
  Run the tests on both even and odd data lengths.
- `query`: number of windows ≤ `max_points + 1`; each window's min/max equals numpy on that
  window's raw slice.
- `nodes_touched` for any range stays at or below ~2 × log2(N).

### Step 3: Event search (`backend/events.py`)

Goal: find every period outside 49.5–50.5 Hz and measure each one.

- **Tree search:** start at the top level. Skip a node if `count == 0` or
  (`min ≥ low` and `max ≤ high`). Otherwise visit its two children. At level 0, record
  out-of-band indices. Count `nodes_visited`.
- **Merge:** out-of-band seconds separated by ≤ `merge_gap_s` (default 30) form one event.
- **Measure each event:**
  - `id`, `start`, `end` (epoch ms), `duration_s` (first to last out-of-band second, inclusive)
  - `direction`: `"under"`, `"over"`, or `"both"`
  - `extreme_hz`: lowest value for under, highest for over
  - `time_of_extreme` (epoch ms)
  - `recovery_s`: seconds from `time_of_extreme` until frequency is back inside the band
  - `max_rocof_hz_per_s`: max absolute second-to-second change within 60 s either side of the
    event. Note in the README that 1-second data makes this a coarse estimate.
- Write events to `results/events.json` and print a summary table.
- Keep `low`, `high` and `merge_gap_s` as function parameters with these defaults.

Acceptance:
- Test: the tree search returns exactly the same out-of-band indices as a numpy boolean mask
  over the raw data, on random synthetic data with injected excursions and NaN gaps.
- The 9 August 2019 event is found near 15:52 UTC. If not, stop and check timezones.
- Report `nodes_visited` vs total points.

### Step 4: Benchmarks (`benchmarks/bench.py`)

Goal: honest measurements of when the tree helps.

Compare three ways to get the min / max / mean of a time range:
1. **Brute force:** boolean mask over the full timestamp array, then nan-min/max/mean.
2. **Binary search + numpy:** `np.searchsorted` to find the slice, then nan-min/max/mean on it.
3. **Segment tree:** `stats()`.

- Range widths: 1 minute, 1 hour, 1 day, 1 week, full dataset. Random start positions,
  50 repetitions each, `time.perf_counter`, report the median.
- Also time the event search: tree search vs numpy mask over all points. Report both time and
  points/nodes touched.
- **Scaling:** repeat the month (tile it) to build datasets of 2^22, 2^24 and 2^26 points
  (add 2^27 only if `psutil` shows enough free RAM; skip any size that won't fit and say so).
  Measure tree build time and full-range `stats` time for each method at each size.
- Write `results/benchmarks.csv` and a log-log plot `results/benchmark_scaling.png`
  (data size vs median time, one line per method).

Expectations: at ~2.7M points numpy may tie or beat the tree. That's a valid finding; report it.
The point of the scaling run is to show how each method grows with size.

### Step 5: API server (`backend/api.py`)

FastAPI, run locally with uvicorn. On startup: load the cached series and tree (build if
missing) and load `results/events.json`. Allow CORS from `http://localhost:5173`.

All times are epoch milliseconds (UTC).

| Endpoint | Params | Returns |
|---|---|---|
| `GET /meta` | none | `{start, end, n_points, n_missing, levels, band: {low, high}}` |
| `GET /query` | `start`, `end`, `width` | `{level, window_seconds, points: [{t, min, max, mean, count}]}` |
| `GET /stats` | `start`, `end` | `{min, max, mean, count, nodes_touched, elapsed_ms}` |
| `GET /events` | none | list of event objects from step 3 |

- Return 400 if `start >= end` or `width` is outside 1–5000.
- Use JSON `null` for NaN.

Acceptance: `tests/test_api.py` using FastAPI's `TestClient` covers each endpoint, the 400
cases, and checks that `/stats` matches `stats()` directly.

### Step 6: Dashboard (`frontend/`)

Vite + React + D3. No other UI or chart libraries.

- **Chart** (full width): time on x, Hz on y.
  - Min–max band (`d3.area`) and mean line (`d3.line`) from `/query`.
  - Dashed reference lines at 49.5, 50.0 and 50.5 Hz.
  - Events from `/events` drawn as shaded vertical spans.
  - Zoom and pan with `d3-zoom`. On each change, request `/query` with
    `width` = chart width in pixels and `/stats` for the visible range.
    Debounce (~100 ms) and cancel stale requests with `AbortController`.
  - Y axis fits the visible data with a little padding.
- **Stats panel:** min, max, mean, count for the visible range, plus the tree level in use
  ("each slice = 2^k seconds") and the `/stats` response time.
- **Events list:** one row per event (start, duration, extreme). Clicking a row zooms the chart
  to that event with 10 minutes of padding on each side.
- Times shown in UTC, labelled as UTC.

Acceptance: I can load the full month, zoom smoothly into the 9 August event down to individual
seconds, and the stats panel updates as I move.

### Step 7: README report (`README.md`)

Sections, written concisely (I'll edit the wording):
1. **The question** (from the top of this file)
2. **Data** (NESO link, the ±0.5 Hz statutory limit with citation, data quality from step 0)
3. **Approach** (a short explanation of the segment tree with a small text diagram)
4. **Results** (events table; profile of the 9 August event; leave a placeholder for a
   screenshot I'll take)
5. **Performance** (benchmark table and `results/benchmark_scaling.png`, with an honest
   discussion of where the tree wins and where it doesn't)
6. **What would break at larger scale** (data arriving live and appending to the tree,
   much higher sample rates, thousands of streams, data larger than memory, out-of-order
   or late-arriving data)
7. **How to run** (backend, tests, benchmarks, frontend)

---

## Decisions (defaults — edit if you disagree)

- Gaps are kept as NaN and excluded from stats (no interpolation).
- No padding to a power of two. Windows are aligned to powers of two; the last window at
  each level may be partial (at most one empty slot added per level during the build).
- Event band 49.5–50.5 Hz. No second, tighter band unless I ask for one.
- `merge_gap_s` = 30; RoCoF window = ±60 s.
- Timestamps: UTC internally and in the UI; epoch milliseconds in the API.
- Frontend in plain JavaScript (switch to TypeScript if I say so).
- Benchmark sizes up to 2^26 points, 2^27 only if RAM allows.

---

## Data notes

From `python -m backend.loader` (step 0):

| Field | Value |
|---|---|
| File | `data/raw/f 2019 8.csv` (plain CSV, 2,678,401 lines incl. header) |
| Columns | `dtm` (object), `f` (float64) |
| Timezone | Explicit UTC offset in the file, e.g. `2019-08-01 01:00:00 +0100` (BST) — converted to UTC |
| First timestamp (UTC) | 2019-08-01T00:00:00+00:00 |
| Last timestamp (UTC) | 2019-08-31T23:59:59+00:00 |
| Raw rows | 2,678,400 |
| Duplicate timestamps | 0 |
| Grid length (seconds) | 2,678,400 |
| Missing seconds | 0 |
| Longest gap | 0 s |
| Value range | 48.787 – 50.274 Hz |

Notes:
- The file is a complete, gap-free UTC calendar month: 31 x 86,400 = 2,678,400 seconds, one
  reading per second, so `len(values)` == seconds(last - first) + 1 with no NaNs.
- The raw file is in **BST (+0100)**, so local 01:00:00 on 1 Aug is 00:00:00 UTC. Reading these
  timestamps as naive local time would shift everything by an hour and make the 9 August event
  appear at 16:52 instead of 15:52.
- `value_min = 48.787 Hz` is well below the 49.5 Hz statutory floor — consistent with the
  9 August power cut. `value_max = 50.274 Hz` never reaches the 50.5 Hz ceiling, so every event
  in August 2019 should come out as `direction == "under"`.
- Gap handling (NaN, no interpolation) is implemented and tested, but there are no real gaps in
  this file; the NaN paths are only covered by synthetic tests.

From `python -m backend.tree` (step 1):

| Field | Value |
|---|---|
| Levels | 23 (0..22); level 0 = raw array, level 22 = 1 window of 4,194,304 s |
| Windows at level 1 | 1,339,200 (N/2) |
| Build time | 53 ms (vectorized, bottom-up) |
| Tree memory | 53.6 MB = 5.0x the 10.7 MB raw array |
| Process RSS delta during build | 93.7 MB (includes transient pair buffers) |
| Top-level count | 2,678,400 == valid points (no NaNs in this month) |
| Top-level min / max / mean | 48.787 / 50.274 / 49.9997 Hz |
| Cache | `data/processed/tree.npz`, 53.6 MB, keys `min_k`/`max_k`/`sum_k`/`count_k` |

Notes:
- The 5x memory overhead is mostly level 1: 17 bytes per window (float32 min + float32
  max + float64 sum + int32 count) over N/2 windows = 8.5 bytes per raw second, vs 4 bytes
  for the float32 value itself. Levels above that halve each time, so levels 2..22 together
  add only ~26 MB. `sum` is float64 deliberately: adding 2.7M float32 values in float32
  loses precision in the mean.
- Padding costs at most one empty slot per level (23 total), not a power-of-two pad, which
  would round 2,678,400 up to 4,194,304 and add ~1.5M phantom windows.
- Level 0 is derived on demand (`tree.nodes(0, lo, hi)`) rather than stored, saving ~32 MB
  of redundant min/max/sum/count arrays.
- Top-level min/max/mean match `np.nanmin`/`nanmax`/`nanmean` over the whole month, and the
  min of 48.787 Hz is the 9 August excursion showing through the summaries.

From `python -m backend.query` (step 2):

| Range | level chosen (max_points=800) | window | points returned |
|---|---|---|---|
| 1 minute | 0 | 1 s | 60 |
| 1 hour | 3 | 8 s | 450 |
| 1 day | 7 | 128 s | 675 |
| 1 week | 10 | 1,024 s | 591 |
| full month | 12 | 4,096 s | 654 |

| `stats` range | min | max | mean | count | nodes touched | time |
|---|---|---|---|---|---|---|
| Full month | 48.787 | 50.274 | 49.9997 | 2,678,400 | 9 | 0.04 ms |
| 9 Aug 15:52 +/- 10 min | 48.787 | 50.248 | 49.9156 | 1,200 | 10 | 0.03 ms |
| 1 Aug 00:00-01:00 UTC | 49.874 | 50.052 | 49.9452 | 3,600 | 4 | 0.01 ms |

Notes:
- The month's minimum (48.787 Hz) falls inside the 9 August 15:52 UTC window, which is the
  first independent confirmation of the event from the tree rather than from the raw file.
- Reading the whole month exactly costs **9 nodes** out of 2.7M points, and the worst random
  range stays under 2 x log2(N) + 2 = 46.
- `stats` stays correct with ceil-sized levels because both boundaries remain exact multiples
  of the current window size, so a taken node never reaches past N; the partial last window is
  never over-counted. Verified exhaustively on all 105 ranges of a 13-point series.
- Mean references in tests use `np.nanmean` on a **float64** cast: `nanmean` over a float32
  array accumulates in float32 and drifts past the 1e-9 tolerance over millions of points, so
  the float64 result is the more accurate ground truth (this is why the tree stores `sum` as
  float64).
- Empty windows and empty ranges return `None`, not NaN, so step 5 needs no NaN scrubbing.

From `python -m backend.events` (step 3):

| Field | Value |
|---|---|
| Band | 49.5-50.5 Hz, `merge_gap_s=30` |
| Events found in August 2019 | **1** |
| Out-of-band seconds | 142 |
| Nodes visited | 323 of 2,678,400 points (0.0121%, ~8,300x fewer) |
| Search time | 1.6 ms |
| Matches a full numpy mask | yes, identical 142 indices |

The single event (`results/events.json`):

| id | start (UTC) | duration_s | direction | extreme_hz | time_of_extreme (UTC) | recovery_s | max_rocof_hz_per_s |
|---|---|---|---|---|---|---|---|
| 0 | 2019-08-09 15:52:38 | 142 | under | 48.787 | 2019-08-09 15:53:49 | 71 | 0.151 |

Notes:
- The 9 August power cut is confirmed from the data at **15:52:38 UTC**, 38 seconds after the
  ~15:52 UTC figure in the brief, with no timezone adjustment. It is the *only* excursion in
  the entire month, and it is downward, as `value_max = 50.274` predicted.
- Shape of the event (sampled every 10 s): ~50.01 Hz until 15:52:28, then a drop to a plateau
  around 49.12-49.22, then a second deeper fall to 48.787 Hz at 15:53:49, then recovery back
  through 49.5 Hz at 15:55:00. The two-stage decline is consistent with two generators being
  lost in sequence, and 48.8 Hz is the level at which low-frequency demand disconnection trips.
- `end` is **exclusive**: `end - start == duration_s * 1000`, so an event's `(start, end)` can
  be passed straight to `query()`/`stats()`. This matches the half-open convention used
  everywhere else.
- `recovery_s = 71` is measured from the extreme to the first in-band second *after the whole
  merged event*, so a momentary touch of 49.5 Hz mid-event cannot shorten it. It steps over
  NaN seconds if the data is missing right after an event.
- `max_rocof_hz_per_s = 0.151` is a coarse **lower bound**: 1-second sampling cannot resolve a
  transient that real RoCoF relays measure over ~500 ms windows.
- Pruning is what makes this cheap: a node whose [min, max] fits inside the band, or whose
  count is 0, is rejected whole. Flat in-band data visits exactly **one** node (the root).
- `events.json` records the band and merge gap used, so a stale file can't be mistaken for a
  different threshold; a test asserts the committed file matches a fresh search.

From `python -m benchmarks.bench` (step 4). Machine: 17.2 GB RAM, ~83% in use by other
apps during the run. Median of 50 reps (10 at 2^24+), random start positions, `perf_counter`.

**Range sweep on the real month (2,678,400 points), median ms:**

| Range | brute force | binary search | tree | tree nodes |
|---|---|---|---|---|
| 1 minute | 1.007 | **0.013** | 0.022 | 7 |
| 1 hour | 1.339 | **0.018** | 0.045 | 12 |
| 1 day | 1.445 | 0.089 | **0.052** | 14 |
| 1 week | 2.007 | 0.540 | **0.058** | 18 |
| full dataset | 4.590 | 2.308 | **0.022** | 9 |

**Event search (whole month):** numpy mask 1.880 ms touching 2,678,400 points;
tree 0.614 ms visiting 323 nodes. ~3x faster while looking at ~8,300x less.

**Scaling (full range):**

| Points | tree build | tree (aligned) | tree (unaligned) | binary search | brute force |
|---|---|---|---|---|---|
| 2,678,400 | 39 ms | 0.022 (9 nodes) | 0.082 (35 nodes) | 2.58 | 5.14 |
| 4,194,304 (2^22) | 70 ms | 0.005 (1 node) | 0.101 (42 nodes) | 4.25 | 8.90 |
| 16,777,216 (2^24) | 297 ms | 0.005 (1 node) | 0.110 (46 nodes) | 18.07 | 36.95 |
| 67,108,864 (2^26) | **skipped** | needs ~2.68 GB, budget 1.94 GB (70% of free RAM) | | | |
| 134,217,728 (2^27) | **skipped** | needs ~5.37 GB | | | |

Notes:
- **At narrow ranges numpy wins and the tree loses.** For a 1-minute range binary search is
  ~1.7x faster than the tree: slicing 60 contiguous float32s is less work than walking 7 tree
  nodes in Python. The crossover is around **1 day**; past that the tree pulls away, and at
  full range it is ~100x faster than binary search. This is the honest finding CLAUDE.md
  predicted, and it is a *Python-overhead* result, not an algorithmic one.
- **Brute force is flat-ish and always slow** (1.0-4.6 ms) because it masks all 2.7M points
  regardless of how little data is asked for. Its cost tracks the dataset, not the question.
- **Growth rates are the real result.** 4x more data costs brute force ~4.2x and binary
  search ~4.3x more time, but the tree ~1.3x (35 -> 46 nodes). The lines diverge on the
  log-log plot and would keep diverging.
- **The "aligned" tree row is a measurement artifact worth knowing.** At a power-of-two N the
  full range *is* the root node, so `stats` touches 1 node and finishes in 5 microseconds.
  That is real but unrepresentative, so the benchmark also measures the same span offset by
  one second, which forces the full O(log N) decomposition. The plot shows both; the solid
  line is the honest one.
- **Tree build is a real one-off cost** (~18 ns/point, so 39 ms for the month, 297 ms at
  2^24). It pays for itself after roughly 8 full-range brute-force queries, but it is pure
  overhead for a dataset queried once.
- **2^26 and 2^27 were skipped for RAM, not for convenience.** At 40 bytes/point (4 raw + 20
  tree + transient build buffers) 2^26 needs ~2.7 GB against a 1.94 GB budget (70% of free
  RAM). The script estimates before each size, recalibrates from measured peak RSS, and
  records the skip in the CSV. Re-running with more free memory will include them.
- Fairness: the baselines get an int64 epoch-ms timestamp array (their fastest option), built
  outside every timed region, and all three methods see the same random starts. The script
  cross-checks that all three return identical answers before timing any of them
  (`methods agree: True`).
- Three plots, all drawn from `benchmarks.csv`:
  - `results/benchmark_scaling.png` -- x is **dataset size** (2.7M -> 16.8M), always a
    full-range query. Answers "how does each method grow as the data grows?"
  - `results/benchmark_ranges_log.png` -- x is **how much of the month you ask for**
    (1 min -> full), log y. The binary-search/tree crossover at ~1 day is a visible
    line intersection. This is the clearest single chart of the two-regime result.
  - `results/benchmark_ranges_panels.png` -- the same range sweep on three stacked
    **linear** panels (0-4.7 / 0-0.6 / 0-0.07 ms), each zooming into the floor of the one
    above. One linear axis cannot hold 0.013 ms and 4.59 ms at once; the stack keeps the
    axis linear and still reaches the crossover.
- `python -m benchmarks.bench --plot-only` redraws all three PNGs from the CSV without
  re-measuring.

## Progress

- [x] Step 0: Load and check the data
- [x] Step 1: Build the segment tree
- [x] Step 2: Query functions
- [x] Step 3: Event search
- [x] Step 4: Benchmarks
- [ ] Step 5: API server
- [ ] Step 6: Dashboard
- [ ] Step 7: README report
