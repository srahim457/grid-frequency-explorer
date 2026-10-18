"""Load NESO grid frequency CSVs into a regular 1-second UTC series.

The tree in step 1 assumes one slot per second with no holes, so the loader's
job is to turn an irregular CSV (duplicates, missing seconds, a local-time
offset) into a fixed-stride array where index == seconds since t0.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CACHE_PATH = PROCESSED_DIR / "series.npz"


@dataclass
class DataReport:
    """What we learned about the raw file, for the README and sanity checks."""

    files: list[str]
    columns: list[str]
    dtypes: dict[str, str]
    tz_source: str
    first_utc: pd.Timestamp
    last_utc: pd.Timestamp
    raw_rows: int
    duplicate_timestamps: int
    n_points: int
    n_missing: int
    longest_gap_s: int
    longest_gap_start_utc: pd.Timestamp | None
    value_min: float
    value_max: float

    def to_text(self) -> str:
        gap_at = (
            self.longest_gap_start_utc.isoformat()
            if self.longest_gap_start_utc is not None
            else "n/a"
        )
        return "\n".join(
            [
                f"files:                {', '.join(self.files)}",
                f"columns:              {self.columns}",
                f"dtypes (raw):         {self.dtypes}",
                f"timezone:             {self.tz_source}",
                f"first timestamp (UTC):{self.first_utc.isoformat()}",
                f"last timestamp (UTC): {self.last_utc.isoformat()}",
                f"raw rows:             {self.raw_rows:,}",
                f"duplicate timestamps: {self.duplicate_timestamps:,} (first kept)",
                f"grid length (seconds):{self.n_points:,}",
                f"missing seconds:      {self.n_missing:,}",
                f"longest gap:          {self.longest_gap_s} s starting {gap_at}",
                f"value range (Hz):     {self.value_min:.3f} .. {self.value_max:.3f}",
            ]
        )


def _read_one(path: Path) -> pd.DataFrame:
    """Read a single CSV, or the first CSV inside a zip."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not names:
                raise ValueError(f"no CSV inside {path.name}")
            with zf.open(names[0]) as fh:
                return pd.read_csv(io.TextIOWrapper(fh, "utf-8"))
    return pd.read_csv(path)


def _pick_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Find the timestamp and frequency columns.

    NESO has shipped these under a few different names over the years, so match
    on substrings rather than hard-coding one spelling.
    """
    time_col = next(
        (c for c in df.columns if c.strip().lower() in {"dtm", "datetime", "timestamp", "date"}),
        None,
    )
    freq_col = next(
        (c for c in df.columns if c.strip().lower() in {"f", "frequency", "freq"}),
        None,
    )
    if time_col is None or freq_col is None:
        raise ValueError(f"could not identify time/frequency columns in {list(df.columns)}")
    return time_col, freq_col


def load_raw(raw_dir: Path = RAW_DIR) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Concatenate every CSV/zip in raw_dir, sorted by name for reproducibility."""
    paths = sorted(p for p in raw_dir.iterdir() if p.suffix.lower() in {".csv", ".zip"})
    if not paths:
        raise FileNotFoundError(f"no .csv or .zip files in {raw_dir}")
    frames = [_read_one(p) for p in paths]
    raw = pd.concat(frames, ignore_index=True)
    dtypes = {c: str(dt) for c, dt in raw.dtypes.items()}
    return raw, [p.name for p in paths], dtypes


def build_series(
    raw: pd.DataFrame,
    files: list[str],
    dtypes: dict[str, str],
) -> tuple[pd.Timestamp, np.ndarray, DataReport]:
    """Turn a raw frame into (t0, values, report) on a complete 1-second UTC grid."""
    time_col, freq_col = _pick_columns(raw)
    columns = list(raw.columns)
    raw_rows = len(raw)

    # utc=True both parses the embedded offset (e.g. "+0100") and normalises to
    # UTC; a naive file would be read as UTC, which we flag in the report.
    ts = pd.to_datetime(raw[time_col], utc=True, format="mixed")
    sample = str(raw[time_col].iloc[0])
    has_offset = bool(pd.Series([sample]).str.contains(r"[+-]\d{2}:?\d{2}$|Z$").iloc[0])
    tz_source = (
        f"explicit UTC offset in the file (sample: {sample!r}) -> converted to UTC"
        if has_offset
        else f"no offset in the file (sample: {sample!r}) -> assumed UTC"
    )

    values = pd.to_numeric(raw[freq_col], errors="coerce")
    s = pd.Series(values.to_numpy(), index=pd.DatetimeIndex(ts)).sort_index(kind="stable")

    duplicate_timestamps = int(s.index.duplicated(keep="first").sum())
    s = s[~s.index.duplicated(keep="first")]

    first_utc, last_utc = s.index[0], s.index[-1]
    grid = pd.date_range(first_utc, last_utc, freq="1s", tz="UTC")
    s = s.reindex(grid)  # missing seconds become NaN; no interpolation

    arr = s.to_numpy(dtype=np.float32)
    missing = np.isnan(arr)
    n_missing = int(missing.sum())
    longest_gap_s, longest_gap_start = _longest_run(missing)

    report = DataReport(
        files=files,
        columns=columns,
        dtypes=dtypes,
        tz_source=tz_source,
        first_utc=first_utc,
        last_utc=last_utc,
        raw_rows=raw_rows,
        duplicate_timestamps=duplicate_timestamps,
        n_points=len(arr),
        n_missing=n_missing,
        longest_gap_s=longest_gap_s,
        longest_gap_start_utc=grid[longest_gap_start] if longest_gap_start is not None else None,
        value_min=float(np.nanmin(arr)),
        value_max=float(np.nanmax(arr)),
    )
    return first_utc, arr, report


def _longest_run(mask: np.ndarray) -> tuple[int, int | None]:
    """Length and start index of the longest run of True in a boolean mask."""
    if not mask.any():
        return 0, None
    # Diff of the padded mask marks run starts (+1) and ends (-1).
    padded = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    starts, ends = edges[0::2], edges[1::2]
    lengths = ends - starts
    best = int(np.argmax(lengths))
    return int(lengths[best]), int(starts[best])


def load_series(
    raw_dir: Path = RAW_DIR,
    cache_path: Path = CACHE_PATH,
    use_cache: bool = True,
) -> tuple[pd.Timestamp, np.ndarray]:
    """Return (t0 UTC, float32 values) for the 1-second grid, caching to npz."""
    if use_cache and cache_path.exists():
        with np.load(cache_path) as z:
            t0 = pd.Timestamp(int(z["t0_ms"]), unit="ms", tz="UTC")
            return t0, z["values"]

    raw, files, dtypes = load_raw(raw_dir)
    t0, values, report = build_series(raw, files, dtypes)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        t0_ms=np.int64(t0.value // 1_000_000),
        values=values,
        report=report.to_text(),
    )
    return t0, values


def main() -> None:
    raw, files, dtypes = load_raw()
    t0, values, report = build_series(raw, files, dtypes)
    print(report.to_text())
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CACHE_PATH,
        t0_ms=np.int64(t0.value // 1_000_000),
        values=values,
        report=report.to_text(),
    )
    print(f"\ncached -> {CACHE_PATH}")

    expected = int((report.last_utc - report.first_utc).total_seconds()) + 1
    print(f"length check: len(values)={len(values):,} expected={expected:,} ok={len(values) == expected}")


if __name__ == "__main__":
    main()
