"""Load GTFS static feeds (and later, GPS point tables) into plain DataFrames.

Why this module exists
----------------------
Every later stage needs the same four ideas handled once, correctly:

1. GTFS is a folder (or zip) of CSV files, with no header dtypes and lots of
   numeric-looking IDs. If pandas guesses `stop_id` is an int, joins silently
   break when one file has "0123" and another has 123. So IDs are always loaded
   as text.
2. Times in GTFS are *relative to noon-minus-12h of the service day*, which
   means they may exceed 24:00:00 (a trip starting at 00:10 on a night service
   can be written 24:10:00). `parse_gtfs_time` handles that.
3. A feed may be a `.zip` or an already-extracted directory. `resolve_feed_dir`
   accepts both and unzips into `data/raw/<name>/` so the raw download is kept.
4. Some feeds (including the trimmed Delhi copy used in Stage 1) omit optional
   columns or tables entirely. Loaders must not crash on that: callers check for
   the columns they need.

Nothing here downloads anything: you put files in `data/raw/`, code reads them.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Project paths - resolved from this file, so scripts work from any cwd.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Tables we care about. `shapes`, `fare_*`, `frequencies`, `transfers` are not
# needed for clustering, so they are ignored if present.
GTFS_TABLES = ("agency", "calendar", "routes", "trips", "stops", "stop_times")
REQUIRED_GTFS_TABLES = ("routes", "trips", "stops", "stop_times")

# Columns that must stay as text (identifiers and time strings).
TEXT_COLUMNS = {
    "agency": ["agency_id", "agency_timezone"],
    "calendar": ["service_id"],
    "routes": ["route_id", "agency_id", "route_short_name", "route_long_name"],
    "trips": ["route_id", "service_id", "trip_id", "shape_id", "direction_id"],
    "stops": ["stop_id", "stop_code", "stop_name", "parent_station"],
    "stop_times": ["trip_id", "stop_id"],
}


def resolve_feed_dir(path: str | Path) -> Path:
    """Return a directory containing the GTFS `.txt` files.

    Accepts either an extracted feed directory or a `.zip` archive. A zip is
    extracted to `data/raw/<zip stem>/` (never overwriting an existing folder),
    which keeps the raw artefact and the working copy separate.
    """
    path = Path(path)
    if path.is_dir():
        return path
    if path.suffix.lower() == ".zip":
        target = RAW_DIR / path.stem
        target.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path) as zf:
            zf.extractall(target)
        # Some zips wrap the feed in an inner folder; step into it if so.
        if not (target / "stops.txt").exists():
            inner = [p for p in target.iterdir() if p.is_dir() and (p / "stops.txt").exists()]
            if inner:
                return inner[0]
        return target
    raise FileNotFoundError(f"GTFS path is neither a directory nor a .zip: {path}")


def load_gtfs_table(feed_dir: str | Path, table: str) -> pd.DataFrame | None:
    """Load one GTFS file as a DataFrame, or None if that file is absent."""
    path = Path(feed_dir) / f"{table}.txt"
    if not path.exists():
        return None
    return pd.read_csv(
        path,
        dtype={c: "string" for c in TEXT_COLUMNS.get(table, [])},
        # GTFS is UTF-8 with optional BOM; keep it simple and explicit.
        encoding="utf-8-sig",
        low_memory=False,
    )


def load_gtfs(feed_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load every GTFS table that exists in the feed.

    Returns a dict of {table_name: DataFrame}. Missing optional tables are
    simply absent from the dict. Missing *required* tables raise, because the
    pipeline cannot do anything sensible without them.
    """
    feed_dir = resolve_feed_dir(feed_dir)
    feed: dict[str, pd.DataFrame] = {}
    for table in GTFS_TABLES:
        df = load_gtfs_table(feed_dir, table)
        if df is not None:
            feed[table] = df

    missing = [t for t in REQUIRED_GTFS_TABLES if t not in feed]
    if missing:
        raise ValueError(f"GTFS feed at {feed_dir} is missing required tables: {missing}")
    return feed


def parse_gtfs_time(series: pd.Series) -> pd.Series:
    """Convert GTFS `HH:MM:SS` strings to pandas Timedelta, allowing hours >= 24.

    `pd.to_timedelta("25:10:00")` is ambiguous across pandas versions, so we
    split the string ourselves. Invalid/empty entries become NaT rather than 0,
    because "no time" and "midnight" must not be confused.
    """
    raw = series.astype("string").str.strip()
    parts = raw.str.split(":", expand=True)
    if parts.shape[1] != 3:
        raise ValueError("Expected HH:MM:SS strings")

    hours = pd.to_numeric(parts[0], errors="coerce")
    minutes = pd.to_numeric(parts[1], errors="coerce")
    seconds = pd.to_numeric(parts[2], errors="coerce")

    total = (
        pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
        + pd.to_timedelta(seconds, unit="s")
    )
    # Preserve "missing" as missing.
    return total.where(hours.notna() & minutes.notna() & seconds.notna())
