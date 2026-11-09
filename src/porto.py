"""Load Porto taxi trajectories (UCI 339, CC BY 4.0) as a GPS point table.

Why this source
---------------
The Delhi OTD static zip could not be obtained (its file host refuses connections),
and no GTFS-Realtime key was available. Porto gives what Stage 3 needs and the
synthetic fixture cannot provide: **real timestamps**. Without real times, hour-of-day
and day-of-week features are fiction.

Limitation to state everywhere results are reported: these are TAXIS, not buses.
No fixed stops, no timetable, no schedule delay; dwell patterns reflect kerbside
waiting and traffic, not boarding. It is a proxy for vehicle mobility, not for bus
service.

File facts (per the UCI dataset page, verified on download):
  * 1,710,671 trips, 442 taxis, 2013-07-01 .. 2014-06-30
  * POLYLINE is one [lon, lat] pair every 15 seconds, first = origin, last = destination
  * TIMESTAMP is the trip start, unix seconds (UTC)

Sampling
--------
Parsing all 1.7 M polylines would cost ~100 M points and tens of GB in memory, far
more than the question needs. We take a deterministic systematic sample of trips
(every k-th row, seeded) and document the exact size in the output metadata.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.io import PROJECT_ROOT

PORTO_ZIP = PROJECT_ROOT / "data" / "raw" / "porto" / "porto_taxi_ecml2015.zip"
SAMPLE_INTERVAL_S = 15                     # documented sampling of the source
PORTO_TZ = "Europe/Lisbon"                 # local wall-clock time for hour-of-day features
N_SAMPLE_TRIPS = 60_000                    # deterministic cap, ~3.5% of the corpus


def _resolve_train_csv_member(zip_path: Path) -> tuple[Path, str]:
    """Return (zip to open, member name) for train.csv.

    The UCI download is a zip whose payload is *another* zip (train.csv.zip), so a
    single `open("train.csv")` fails. Handle both layouts instead of assuming one.
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        if "train.csv" in names:
            return zip_path, "train.csv"
        inner = next((n for n in names if n.endswith("train.csv.zip")), None)
        if inner is None:
            raise FileNotFoundError(f"no train.csv or train.csv.zip inside {zip_path}; found {names}")
        # Extract the inner archive once, next to the download.
        target_dir = zip_path.parent
        inner_path = target_dir / Path(inner).name
        if not inner_path.exists():
            with zf.open(inner) as src, open(inner_path, "wb") as dst:
                dst.write(src.read())
    with zipfile.ZipFile(inner_path) as inner_zf:
        member = next(n for n in inner_zf.namelist() if n.endswith("train.csv"))
    return inner_path, member


def _read_sampled_chunks(zip_path: Path, chunksize: int = 120_000,
                         n_target: int = N_SAMPLE_TRIPS, seed: int = 42) -> pd.DataFrame:
    """Stream train.csv and keep a systematic, seeded sample of rows."""
    archive, member = _resolve_train_csv_member(zip_path)
    with zipfile.ZipFile(archive) as zf:
        keep: list[pd.DataFrame] = []
        kept = 0
        scanned = 0
        with zf.open(member) as fh:
            reader = pd.read_csv(
                fh,
                chunksize=chunksize,
                usecols=["TRIP_ID", "TAXI_ID", "TIMESTAMP", "POLYLINE", "MISSING_DATA"],
                dtype={"TRIP_ID": "string", "POLYLINE": "string"},
            )
            for chunk in reader:
                if kept >= n_target:
                    break
                scanned += len(chunk)
                take = min(n_target - kept, max(len(chunk) // 28, 1))
                keep.append(chunk.sample(n=take, random_state=seed + kept))
                kept += take
    sampled = pd.concat(keep, ignore_index=True)
    sampled.attrs["rows_scanned"] = scanned
    return sampled


def load_porto_points(zip_path: str | Path = PORTO_ZIP,
                      n_target: int = N_SAMPLE_TRIPS,
                      seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """Return (points DataFrame, metadata dict).

    Output columns: trip_id, vehicle_id (=TAXI_ID), timestamp_utc, timestamp_local,
    lat, lon, trip_point_index, data_source.

    Each trip's pings are 15 s apart; the timestamp of ping k is
    start + 15k seconds (documented by the dataset, not inferred by us).
    """
    zip_path = Path(zip_path)
    raw = _read_sampled_chunks(zip_path, n_target=n_target, seed=seed)

    n_read = len(raw)
    scanned = int(raw.attrs.get("rows_scanned", 0))
    # Trips with MISSING_DATA=True have gaps in the polyline; their 15 s spacing
    # assumption is false, so they must not be used to compute speeds.
    usable = raw[~raw["MISSING_DATA"].astype(str).str.strip().str.upper().eq("TRUE")]

    trip_ids, taxi_ids, lats, lons, ts_list, idx_list = [], [], [], [], [], []
    skipped = 0
    for row in usable.itertuples(index=False):
        try:
            poly = json.loads(row.POLYLINE)
        except (TypeError, ValueError):
            skipped += 1
            continue
        if not isinstance(poly, list) or len(poly) < 2:
            skipped += 1
            continue
        arr = np.asarray(poly, dtype=float)          # (n, 2) as [lon, lat]
        n = len(arr)
        trip_ids.append(np.repeat(row.TRIP_ID, n))
        taxi_ids.append(np.repeat(row.TAXI_ID, n))
        lons.append(arr[:, 0])
        lats.append(arr[:, 1])
        ts_list.append(row.TIMESTAMP + np.arange(n) * SAMPLE_INTERVAL_S)
        idx_list.append(np.arange(n))

    points = pd.DataFrame({
        "trip_id": np.concatenate(trip_ids).astype(str),
        "vehicle_id": np.concatenate(taxi_ids).astype(str),
        "timestamp_utc": pd.to_datetime(np.concatenate(ts_list), unit="s", utc=True),
        "lat": np.concatenate(lats),
        "lon": np.concatenate(lons),
        "trip_point_index": np.concatenate(idx_list),
    })
    # Local wall clock matters for "morning peak": a UTC hour would shift Lisbon's
    # peak by one hour in summer. Keep both, use local for features.
    points["timestamp_local"] = points["timestamp_utc"].dt.tz_convert(PORTO_TZ)
    points["data_source"] = "porto_taxi_uci339"

    meta = {
        "source": "Porto taxi trajectories, UCI ML Repository dataset 339 (ECML/PKDD 2015)",
        "licence": "CC BY 4.0 (as stated on the UCI dataset page)",
        "zip": str(zip_path),
        "sampling": "systematic seeded sample of trips (every ~28th row within each chunk)",
        "rows_sampled_before_parsing": int(n_read),
        "rows_scanned": int(scanned),
        "trips_sampled": int(points["trip_id"].nunique()),
        "trips_skipped_unparsable": int(skipped),
        "points": int(len(points)),
        "missing_data_trips_dropped": int((raw["MISSING_DATA"].astype(str).str.upper() == "TRUE").sum()),
        "sampling_interval_s": SAMPLE_INTERVAL_S,
        "timezone_local": PORTO_TZ,
        "limitation": "taxis, not buses: no fixed stops, no timetable, no schedule delay",
    }
    return points, meta


if __name__ == "__main__":
    pts, meta = load_porto_points()
    print(json.dumps(meta, indent=2))
    print(pts.head())
