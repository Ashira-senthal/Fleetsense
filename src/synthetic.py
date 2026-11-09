"""Build a SYNTHETIC bus GPS point table for testing the cleaning pipeline.

WHY THIS FILE EXISTS
--------------------
The Delhi GTFS copy we hold has no arrival/departure times and one trip per
route, so no real or plausible timetable can be derived from it. Stage 2 needs
*points with timestamps* to test jump removal, so this module manufactures them
by walking along each trip's ordered stop coordinates.

EVERYTHING PRODUCED HERE IS SYNTHETIC.
  * positions    : straight-line interpolation between real stop coordinates
  * timestamps   : invented (fixed 30 s sampling, invented average speed)
  * defects      : deliberately injected teleports, duplicates, out-of-city
                   points and out-of-order timestamps, so the cleaner has known
                   ground truth to be scored against
  * no traffic, no signal loss modelling, no real dwell behaviour

Do NOT report any mobility finding from this table. It is a unit test with
coordinates attached. Real analysis starts when the official Delhi zip (with
times) or the Porto trajectories are in place.

CLI
---
    python -m src.synthetic --feed data/raw/delhi_gtfs_trimmed \
        --out data/interim/points_synthetic_raw.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.io import PROJECT_ROOT, load_gtfs

# Invented, documented constants - see module docstring.
SAMPLE_INTERVAL_S = 30          # one GPS ping every 30 s (typical AVL cadence)
CRUISE_SPEED_KMH = 18.0         # invented city-bus average, NOT measured
DWELL_PER_STOP_S = 25           # invented dwell so stop-adjacent features have something to see
GPS_NOISE_DEG = 0.00009         # ~10 m sigma, the order of consumer GPS error

# Injected defect rates (fractions of all rows).
RATE_JUMP = 0.005
RATE_DUPLICATE = 0.005
RATE_OUTSIDE = 0.002
RATE_TIME_SWAP = 0.003

DATA_SOURCE_LABEL = "synthetic_test_fixture"


def _walk_polyline(coords: np.ndarray, step_m: float) -> list[dict]:
    """Place points every `step_m` metres along a stop-to-stop polyline.

    `coords` is (n_stops, 2) of (lat, lon) in trip order. Between stops we drop
    evenly spaced points; at each stop we add `DWELL_PER_STOP_S` worth of
    identical positions, because a bus that never dwells would trivialise the
    dwell-time feature in Stage 3.

    Distances use an equirectangular approximation: fine for a few hundred
    metres inside one city, and avoids calling haversine thousands of times.
    """
    points: list[dict] = []
    m_per_deg_lat = 111_320.0

    for i in range(len(coords) - 1):
        lat0, lon0 = coords[i]
        lat1, lon1 = coords[i + 1]
        m_per_deg_lon = 111_320.0 * np.cos(np.radians((lat0 + lat1) / 2))

        seg_m = np.hypot((lat1 - lat0) * m_per_deg_lat, (lon1 - lon0) * m_per_deg_lon)
        n_inner = max(int(seg_m // step_m), 0)

        # Dwell at the start stop of this segment.
        points.extend([{"lat": lat0, "lon": lon0, "dwell": True}] * max(int(DWELL_PER_STOP_S / SAMPLE_INTERVAL_S), 1))
        for k in range(1, n_inner + 1):
            frac = k / (n_inner + 1)
            points.append({"lat": lat0 + frac * (lat1 - lat0), "lon": lon0 + frac * (lon1 - lon0), "dwell": False})

    lat_last, lon_last = coords[-1]
    points.extend([{"lat": lat_last, "lon": lon_last, "dwell": True}])
    return points


def make_fixture_points(feed_dir: str | Path, random_state: int = 42) -> pd.DataFrame:
    """Return a synthetic GPS point table with known injected defects.

    Columns
    -------
    vehicle_id, trip_id, route_id : strings
    timestamp                     : pandas datetime (timezone-naive, 'Asia/Kolkata' wall clock)
    lat, lon                      : float degrees (WGS84)
    is_dwell                      : True if the ping was generated at a stop (synthetic truth)
    defect_*                      : booleans marking deliberately injected defects (ground truth)
    data_source                   : constant "synthetic_test_fixture"
    """
    rng = np.random.default_rng(random_state)
    feed = load_gtfs(feed_dir)

    stops = feed["stops"].copy()
    stops["stop_id"] = stops["stop_id"].astype(str)
    stops["lat"] = pd.to_numeric(stops["stop_lat"], errors="coerce")
    stops["lon"] = pd.to_numeric(stops["stop_lon"], errors="coerce")
    stops = stops.dropna(subset=["lat", "lon"]).drop_duplicates("stop_id")
    coord = stops.set_index("stop_id")[["lat", "lon"]]

    # Order stop_times by stop_sequence and keep the coordinates that exist.
    st = feed["stop_times"].copy()
    st["stop_id"] = st["stop_id"].astype(str)
    st["trip_id"] = st["trip_id"].astype(str)
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    st = st.dropna(subset=["stop_sequence"]).sort_values(["trip_id", "stop_sequence"])

    trips = feed["trips"].copy()
    trips["trip_id"] = trips["trip_id"].astype(str)
    trips["route_id"] = trips["route_id"].astype(str)
    trip_route = trips.set_index("trip_id")["route_id"].to_dict()

    step_m = CRUISE_SPEED_KMH * 1000 / 3600 * SAMPLE_INTERVAL_S  # metres between pings

    rows: list[pd.DataFrame] = []
    start = pd.Timestamp("2024-06-03 05:30:00")  # a Monday, invented
    trip_counter = 0

    for trip_id, grp in st.groupby("trip_id", sort=False):
        ids = [s for s in grp["stop_id"] if s in coord.index]
        if len(ids) < 2:
            continue
        coords = coord.loc[ids].to_numpy()

        pts = _walk_polyline(coords, step_m=step_m)
        if len(pts) < 2:
            continue

        n = len(pts)
        # Stagger departures so vehicles do not all move at once.
        dep = start + pd.Timedelta(minutes=7 * (trip_counter % 200))
        trip_counter += 1

        rows.append(pd.DataFrame({
            "vehicle_id": f"V{trip_route.get(trip_id, 'NA')}",
            "trip_id": trip_id,
            "route_id": trip_route.get(trip_id, "NA"),
            "timestamp": dep + pd.to_timedelta(np.arange(n) * SAMPLE_INTERVAL_S, unit="s"),
            "lat": [p["lat"] for p in pts],
            "lon": [p["lon"] for p in pts],
            "is_dwell": [p["dwell"] for p in pts],
        }))

    df = pd.concat(rows, ignore_index=True)

    # --- realistic GPS noise on every ping -----------------------------------
    df["lat"] = df["lat"] + rng.normal(0, GPS_NOISE_DEG, len(df))
    df["lon"] = df["lon"] + rng.normal(0, GPS_NOISE_DEG, len(df))

    # --- inject defects, recording ground truth ------------------------------
    n = len(df)
    df["defect_jump"] = False
    df["defect_duplicate"] = False
    df["defect_outside"] = False
    df["defect_time"] = False

    jump_idx = rng.choice(n, size=int(n * RATE_JUMP), replace=False)
    df.loc[jump_idx, "lat"] += rng.normal(0, 0.02, len(jump_idx))   # ~2 km teleport
    df.loc[jump_idx, "lon"] += rng.normal(0, 0.02, len(jump_idx))
    df.loc[jump_idx, "defect_jump"] = True

    # Duplicates: append entire copies of existing rows. Same vehicle, same
    # timestamp, same position - i.e. the same ping delivered twice by the AVL
    # feed. Note deliberately what is NOT injected here: a repeated *position*
    # with a later timestamp, because a bus dwelling at a stop produces exactly
    # that and must never be deleted as a "duplicate".
    dup_idx = rng.choice(n, size=int(n * RATE_DUPLICATE), replace=False)
    dup_rows = df.iloc[dup_idx].copy()
    dup_rows["defect_duplicate"] = True
    df = pd.concat([df, dup_rows], ignore_index=True)
    n = len(df)

    out_idx = rng.choice(n, size=int(n * RATE_OUTSIDE), replace=False)
    # Place outside points clearly below the network's southern edge, so the
    # bounding-box rule (which runs before the speed rule) catches them and
    # removals can be attributed to the right rule.
    lat_south = float(coord["lat"].min()) - 0.25
    df.loc[out_idx, "lat"] = lat_south
    df.loc[out_idx, "defect_outside"] = True

    # Out-of-order timestamps: swap the time of a row with its neighbour.
    swap_idx = rng.choice(np.arange(0, n - 1), size=int(n * RATE_TIME_SWAP), replace=False)
    ts = df["timestamp"].to_numpy().copy()
    a = ts[swap_idx].copy()
    b = ts[swap_idx + 1].copy()
    ts[swap_idx] = b
    ts[swap_idx + 1] = a
    df["timestamp"] = ts
    df.loc[swap_idx, "defect_time"] = True

    df["data_source"] = DATA_SOURCE_LABEL
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)  # scramble row order
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the SYNTHETIC GPS test fixture")
    parser.add_argument("--feed", default="data/raw/delhi_gtfs_trimmed")
    parser.add_argument("--out", default="data/interim/points_synthetic_raw.parquet")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = make_fixture_points(args.feed, random_state=args.seed)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    print(f"SYNTHETIC fixture written to {out}")
    print(f"rows={len(df):,}  vehicles={df['vehicle_id'].nunique()}  trips={df['trip_id'].nunique()}")
    print("injected defects:",
          {c: int(df[c].sum()) for c in ("defect_jump", "defect_duplicate", "defect_outside", "defect_time")})
    print(f"timestamp range: {df['timestamp'].min()} .. {df['timestamp'].max()}")


if __name__ == "__main__":
    main()
