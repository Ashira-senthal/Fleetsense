"""Stage 2 - cleaning GPS point tables.

What this module removes, in this order (order matters, see the docstring of
`clean_points`):

1. rows with missing coordinates or timestamps
2. duplicate pings - same vehicle, same timestamp
3. points outside the city bounding box
4. out-of-order timestamps for the same vehicle (negative time steps)
5. impossible GPS jumps - a step whose implied speed exceeds a bus

Every threshold is a named constant with the reasoning written next to it.
Nothing here is tuned to make the numbers look good: the thresholds come from
GPS error characteristics and bus physics, and are then *checked* against the
synthetic fixture, which has known injected defects.

DISTANCE METRIC: haversine, never Euclidean on lat/lon. One degree of longitude
is 111 km at the equator but only ~97.6 km at Delhi's latitude (28.6 N), so
Euclidean distance in degrees would systematically shrink east-west distances
and bias every speed, segment length and cluster shape.

    a = sin^2(dphi/2) + cos(phi1) * cos(phi2) * sin^2(dlambda/2)
    d = 2 * R * arcsin(sqrt(a)),        R = 6371.0088 km

with phi = latitude, lambda = longitude in radians. The arcsin form is used
instead of atan2 because it is numerically stable for the short distances we
care about here (tens to hundreds of metres).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

# --- Thresholds and why -----------------------------------------------------
# A city bus physically cannot average 100 km/h between two pings, including
# across an expressway. Delhi city bus operations are well below this; the
# threshold is set high on purpose so that only *impossible* steps are removed
# and genuinely fast segments survive. If your agency publishes a governor or
# speed-limit policy, replace this with that number.
MAX_SPEED_KMH = 100.0

# GPS consumer receivers are accurate to roughly 5-15 m. At 30 s sampling and
# 30 km/h, two consecutive true pings are ~250 m apart, i.e. an order of
# magnitude above the noise, so a step far above the speed threshold cannot be
# explained by noise: it is a teleport (multipath, tunnel exit, bad fix).
# Consequence: jumps must be *dropped*, not smoothed, because the point is wrong.
BBOX_MARGIN_DEG = 0.05   # ~5.5 km around the observed stop extent

# A single teleport creates an impossible step *into* the bad point and another
# one *out of* it, so flagging "every fast step" deletes the bad point AND its
# innocent neighbour. Measured on the synthetic fixture: 3 passes still left a
# 127.8 km/h step and removed 3,953 rows for 2,156 injected jumps; 5 passes
# converged (max residual speed 99.94 km/h, 3,956 removals). Hence 5, and the
# spike rule in `_jump_mask` below.
MAX_JUMP_PASSES = 5


def haversine_km(lat1, lon1, lat2, lon2, radius_km: float = EARTH_RADIUS_KM) -> np.ndarray:
    """Great-circle distance in km between two sets of WGS84 points (vectorised)."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))

    dphi = lat2 - lat1
    dlam = lon2 - lon1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlam / 2.0) ** 2
    # np.clip guards against a > 1 from floating point, which would give NaN.
    return 2.0 * radius_km * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bbox_from_stops(stops: pd.DataFrame, margin_deg: float = BBOX_MARGIN_DEG) -> dict:
    """City bounding box = observed stop extent + a margin.

    Deriving the box from the feed's own stops (rather than a hard-coded map
    rectangle) keeps the filter honest when you switch city or agency: any point
    far outside the network that operates the buses is a data error, not a place.
    """
    lat = pd.to_numeric(stops["stop_lat"], errors="coerce").dropna()
    lon = pd.to_numeric(stops["stop_lon"], errors="coerce").dropna()
    return {
        "lat_min": float(lat.min() - margin_deg),
        "lat_max": float(lat.max() + margin_deg),
        "lon_min": float(lon.min() - margin_deg),
        "lon_max": float(lon.max() + margin_deg),
        "margin_deg": margin_deg,
        "source": "observed stop extent + margin",
    }


def add_step_metrics(points: pd.DataFrame, group_col: str = "vehicle_id",
                     time_col: str = "timestamp") -> pd.DataFrame:
    """Sort by vehicle+time and add step distance, time step and implied speed.

    `shift(1)` within each vehicle gives the previous ping of *that* vehicle;
    without the groupby, a step would be computed between two different buses
    and produced nonsense speeds.
    """
    # time_col exists because real feeds name their time column differently
    # (Porto: timestamp_utc), while the maths only needs *a* time column.
    df = points.sort_values([group_col, time_col], kind="mergesort").reset_index(drop=True)

    prev_lat = df.groupby(group_col)["lat"].shift(1)
    prev_lon = df.groupby(group_col)["lon"].shift(1)
    prev_ts = df.groupby(group_col)[time_col].shift(1)

    df["step_km"] = haversine_km(prev_lat, prev_lon, df["lat"], df["lon"])
    df["step_s"] = (df[time_col] - prev_ts).dt.total_seconds()
    # speed = distance / time, guarding against divide-by-zero.
    df["speed_kmh"] = np.where(df["step_s"] > 0, df["step_km"] / (df["step_s"] / 3600.0), np.nan)
    return df


def _jump_mask(df: pd.DataFrame, max_speed_kmh: float, group_col: str,
               time_col: str = "timestamp") -> pd.Series:
    """Boolean mask of points that are physically impossible *for a bus*.

    The rule needs corroboration, and that is the whole point:

      * a point whose step IN is impossible and whose step OUT is normal is
        usually an innocent point that follows a bad fix - keep it;
      * a point whose steps IN *and* OUT are both impossible is a spike - the
        GPS fix is wrong and the point must go;
      * the last point of a vehicle has no outgoing step, so an impossible
        incoming step is all the evidence we can get - drop it.

    `df` must already be sorted by (group, timestamp), which `add_step_metrics`
    guarantees.
    """
    g = df.groupby(group_col, sort=False)
    lat_next = g["lat"].shift(-1)
    lon_next = g["lon"].shift(-1)
    ts_next = g[time_col].shift(-1)

    step_next_km = haversine_km(df["lat"], df["lon"], lat_next, lon_next)
    step_next_s = (ts_next - df[time_col]).dt.total_seconds()
    speed_out = np.where(step_next_s > 0, step_next_km / (step_next_s / 3600.0), np.nan)

    speed_in = df["speed_kmh"].to_numpy()
    is_last = lat_next.isna().to_numpy()

    spike = (speed_in > max_speed_kmh) & (speed_out > max_speed_kmh)
    tail = (speed_in > max_speed_kmh) & is_last
    return pd.Series(spike | tail, index=df.index)


def clean_points(
    points: pd.DataFrame,
    bbox: dict,
    *,
    max_speed_kmh: float = MAX_SPEED_KMH,
    group_col: str = "vehicle_id",
    time_col: str = "timestamp",
) -> tuple[pd.DataFrame, dict]:
    """Run the whole cleaning chain and return (clean points, report).

    The order is deliberate:
      * drop missing first, because every later step assumes usable numbers;
      * de-duplicate before measuring speed, because a repeated timestamp gives
        step_s = 0 and an infinite speed that would be misread as a jump;
      * apply the bounding box before speed, because a point 60 km outside the
        city is caught by a cheaper and more explainable rule;
      * remove jumps last and iteratively, because each removal changes the
        step of its neighbours.
    """
    report: dict = {"thresholds": {"max_speed_kmh": max_speed_kmh, "bbox": bbox,
                                   "max_jump_passes": MAX_JUMP_PASSES}}
    df = points.copy()
    report["rows_in"] = int(len(df))

    # --- 1. missing coordinates / timestamps --------------------------------
    before = len(df)
    df = df.dropna(subset=["lat", "lon", time_col, group_col])
    report["dropped_missing"] = int(before - len(df))

    # --- 2. duplicate pings: same vehicle, same timestamp -------------------
    before = len(df)
    # Two sub-cases, both fatal: an exact repeated row, and two different
    # positions for the same vehicle at the same second (the AVL feed cannot
    # have been in two places). Keep the first, count the rest.
    exact_dupes = df.duplicated(subset=[group_col, time_col, "lat", "lon"]).sum()
    df = df.drop_duplicates(subset=[group_col, time_col, "lat", "lon"], keep="first")
    conflicting = df.duplicated(subset=[group_col, time_col]).sum()
    df = df.drop_duplicates(subset=[group_col, time_col], keep="first")
    report["dropped_exact_duplicate_pings"] = int(exact_dupes)
    report["dropped_conflicting_timestamps"] = int(conflicting)
    report["dropped_duplicates_total"] = int(before - len(df))

    # --- 3. bounding box ----------------------------------------------------
    before = len(df)
    in_box = (
        df["lat"].between(bbox["lat_min"], bbox["lat_max"])
        & df["lon"].between(bbox["lon_min"], bbox["lon_max"])
    )
    df = df[in_box]
    report["dropped_outside_bbox"] = int(before - len(df))

    # --- 4. out-of-order timestamps ----------------------------------------
    df = add_step_metrics(df, group_col=group_col, time_col=time_col)
    before = len(df)
    # A non-positive step is a duplicate second or a clock that went backwards;
    # both make speed undefined and must not be interpreted as movement.
    df = df[~(df["step_s"].notna() & (df["step_s"] <= 0))]
    report["dropped_nonpositive_timestep"] = int(before - len(df))

    # --- 5. impossible jumps (iterative, spike-aware) ----------------------
    total_jumps = 0
    passes_used = 0
    for i in range(MAX_JUMP_PASSES):
        df = add_step_metrics(df, group_col=group_col, time_col=time_col)
        jump = _jump_mask(df, max_speed_kmh, group_col, time_col)
        n_jump = int(jump.sum())
        passes_used = i + 1
        if n_jump == 0:
            passes_used = i
            break
        df = df[~jump]
        total_jumps += n_jump
    report["dropped_impossible_jumps"] = total_jumps
    report["jump_passes_used"] = passes_used

    # --- final state --------------------------------------------------------
    df = add_step_metrics(df, group_col=group_col, time_col=time_col)
    # Residual impossible steps are FLAGGED, not deleted. Reason: when a step is
    # impossible, position alone cannot tell which of its two endpoints is the
    # bad fix - the teleport may sit near its predecessor, making the innocent
    # successor look like the culprit. Deleting both endpoints would erase real
    # data (measured: ~2 bad rows per injected defect). Instead the step is
    # marked, and Stage 3 computes speed features only from plausible steps.
    df["step_implausible"] = df["speed_kmh"] > max_speed_kmh
    report["residual_implausible_steps"] = int(df["step_implausible"].sum())
    report["residual_implausible_vehicles"] = int(df.loc[df["step_implausible"], group_col].nunique())

    report["rows_out"] = int(len(df))
    report["pct_removed"] = round(100 * (1 - len(df) / max(report["rows_in"], 1)), 3)

    def _speed_stats(frame: pd.DataFrame) -> dict:
        s = frame["speed_kmh"].dropna()
        if s.empty:
            return {}
        return {k: round(float(v), 2) for k, v in
                {"p50": s.quantile(0.50), "p95": s.quantile(0.95), "p99": s.quantile(0.99), "max": s.max()}.items()}

    report["speed_kmh_before"] = _speed_stats(add_step_metrics(points, group_col=group_col, time_col=time_col))
    report["speed_kmh_after"] = _speed_stats(df)
    return df, report


def score_against_known_defects(raw: pd.DataFrame, clean: pd.DataFrame,
                                id_col: str = "trip_id") -> dict:
    """Measure how well cleaning removed the fixture's injected defects.

    This is only possible because the synthetic fixture records ground truth.
    On real data there is no such column, and the equivalent check is a manual
    look at the removed points on a map.
    """
    defect_cols = [c for c in raw.columns if c.startswith("defect_")]
    if not defect_cols:
        return {}

    key = ["vehicle_id", "timestamp", "lat", "lon"]
    clean_keys = set(map(tuple, clean[key].itertuples(index=False, name=None)))
    raw_keyed = raw.reset_index(drop=True)[key].itertuples(index=False, name=None)
    survived_mask = np.fromiter((k in clean_keys for k in raw_keyed), dtype=bool, count=len(raw))

    scores = {}
    for col in defect_cols:
        mask = raw[col].to_numpy(dtype=bool)
        n_injected = int(mask.sum())

        if col == "defect_duplicate":
            # Key matching cannot score duplicates: the surviving copy is
            # byte-identical to the one that was removed. The right question is
            # "are any duplicate keys left?", not "did this exact row survive?".
            scores[col] = {
                "injected": n_injected,
                "duplicate_keys_remaining_in_clean": int(
                    clean.duplicated(subset=["vehicle_id", "timestamp"]).sum()
                ),
                "recall": 1.0 if not clean.duplicated(subset=["vehicle_id", "timestamp"]).any() else None,
            }
            continue

        if col == "defect_time":
            # Out-of-order timestamps are REPAIRED by sorting, not deleted:
            # the observation is real, only its order in the feed was wrong.
            # So "recall" is the wrong metric; the right one is whether any
            # non-positive time step survives.
            scores[col] = {
                "injected": n_injected,
                "nonpositive_steps_after_clean": int((clean["step_s"] <= 0).sum()),
                "note": "repaired by sorting; rows intentionally kept",
            }
            continue

        survivors = int(survived_mask[mask].sum())
        scores[col] = {
            "injected": n_injected,
            "survived_cleaning": survivors,
            "recall": round(1 - survivors / n_injected, 4) if n_injected else None,
        }
    return scores


def snap_to_network(points: pd.DataFrame, graph, lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    """OPTIONAL: snap points to the nearest OSM road edge.

    Not run in this project yet, on purpose:
      * it needs an OSM download for the whole city (minutes and hundreds of MB),
      * it moves points onto roads a bus may not have used (a GPS error near a
        flyover can snap onto the wrong level),
      * clustering works on stop/hex aggregates, where snapping changes little.

    If you enable it, keep the raw coordinates in the table (`lat_raw`,
    `lon_raw`) so Stage 2 can be audited and reverted.
    """
    import osmnx as ox  # local import: keeps the core pipeline importable without OSM

    if graph is None:
        raise ValueError("snap_to_network needs an OSMnx graph; see Stage 7 / osmnx.graph_from_place")
    snapped = ox.distance.nearest_edges(graph, X=points[lon_col].to_numpy(), Y=points[lat_col].to_numpy())
    out = points.copy()
    out["snapped_u"], out["snapped_v"], out["snapped_key"] = zip(*snapped)
    return out
