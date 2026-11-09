"""Stage 3 - feature engineering for unsupervised mobility clustering.

The unit of analysis decides everything downstream, so it is stated once, here:

* **H3 resolution 8 cell (~0.74 km2)** - the spatial unit. It is a regular hexagon
  grid, so every cell has the same area and six equidistant neighbours; a square
  lat/lon grid would have cells that shrink with latitude and 4 vs 8 neighbours.
* **stop** - only available where a GTFS feed exists (Delhi). Porto has no stops,
  so `stop_features` exists but is not used in this run.
* **trip** - one vehicle journey (Porto trip, or a GTFS trip when real bus data exists).

WHY CYCLIC TIME ENCODING
------------------------
Hour-of-day 23 and 0 are one hour apart but 23 units apart as plain numbers, and a
clustering algorithm using Euclidean distance would treat midnight and 23:00 as
maximally dissimilar. Map the clock onto a circle instead:

    theta = 2*pi*h/24,   h_sin = sin(theta),   h_cos = cos(theta)

Then 23:00 and 00:00 are neighbours on the circle. Both components are kept: sin
alone cannot distinguish 03:00 from 09:00 (same sine), the pair (sin, cos) is unique
per hour. The same transform with period 7 is used for day-of-week.

WHY STANDARDISE
---------------
Every distance-based algorithm computes squared differences summed over features:

    d^2(x, y) = sum_j (x_j - y_j)^2

so a feature measured in km/h (0-80) dominates a feature measured as a share (0-1)
purely because of its units. z-scoring removes the units and the scale:

    z_j = (x_j - mean_j) / sd_j

This is *not* making the data normal, and it does not fix outliers - that is what
log1p on skewed counts is for. After scaling, one unit = one standard deviation of
that feature, so no single column wins by accident. Parameters mean_j and sd_j are
returned so that cluster profiles can be reported back in original units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import h3

from src.cleaning import add_step_metrics, haversine_km

# --- Thresholds -------------------------------------------------------------
# A bus or taxi below walking pace is "not making progress". Consumer GPS jitter
# puts a genuinely stopped vehicle at 0-2 km/h, so 5 km/h tolerates drift while
# still excluding slow creep in traffic. Used for dwell time.
DWELL_SPEED_KMH = 5.0
# "Slow" for the stop-and-go feature: below typical free-flow city speed.
SLOW_SPEED_KMH = 12.0
# H3 resolution 8: ~0.74 km2 per hexagon - about a city block cluster, fine enough
# to separate neighbouring hotspots, coarse enough to avoid near-empty cells.
H3_RESOLUTION = 8

NIGHT_HOURS = tuple(range(22, 24)) + tuple(range(0, 6))   # 22:00-05:59
AM_PEAK_HOURS = (7, 8, 9)
PM_PEAK_HOURS = (17, 18, 19)


# ---------------------------------------------------------------------------
# 1. Time and grid encoding
# ---------------------------------------------------------------------------
def cyclic_encode(values: pd.Series | np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray]:
    """Map a wrapped quantity onto the unit circle: (sin, cos) of 2*pi*x/period."""
    theta = 2.0 * np.pi * np.asarray(values, dtype=float) / period
    return np.sin(theta), np.cos(theta)


def add_time_features(points: pd.DataFrame, time_col: str = "timestamp_local") -> pd.DataFrame:
    """Add hour, weekday, weekend flag, peak/night flags and their cyclic encoding."""
    out = points.copy()
    ts = out[time_col]
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)          # wall-clock values are what we want here
    out["timestamp"] = ts

    out["hour"] = ts.dt.hour
    out["weekday"] = ts.dt.weekday            # 0 = Monday
    out["is_weekend"] = out["weekday"] >= 5
    out["is_night"] = out["hour"].isin(NIGHT_HOURS)
    out["is_am_peak"] = out["hour"].isin(AM_PEAK_HOURS) & ~out["is_weekend"]
    out["is_pm_peak"] = out["hour"].isin(PM_PEAK_HOURS) & ~out["is_weekend"]
    out["hour_of_week"] = out["weekday"] * 24 + out["hour"]

    out["hour_sin"], out["hour_cos"] = cyclic_encode(out["hour"], 24)
    out["dow_sin"], out["dow_cos"] = cyclic_encode(out["weekday"], 7)
    return out


def add_h3_cell(points: pd.DataFrame, resolution: int = H3_RESOLUTION,
                lat_col: str = "lat", lon_col: str = "lon") -> pd.DataFrame:
    """Attach the H3 cell id (resolution 8) of every point."""
    out = points.copy()
    out["h3_cell"] = [
        h3.latlng_to_cell(lat, lon, resolution)
        for lat, lon in zip(out[lat_col].to_numpy(), out[lon_col].to_numpy())
    ]
    out["h3_resolution"] = resolution
    return out


def prepare_points(points: pd.DataFrame, group_col: str = "trip_id") -> pd.DataFrame:
    """Add step metrics (if absent), time features and the H3 cell in one call."""
    needed = {"step_km", "step_s", "speed_kmh"}
    df = points if needed.issubset(points.columns) else add_step_metrics(points, group_col=group_col)
    df = add_time_features(df)
    df = add_h3_cell(df)

    # A step is usable for speed statistics only if it exists, has positive
    # duration, and was not flagged as physically impossible in Stage 2.
    plausible = df["speed_kmh"] > 0
    if "step_implausible" in df.columns:
        plausible &= ~df["step_implausible"].fillna(False)
    df["step_valid"] = plausible.fillna(False)
    return df


# ---------------------------------------------------------------------------
# 2. Spatial aggregation: one row per H3 cell
# ---------------------------------------------------------------------------
def cell_features(points: pd.DataFrame, cell_col: str = "h3_cell") -> pd.DataFrame:
    """Aggregate point-level data into one feature row per hexagon.

    Every feature is a *description of behaviour*, not a raw count where a rate
    is available, because cells differ in traffic volume.
    """
    p = points
    valid = p[p["step_valid"]]

    g = p.groupby(cell_col)
    feats = pd.DataFrame({
        "n_points": g.size(),
        "n_visits": g["trip_id"].nunique(),
        "n_vehicles": g["vehicle_id"].nunique(),
        "lat_centroid": g["lat"].mean(),
        "lon_centroid": g["lon"].mean(),
        "night_share": g["is_night"].mean(),
        "am_peak_share": g["is_am_peak"].mean(),
        "pm_peak_share": g["is_pm_peak"].mean(),
        "weekend_share": g["is_weekend"].mean(),
        "hour_sin_mean": g["hour_sin"].mean(),
        "hour_cos_mean": g["hour_cos"].mean(),
    })

    # Dwell: seconds spent at below-walking pace, summed over valid steps.
    dwell = (valid["step_s"].where(valid["speed_kmh"] < DWELL_SPEED_KMH, 0.0)
             .groupby(valid[cell_col]).sum())
    feats["dwell_s_total"] = dwell
    feats["dwell_s_per_visit"] = feats["dwell_s_total"] / feats["n_visits"].clip(lower=1)

    # Speed profile over valid steps only.
    vg = valid.groupby(cell_col)["speed_kmh"]
    feats["speed_mean"] = vg.mean()
    feats["speed_var"] = vg.var()
    feats["speed_p95"] = vg.quantile(0.95)
    feats["speed_cv"] = feats["speed_var"] ** 0.5 / feats["speed_mean"].replace(0, np.nan)

    feats["slow_share"] = (valid["speed_kmh"] < SLOW_SPEED_KMH).groupby(valid[cell_col]).mean()

    # Modal hour of activity (mode of the hour histogram), for human-readable naming.
    peak = (p.groupby([cell_col, "hour"]).size().rename("n").reset_index()
             .sort_values([cell_col, "n"], ascending=[True, False])
             .drop_duplicates(cell_col).set_index(cell_col)["hour"])
    feats["peak_hour"] = peak

    feats["data_source"] = p["data_source"].iloc[0] if "data_source" in p.columns else "unknown"
    return feats.sort_values("n_visits", ascending=False)


# ---------------------------------------------------------------------------
# 3. Trip-level features
# ---------------------------------------------------------------------------
def _count_episodes(flag: pd.Series) -> int:
    """Number of contiguous True runs (used to count stop-and-go episodes)."""
    f = flag.fillna(False).to_numpy(dtype=bool)
    if f.size == 0:
        return 0
    return int(f[0]) + int(np.sum(f[1:] & ~f[:-1]))


def trip_features(points: pd.DataFrame, group_col: str = "trip_id") -> pd.DataFrame:
    """One row per trip: duration, distance, dwell, stop-and-go behaviour."""
    p = points
    valid = p[p["step_valid"]]

    g = p.groupby(group_col)
    feats = pd.DataFrame({
        "start_time": g["timestamp"].min(),
        "end_time": g["timestamp"].max(),
        "n_points": g.size(),
    })
    feats["duration_s"] = (feats["end_time"] - feats["start_time"]).dt.total_seconds()
    feats["distance_km"] = valid.groupby(valid[group_col])["step_km"].sum()
    feats["dwell_s"] = (valid["step_s"].where(valid["speed_kmh"] < DWELL_SPEED_KMH, 0.0)
                        .groupby(valid[group_col]).sum())

    vg = valid.groupby(valid[group_col])["speed_kmh"]
    feats["speed_mean"] = vg.mean()
    feats["speed_p95"] = vg.quantile(0.95)
    feats["speed_sd"] = vg.std()
    feats["stopgo_cv"] = feats["speed_sd"] / feats["speed_mean"].replace(0, np.nan)

    # Dwell episodes: how many separate times the vehicle fell below walking pace.
    feats["n_dwell_episodes"] = (valid.assign(dwell=valid["speed_kmh"] < DWELL_SPEED_KMH)
                                 .groupby(group_col)["dwell"].apply(_count_episodes))

    feats["mean_speed_overall"] = feats["distance_km"] / (feats["duration_s"] / 3600.0)
    feats["start_hour"] = feats["start_time"].dt.hour
    feats["weekday"] = feats["start_time"].dt.weekday
    feats["is_weekend"] = feats["weekday"] >= 5
    feats["start_hour_sin"], feats["start_hour_cos"] = cyclic_encode(feats["start_hour"], 24)
    feats["data_source"] = p["data_source"].iloc[0] if "data_source" in p.columns else "unknown"
    return feats


# ---------------------------------------------------------------------------
# 4. Weekly activity profile (input for time-series clustering in Stage 5)
# ---------------------------------------------------------------------------
def hour_of_week_profile(points: pd.DataFrame, cell_col: str = "h3_cell",
                         normalise: bool = True) -> pd.DataFrame:
    """Matrix of cells x 168 hourly buckets (Mon 00:00 .. Sun 23:00).

    Normalising each row to sum to 1 keeps *shape* (when a place is busy) separate
    from *size* (how busy overall). DTW/KShape clustering compares shapes, so
    without this the busiest cells would form their own cluster trivially.
    """
    counts = (points.groupby([cell_col, "hour_of_week"]).size()
              .unstack(fill_value=0)
              .reindex(columns=range(168), fill_value=0))
    if normalise:
        row_sum = counts.sum(axis=1).replace(0, np.nan)
        return counts.div(row_sum, axis=0).fillna(0.0)
    return counts


# ---------------------------------------------------------------------------
# 5. Standardisation
# ---------------------------------------------------------------------------
def standardise(frame: pd.DataFrame, cols: list[str], log1p_cols: tuple[str, ...] = ()) -> tuple[pd.DataFrame, dict]:
    """z-score the selected columns; return (scaled frame, params for inversion).

    log1p is applied first to columns of skewed counts (visits, dwell seconds),
    because a single arterial cell with 10x the median visits would otherwise sit
    so far out that every other cell looks identical after scaling.
    """
    out = frame.copy()
    params: dict[str, dict] = {}

    for c in log1p_cols:
        out[c] = np.log1p(out[c].clip(lower=0))
        params[c] = {"log1p": True}

    for c in cols:
        mu = float(out[c].mean())
        sd = float(out[c].std(ddof=0))
        if not np.isfinite(sd) or sd == 0:
            # Constant column carries no information; z=0 for everyone.
            out["z_" + c] = 0.0
            params[c] = {"mean": mu, "sd": 0.0, "note": "zero variance, z set to 0"}
            continue
        out["z_" + c] = (out[c] - mu) / sd
        params[c] = {"mean": mu, "sd": sd}

    return out, params


def check_haversine_against_reference() -> pd.DataFrame:
    """Small self-test table: haversine vs a known landmark distance.

    Lisbon (38.7223 N, -9.1393) to Porto (41.1579 N, -8.6291) is about 274 km
    great-circle. If our function disagrees by more than a few km, the metric is
    broken and every distance feature is wrong.
    """
    d = haversine_km(38.7223, -9.1393, 41.1579, -8.6291)
    return pd.DataFrame({"route": ["Lisbon->Porto"], "haversine_km": [round(float(d), 2)],
                         "reference_km_approx": [274.0]})
