"""Stage 7 - interpretation: cluster profiles, OSM POI context, names, anomalies.

WHAT THIS MODULE DOES, AND WHAT IT REFUSES TO DO
-----------------------------------------------
It turns label tables into sentences a human can check:
  * profile each cluster (mean of every feature, size, peak hour, location),
  * attach OpenStreetMap POI context (schools, markets, transport, ...) so that
    "high dwell + morning peak" can be read as "market/school-adjacent",
  * give every cluster a short name built from *measured* ranks, never from
    intuition about what a cluster "should" be,
  * flag anomalies in three different senses (behaviour, rhythm, day),
  * write down limitations and privacy rules.

Names are a *reading aid*, not ground truth. There are no labels in this project,
so a name is a hypothesis stated in the clearest possible form and printed next to
the numbers that produced it (`description`).

Robust z-scores are used throughout for outlier work:

    z = (x - median) / (1.4826 * MAD),      MAD = median(|x - median(x)|)

The 1.4826 factor makes MAD a consistent estimator of sigma for normal data, and
unlike mean/sd the pair is not dragged around by the very outliers we are hunting.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
H3_RES = 8

# POI categories asked of OpenStreetMap. Each is a separate Overpass query, cached on disk.
# Chosen so that the behaviour axes we measured can plausibly be explained by them:
# congestion (retail, industry, hospitality), footfall (education, market), transit
# (transport), and civic/health as controls.
POI_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "education": {"amenity": ["school", "university", "college", "kindergarten"]},
    "market": {"amenity": ["marketplace"], "shop": ["supermarket", "mall", "department_store"]},
    "transport": {"railway": ["station", "halt"], "highway": ["bus_stop"],
                  "amenity": ["bus_station", "ferry_terminal"]},
    "health": {"amenity": ["hospital", "clinic", "pharmacy", "doctors"]},
    "hospitality": {"amenity": ["restaurant", "cafe", "bar", "pub", "fast_food"]},
    "civic": {"amenity": ["bank", "post_office", "police", "townhall", "courthouse"],
              "office": ["government"]},
    "worship": {"amenity": ["place_of_worship"]},
    "retail_area": {"landuse": ["retail", "commercial"]},
    "industry": {"landuse": ["industrial"], "amenity": ["fuel"]},
}

# Feature groups used to build the two naming axes. Kept explicit so a reader can
# disagree with the grouping and change it in one place.
CONGESTION_FEATURES = ["slow_share", "speed_cv", "dwell_s_per_visit"]   # all "higher = slower"
FLUIDITY_FEATURES = ["speed_mean", "speed_p95"]                          # "higher = freer"
VOLUME_FEATURES = ["n_visits", "n_vehicles"]
TEMPORAL_FEATURES = {"am": "am_peak_share", "pm": "pm_peak_share",
                     "night": "night_share", "weekend": "weekend_share"}


# ---------------------------------------------------------------------------
# small statistics helpers
# ---------------------------------------------------------------------------
def robust_z(x: np.ndarray) -> np.ndarray:
    """(x - median) / (1.4826 * MAD); zeros where the spread is zero."""
    x = np.asarray(x, dtype=float)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad == 0:
        return np.zeros_like(x)
    return (x - med) / (1.4826 * mad)


def zscore_rows(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Standardise the given columns *across clusters* (so clusters can be compared)."""
    out = df[cols].astype(float).copy()
    sd = out.std(ddof=0).replace(0, np.nan)
    return (out - out.mean()) / sd


# ---------------------------------------------------------------------------
# OSM POIs
# ---------------------------------------------------------------------------
def bbox_from_cells(cells: pd.DataFrame, pad: float = 0.01) -> tuple[float, float, float, float]:
    """(west, south, east, north) of the analysed area, with a small pad in degrees."""
    return (float(cells["lon_centroid"].min()) - pad, float(cells["lat_centroid"].min()) - pad,
            float(cells["lon_centroid"].max()) + pad, float(cells["lat_centroid"].max()) + pad)


REPO_ROOT = Path(__file__).resolve().parents[1]      # .../Fleetsense


def fetch_pois(bbox, cache_dir: str | Path | None = None, pause: float = 2.0,
               categories: dict | None = None) -> pd.DataFrame:
    """Download POIs per category from OSM (Overpass), caching each category to parquet.

    Caching matters for two reasons: Overpass is a shared volunteer service (do not
    hammer it), and the notebook must be re-runnable offline once the data is fetched.
    A failed category is skipped with a warning rather than crashing the stage, so a
    temporary Overpass failure degrades the POI table instead of the whole analysis.

    The cache path is anchored to the *repository*, not the current directory: a notebook
    runs with cwd=notebooks/, and a relative path there silently misses the cache and
    re-downloads everything (this happened once - nine Overpass queries, no fun).
    """
    import osmnx as ox
    from shapely.geometry import box

    categories = categories or POI_CATEGORIES
    cache_dir = Path(cache_dir) if cache_dir is not None else REPO_ROOT / "data" / "raw" / "osm"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.cache_folder = str(cache_dir / "overpass_cache")
    poly = box(bbox[0], bbox[1], bbox[2], bbox[3])
    ox.settings.requests_timeout = 120
    frames = []

    for cat, tags in categories.items():
        cache = cache_dir / f"pois_{cat}.parquet"
        if cache.exists():
            frames.append(pd.read_parquet(cache))
            continue
        try:
            gdf = ox.features.features_from_polygon(poly, tags=tags)
        except Exception as exc:                                  # network / Overpass hiccup
            print(f"  [warn] POI category '{cat}' failed: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        if len(gdf) == 0:
            continue
        # Centroid of each polygon/line in degrees. At this scale (tens of km) the
        # distortion of a degree-space centroid is far below one H3 cell (0.74 km2).
        cent = gdf.geometry.representative_point()
        df = pd.DataFrame({"lat": cent.y.to_numpy(), "lon": cent.x.to_numpy(), "category": cat})
        df = df.dropna().astype({"lat": float, "lon": float})
        df.to_parquet(cache, index=False)
        frames.append(df)
        print(f"  fetched {len(df):5d} '{cat}' POIs")
        time.sleep(pause)                                          # be polite to Overpass

    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "category"])
    return pd.concat(frames, ignore_index=True)


def poi_cell_counts(pois: pd.DataFrame, res: int = H3_RES) -> pd.DataFrame:
    """Count POIs per H3 cell per category -> wide table indexed by h3_cell."""
    import h3

    if pois.empty:
        return pd.DataFrame(columns=list(POI_CATEGORIES))
    cells = [h3.latlng_to_cell(la, lo, res) for la, lo in zip(pois["lat"], pois["lon"])]
    tab = (pd.DataFrame({"h3_cell": cells, "category": pois["category"]})
           .value_counts(["h3_cell", "category"]).unstack(fill_value=0))
    return tab.reindex(columns=[c for c in POI_CATEGORIES if c in tab.columns], fill_value=0)


# ---------------------------------------------------------------------------
# cluster profiles
# ---------------------------------------------------------------------------
PROFILE_FEATURES = ["n_visits", "n_vehicles", "dwell_s_per_visit", "speed_mean", "speed_p95",
                    "speed_cv", "slow_share", "night_share", "am_peak_share", "pm_peak_share",
                    "weekend_share"]


def cluster_profiles(cells: pd.DataFrame, labels: pd.Series, res: int = H3_RES) -> pd.DataFrame:
    """One row per cluster: size, centroid, mean of each behaviour feature."""
    df = cells.copy()
    df["cluster"] = labels.reindex(df["h3_cell"]).to_numpy()
    df = df[df["cluster"].notna()]
    df["cluster"] = df["cluster"].astype(int)

    g = df.groupby("cluster")
    prof = g[PROFILE_FEATURES].mean().round(4)
    prof.insert(0, "n_cells", g.size())
    prof.insert(1, "share_cells", (g.size() / len(df)).round(4))
    prof["n_points_total"] = g["n_points"].sum()
    prof["lat_centroid"] = g["lat_centroid"].mean().round(5)
    prof["lon_centroid"] = g["lon_centroid"].mean().round(5)
    prof["peak_hour"] = g["peak_hour"].agg(lambda s: int(s.mode().iloc[0]))
    prof = prof.reset_index()
    prof.insert(0, "algorithm", "kmeans")
    return prof


def cluster_poi_profile(cells: pd.DataFrame, labels: pd.Series,
                        poi_tab: pd.DataFrame) -> pd.DataFrame:
    """POI counts per cluster, as a total and as a per-cell density.

    Density (POIs per cell) is the comparable number: a cluster with three times as
    many cells would otherwise "have more schools" for purely geometric reasons.
    """
    df = cells[["h3_cell"]].copy()
    df["cluster"] = labels.reindex(df["h3_cell"]).to_numpy()
    df = df[df["cluster"].notna()].astype({"cluster": int})
    cats = [c for c in poi_tab.columns]
    joined = df.join(poi_tab, on="h3_cell").fillna({c: 0 for c in cats})

    g = joined.groupby("cluster")[cats].sum()
    out = pd.DataFrame({"cluster": g.index})
    out["poi_total"] = g.sum(axis=1).to_numpy()
    for c in cats:
        out[f"{c}_total"] = g[c].to_numpy()
        out[f"{c}_per_cell"] = (g[c] / joined.groupby("cluster").size()).round(3).to_numpy()
    out["poi_per_cell"] = (out["poi_total"] / joined.groupby("cluster").size()).round(3)
    return out


def cluster_hourly_profile(counts: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Mean pings per cell for each cluster over the 168 hours of the week.

    Averaging over the cluster's cells (not summing) keeps the curve comparable
    between a 23-cell cluster and a 470-cell one.
    """
    lab = labels.reindex(counts.index)
    keep = lab.notna()
    df = counts.loc[keep]
    grp = lab[keep].astype(int)
    prof = df.groupby(grp).mean().round(3)
    prof.index.name = "cluster"
    return prof


# ---------------------------------------------------------------------------
# naming - thresholds live here, documented
# ---------------------------------------------------------------------------
def name_clusters(prof: pd.DataFrame, poi_prof: pd.DataFrame | None = None,
                  poi_exclude: tuple[str, ...] = ("transport",),
                  poi_ratio: float = 2.0) -> pd.DataFrame:
    """Attach `name`, `label` and `description` built from measured differences.

    Two behaviour axes, each the mean of standardised features across clusters:
      * congestion_index = slow_share, speed_cv, dwell_s_per_visit, -speed_mean (higher = slower)
      * volume_index     = n_visits, n_vehicles                                   (higher = busier)
    Each quadrant of (volume, congestion) gets its own phrase - the classic
    throughput/speed trade-off in traffic engineering.

    Then two qualifiers:
      * rhythm_modifier - the time window with the largest share inside the cluster,
        kept only if it beats the median cluster's share for that same window;
      * poi_context - the POI category where this cluster is most extreme *relative to
        the other clusters* (density / median density >= 2).

    `transport` is excluded from the POI phrase on purpose: bus stops are everywhere a
    road is, so their density tracks the road network and would tag every cluster
    identically. It stays in the POI table, just not in the name.
    """
    out = prof.copy()
    z = zscore_rows(out, CONGESTION_FEATURES + FLUIDITY_FEATURES + VOLUME_FEATURES)
    out["congestion_index"] = (z[CONGESTION_FEATURES].mean(axis=1)
                              - z[FLUIDITY_FEATURES].mean(axis=1)).round(3)
    out["volume_index"] = z[VOLUME_FEATURES].mean(axis=1).round(3)

    # --- POI context: compare each category ACROSS CLUSTERS, not across categories ---
    poi_ctx: dict[int, str] = {int(c): "" for c in out["cluster"]}
    if poi_prof is not None and not poi_prof.empty:
        cats = [c for c in POI_CATEGORIES
                if f"{c}_per_cell" in poi_prof and c not in poi_exclude]
        med = {c: float(poi_prof[f"{c}_per_cell"].median()) for c in cats}
        for _, prow in poi_prof.iterrows():
            best, best_ratio = None, 0.0
            for c in cats:
                ratio = float(prow[f"{c}_per_cell"]) / max(med[c], 1e-9)
                if ratio > best_ratio:
                    best, best_ratio = c, ratio
            if best is not None and best_ratio >= poi_ratio:
                poi_ctx[int(prow["cluster"])] = f"{best}-dense ({best_ratio:.1f}x median)"

    names, labels, mods, descriptions = [], [], [], []
    for _, row in out.iterrows():
        busy = row["volume_index"] >= 0
        # Congestion is split into three bands, not two: sign alone gave two clusters with
        # the same name although one runs at 34 km/h and the other at 25 km/h. The cuts are
        # +/-1 index unit (the index is a mean of standardised features, so 1 = one sd of
        # the composite); they are stated here rather than tuned.
        ci = row["congestion_index"]
        band = "severe" if ci > 1.0 else "mild" if ci > -0.5 else "fluid"
        base = {
            (True, "severe"): "Congested high-traffic core",
            (True, "mild"): "Busy mid-speed corridors",
            (True, "fluid"): "Fluid high-traffic corridors",
            (False, "severe"): "Severe stop-and-go pockets",
            (False, "mild"): "Slow low-traffic pockets",
            (False, "fluid"): "Quiet free-flowing fringe",
        }[(busy, band)]

        shares = {k: row[col] for k, col in TEMPORAL_FEATURES.items()}
        top_key, top_val = max(shares.items(), key=lambda kv: kv[1])
        col = TEMPORAL_FEATURES[top_key]
        phrase = {"am": "morning-peaked", "pm": "evening-peaked",
                  "night": "late-night", "weekend": "weekend-heavy"}[top_key]
        modifier = phrase if top_val > out[col].median() else "all-day"

        ctx = poi_ctx.get(int(row["cluster"]), "")
        names.append(base)
        mods.append(modifier)
        extra = ", ".join(x for x in (modifier, ctx.split(" (")[0] if ctx else "") if x)
        labels.append(f"{base} ({extra})" if extra else base)
        descriptions.append(
            f"{int(row['n_cells'])} cells ({row['share_cells']:.0%} of the city); "
            f"peak hour {int(row['peak_hour']):02d}:00; mean speed {row['speed_mean']:.1f} km/h; "
            f"slow steps {row['slow_share']:.0%}; stop-and-go CV {row['speed_cv']:.2f}; "
            f"dwell per visit {row['dwell_s_per_visit']:.0f} s; "
            f"visits per cell {row['n_visits']:.0f}."
        )

    out["name"] = names
    out["rhythm_modifier"] = mods
    out["poi_context"] = [poi_ctx.get(int(c), "") for c in out["cluster"]]
    out["label"] = labels
    out["description"] = descriptions
    return out


def _count_runs(above: np.ndarray) -> int:
    """Number of contiguous True runs in a boolean vector."""
    edges = np.diff(np.r_[0, above.astype(int), 0])
    return int((edges == 1).sum())


def name_rhythms(rhythm_prof: pd.DataFrame) -> pd.DataFrame:
    """Name the TS-K-Means clusters by WHEN and HOW their weekly activity sits.

    Three measured properties, because peak hour alone gave every cluster the same name:
      * weekend_share  = activity in hours 120-167 / total  (>0.30 weekend-weighted,
                         <0.12 weekday-only, else all-week)
      * peak hour      = argmax of the 168-hour profile
      * peak shape     = single or twin (>=2 weekday runs above 80% of the weekday max)
      * activity level = cluster mean total vs the median cluster (>3x high, <0.5x low)
    """
    rows = []
    totals = rhythm_prof.sum(axis=1).astype(float)
    med_total = float(totals.median())

    for cid, row in rhythm_prof.iterrows():
        v = row.to_numpy(dtype=float)
        tot = float(v.sum()) or 1.0
        weekend_share = float(v[120:].sum() / tot)      # Sat+Sun
        h = int(np.argmax(v))
        hour = h % 24

        wk_mean = v[:120].reshape(5, 24).mean(axis=0)    # average weekday by hour
        window = wk_mean[5:23]                           # ignore pre-dawn floor
        above = window >= 0.8 * window.max() if window.max() > 0 else np.zeros_like(window, bool)
        n_runs = _count_runs(above)

        wk_phrase = ("weekend-weighted" if weekend_share >= 0.30
                     else "weekday-only" if weekend_share <= 0.12 else "all-week")
        peak_phrase = ("night" if hour < 6 else "morning" if hour < 11
                       else "midday" if hour < 16 else "evening" if hour < 21
                       else "late-evening")
        shape = "twin-peak" if n_runs >= 2 else "single-peak"

        t = float(totals.loc[cid])
        level = ("high-volume" if t > 3 * med_total
                 else "low-volume" if t < 0.5 * med_total else "mid-volume")

        rows.append({
            "rhythm_cluster": int(cid),
            "rhythm_name": f"{wk_phrase}, {peak_phrase} {shape} ({hour:02d}:00 peak, {level})",
            "peak_hour": hour,
            "n_peaks": n_runs,
            "weekend_share": round(weekend_share, 3),
            "activity_level": level,
            "mean_total_activity": round(t, 1),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# anomalies - three independent definitions, all stated
# ---------------------------------------------------------------------------
def spatial_anomalies(pca: pd.DataFrame, labels: pd.Series, top_frac: float = 0.02,
                      z_min: float = 2.0) -> pd.DataFrame:
    """Cells that sit far from their own cluster centre in PC1-PC5 space.

    d_i = || x_i - mu_{k(i)} ||_2, then robust z within the whole sample.
    Flag = among the top `top_frac` of d AND robust z >= z_min. Two conditions, because
    either alone misbehaves: a rank cut always flags someone even when nothing is odd,
    and a z cut can flag half the city when the distribution is heavy-tailed.

    For each flagged cell the driver features are reported (largest |z| vs cluster mean),
    so the flags can be argued with rather than believed.
    """
    X = pca[[f"PC{i}" for i in range(1, 6)]].to_numpy()
    lab = labels.reindex(pca["h3_cell"]).to_numpy()
    keep = ~pd.isna(lab)
    X, lab = X[keep], lab[keep].astype(int)

    cent = {k: X[lab == k].mean(axis=0) for k in np.unique(lab)}
    d = np.array([np.linalg.norm(x - cent[k]) for x, k in zip(X, lab)])
    z = robust_z(d)
    cutoff = np.quantile(d, 1 - top_frac)
    flag = (d >= cutoff) & (z >= z_min)

    return pd.DataFrame({"h3_cell": pca["h3_cell"].to_numpy()[keep], "cluster": lab,
                         "distance_to_centroid": d.round(4), "anomaly_z_spatial": z.round(3),
                         "is_anomaly_spatial": flag})


def temporal_anomalies(profile: pd.DataFrame, labels: pd.Series, top_frac: float = 0.02,
                       z_min: float = 2.0) -> pd.DataFrame:
    """Cells whose weekly rhythm disagrees with their cluster's rhythm.

    s_i = || p_i - mean_{k(i)}(p) ||_2 on row-normalised 168-hour profiles (the same
    representation TS-K-Means clustered). A cell can behave typically on average and
    still be the only one awake at 04:00; this catches exactly that.
    """
    lab = labels.reindex(profile.index)
    keep = lab.notna()
    P = profile.loc[keep].to_numpy(dtype=float)
    lab = lab[keep].astype(int)

    mean_prof = {k: P[(lab == k).to_numpy()].mean(axis=0) for k in np.unique(lab)}
    d = np.array([np.linalg.norm(p - mean_prof[k]) for p, k in zip(P, lab)])
    z = robust_z(d)
    cutoff = np.quantile(d, 1 - top_frac)
    flag = (d >= cutoff) & (z >= z_min)

    return pd.DataFrame({"h3_cell": profile.index[keep], "cluster": lab.to_numpy(),
                         "profile_distance": d.round(5), "anomaly_z_temporal": z.round(3),
                         "is_anomaly_temporal": flag})


def driver_features(cells: pd.DataFrame, labels: pd.Series, top_n: int = 3) -> pd.DataFrame:
    """For every cell, the features where it departs most from its own cluster mean."""
    feats = ["dwell_s_per_visit", "speed_mean", "speed_cv", "slow_share", "n_visits",
             "night_share", "am_peak_share", "pm_peak_share", "weekend_share"]
    df = cells.copy()
    df["cluster"] = labels.reindex(df["h3_cell"]).to_numpy()

    stats = df.groupby("cluster")[feats].agg(["mean", "std"])
    z = pd.DataFrame(index=df.index)
    for f in feats:
        mu = df["cluster"].map(stats[(f, "mean")])
        sd = df["cluster"].map(stats[(f, "std")]).replace(0, np.nan)
        z[f] = (df[f] - mu) / sd

    reasons = []
    for _, row in z.iterrows():
        s = row.dropna().abs().sort_values(ascending=False).head(top_n)
        reasons.append("; ".join(f"{f} {'+' if row[f] > 0 else '-'}{abs(row[f]):.1f}sd"
                                 for f in s.index))
    return pd.DataFrame({"h3_cell": df["h3_cell"], "drivers": reasons})


def day_anomalies(trips: pd.DataFrame, z_min: float = 2.5,
                  min_trips: int = 30) -> pd.DataFrame:
    """Days that deviate from the usual, with the sample-size caveat built in.

    Two signals per calendar day:
      * volume: Poisson z = (n - mu_weekday) / sqrt(mu_weekday), mu = median count for
        that weekday. Poisson is the right null for arrival counts and it widens the
        interval for small days automatically - which is the point.
        BUT taxi demand is over-dispersed: its variance is larger than its mean, so a
        raw Poisson z fires far too often (measured here: phi = 3.2, i.e. 50 flagged days
        out of 359 instead of 9). The dispersion factor is therefore estimated from the
        data and the z is scaled by sqrt(phi) - a quasi-Poisson correction:

            phi = mean( (n - mu)^2 / mu ) over assessable days, floored at 1
            z   = (n - mu) / sqrt(mu * phi)

        This is estimated, not assumed to be 1; the value is written to the output.
      * speed: robust z of the day's median speed against all days. It flags BOTH
        directions - a much slower day is congestion, a much faster day is unusually
        free-flowing (typically a holiday) - so the output carries `speed_direction`
        instead of pretending every deviation is congestion.

    `min_trips` suppresses days where the sample is too thin to say anything: this
    dataset is a ~1/28 systematic sample of all Porto taxi trips, so a day with 5 trips
    is noise, not a finding. Days below the floor are reported with flag=False and a
    reason, never silently dropped.
    """
    df = trips.copy()
    ts = pd.to_datetime(df["start_time"])
    df["date"] = ts.dt.normalize()
    df["dow"] = ts.dt.dayofweek

    daily = df.groupby("date").agg(n_trips=("trip_id", "nunique"),
                                   median_speed=("speed_mean", "median"),
                                   median_duration=("duration_s", "median"),
                                   dow=("dow", "first")).reset_index()
    med_by_dow = daily.groupby("dow")["n_trips"].median()
    mu = daily["dow"].map(med_by_dow).astype(float)
    daily["poisson_z"] = ((daily["n_trips"] - mu) / np.sqrt(mu)).round(3)

    enough = daily["n_trips"] >= min_trips
    # Over-dispersion from the data itself: mean squared standardised residual.
    phi = float(np.mean(daily.loc[enough, "poisson_z"].to_numpy() ** 2)) if enough.any() else 1.0
    phi = max(phi, 1.0)
    daily["dispersion_phi"] = round(phi, 3)
    daily["volume_z"] = (daily["poisson_z"] / np.sqrt(phi)).round(3)

    daily["speed_robust_z"] = robust_z(daily["median_speed"].to_numpy()).round(3)
    daily["is_anomaly_volume"] = enough & (daily["volume_z"].abs() >= z_min)
    daily["is_anomaly_speed"] = enough & (daily["speed_robust_z"].abs() >= z_min)
    daily["speed_direction"] = np.where(
        ~daily["is_anomaly_speed"], "",
        np.where(daily["speed_robust_z"] < 0, "slower", "faster"))
    daily["note"] = np.where(enough, "", f"< {min_trips} sampled trips: not assessable")
    daily["median_speed"] = daily["median_speed"].round(2)
    daily["median_duration"] = daily["median_duration"].round(0)
    return daily.drop(columns=["dow"])


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def plot_profiles(prof: pd.DataFrame, hourly: pd.DataFrame, out_path: str | Path) -> None:
    """Left: cluster means as z-scores. Right: mean weekly activity per cluster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    feats = ["dwell_s_per_visit", "speed_mean", "speed_cv", "slow_share", "n_visits",
             "night_share", "am_peak_share", "pm_peak_share", "weekend_share"]
    z = zscore_rows(prof, feats).T
    z.columns = [f"C{c}" for c in prof["cluster"]]

    # Layout: heatmap on the left, one small panel per cluster on the right. A single
    # shared-axis line chart hid three of four clusters behind the busiest one (their
    # means are ~5 pings/hour against ~175), so each panel gets its own y-scale and its
    # own peak value in the title - shape is what we want to compare, not absolute level.
    n = len(hourly)
    fig = plt.figure(figsize=(15, 6.8))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1, 1], wspace=0.30, hspace=0.55)
    ax1 = fig.add_subplot(gs[:, 0])
    panel_axes = [fig.add_subplot(gs[i // 2, 1 + i % 2]) for i in range(min(n, 4))]

    im = ax1.imshow(z.to_numpy(), cmap="Greys", aspect="auto")
    ax1.set_xticks(range(z.shape[1]), z.columns)
    ax1.set_yticks(range(len(feats)), feats, fontsize=8)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            v = z.iloc[i, j]
            ax1.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                     color="white" if v > 0.8 else "black")
    ax1.set_title("cluster profile (z vs other clusters)")
    fig.colorbar(im, ax=ax1, fraction=0.046, label="z")

    names = prof.set_index("cluster")["label"]
    for ax, (cid, row) in zip(panel_axes, hourly.iterrows()):
        v = row.to_numpy(dtype=float)
        ax.plot(v, color="#111111", lw=1.3)
        ax.fill_between(range(len(v)), v, color="#cccccc", alpha=0.5, lw=0)
        for d in range(0, 168, 24):
            ax.axvline(d, color="#eeeeee", lw=0.8, zorder=0)
        ax.set_xticks(range(0, 168, 24), ["M", "T", "W", "T", "F", "S", "S"])
        ax.set_title(f"C{cid}  {str(names.get(cid, ''))[:34]}\npeak {v.max():.0f} pings/cell/h",
                     fontsize=8)
        ax.tick_params(labelsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    if panel_axes:
        panel_axes[0].set_ylabel("pings / cell / hour", fontsize=8)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_poi(poi_prof: pd.DataFrame, prof: pd.DataFrame, out_path: str | Path) -> None:
    """POIs per cell by category, cluster by cluster."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # `transport` is plotted nowhere: bus stops follow the road network, so they swamp
    # every other category and say nothing about what happens in a cell. It stays in
    # the POI table (outputs/cluster_pois.csv) for anyone who wants it.
    cats = [c for c in POI_CATEGORIES if f"{c}_per_cell" in poi_prof and c != "transport"]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    width = 0.8 / max(len(poi_prof), 1)
    greys = ["#111111", "#555555", "#999999", "#cccccc", "#7a7a7a", "#333333"]
    names = prof.set_index("cluster")["label"]
    for i, (_, row) in enumerate(poi_prof.iterrows()):
        vals = [row[f"{c}_per_cell"] for c in cats]
        xs = np.arange(len(cats)) + i * width
        ax.bar(xs, vals, width=width, color=greys[i % len(greys)],
               label=f"C{int(row['cluster'])}: {names.get(row['cluster'], '')}")
    ax.set_xticks(np.arange(len(cats)) + width * (len(poi_prof) - 1) / 2, cats, fontsize=8)
    ax.set_ylabel("POIs per H3 cell")
    ax.set_title("OSM POI density by cluster (OpenStreetMap via osmnx; bus stops excluded)\n"
                 "POIs per cell, so cluster area does not decide the winner", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, ncol=1, loc="upper center",
              bbox_to_anchor=(0.5, -0.14), title="cluster", title_fontsize=7.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_anomalies(cells: pd.DataFrame, ann: pd.DataFrame, out_path: str | Path) -> None:
    """Map of flagged cells over a faint grey background of all cells, plus score histograms."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [1.1, 1]})
    ax1.scatter(cells["lon_centroid"], cells["lat_centroid"], s=6, c="#d9d9d9", label="cells")
    s = ann.merge(cells[["h3_cell", "lat_centroid", "lon_centroid"]], on="h3_cell", how="left")
    spa = s[s["is_anomaly_spatial"]]
    tmp = s[s["is_anomaly_temporal"]]
    both = s[s["is_anomaly_spatial"] & s["is_anomaly_temporal"]]
    ax1.scatter(tmp["lon_centroid"], tmp["lat_centroid"], s=26, facecolors="none",
                edgecolors="#777777", label=f"rhythm anomaly ({len(tmp)})")
    ax1.scatter(spa["lon_centroid"], spa["lat_centroid"], s=26, marker="^", c="#111111",
                label=f"behaviour anomaly ({len(spa)})")
    ax1.scatter(both["lon_centroid"], both["lat_centroid"], s=70, facecolors="none",
                edgecolors="#111111", linewidths=1.2, label=f"both ({len(both)})")
    ax1.set_aspect(1 / np.cos(np.radians(cells["lat_centroid"].mean())))
    ax1.set_xlabel("longitude")
    ax1.set_ylabel("latitude")
    ax1.set_title("flagged cells (grey = all analysed cells)")
    ax1.legend(fontsize=7, frameon=False)

    ax2.hist(ann["anomaly_z_spatial"], bins=40, color="#bbbbbb", label="behaviour z")
    ax2.hist(ann["anomaly_z_temporal"], bins=40, color="#333333", alpha=0.55,
             label="rhythm z")
    ax2.axvline(2.0, color="#111111", ls="--", lw=1, label="z = 2 gate")
    ax2.set_xlabel("robust z of distance to own cluster")
    ax2.set_ylabel("cells")
    ax2.set_title("anomaly score distributions")
    ax2.legend(fontsize=7, frameon=False)
    for s_ in ("top", "right"):
        ax2.spines[s_].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# documentation written by the pipeline itself
# ---------------------------------------------------------------------------
def write_limitations(path: str | Path, meta: dict) -> None:
    """Write the limitations + privacy note, with the real dataset name filled in."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = f"""# Limitations and privacy

_Generated by `src/interpret.py` (Stage 7). Dataset: `{meta['data_source']}`,
synthetic: **{meta['synthetic']}**._

## 1. What the data is not
{meta['limitation_text']}

* **No ground truth.** Nothing in this project knows what a "footfall hotspot" is. The
  clusters are a description of the data, not a validated classification. Every name
  ("morning-peaked congested core") is a hypothesis written next to its numbers.
* **Sampling, not census.** The trip table is a systematic sample of the source file, so
  absolute counts are not city totals. Relative comparisons between cells and clusters
  are what the analysis supports.
* **Seasonality is collapsed.** One year of trips is folded into a single 168-hour week,
  so summer/winter differences are averaged into the same hour-of-week bucket.
* **Only moving vehicles are seen.** Taxis appear where they drive. A quiet residential
  street may be quiet because nobody goes there *or* because no taxi did; the data cannot
  separate the two. This is the classic exposure-bias problem in trajectory mining.
* **No people, no boarding.** Vehicle positions are not passenger counts. Dwell means
  below walking pace, which includes waiting at a light, parking, and picking up a fare.

## 2. GPS and processing error
* Consumer GPS is accurate to roughly 5-15 m; urban canyons and tunnels are worse.
* The 100 km/h speed filter removes physically impossible steps, but it cannot repair a
  plausible-looking wrong fix. Cleaning reduces error, it does not eliminate it.
* H3 resolution 8 puts every point in a ~0.74 km2 hexagon. Anything that happens within
  one hexagon is, by construction, invisible.

## 3. Statistical limitations
* Internal indices (silhouette, Davies-Bouldin, Calinski-Harabasz, DBCV) measure
  *geometry*, not usefulness. A high silhouette means compact and separated, not correct.
* k was chosen with an inertia knee and cross-checked with BIC, because silhouette is
  flat across k=2-5 here; another analyst could defend a different k.
* Anomalies are defined by an explicit rule (top 2 % of a distance score AND robust
  z >= 2). Change the rule and the list changes. They are *candidates for inspection*.
* Clusters are computed on aggregate cells, so a cluster's "mean speed" is an average of
  averages and is not a travel-time estimate for any real trip.

## 4. Privacy
* **Inputs are vehicle trajectories**, already pseudonymised (`vehicle_id` hashed /
  hashed by the source). No names, plates, phone numbers or addresses are used.
* **Outputs are aggregates only**: H3 cells of ~0.74 km2, cluster profiles, and anomaly
  flags on cells. No individual vehicle track is published in `outputs/`.
* **No re-identification is attempted** and no auxiliary dataset is joined to try.
* **Minimum support.** Cell-level statistics are only clustered when a cell has at least
  the Stage-3 visit threshold, so a single trip cannot define a location's profile.
* **Tier 2 (footfall, if built) is count-only**: a pretrained person detector outputs
  counts per frame; no faces, no gait, no tracking of individuals across frames, and only
  aggregated counts leave the pipeline. Public street/vehicle video only.

## 5. What would make these conclusions stronger
1. Real **bus** GPS with GTFS schedules (the original Tier 1 target), giving stop-level
   departures and schedule delay.
2. An independent **footfall** source (e.g. a city pedestrian counter) to test whether
   high-dwell clusters really are high-footfall places.
3. A second year of data to test whether clusters are stable across years, not just
   across bootstrap resamples.
"""
    Path(path).write_text(text)


def interpret_meta(meta_in: dict) -> dict:
    """Small provenance block that downstream consumers (Streamlit) can trust."""
    return {
        "stage": 7,
        "data_source": meta_in.get("data_source"),
        "synthetic": meta_in.get("synthetic"),
        "algorithm": "kmeans (k=4) on PC1-PC5; tskmeans on 168-h profiles as orthogonal axis",
        "naming_rule": "quadrant of (volume_index, congestion_index) + temporal modifier + POI phrase",
        "anomaly_rules": {
            "spatial": "top 2% of distance to own centroid in PC1-PC5 AND robust z >= 2",
            "temporal": "top 2% of L2 distance to own cluster mean 168-h profile AND robust z >= 2",
            "daily": "|Poisson z| >= 2.5 (or robust z of median speed) with >= 30 sampled trips",
        },
        "poi_source": "OpenStreetMap via osmnx (cached in data/raw/osm/)",
        "limitation": "taxis, not buses: no fixed stops, no timetable, no schedule delay",
        "random_state": RANDOM_STATE,
    }
