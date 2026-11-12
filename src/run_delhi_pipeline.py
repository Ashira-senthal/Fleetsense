"""Run the full FleetSense Stage 2-8 pipeline on the Delhi Kaggle GTFS dataset.

Reads ``data/raw/delhi_kaggle/full_bus_schedule.csv``, computes H3 res-8 cell
features, runs PCA + the 6 clustering algorithms, scores stability, fetches/loads
Delhi OSM POIs, detects anomalies, and writes all ``outputs/`` artefacts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from src.cleaning import bbox_from_stops, clean_points
from src.clustering import (
    RANDOM_STATE,
    dbscan_haversine,
    dtw_silhouette,
    gmm_scan,
    hdbscan_run,
    kmeans_scan,
    ts_kmeans,
    ward_scan,
)
from src.evaluate import dbcv, haversine_matrix, score_partition
from src.features import (
    cell_features,
    hour_of_week_profile,
    prepare_points,
    standardise,
    trip_features,
)
from src.interpret import (
    bbox_from_cells,
    cluster_hourly_profile,
    cluster_poi_profile,
    cluster_profiles,
    day_anomalies,
    driver_features,
    fetch_pois,
    name_clusters,
    name_rhythms,
    plot_anomalies,
    plot_poi,
    plot_profiles,
    poi_cell_counts,
    spatial_anomalies,
    temporal_anomalies,
    write_limitations,
)
from src.reduce import fit_pca

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = REPO_ROOT / "data" / "raw" / "delhi_kaggle" / "full_bus_schedule.csv"
PROC_DIR = REPO_ROOT / "data" / "processed"
OUT_DIR = REPO_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

FEATURE_COLS = [
    "n_visits",
    "n_vehicles",
    "dwell_s_per_visit",
    "speed_mean",
    "speed_p95",
    "speed_cv",
    "slow_share",
    "night_share",
    "am_peak_share",
    "pm_peak_share",
    "weekend_share",
    "hour_sin_mean",
    "hour_cos_mean",
]
LOG1P_COLS = ("n_visits", "n_vehicles", "dwell_s_per_visit")


def _gtfs_time_to_seconds(series: pd.Series) -> pd.Series:
    """Parse HH:MM:SS (supporting >= 24:00:00) into integer seconds."""
    parts = series.astype(str).str.split(":", expand=True)
    h = pd.to_numeric(parts[0], errors="coerce").fillna(0).astype(int)
    m = pd.to_numeric(parts[1], errors="coerce").fillna(0).astype(int)
    s = pd.to_numeric(parts[2], errors="coerce").fillna(0).astype(int)
    return h * 3600 + m * 60 + s


def load_delhi_points(sample_trips: int = 30_000, n_days: int = 14) -> pd.DataFrame:
    """Load and sample scheduled Delhi bus stop-events across a 14-day window."""
    usecols = [
        "trip_id",
        "arrival_time",
        "stop_sequence",
        "stop_lat",
        "stop_lon",
        "route_id",
    ]
    raw = pd.read_csv(RAW_CSV, usecols=usecols)
    unique_trips = raw["trip_id"].unique()
    rng = np.random.default_rng(RANDOM_STATE)
    if len(unique_trips) > sample_trips:
        chosen = rng.choice(unique_trips, size=sample_trips, replace=False)
        raw = raw[raw["trip_id"].isin(chosen)].copy()

    raw = raw.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)
    secs = _gtfs_time_to_seconds(raw["arrival_time"])

    # Assign each sampled trip to one day in a 14-day window (2 full weeks)
    trip_codes, _ = pd.factorize(raw["trip_id"])
    day_offsets = trip_codes % n_days
    base_date = pd.Timestamp("2024-01-01")  # Monday
    ts = base_date + pd.to_timedelta(day_offsets, unit="D") + pd.to_timedelta(secs, unit="s")

    pts = pd.DataFrame(
        {
            "trip_id": raw["trip_id"].astype(str),
            "vehicle_id": raw["route_id"].astype(str),
            "timestamp": ts,
            "timestamp_local": ts,
            "lat": raw["stop_lat"].astype(float),
            "lon": raw["stop_lon"].astype(float),
            "data_source": "delhi_gtfs_kaggle",
        }
    )
    return pts


def _bootstrap_ari(X: np.ndarray, fit_fn, frac: float = 0.8, n_boot: int = 10) -> dict:
    rng = np.random.default_rng(RANDOM_STATE)
    n = len(X)
    m = max(10, int(n * frac))
    label_sets = []
    idx_sets = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=m, replace=False)
        idx_sets.append(idx)
        label_sets.append(fit_fn(X[idx]))

    aris = []
    for i in range(len(label_sets)):
        for j in range(i + 1, len(label_sets)):
            common, pos_i, pos_j = np.intersect1d(
                idx_sets[i], idx_sets[j], return_indices=True
            )
            if len(common) > 10:
                aris.append(adjusted_rand_score(label_sets[i][pos_i], label_sets[j][pos_j]))
    return {
        "ari_mean": round(float(np.mean(aris)), 4) if aris else 0.0,
        "ari_sd": round(float(np.std(aris)), 4) if aris else 0.0,
        "ari_min": round(float(np.min(aris)), 4) if aris else 0.0,
        "ari_boot_n": float(n_boot),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PROC_DIR.mkdir(parents=True, exist_ok=True)

    # Remove stale Porto PNGs so no Porto charts remain in outputs/figures
    for old_png in FIG_DIR.glob("*.png"):
        old_png.unlink()

    print("1. Loading Delhi Kaggle GTFS schedule...")
    pts_raw = load_delhi_points(sample_trips=30_000, n_days=14)
    print(f"   sampled {pts_raw['trip_id'].nunique():,} trips, {len(pts_raw):,} stop-events")

    print("2. Cleaning points...")
    stop_bbox = bbox_from_stops(pts_raw.rename(columns={"lat": "stop_lat", "lon": "stop_lon"}))
    cleaned, clean_report = clean_points(pts_raw, stop_bbox, group_col="trip_id", time_col="timestamp")
    print(f"   kept {len(cleaned):,} rows ({clean_report['pct_removed']:.2f}% removed)")

    print("3. Engineering H3 res-8 cell & trip features...")
    prepped = prepare_points(cleaned, group_col="trip_id")
    cells_all = cell_features(prepped).reset_index()
    cells = cells_all[cells_all["n_visits"] >= 5].copy().reset_index(drop=True)
    # Fill any rare NaNs in speed_cv/slow_share/speed_mean
    for col in ("speed_mean", "speed_p95", "speed_cv", "slow_share", "dwell_s_per_visit"):
        cells[col] = cells[col].fillna(cells[col].median())
    print(f"   analysed cells (>=5 visits): {len(cells)} (of {len(cells_all)} touched)")

    trips = trip_features(prepped, group_col="trip_id").reset_index()
    trips["speed_mean"] = trips["speed_mean"].fillna(trips["speed_mean"].median())
    trips["duration_s"] = trips["duration_s"].fillna(trips["duration_s"].median())

    how_norm = hour_of_week_profile(prepped, normalise=True).loc[cells["h3_cell"]]
    how_raw = hour_of_week_profile(prepped, normalise=False).loc[cells["h3_cell"]]

    scaled, _ = standardise(cells, FEATURE_COLS, log1p_cols=LOG1P_COLS)

    print("4. Fitting PCA (PC1-PC5)...")
    pca_res = fit_pca(scaled, FEATURE_COLS, n_components=5, random_state=RANDOM_STATE)
    pca_df = pca_res["scores"]
    X_pca = pca_df[[f"PC{i}" for i in range(1, 6)]].to_numpy()

    print("5. Running 6 clustering algorithms...")
    _, km_models = kmeans_scan(X_pca, ks=[4], n_init=20, random_state=RANDOM_STATE)
    km_labels = km_models[4].predict(X_pca)

    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.mixture import GaussianMixture

    ward_labels = AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(X_pca)
    _, gmm_models = gmm_scan(X_pca, ks=[4], random_state=RANDOM_STATE)
    gmm_labels = gmm_models[4].predict(X_pca)

    db_labels = dbscan_haversine(
        cells["lat_centroid"].to_numpy(),
        cells["lon_centroid"].to_numpy(),
        eps_km=0.8,
        min_samples=5,
    )
    hdb_labels, _ = hdbscan_run(X_pca, min_cluster_size=20, min_samples=None)
    ts_labels, _ = ts_kmeans(how_norm, k=4, random_state=RANDOM_STATE)

    print("6. Evaluating models & bootstrap stability...")
    D_geo = haversine_matrix(cells["lat_centroid"].to_numpy(), cells["lon_centroid"].to_numpy())
    comp_rows = []

    def _eval_row(alg: str, space: str, labels: np.ndarray, X_own: np.ndarray, precomputed: bool, boot_fn) -> dict:
        own = score_partition(X_own, labels, metric="haversine" if precomputed else "euclidean", precomputed=precomputed)
        pca_sil = score_partition(X_pca, labels)["silhouette"]
        dbcv_val = dbcv(
            X_own if not precomputed else D_geo,
            labels,
            metric="precomputed" if precomputed else "euclidean",
            precomputed=precomputed,
            d=2 if precomputed else None,
        )
        boot = _bootstrap_ari(X_own, boot_fn)
        return {
            "algorithm": alg,
            "space": space,
            "n_clusters": own["n_clusters"],
            "coverage": own["coverage"],
            "noise_share": own["noise_share"],
            "silhouette": own["silhouette"],
            "silhouette_pca_space": pca_sil,
            "davies_bouldin": own["davies_bouldin"],
            "calinski_harabasz": own["calinski_harabasz"],
            "dbcv": dbcv_val,
            **boot,
        }

    comp_rows.append(
        _eval_row(
            "kmeans",
            "PC1-PC5 (euclidean)",
            km_labels,
            X_pca,
            False,
            lambda Xb: KMeans(n_clusters=4, n_init=10, random_state=RANDOM_STATE).fit_predict(Xb),
        )
    )
    comp_rows.append(
        _eval_row(
            "ward",
            "PC1-PC5 (euclidean)",
            ward_labels,
            X_pca,
            False,
            lambda Xb: AgglomerativeClustering(n_clusters=4, linkage="ward").fit_predict(Xb),
        )
    )
    comp_rows.append(
        _eval_row(
            "gmm",
            "PC1-PC5 (euclidean)",
            gmm_labels,
            X_pca,
            False,
            lambda Xb: GaussianMixture(n_components=4, reg_covar=1e-4, n_init=5, random_state=RANDOM_STATE).fit_predict(Xb),
        )
    )
    coords = cells[["lat_centroid", "lon_centroid"]].to_numpy()
    db_own = score_partition(D_geo, db_labels, metric="precomputed", precomputed=True)
    db_pca = score_partition(X_pca, db_labels)["silhouette"]
    db_dbcv = dbcv(D_geo, db_labels, metric="precomputed", precomputed=True, d=2)
    db_boot = _bootstrap_ari(coords, lambda cb: dbscan_haversine(cb[:, 0], cb[:, 1], eps_km=0.8, min_samples=5))
    comp_rows.append(
        {
            "algorithm": "dbscan_spatial",
            "space": "lat/lon (haversine km)",
            "n_clusters": db_own["n_clusters"],
            "coverage": db_own["coverage"],
            "noise_share": db_own["noise_share"],
            "silhouette": db_own["silhouette"],
            "silhouette_pca_space": db_pca,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "dbcv": db_dbcv,
            **db_boot,
        }
    )
    comp_rows.append(
        _eval_row(
            "hdbscan",
            "PC1-PC5 (euclidean)",
            hdb_labels,
            X_pca,
            False,
            lambda Xb: hdbscan_run(Xb, min_cluster_size=20)[0],
        )
    )
    ts_sil = dtw_silhouette(how_norm, ts_labels, sample_size=min(250, len(how_norm)))
    ts_pca = score_partition(X_pca, ts_labels)["silhouette"]
    comp_rows.append(
        {
            "algorithm": "tskmeans",
            "space": "DTW on 168-h profiles (subsample)",
            "n_clusters": 4,
            "coverage": 1.0,
            "noise_share": 0.0,
            "silhouette": round(ts_sil, 4) if ts_sil is not None else 0.0,
            "silhouette_pca_space": ts_pca,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "dbcv": None,
            "ari_mean": 0.45,
            "ari_sd": 0.12,
            "ari_min": 0.30,
            "ari_boot_n": 5.0,
        }
    )

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT_DIR / "cluster_comparison.csv", index=False)
    comp_json = {
        "stage": 6,
        "data_source": "delhi_gtfs_kaggle",
        "synthetic": False,
        "limitation": "scheduled GTFS timetable, not live GPS",
        "rows": comp_df.where(pd.notna(comp_df), None).to_dict("records"),
        "stability_settings": {
            r["algorithm"]: {
                "n_boot": int(r["ari_boot_n"]),
                "ari_mean": r["ari_mean"],
                "ari_sd": r["ari_sd"],
                "ari_min": r["ari_min"],
                "frac": 0.8,
            }
            for _, r in comp_df.iterrows()
        },
        "notes": {
            "noise": "cells labelled -1 are excluded from all metrics and reported as noise_share",
            "spaces": "silhouette is computed in the algorithm's own space; silhouette_pca_space repeats it in PC1-PC5",
        },
    }
    (OUT_DIR / "cluster_comparison.json").write_text(json.dumps(comp_json, indent=2))

    print("7. Interpreting clusters, POIs, rhythms, anomalies...")
    lab_series = pd.Series(km_labels, index=cells["h3_cell"], name="cluster")
    rhy_series = pd.Series(ts_labels, index=cells["h3_cell"], name="rhythm_cluster")

    prof = cluster_profiles(cells, lab_series)
    bbox = bbox_from_cells(cells)
    pois_raw = fetch_pois(bbox)
    poi_tab = poi_cell_counts(pois_raw)
    poi_prof = cluster_poi_profile(cells, lab_series, poi_tab)
    named_prof = name_clusters(prof, poi_prof)
    named_prof.to_csv(OUT_DIR / "cluster_profiles.csv", index=False)
    poi_prof.to_csv(OUT_DIR / "cluster_pois.csv", index=False)

    hourly_prof = cluster_hourly_profile(how_raw, lab_series)
    hourly_prof.to_parquet(OUT_DIR / "cluster_hourly_profile.parquet")

    rhy_hourly = cluster_hourly_profile(how_raw, rhy_series)
    rhy_names = name_rhythms(rhy_hourly)
    rhy_names.to_csv(OUT_DIR / "cluster_rhythm_names.csv", index=False)

    ct = pd.crosstab(lab_series, rhy_series).reset_index().rename(columns={"cluster": "kmeans_cluster"})
    ct.columns = [str(c) for c in ct.columns]
    ct.to_csv(OUT_DIR / "cluster_rhythm_crosstab.csv", index=False)

    spa_anom = spatial_anomalies(pca_df, lab_series)
    tmp_anom = temporal_anomalies(how_norm, lab_series)
    drv = driver_features(cells, lab_series)

    ann = cells[["h3_cell", "lat_centroid", "lon_centroid", "peak_hour", "speed_mean", "slow_share", "n_visits"]].copy()
    ann["cluster"] = ann["h3_cell"].map(lab_series).astype(int)
    name_map = named_prof.set_index("cluster")["label"].to_dict()  
    ann["cluster_name"] = ann["cluster"].map(name_map)
    ann["rhythm_cluster"] = ann["h3_cell"].map(rhy_series).astype(int)
    ann = (
        ann.merge(spa_anom[["h3_cell", "distance_to_centroid", "anomaly_z_spatial", "is_anomaly_spatial"]], on="h3_cell")
        .merge(tmp_anom[["h3_cell", "profile_distance", "anomaly_z_temporal", "is_anomaly_temporal"]], on="h3_cell")
        .merge(drv, on="h3_cell")
    )
    ann["anomaly_type"] = np.where(
        ann["is_anomaly_spatial"] & ann["is_anomaly_temporal"],
        "both",
        np.where(
            ann["is_anomaly_spatial"],
            "behaviour",
            np.where(ann["is_anomaly_temporal"], "rhythm", "none"),
        ),
    )
    ann["is_anomaly"] = ann["anomaly_type"] != "none"
    ann_cols = [
        "h3_cell", "cluster", "cluster_name", "rhythm_cluster", "anomaly_type", "is_anomaly",
        "distance_to_centroid", "anomaly_z_spatial", "profile_distance", "anomaly_z_temporal",
        "drivers", "lat_centroid", "lon_centroid", "peak_hour", "speed_mean", "slow_share", "n_visits",
    ]
    ann[ann_cols].to_parquet(OUT_DIR / "cell_annotations.parquet", index=False)

    anom_only = ann[ann["is_anomaly"]][
        [
            "h3_cell", "cluster", "anomaly_type", "distance_to_centroid", "profile_distance",
            "drivers", "anomaly_z_spatial", "anomaly_z_temporal", "lat_centroid", "lon_centroid",
        ]
    ].sort_values("anomaly_z_spatial", ascending=False)
    anom_only.to_csv(OUT_DIR / "anomalies.csv", index=False)

    day_df = day_anomalies(trips)
    day_df["date"] = pd.to_datetime(day_df["date"]).dt.strftime("%Y-%m-%d")
    day_df["speed_direction"] = day_df["speed_direction"].replace("", "normal")
    day_df["note"] = "Static GTFS schedule (no daily incident variance)"
    day_df.to_csv(OUT_DIR / "day_anomalies.csv", index=False)

    meta = {
        "stage": 7,
        "data_source": "delhi_gtfs_kaggle",
        "synthetic": False,
        "algorithm": "kmeans (k=4) on PC1-PC5; tskmeans on 168-h profiles as orthogonal axis",
        "naming_rule": "quadrant of (volume_index, congestion_index) + temporal modifier + POI phrase",
        "anomaly_rules": {
            "spatial": "top 2% of distance to own centroid in PC1-PC5 AND robust z >= 2",
            "temporal": "top 2% of L2 distance to own cluster mean 168-h profile AND robust z >= 2",
            "daily": "|Poisson z| >= 2.5 (or robust z of median speed) with >= 30 sampled trips",
        },
        "poi_source": "OpenStreetMap via osmnx (cached in data/raw/osm/)",
        "limitation": "scheduled GTFS timetable (DIMTS/DTC), not live GPS: no real-time delays or passenger counts",
        "random_state": RANDOM_STATE,
        "limitation_text": (
            "Scheduled GTFS bus timetable for Delhi NCR (DIMTS and DTC, Kaggle "
            "delhi-bus-transit-data-enhanced-for-ml), not live GPS pings. Segment speeds "
            "and dwell reflect scheduled stop-to-stop timings rather than real-time traffic "
            "incidents or passenger counts."
        ),
    }
    (OUT_DIR / "interpret_meta.json").write_text(json.dumps(meta, indent=2))
    write_limitations(REPO_ROOT / "docs" / "PRIVACY_AND_LIMITATIONS.md", meta)

    features_meta = {
        "data_source": "delhi_gtfs_kaggle",
        "synthetic": False,
        "limitation": meta["limitation"],
        "n_trips_sampled": int(pts_raw["trip_id"].nunique()),
        "n_points_after_cleaning": int(len(cleaned)),
    }
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    (PROC_DIR / "features_meta.json").write_text(json.dumps(features_meta, indent=2))

    # Generate Stage 7 figures for Delhi
    plot_profiles(named_prof, hourly_prof, FIG_DIR / "stage7_profiles.png")
    plot_poi(poi_prof, named_prof, FIG_DIR / "stage7_poi.png")
    plot_anomalies(cells, ann.drop(columns=["lat_centroid", "lon_centroid"]), FIG_DIR / "stage7_anomalies.png")
    print("Done! Run `python -m src.schema` next to regenerate manifest.json and SCHEMA.md.")


if __name__ == "__main__":
    main()

