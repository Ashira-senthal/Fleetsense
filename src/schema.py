"""Stage 8 - the outputs contract for FleetSense.

This module is the single source of truth for what lives in `outputs/`. It does three
things and nothing else:

1. **Declares** every file, column, dtype, unit and meaning (`FILES` below).
2. **Validates** the real directory against that declaration, including cross-file
   consistency (a cluster id in one file must exist in another; share of cells must sum
   to 1; the anomaly CSV must contain exactly the flagged rows).
3. **Emits** two artefacts from the same declaration:
     * `outputs/manifest.json` - machine-readable, for the Streamlit app to check.
     * `outputs/SCHEMA.md`     - human-readable, for a reader or examiner.

Why bother: a dashboard that silently reads a renamed column renders an empty chart, and
a reader cannot tell a column that changed meaning from one that changed name. The
schema turns both into a loud failure at build time instead of a quiet lie at view time.

Usage:
    python -m src.schema            # validate, then write manifest.json + SCHEMA.md
    python -m src.schema --check    # validate only (non-zero exit on failure)
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs"
PROC = REPO_ROOT / "data" / "processed"

# ---------------------------------------------------------------------------
# Column vocabulary: (kind, unit, description)
# kind is one of str / int / float / bool and is checked against the real dtype family.
# ---------------------------------------------------------------------------
FILES: dict[str, dict] = {
    "cluster_profiles.csv": {
        "format": "csv",
        "key": "cluster",
        "description": "One row per cluster: size, location, mean of every behaviour "
                       "feature, generated name and a one-sentence description.",
        "columns": {
            "algorithm": ("str", "-", "Clustering that produced these labels ('kmeans')."),
            "cluster": ("int", "id", "Cluster id, 0-based."),
            "n_cells": ("int", "H3 cells", "Cells in the cluster."),
            "share_cells": ("float", "fraction 0-1", "Share of all clustered cells."),
            "n_visits": ("float", "distinct trips per cell", "Mean over the cluster's cells."),
            "n_vehicles": ("float", "vehicles per cell", "Mean over the cluster's cells."),
            "dwell_s_per_visit": ("float", "seconds", "Mean below-5 km/h time per visit."),
            "speed_mean": ("float", "km/h", "Mean of valid step speeds."),
            "speed_p95": ("float", "km/h", "95th percentile of valid step speeds."),
            "speed_cv": ("float", "ratio sd/mean", "Stop-and-go signature; higher = more "
                                                   "accelerate/brake cycling."),
            "slow_share": ("float", "fraction 0-1", "Share of valid steps below 12 km/h."),
            "night_share": ("float", "fraction 0-1", "Share of pings in 00:00-05:59."),
            "am_peak_share": ("float", "fraction 0-1", "Share of pings in 07:00-09:59."),
            "pm_peak_share": ("float", "fraction 0-1", "Share of pings in 16:00-18:59."),
            "weekend_share": ("float", "fraction 0-1", "Share of pings on Sat/Sun."),
            "n_points_total": ("int", "GPS pings", "All pings inside the cluster."),
            "lat_centroid": ("float", "degrees WGS84", "Mean latitude of the cluster's cells."),
            "lon_centroid": ("float", "degrees WGS84", "Mean longitude of the cluster's cells."),
            "peak_hour": ("int", "hour 0-23", "Modal peak hour of the cluster's cells."),
            "congestion_index": ("float", "z units", "Mean z of (slow_share, speed_cv, "
                                                     "dwell) minus mean z of (speed_mean, speed_p95); "
                                                     "higher = slower."),
            "volume_index": ("float", "z units", "Mean z of (n_visits, n_vehicles); higher = "
                                                 "busier."),
            "name": ("str", "-", "Short generated name (behaviour quadrant)."),
            "rhythm_modifier": ("str", "-", "Time window that dominates, or 'all-day'."),
            "poi_context": ("str", "-", "POI category where this cluster is most extreme, "
                                        "with its ratio to the median cluster."),
            "label": ("str", "-", "Name plus qualifiers, for display."),
            "description": ("str", "-", "One sentence quoting the measured numbers."),
        },
    },
    "cluster_pois.csv": {
        "format": "csv",
        "key": "cluster",
        "description": "OpenStreetMap POI counts per cluster, absolute and per cell. "
                       "Per-cell density is the comparable number (cluster area differs).",
        "columns": {          # prefix pattern documented; validated by pattern below
            "cluster": ("int", "id", "Cluster id."),
            "poi_total": ("float", "POIs", "All POIs in the cluster's cells."),
            "poi_per_cell": ("float", "POIs per cell", "poi_total / n_cells."),
            "<category>_total": ("float", "POIs", "POIs of that category in the cluster."),
            "<category>_per_cell": ("float", "POIs per cell", "Total / n_cells."),
        },
        "patterns": {
            r"^[a-z_]+_total$": ("float", "POIs", "Category total."),
            r"^[a-z_]+_per_cell$": ("float", "POIs per cell", "Category density."),
        },
        "categories": ["education", "market", "transport", "health", "hospitality",
                       "civic", "worship", "retail_area", "industry"],
    },
    "cluster_hourly_profile.parquet": {
        "format": "parquet",
        "key": "cluster (index)",
        "description": "Cluster x hour-of-week activity matrix: what the dashboard's weekly "
                       "chart needs, one row per cluster, no aggregation left to do in the app.",
        "wide": {"n": 168, "dtype": "float",
                 "unit": "mean pings per cell per hour",
                 "description": "Column j = hour-of-week j, where j = weekday*24 + hour and "
                                "j = 0 is Monday 00:00. Averaged over the cluster's cells, so "
                                "clusters of different size are comparable."},
        "columns": {},
    },
    "cluster_rhythm_names.csv": {
        "format": "csv",
        "key": "rhythm_cluster",
        "description": "Name and measured properties of each TS-K-Means weekly-rhythm cluster.",
        "columns": {
            "rhythm_cluster": ("int", "id", "Rhythm cluster id (from time-series K-Means)."),
            "rhythm_name": ("str", "-", "Generated name: weekend share, peak hour, shape, level."),
            "peak_hour": ("int", "hour 0-23", "Argmax hour of the 168-hour profile."),
            "n_peaks": ("int", "count", "Weekday runs above 80% of the weekday maximum; "
                                        ">= 2 means a twin peak."),
            "weekend_share": ("float", "fraction 0-1", "Activity on Sat/Sun / total."),
            "activity_level": ("str", "-", "high-volume (>3x median), mid, or low (<0.5x)."),
            "mean_total_activity": ("float", "pings per cell summed over the week", "Volume."),
        },
    },
    "cluster_rhythm_crosstab.csv": {
        "format": "csv",
        "key": "kmeans_cluster",
        "description": "Cell counts crossing the behaviour clusters with the rhythm clusters. "
                       "A row spread across several rhythm columns is a behaviour that happens "
                       "at different times of week in different places.",
        "columns": {"kmeans_cluster": ("int", "id", "Behaviour cluster id.")},
        "patterns": {r"^[0-9]+$": ("int", "cells", "Count of cells in that rhythm cluster.")},
    },
    "cell_annotations.parquet": {
        "format": "parquet",
        "key": "h3_cell",
        "description": "The join table every map needs: one row per analysed H3 cell with its "
                       "behaviour cluster, rhythm cluster, anomaly scores and flags.",
        "columns": {
            "h3_cell": ("str", "H3 index res 8", "Hexagon id (~0.74 km2)."),
            "cluster": ("int", "id", "Behaviour cluster."),
            "cluster_name": ("str", "-", "Display label of the behaviour cluster."),
            "rhythm_cluster": ("int", "id", "Weekly-rhythm cluster of this cell."),
            "anomaly_type": ("str", "enum", "none | behaviour | rhythm | both."),
            "is_anomaly": ("bool", "flag", "True if anomaly_type != 'none'."),
            "distance_to_centroid": ("float", "PC1-PC5 units", "Distance to its own cluster "
                                                              "centre in the clustering space."),
            "anomaly_z_spatial": ("float", "robust z", "That distance as a robust z."),
            "profile_distance": ("float", "L2 on normalised 168-vector", "Distance from its "
                                                                         "cluster's mean rhythm."),
            "anomaly_z_temporal": ("float", "robust z", "That distance as a robust z."),
            "drivers": ("str", "-", "Features where this cell departs most from its cluster, "
                                    "e.g. 'speed_cv +5.0sd'."),
            "lat_centroid": ("float", "degrees WGS84", "Cell centre latitude."),
            "lon_centroid": ("float", "degrees WGS84", "Cell centre longitude."),
            "peak_hour": ("int", "hour 0-23", "Modal hour of the cell."),
            "speed_mean": ("float", "km/h", "Cell mean speed."),
            "slow_share": ("float", "fraction 0-1", "Cell share of steps below 12 km/h."),
            "n_visits": ("int", "distinct trips", "Trips touching the cell."),
        },
    },
    "anomalies.csv": {
        "format": "csv",
        "key": "h3_cell",
        "description": "Only the flagged cells (behaviour / rhythm / both), with coordinates "
                       "and drivers. A subset of cell_annotations.parquet.",
        "columns": {
            "h3_cell": ("str", "H3 index res 8", "Hexagon id."),
            "cluster": ("int", "id", "Behaviour cluster."),
            "anomaly_type": ("str", "enum", "behaviour | rhythm | both."),
            "distance_to_centroid": ("float", "PC1-PC5 units", "Distance to own centre."),
            "profile_distance": ("float", "L2 on normalised 168-vector", "Rhythm distance."),
            "drivers": ("str", "-", "Largest departures from the cluster mean."),
            "anomaly_z_spatial": ("float", "robust z", "Behaviour score."),
            "anomaly_z_temporal": ("float", "robust z", "Rhythm score."),
            "lat_centroid": ("float", "degrees WGS84", "Latitude."),
            "lon_centroid": ("float", "degrees WGS84", "Longitude."),
        },
    },
    "day_anomalies.csv": {
        "format": "csv",
        "key": "date",
        "description": "One row per calendar day: sampled trip count, median speed, the "
                       "dispersion-corrected z-scores and the flags. Days with too few "
                       "sampled trips are marked 'not assessable' in `note`.",
        "columns": {
            "date": ("str", "ISO date", "Calendar day (YYYY-MM-DD)."),
            "n_trips": ("int", "trips", "Sampled trips starting that day (NOT a city total)."),
            "median_speed": ("float", "km/h", "Median trip mean speed that day."),
            "median_duration": ("float", "seconds", "Median trip duration that day."),
            "poisson_z": ("float", "z", "Raw (n - mu)/sqrt(mu) against the same weekday."),
            "dispersion_phi": ("float", "ratio", "Estimated over-dispersion, same value on "
                                                "every row (~3.2 here)."),
            "volume_z": ("float", "z", "Poisson z scaled by sqrt(phi); the one to use."),
            "speed_robust_z": ("float", "robust z", "Median speed vs all days."),
            "is_anomaly_volume": ("bool", "flag", "|volume_z| >= 2.5 with >= 30 trips."),
            "is_anomaly_speed": ("bool", "flag", "|speed_robust_z| >= 2.5 with >= 30 trips "
                                                 "(either direction)."),
            "speed_direction": ("str", "enum", "slower | faster | empty when not flagged."),
            "note": ("str", "-", "Why a day was not assessed, else empty."),
        },
    },
    "cluster_comparison.csv": {
        "format": "csv",
        "key": "algorithm",
        "description": "Stage 6 metric table: six algorithms, each scored in the space it "
                       "actually clustered, plus bootstrap stability.",
        "columns": {
            "algorithm": ("str", "enum", "kmeans | ward | gmm | dbscan_spatial | hdbscan | tskmeans."),
            "space": ("str", "-", "Feature space the algorithm used."),
            "n_clusters": ("int", "count", "Clusters found (excluding noise)."),
            "coverage": ("float", "fraction 0-1", "Share of cells that received a label."),
            "noise_share": ("float", "fraction 0-1", "1 - coverage."),
            "silhouette": ("float", "-1..1", "Mean silhouette in the algorithm's own space."),
            "silhouette_pca_space": ("float", "-1..1", "Same labels scored in PC1-PC5, so the "
                                                       "algorithms are comparable."),
            "davies_bouldin": ("float", ">=0", "Lower is better; NaN where undefined."),
            "calinski_harabasz": ("float", ">=0", "Higher is better; NaN where undefined."),
            "dbcv": ("float", "<=1", "Density-based validity; only meaningful for DBSCAN/HDBSCAN."),
            "ari_mean": ("float", "-1..1", "Mean ARI over bootstrap refits."),
            "ari_sd": ("float", ">=0", "Standard deviation of those ARIs."),
            "ari_min": ("float", "-1..1", "Worst bootstrap ARI."),
            "ari_boot_n": ("float", "count", "Number of bootstrap pairs (float because some "
                                             "rows are NaN in CSV)."),
        },
    },
    "cluster_comparison.json": {
        "format": "json",
        "description": "The same comparison with settings and notes, for programs.",
        "required_keys": ["stage", "data_source", "synthetic", "rows", "stability_settings"],
    },
    "interpret_meta.json": {
        "format": "json",
        "description": "Provenance for the interpretation stage: source, synthetic flag, "
                       "naming rule, anomaly rules, random seed.",
        "required_keys": ["stage", "data_source", "synthetic", "algorithm", "anomaly_rules",
                          "limitation", "random_state"],
    },
    "cleaning_report.json": {
        "format": "json",
        "description": "Stage 2 report from the LABELLED SYNTHETIC FIXTURE (synthetic: true "
                       "inside the file): thresholds, rows removed per rule, and the score "
                       "against known injected defects. Kept as a regression test of the "
                       "cleaning rules. The cleaning actually applied to the Porto data is "
                       "recorded in data/processed/features_meta.json -> 'cleaning'.",
        "provenance": {"data_source": "synthetic_test_fixture", "synthetic": True,
                       "note": "test fixture, not an input to any result in this directory"},
        "required_keys": ["stage", "data_source", "synthetic", "report"],
    },
    "data_audit.json": {
        "format": "json",
        "description": "Stage 1 audit of the Delhi OTD trimmed GTFS feed (the first dataset "
                       "attempt): missing values, duplicates, time zones, coordinate ranges, "
                       "calendar coverage, findings. Not the data behind the results here - "
                       "those use Porto taxi GPS, audited in data/processed/features_meta.json.",
        "provenance": {"data_source": "delhi_otd_trimmed", "synthetic": False,
                       "note": "earlier data attempt; retained as evidence of why the fallback "
                               "was taken"},
        "required_keys": ["stage", "synthetic", "tables", "integrity", "findings"],
    },
}

EXPECTED_FIGURES = [
    "stage1_stop_coverage.png", "stage2_removed_points.png", "stage2_speed_before_after.png",
    "stage3_cell_activity.png", "stage3_cyclic_encoding.png", "stage3_scaling_effect.png",
    "stage3_weekly_profile.png", "stage4_correlation.png", "stage4_loadings.png",
    "stage4_pca_umap.png", "stage4_scree.png", "stage5_cluster_maps.png",
    "stage5_dbscan_selection.png", "stage5_dtw_profiles.png", "stage5_gmm_bic.png",
    "stage5_kmeans_selection.png", "stage6_agreement.png", "stage6_comparison.png",
    "stage6_k_selection.png", "stage7_anomalies.png", "stage7_poi.png",
    "stage7_profiles.png",
]

def _kind(dtype) -> str:
    """Map a pandas dtype to one of the four declared families.

    Prefix matching, not a dict lookup: pandas reports 'int64', 'float64', and pandas 3
    can report 'int64[pyarrow]' or 'string[python]', so an exact-match table silently
    fails every column (as my first version did).
    """
    s = str(dtype)
    for prefix, kind in (("uint", "int"), ("int", "int"), ("float", "float"),
                         ("bool", "bool"), ("datetime", "datetime"),
                         ("object", "str"), ("string", "str")):
        if s.startswith(prefix):
            return kind
    return s


def _load(path: Path):
    """Load by extension. JSON is read as JSON: pd.read_json would reshape nested
    metadata files into a frame and lose the top-level keys we validate."""
    if path.suffix == ".json":
        return json.load(open(path))
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
def check_file(name: str, spec: dict) -> tuple[list[str], dict]:
    """Validate one declared file. Returns (problems, observed metadata)."""
    problems: list[str] = []
    path = OUT / name
    meta: dict = {"format": spec["format"], "exists": path.exists()}

    if not path.exists():
        return [f"{name}: MISSING"], meta

    meta["bytes"] = path.stat().st_size
    try:
        df = _load(path)
    except Exception as exc:
        return [f"{name}: unreadable ({type(exc).__name__}: {exc})"], meta

    if isinstance(df, pd.DataFrame):
        meta["rows"], meta["cols"] = int(len(df)), int(df.shape[1])
        if len(df) == 0:
            problems.append(f"{name}: empty table")

        # wide tables (cluster x 168) are declared as columns 0..167
        if "wide" in spec:
            w = spec["wide"]
            expected = [str(i) for i in range(w["n"])]
            missing = set(expected) - set(map(str, df.columns))
            extra = set(map(str, df.columns)) - set(expected)
            if missing:
                problems.append(f"{name}: {len(missing)} hour columns missing")
            if extra:
                problems.append(f"{name}: unexpected columns {sorted(extra)[:5]}")
            wrong = [_kind(df[c].dtype) != w["dtype"] for c in df.columns if str(c) in expected]
            if any(wrong):
                problems.append(f"{name}: {sum(wrong)} hour columns are not {w['dtype']}")
            if spec.get("key", "").endswith("(index)") and df.index.name != "cluster":
                problems.append(f"{name}: index name is {df.index.name!r}, expected 'cluster'")
        else:
            declared = spec["columns"]
            for col, (kind, _unit, _desc) in declared.items():
                if "<" in col:                      # pattern placeholder, handled below
                    continue
                if col not in df.columns:
                    problems.append(f"{name}: missing column '{col}'")
                elif _kind(df[col].dtype) != kind:
                    problems.append(f"{name}: column '{col}' is {df[col].dtype}, expected {kind}")
            known = {c for c in declared if "<" not in c}
            for pat, (kind, _u, _d) in spec.get("patterns", {}).items():
                import re
                matched = [c for c in df.columns if re.match(pat, str(c))]
                if not matched:
                    problems.append(f"{name}: no columns match pattern {pat}")
                bad = [c for c in matched if _kind(df[c].dtype) != kind]
                if bad:
                    problems.append(f"{name}: {len(bad)} pattern-matched columns not {kind}")
                known |= set(matched)
            unknown = [c for c in df.columns if c not in known]
            if unknown:
                problems.append(f"{name}: undeclared columns {unknown[:6]}")

            # primary key uniqueness
            key = spec.get("key")
            if key in df.columns and df[key].duplicated().any():
                problems.append(f"{name}: duplicate values in key '{key}'")

        meta["columns"] = {str(c): _kind(df[c].dtype) for c in df.columns}
    else:                                            # json
        meta["keys"] = list(df) if isinstance(df, dict) else None
        missing = [k for k in spec.get("required_keys", []) if k not in (df or {})]
        if missing:
            problems.append(f"{name}: missing keys {missing}")

    return problems, meta


def _has(df, *cols) -> bool:
    """True when df exists and has all the named columns (a corrupt file must not crash)."""
    return df is not None and all(c in df.columns for c in cols)


def cross_checks(frames: dict) -> list[str]:
    """Consistency BETWEEN files - the failures a per-file check cannot see.

    Wrapped in try/except: if a file is so broken that a check itself throws, that is a
    finding to report, not a reason for the validator to die with a traceback.
    """
    try:
        return _cross_checks(frames)
    except Exception as exc:
        return [f"cross-checks aborted: {type(exc).__name__}: {exc}"]


def _cross_checks(frames: dict) -> list[str]:
    p: list[str] = []
    prof = frames.get("cluster_profiles.csv")
    ann = frames.get("cell_annotations.parquet")
    anom = frames.get("anomalies.csv")
    hourly = frames.get("cluster_hourly_profile.parquet")
    pois = frames.get("cluster_pois.csv")
    day = frames.get("day_anomalies.csv")
    comp = frames.get("cluster_comparison.csv")

    if _has(prof, "share_cells"):
        # share_cells is stored rounded to 4 dp, so the sum can drift by ~2e-4 across 4
        # clusters; 1e-3 is tight enough to catch a real omission (a whole cluster)
        # and loose enough to ignore rounding.
        total = float(prof["share_cells"].sum())
        if abs(total - 1.0) > 1e-3:
            p.append(f"cluster_profiles: share_cells sums to {total:.6f}, expected 1.0")
    if _has(prof, "cluster") and prof["cluster"].duplicated().any():
        p.append("cluster_profiles: duplicated cluster ids")

    if _has(prof, "cluster", "n_cells") and _has(ann, "cluster"):
        if set(ann["cluster"].astype(int)) != set(prof["cluster"].astype(int)):
            p.append("cluster ids differ between cluster_profiles and cell_annotations")
        if int(prof["n_cells"].sum()) != len(ann):
            p.append(f"n_cells sums to {int(prof['n_cells'].sum())} but cell_annotations "
                     f"has {len(ann)} rows")

    if _has(ann, "is_anomaly") and anom is not None:
        flagged = int(ann["is_anomaly"].sum())
        if flagged != len(anom):
            p.append(f"anomalies.csv has {len(anom)} rows, cell_annotations flags {flagged}")
    if _has(ann, "h3_cell") and ann["h3_cell"].duplicated().any():
        p.append("cell_annotations: duplicated h3_cell")
    if _has(ann, "anomaly_type"):
        bad = set(ann["anomaly_type"]) - {"none", "behaviour", "rhythm", "both"}
        if bad:
            p.append(f"cell_annotations: unexpected anomaly_type values {bad}")
        if _has(ann, "is_anomaly"):
            if not ((ann["anomaly_type"] == "none") == (~ann["is_anomaly"])).all():
                p.append("cell_annotations: is_anomaly disagrees with anomaly_type")

    if hourly is not None and _has(prof, "cluster"):
        if set(hourly.index.astype(int)) != set(prof["cluster"].astype(int)):
            p.append("cluster_hourly_profile: cluster ids differ from cluster_profiles")
    if _has(pois, "cluster") and _has(prof, "cluster"):
        if set(pois["cluster"].astype(int)) != set(prof["cluster"].astype(int)):
            p.append("cluster_pois: cluster ids differ from cluster_profiles")

    if _has(day, "date") and day["date"].duplicated().any():
        p.append("day_anomalies: duplicated dates")
    if _has(day, "is_anomaly_speed", "speed_direction"):
        flagged_dir = day.loc[day["is_anomaly_speed"], "speed_direction"]
        if flagged_dir.isna().any() or (flagged_dir == "").any():
            p.append("day_anomalies: a speed flag has no direction")

    if _has(comp, "coverage", "noise_share"):
        gap = (comp["coverage"] + comp["noise_share"] - 1).abs().max()
        if gap > 1e-6:
            p.append(f"cluster_comparison: coverage + noise_share != 1 (max gap {gap:.2e})")
    if _has(comp, "silhouette"):
        within = comp["silhouette"].dropna()
        if ((within < -1) | (within > 1)).any():
            p.append("cluster_comparison: silhouette outside [-1, 1]")
    return p


def provenance() -> dict:
    """Where the numbers came from, read from the pipeline's own metadata."""
    meta = {}
    f = PROC / "features_meta.json"
    if f.exists():
        m = json.load(open(f))
        meta = {"data_source": m.get("data_source"), "synthetic": m.get("synthetic"),
                "limitation": m.get("limitation")}
    return meta


# ---------------------------------------------------------------------------
# artefacts
# ---------------------------------------------------------------------------
def write_manifest(meta: dict, checks: dict, path: Path) -> None:
    manifest = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage": 8,
        "provenance": provenance(),
        "read_contract": "The dashboard reads only this directory. Files are outputs of "
                         "the notebooks in ./notebooks and must not be edited by hand.",
        "files": {},
        "figures": EXPECTED_FIGURES,
        "checks": checks,
    }
    for name, spec in FILES.items():
        entry = {"format": spec["format"], "description": spec["description"],
                 "provenance": spec.get("provenance", provenance()),
                 **{k: v for k, v in meta.get(name, {}).items() if k != "columns"}}
        if spec["format"] == "json":
            entry["required_keys"] = spec.get("required_keys", [])
        else:
            entry["columns"] = {
                col: {"dtype": kind, "unit": unit, "description": desc}
                for col, (kind, unit, desc) in spec.get("columns", {}).items()
            }
            if "wide" in spec:
                entry["wide"] = spec["wide"]
            if "patterns" in spec:
                entry["patterns"] = {k: {"dtype": v[0], "unit": v[1], "description": v[2]}
                                     for k, v in spec["patterns"].items()}
            if "categories" in spec:
                entry["categories"] = spec["categories"]
        manifest["files"][name] = entry
    path.write_text(json.dumps(manifest, indent=2))


def write_schema_md(path: Path) -> None:
    lines = [
        "# `outputs/` schema — FleetSense",
        "",
        "_Generated by `python -m src.schema`. Do not edit by hand: change `src/schema.py` "
        "and re-run, so the validator and this document cannot disagree._",
        "",
        f"Data provenance: **{provenance().get('data_source')}**, "
        f"synthetic: **{provenance().get('synthetic')}**. "
        f"Limitation: {provenance().get('limitation')}",
        "",
        "All coordinates are WGS84 degrees. `NaN`/empty means *not defined for that row* "
        "(for example Davies–Bouldin for a precomputed distance matrix), never zero.",
        "",
    ]
    for name, spec in FILES.items():
        lines += [f"## `{name}`", "", spec["description"], "",
                  f"* format: `{spec['format']}`  ·  key: `{spec.get('key', '-')}`"]
        if spec.get("provenance"):
            pr = spec["provenance"]
            lines.append(f"* provenance: `{pr['data_source']}`, synthetic: "
                         f"**{pr['synthetic']}** — {pr.get('note', '')}")
        lines.append("")
        if spec.get("wide"):
            w = spec["wide"]
            lines += [f"{w['n']} columns `0`–`{w['n'] - 1}`, dtype `{w['dtype']}`, unit "
                      f"**{w['unit']}**: {w['description']}", ""]
        if spec.get("columns"):
            lines += ["| column | dtype | unit | meaning |", "|---|---|---|---|"]
            for col, (kind, unit, desc) in spec["columns"].items():
                lines.append(f"| `{col}` | {kind} | {unit} | {desc} |")
            lines.append("")
        if spec.get("patterns"):
            lines += ["Pattern-matched columns:", ""]
            for pat, (kind, unit, desc) in spec["patterns"].items():
                lines.append(f"* `{pat}` → {kind}, {unit}. {desc}")
            lines.append("")
        if spec.get("categories"):
            lines += [f"Categories: {', '.join('`' + c + '`' for c in spec['categories'])}", ""]
        if spec.get("required_keys"):
            lines += [f"Required top-level keys: {', '.join('`' + k + '`' for k in spec['required_keys'])}", ""]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    import sys
    argv = argv if argv is not None else sys.argv[1:]
    check_only = "--check" in argv

    problems: list[str] = []
    meta: dict = {}
    frames: dict = {}

    for name, spec in FILES.items():
        probs, m = check_file(name, spec)
        problems += probs
        meta[name] = m
        if (OUT / name).exists():
            try:
                obj = _load(OUT / name)
                if isinstance(obj, pd.DataFrame):
                    frames[name] = obj
            except Exception:
                pass

    problems += cross_checks(frames)

    missing_figs = [f for f in EXPECTED_FIGURES if not (OUT / "figures" / f).exists()]
    if missing_figs:
        problems.append(f"missing figures: {missing_figs}")

    print(f"checked {len(FILES)} declared files + {len(EXPECTED_FIGURES)} figures\n")
    for name in FILES:
        m = meta[name]
        status = "ok " if not [p for p in problems if p.startswith(name)] else "FAIL"
        size = f"{m.get('bytes', 0) / 1024:7.1f} KB" if m.get("exists") else "   absent"
        rows = f"rows={m.get('rows', '-'):>4}" if "rows" in m else "         "
        print(f"  [{status}] {name:<34} {size}  {rows}")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print("  -", p)
        return 1

    print("\nall files, columns, dtypes, keys and cross-file checks passed")
    if not check_only:
        checks = {"declared_files": len(FILES), "figures": len(EXPECTED_FIGURES),
                  "passed": True, "validated_utc": datetime.now(timezone.utc)
                  .isoformat(timespec="seconds")}
        write_manifest(meta, checks, OUT / "manifest.json")
        write_schema_md(OUT / "SCHEMA.md")
        print("wrote outputs/manifest.json and outputs/SCHEMA.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
