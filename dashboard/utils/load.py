"""Cached loaders and manifest startup validation for FleetSense outputs."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT: Path = Path(__file__).resolve().parents[2]
OUTPUTS_DIR: Path = REPO_ROOT / "outputs"
DOCS_DIR: Path = REPO_ROOT / "docs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
MANIFEST_FILENAME: str = "manifest.json"
PRIVACY_DOC_FILENAME: str = "PRIVACY_AND_LIMITATIONS.md"

KNOWN_HOLIDAYS: dict[str, str] = {}


@dataclass(frozen=True)
class ManifestValidationResult:
    """Result of validating outputs/ against outputs/manifest.json."""

    passed: bool
    errors: list[str]
    manifest: dict[str, Any]


def _read_table(path: Path, fmt: str) -> pd.DataFrame:
    """Read a tabular output file without mutating schemas."""
    if fmt == "csv":
        return pd.read_csv(path)
    if fmt == "parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported tabular format: {fmt}")


@st.cache_data(ttl=3600, show_spinner=False)
def validate_manifest(outputs_dir: str | Path = OUTPUTS_DIR) -> ManifestValidationResult:
    """Validate that manifest.json exists and every declared file and column is present.

    Checks all declared files in ``manifest['files']``. Missing optional figures do not
    fail startup validation so the app degrades gracefully when figures are absent.
    """
    base = Path(outputs_dir)
    manifest_path = base / MANIFEST_FILENAME
    if not manifest_path.exists():
        return ManifestValidationResult(
            passed=False,
            errors=[f"Missing contract file: {manifest_path}"],
            manifest={},
        )

    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return ManifestValidationResult(
            passed=False,
            errors=[f"Failed to parse {MANIFEST_FILENAME}: {exc}"],
            manifest={},
        )

    declared_files = manifest.get("files")
    if not isinstance(declared_files, dict) or not declared_files:
        return ManifestValidationResult(
            passed=False,
            errors=[f"{MANIFEST_FILENAME} declares no files under 'files'."],
            manifest=manifest,
        )

    errors: list[str] = []
    for filename, spec in declared_files.items():
        fpath = base / filename
        if not fpath.exists():
            errors.append(f"Missing declared output file: {filename}")
            continue

        fmt = spec.get("format")
        if fmt == "json":
            try:
                payload = json.loads(fpath.read_text(encoding="utf-8"))
            except Exception as exc:
                errors.append(f"{filename}: unreadable JSON ({exc})")
                continue
            if not isinstance(payload, dict):
                errors.append(f"{filename}: expected top-level JSON object")
                continue
            for key in spec.get("required_keys", []):
                if key not in payload:
                    errors.append(f"{filename}: missing required key '{key}'")
        elif fmt in ("csv", "parquet"):
            try:
                df = _read_table(fpath, fmt)
            except Exception as exc:
                errors.append(f"{filename}: unreadable {fmt} ({exc})")
                continue

            if df.empty:
                errors.append(f"{filename}: table has 0 rows")

            if "wide" in spec:
                expected_n = int(spec["wide"].get("n", 0))
                col_strs = {str(c) for c in df.columns}
                missing_wide = [str(i) for i in range(expected_n) if str(i) not in col_strs]
                if missing_wide:
                    errors.append(
                        f"{filename}: missing {len(missing_wide)} hour columns "
                        f"(first missing: {missing_wide[0]})"
                    )
            else:
                declared_cols = spec.get("columns", {})
                for col in declared_cols:
                    if "<" in col:
                        continue
                    if col not in df.columns:
                        errors.append(f"{filename}: missing required column '{col}'")

                for pat in spec.get("patterns", {}):
                    matched = [c for c in df.columns if re.match(pat, str(c))]
                    if not matched:
                        errors.append(
                            f"{filename}: no columns matched required pattern '{pat}'"
                        )
        else:
            errors.append(f"{filename}: unknown declared format '{fmt}'")

    return ManifestValidationResult(passed=len(errors) == 0, errors=errors, manifest=manifest)


@st.cache_data(ttl=3600, show_spinner=False)
def load_manifest(outputs_dir: str | Path = OUTPUTS_DIR) -> dict[str, Any]:
    """Load outputs/manifest.json."""
    path = Path(outputs_dir) / MANIFEST_FILENAME
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=3600, show_spinner=False)
def load_cluster_profiles(outputs_dir: str | Path = OUTPUTS_DIR) -> pd.DataFrame:
    """Load outputs/cluster_profiles.csv sorted by cluster id."""
    df = pd.read_csv(Path(outputs_dir) / "cluster_profiles.csv")
    df["cluster"] = df["cluster"].astype(int)
    df["poi_context"] = df["poi_context"].fillna("")
    return df.sort_values("cluster").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_cluster_pois(outputs_dir: str | Path = OUTPUTS_DIR) -> pd.DataFrame:
    """Load outputs/cluster_pois.csv sorted by cluster id."""
    df = pd.read_csv(Path(outputs_dir) / "cluster_pois.csv")
    df["cluster"] = df["cluster"].astype(int)
    return df.sort_values("cluster").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_rhythm_names(outputs_dir: str | Path = OUTPUTS_DIR) -> pd.DataFrame:
    """Load outputs/cluster_rhythm_names.csv sorted by rhythm_cluster id."""
    df = pd.read_csv(Path(outputs_dir) / "cluster_rhythm_names.csv")
    df["rhythm_cluster"] = df["rhythm_cluster"].astype(int)
    return df.sort_values("rhythm_cluster").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_rhythm_crosstab(outputs_dir: str | Path = OUTPUTS_DIR) -> pd.DataFrame:
    """Load outputs/cluster_rhythm_crosstab.csv indexed by kmeans_cluster."""
    df = pd.read_csv(Path(outputs_dir) / "cluster_rhythm_crosstab.csv")
    df["kmeans_cluster"] = df["kmeans_cluster"].astype(int)
    return df.sort_values("kmeans_cluster").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_hourly_profile(outputs_dir: str | Path = OUTPUTS_DIR) -> pd.DataFrame:
    """Load outputs/cluster_hourly_profile.parquet with integer hour columns 0..167."""
    df = pd.read_parquet(Path(outputs_dir) / "cluster_hourly_profile.parquet")
    df.columns = [int(c) for c in df.columns]
    df.index = df.index.astype(int)
    return df.sort_index()


@st.cache_data(ttl=3600, show_spinner=False)
def load_cell_annotations(outputs_dir: str | Path = OUTPUTS_DIR) -> pd.DataFrame:
    """Load outputs/cell_annotations.parquet and enrich with rhythm names and cluster means."""
    base = Path(outputs_dir)
    cells = pd.read_parquet(base / "cell_annotations.parquet")
    cells["cluster"] = cells["cluster"].astype(int)
    cells["rhythm_cluster"] = cells["rhythm_cluster"].astype(int)
    cells["drivers"] = cells["drivers"].fillna("")
    cells["anomaly_score"] = cells[["anomaly_z_spatial", "anomaly_z_temporal"]].max(axis=1)

    rhythms = load_rhythm_names(base)[["rhythm_cluster", "rhythm_name"]]
    profiles = load_cluster_profiles(base)[["cluster", "dwell_s_per_visit", "name", "label"]]
    pois = load_cluster_pois(base)[["cluster", "poi_per_cell"]]

    enriched = (
        cells.merge(rhythms, on="rhythm_cluster", how="left")
        .merge(
            profiles.rename(
                columns={
                    "dwell_s_per_visit": "cluster_dwell_s_per_visit",
                    "name": "cluster_short_name",
                    "label": "cluster_label",
                }
            ),
            on="cluster",
            how="left",
        )
        .merge(
            pois.rename(columns={"poi_per_cell": "cluster_poi_per_cell"}),
            on="cluster",
            how="left",
        )
    )
    return enriched


@st.cache_data(ttl=3600, show_spinner=False)
def load_anomalies(outputs_dir: str | Path = OUTPUTS_DIR) -> pd.DataFrame:
    """Load outputs/anomalies.csv joined with cell_annotations context."""
    base = Path(outputs_dir)
    anom = pd.read_csv(base / "anomalies.csv")
    anom["cluster"] = anom["cluster"].astype(int)
    anom["drivers"] = anom["drivers"].fillna("")

    cells = load_cell_annotations(base)[
        [
            "h3_cell",
            "cluster_name",
            "rhythm_cluster",
            "rhythm_name",
            "speed_mean",
            "slow_share",
            "n_visits",
            "peak_hour",
            "anomaly_score",
        ]
    ]
    return anom.merge(cells, on="h3_cell", how="left")


@st.cache_data(ttl=3600, show_spinner=False)
def load_day_anomalies(outputs_dir: str | Path = OUTPUTS_DIR) -> pd.DataFrame:
    """Load outputs/day_anomalies.csv with derived combined anomaly flag and holiday notes."""
    df = pd.read_csv(Path(outputs_dir) / "day_anomalies.csv")
    df["speed_direction"] = df["speed_direction"].fillna("")
    df["note"] = df["note"].fillna("")
    df["is_anomaly_volume"] = df["is_anomaly_volume"].astype(bool)
    df["is_anomaly_speed"] = df["is_anomaly_speed"].astype(bool)
    df["is_anomaly"] = df["is_anomaly_volume"] | df["is_anomaly_speed"]

    dt = pd.to_datetime(df["date"])
    df["date_dt"] = dt
    df["day_of_week"] = dt.dt.day_name()
    df["holiday"] = df["date"].map(KNOWN_HOLIDAYS).fillna("")

    def _kind(row: pd.Series) -> str:
        if row["is_anomaly_volume"] and row["is_anomaly_speed"]:
            return "volume + speed"
        if row["is_anomaly_volume"]:
            return "volume"
        if row["is_anomaly_speed"]:
            return f"speed ({row['speed_direction']})" if row["speed_direction"] else "speed"
        return "normal"

    df["anomaly_kind"] = df.apply(_kind, axis=1)
    return df.sort_values("date_dt").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_cluster_comparison(outputs_dir: str | Path = OUTPUTS_DIR) -> pd.DataFrame:
    """Load outputs/cluster_comparison.csv."""
    return pd.read_csv(Path(outputs_dir) / "cluster_comparison.csv")


@st.cache_data(ttl=3600, show_spinner=False)
def load_json_output(filename: str, outputs_dir: str | Path = OUTPUTS_DIR) -> dict[str, Any]:
    """Load a JSON metadata or report file from outputs/."""
    path = Path(outputs_dir) / filename
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(ttl=3600, show_spinner=False)
def load_privacy_doc(docs_dir: str | Path = DOCS_DIR) -> str | None:
    """Load docs/PRIVACY_AND_LIMITATIONS.md if present."""
    path = Path(docs_dir) / PRIVACY_DOC_FILENAME
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


@st.cache_data(ttl=3600, show_spinner=False)
def list_figures(
    outputs_dir: str | Path = OUTPUTS_DIR,
    manifest: dict[str, Any] | None = None,
) -> list[Path]:
    """Return existing PNG figure paths, preferring manifest order."""
    fig_dir = Path(outputs_dir) / "figures"
    if not fig_dir.exists():
        return []

    ordered_names = (manifest or {}).get("figures", [])
    paths: list[Path] = []
    seen: set[str] = set()
    for name in ordered_names:
        candidate = fig_dir / name
        if candidate.exists():
            paths.append(candidate)
            seen.add(candidate.name)

    for extra in sorted(fig_dir.glob("*.png")):
        if extra.name not in seen:
            paths.append(extra)
    return paths


@st.cache_data(ttl=3600, show_spinner=False)
def get_overview_kpis(outputs_dir: str | Path = OUTPUTS_DIR) -> dict[str, Any]:
    """Derive headline KPI counts strictly from outputs/ files."""
    base = Path(outputs_dir)
    profiles = load_cluster_profiles(base)
    rhythms = load_rhythm_names(base)
    cells = load_cell_annotations(base)
    anom = load_anomalies(base)
    days = load_day_anomalies(base)

    return {
        "trips_sampled": int(days["n_trips"].sum()),
        "pings_analysed": int(profiles["n_points_total"].sum()),
        "cells_analysed": int(len(cells)),
        "behaviour_clusters": int(len(profiles)),
        "rhythm_clusters": int(len(rhythms)),
        "flagged_cells": int(len(anom)),
        "flagged_days": int(days["is_anomaly"].sum()),
        "flagged_days_volume": int(days["is_anomaly_volume"].sum()),
        "flagged_days_speed": int(days["is_anomaly_speed"].sum()),
        "calendar_days": int(len(days)),
    }

