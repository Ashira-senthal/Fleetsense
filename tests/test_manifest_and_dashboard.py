"""Smoke tests for outputs/ manifest contract, loaders, and Streamlit pages."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = REPO_ROOT / "dashboard"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from dashboard.utils.load import (  # noqa: E402
    OUTPUTS_DIR,
    get_overview_kpis,
    list_figures,
    load_anomalies,
    load_cell_annotations,
    load_cluster_comparison,
    load_cluster_pois,
    load_cluster_profiles,
    load_day_anomalies,
    load_hourly_profile,
    load_json_output,
    load_manifest,
    load_privacy_doc,
    load_rhythm_crosstab,
    load_rhythm_names,
    validate_manifest,
)


def test_manifest_validation_passes_on_real_outputs() -> None:
    """Every file and required column declared in outputs/manifest.json must exist."""
    res = validate_manifest(OUTPUTS_DIR)
    assert res.passed, f"Manifest validation failed: {res.errors}"
    assert res.manifest["checks"]["declared_files"] == 13


def test_all_output_loaders_return_expected_data() -> None:
    """Every output loader succeeds and returns non-empty validated structures."""
    manifest = load_manifest(OUTPUTS_DIR)
    assert "files" in manifest

    profiles = load_cluster_profiles(OUTPUTS_DIR)
    assert len(profiles) == 4

    pois = load_cluster_pois(OUTPUTS_DIR)
    assert len(pois) == 4

    rhythms = load_rhythm_names(OUTPUTS_DIR)
    assert len(rhythms) == 4

    crosstab = load_rhythm_crosstab(OUTPUTS_DIR)
    assert len(crosstab) == 4

    hourly = load_hourly_profile(OUTPUTS_DIR)
    assert hourly.shape == (4, 168)

    cells = load_cell_annotations(OUTPUTS_DIR)
    assert len(cells) > 0
    assert "rhythm_name" in cells.columns

    anom = load_anomalies(OUTPUTS_DIR)
    assert len(anom) >= 0

    days = load_day_anomalies(OUTPUTS_DIR)
    assert len(days) > 0
    assert "is_anomaly" in days.columns

    comp = load_cluster_comparison(OUTPUTS_DIR)
    assert len(comp) == 6

    for jname in ("interpret_meta.json", "cleaning_report.json", "data_audit.json", "cluster_comparison.json"):
        payload = load_json_output(jname, OUTPUTS_DIR)
        assert isinstance(payload, dict)

    kpis = get_overview_kpis(OUTPUTS_DIR)
    assert kpis["cells_analysed"] == len(cells)
    assert kpis["flagged_cells"] == len(anom)
    assert kpis["flagged_days"] == int(days["is_anomaly"].sum())
    assert load_privacy_doc() is not None
    assert len(list_figures(OUTPUTS_DIR, manifest)) >= 1


def test_manifest_validation_detects_missing_column(tmp_path: Path) -> None:
    """Startup validation fails loudly when a declared column is missing."""
    manifest = {
        "files": {
            "sample.csv": {
                "format": "csv",
                "columns": {"required_col": {"dtype": "int"}},
            }
        }
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pd.DataFrame({"wrong_col": [1, 2]}).to_csv(tmp_path / "sample.csv", index=False)

    res = validate_manifest(tmp_path)
    assert not res.passed
    assert any("required_col" in err for err in res.errors)


def test_streamlit_pages_render_without_exceptions() -> None:
    """Run AppTest against the multipage entrypoint and verify zero exceptions."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(DASHBOARD_DIR / "app.py"), default_timeout=15)
    at.run()
    assert not at.exception
    assert not at.error

    for page_file in sorted((DASHBOARD_DIR / "pages").glob("[0-9]_*.py")):
        pt = AppTest.from_file(str(page_file), default_timeout=15)
        pt.run()
        assert not pt.exception, f"Exception in {page_file.name}: {pt.exception}"
        assert not pt.error, f"Error in {page_file.name}: {pt.error}"


def test_no_delhi_gtfs_labels_outside_data_quality_note() -> None:
    """Ensure Delhi/GTFS strings only appear in the single allowed Data quality note."""
    forbidden = ("Delhi bus", "DIMTS", "DTC", "GTFS")
    allowed_file = (DASHBOARD_DIR / "pages" / "7_data_quality.py").resolve()
    violations: list[str] = []

    for py_file in sorted(DASHBOARD_DIR.rglob("*.py")):
        text = py_file.read_text(encoding="utf-8")
        if py_file.resolve() == allowed_file:
            matching_lines = [
                line for line in text.splitlines() if any(tok in line for tok in forbidden)
            ]
            if len(matching_lines) > 1:
                violations.append(
                    f"{py_file.name}: expected at most 1 allowed note line, found {len(matching_lines)}"
                )
        else:
            for lineno, line in enumerate(text.splitlines(), start=1):
                for tok in forbidden:
                    if tok in line:
                        violations.append(f"{py_file.relative_to(DASHBOARD_DIR)}:{lineno}: {tok}")

    assert not violations, "Unexpected Delhi/GTFS references:\n" + "\n".join(violations)


