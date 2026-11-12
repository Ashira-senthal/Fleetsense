"""FleetSense Streamlit dashboard entrypoint.

Reads precomputed outputs from ``outputs/`` only and validates the data contract
against ``outputs/manifest.json`` before rendering any page.
"""

from __future__ import annotations

import streamlit as st

from utils.load import OUTPUTS_DIR, get_overview_kpis, load_manifest, validate_manifest
from utils.theme import render_jargon_popover

st.set_page_config(
    page_title="FleetSense: Delhi bus mobility patterns",
    page_icon=":material/directions_bus:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Startup contract validation — fail loudly if any declared file or column is missing.
validation = validate_manifest(OUTPUTS_DIR)
if not validation.passed:
    st.error(
        "Startup validation against `outputs/manifest.json` failed:\n\n"
        + "\n".join(f"- {err}" for err in validation.errors),
        title="Missing or invalid output files",
        icon=":material/error:",
    )
    st.stop()

# Shared cross-page session state.
st.session_state.setdefault("selected_h3_cell", None)

pages = [
    st.Page("pages/2_cluster_map.py", title="Map", icon=":material/map:", default=True),
    st.Page("pages/3_cluster_profiles.py", title="Clusters", icon=":material/analytics:"),
    st.Page("pages/4_weekly_rhythms.py", title="Weekly rhythm", icon=":material/schedule:"),
    st.Page("pages/5_anomalies.py", title="Anomalies", icon=":material/warning:"),
    st.Page("pages/6_model_comparison.py", title="Method", icon=":material/compare:"),
    st.Page("pages/7_data_quality.py", title="Data quality", icon=":material/verified_user:"),
    st.Page("pages/1_overview.py", title="Overview", icon=":material/dashboard:"),
]

page = st.navigation(pages, position="top")

# Compact global header + KPI strip matching the FleetSense hero layout.
manifest = load_manifest(OUTPUTS_DIR)
kpis = get_overview_kpis(OUTPUTS_DIR)
checks = manifest.get("checks", {})

head_left, head_right = st.columns([3.5, 1], vertical_alignment="center")
with head_left:
    st.title("FleetSense: Delhi bus mobility patterns")
    st.caption(
        "Which parts of Delhi NCR behave alike, and when? Bus schedules (DIMTS & DTC) "
        "grouped into H3 hexagons and clustered without labels."
    )
with head_right:
    with st.container(horizontal=True, horizontal_alignment="right", vertical_alignment="center"):
        st.badge(
            f"Manifest OK ({checks.get('declared_files', 0)} files)",
            icon=":material/check_circle:",
            color="green",
        )
        render_jargon_popover()

kpi_cols = st.columns(6)
kpi_cols[0].metric("trips sampled", f"{kpis['trips_sampled']:,}", border=True, height="stretch")
kpi_cols[1].metric(
    "pings in analysed cells",
    f"{kpis['pings_analysed'] / 1_000_000:.2f}M",
    border=True,
    height="stretch",
)
kpi_cols[2].metric("cells analysed", f"{kpis['cells_analysed']:,}", border=True, height="stretch")
kpi_cols[3].metric(
    "behaviour clusters",
    f"{kpis['behaviour_clusters']}",
    border=True,
    height="stretch",
)
kpi_cols[4].metric("flagged cells", f"{kpis['flagged_cells']:,}", border=True, height="stretch")
kpi_cols[5].metric("flagged days", f"{kpis['flagged_days']:,}", border=True, height="stretch")

page.run()
