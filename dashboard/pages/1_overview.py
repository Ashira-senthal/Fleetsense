"""Overview page: KPI summary, provenance badge, and reading guide."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.load import (
    OUTPUTS_DIR,
    get_overview_kpis,
    load_cluster_profiles,
    load_json_output,
    load_manifest,
)
from utils.theme import cluster_hex

manifest = load_manifest(OUTPUTS_DIR)
meta = load_json_output("interpret_meta.json", OUTPUTS_DIR)
kpis = get_overview_kpis(OUTPUTS_DIR)
profiles = load_cluster_profiles(OUTPUTS_DIR)

provenance = manifest.get("provenance", {})
checks = manifest.get("checks", {})

with st.container(horizontal=True, vertical_alignment="center"):
    st.badge(
        f"Manifest validated ({checks.get('declared_files', 0)} files)",
        icon=":material/check_circle:",
        color="green",
    )
    st.badge(
        f"Source: {provenance.get('data_source', 'unknown')} · synthetic={provenance.get('synthetic', False)}",
        icon=":material/verified_user:",
        color="blue",
    )
    st.caption(f"Contract generated: `{manifest.get('generated_utc', 'unknown')}`")

# KPI row 1 & 2 using equal-height columns
col1, col2, col3 = st.columns(3)
col1.metric("Trips sampled", f"{kpis['trips_sampled']:,}", border=True, height="stretch")
col2.metric(
    "Pings in analysed cells",
    f"{kpis['pings_analysed']:,}",
    border=True,
    height="stretch",
)
col3.metric(
    "H3 cells analysed",
    f"{kpis['cells_analysed']:,}",
    help="Resolution 8 hexagons (~0.74 km²) with at least 5 distinct taxi visits.",
    border=True,
    height="stretch",
)

col4, col5, col6 = st.columns(3)
col4.metric(
    "Clusters",
    f"{kpis['behaviour_clusters']} behaviour · {kpis['rhythm_clusters']} rhythm",
    border=True,
    height="stretch",
)
col5.metric(
    "Flagged cells",
    f"{kpis['flagged_cells']:,}",
    help="Rule-based candidates (top 2% distance AND robust z ≥ 2).",
    border=True,
    height="stretch",
)
col6.metric(
    "Flagged days",
    f"{kpis['flagged_days']:,}",
    help=(
        f"{kpis['flagged_days_volume']} volume + {kpis['flagged_days_speed']} speed flags "
        f"across {kpis['calendar_days']} days (2013-12-25 flagged on both)."
    ),
    border=True,
    height="stretch",
)

st.subheader("How to read this dashboard")
with st.container(border=True):
    st.markdown(
        f"""
        1. **Two independent axes:** Every analysed H3 hexagon (~0.74 km²) receives a
           **behaviour cluster** (`K-Means, k=4` on speed, dwell, stop-and-go CV, and volume)
           and a **weekly rhythm cluster** (`TS-K-Means` on 168-hour activity profiles).
           Keeping them separate lets you see when the same traffic behaviour occurs at
           different times of week.
        2. **Names are hypotheses, not facts:** Labels like *"{profiles.iloc[1]['label']}"* are
           generated from measured z-scores and OpenStreetMap POI ratios. Always read the
           measured numbers alongside each name.
        3. **Anomalies are candidates, not proven incidents:** Cell and day flags come from
           explicit statistical rules (`{meta.get('anomaly_rules', {}).get('spatial', '')}`).
        4. **Taxis, not buses:** {meta.get('limitation_text', '')}
        """
    )

st.subheader("Behaviour clusters at a glance")

summary_rows = []
for _, row in profiles.iterrows():
    cid = int(row["cluster"])
    summary_rows.append(
        {
            "Cluster": f"C{cid}",
            "Colour": cluster_hex(cid),
            "Generated name": row["label"],
            "Cells": int(row["n_cells"]),
            "Share": float(row["share_cells"]),
            "Mean speed (km/h)": float(row["speed_mean"]),
            "Slow steps": float(row["slow_share"]),
            "Dwell / visit (s)": float(row["dwell_s_per_visit"]),
            "Mean visits / cell": float(row["n_visits"]),
        }
    )

summary_df = pd.DataFrame(summary_rows)
st.dataframe(
    summary_df,
    column_config={
        "Colour": st.column_config.TextColumn("Palette hex"),
        "Share": st.column_config.NumberColumn("City share", format="percent"),
        "Mean speed (km/h)": st.column_config.NumberColumn(format="%.1f km/h"),
        "Slow steps": st.column_config.NumberColumn("Slow (<12 km/h)", format="percent"),
        "Dwell / visit (s)": st.column_config.NumberColumn(format="%.0f s"),
        "Mean visits / cell": st.column_config.NumberColumn(format="%.0f"),
    },
    hide_index=True,
    alt="Summary table of the four Porto taxi behaviour clusters",
)
