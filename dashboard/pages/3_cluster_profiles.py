"""Cluster profiles, normalized feature comparison, and POI breakdown."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from utils.load import (
    OUTPUTS_DIR,
    load_cluster_pois,
    load_cluster_profiles,
)
from utils.theme import cluster_hex, plotly_template

profiles = load_cluster_profiles(OUTPUTS_DIR)
pois = load_cluster_pois(OUTPUTS_DIR)

merged = profiles.merge(pois[["cluster", "poi_per_cell", "poi_total"]], on="cluster", how="left")

st.info(
    "**Names are hypotheses, not ground truth:** every generated label is printed next to "
    "the measured numbers from `outputs/cluster_profiles.csv`.",
    icon=":material/lightbulb:",
)

# 1. One card per cluster (2x2 grid for responsiveness)
st.subheader("Cluster summary cards")
for row_start in range(0, len(merged), 2):
    cols = st.columns(2)
    for col, (_, row) in zip(cols, merged.iloc[row_start : row_start + 2].iterrows()):
        cid = int(row["cluster"])
        hex_c = cluster_hex(cid)
        with col.container(border=True, height="stretch"):
            st.markdown(
                f"<span style='display:inline-block;width:12px;height:12px;"
                f"background:{hex_c};border-radius:2px;margin-right:6px;'></span>"
                f"**Cluster {cid} — {row['label']}**",
                unsafe_allow_html=True,
            )
            st.caption(f"Evidence: {row['description']}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Cells", f"{int(row['n_cells'])} ({row['share_cells']:.0%})")
            m2.metric("Mean speed", f"{row['speed_mean']:.1f} km/h")
            m3.metric("Slow steps", f"{row['slow_share']:.0%}")
            m4, m5, m6 = st.columns(3)
            m4.metric("Speed CV", f"{row['speed_cv']:.2f}")
            m5.metric("Dwell / visit", f"{row['dwell_s_per_visit']:.0f} s")
            m6.metric("POIs / cell", f"{row['poi_per_cell']:.1f}")
            if row["poi_context"]:
                st.caption(f"Notable POI contrast: `{row['poi_context']}`")

# 2. Normalized feature comparison (Grouped Bar chart)
st.subheader("Feature comparison across clusters (0–1 min-max normalised)")

compare_features = {
    "speed_mean": "Mean speed",
    "slow_share": "Slow-step share",
    "speed_cv": "Stop-and-go CV",
    "dwell_s_per_visit": "Dwell / visit",
    "n_visits": "Visits / cell",
    "weekend_share": "Weekend share",
    "night_share": "Night share",
    "poi_per_cell": "POIs / cell",
}

norm_df = merged[["cluster", "label"]].copy()
for col_name in compare_features:
    vals = merged[col_name].astype(float)
    span = vals.max() - vals.min()
    norm_df[col_name] = (vals - vals.min()) / span if span > 0 else 0.5

fig_comp = go.Figure()
cats = list(compare_features.values())
keys = list(compare_features.keys())

for _, row in norm_df.iterrows():
    cid = int(row["cluster"])
    vals = [float(row[k]) for k in keys]
    fig_comp.add_trace(
        go.Bar(
            name=f"C{cid}: {row['label']}",
            x=cats,
            y=vals,
            marker_color=cluster_hex(cid),
        )
    )

fig_comp.update_layout(
    barmode="group",
    yaxis_title="Normalised score (0 = lowest cluster, 1 = highest)",
    template=plotly_template(),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=60, b=40, l=40, r=20),
)

st.plotly_chart(
    fig_comp,
    alt="Normalized comparison of speed, dwell, volume, time-of-week, and POI density across clusters",
)

# 3. POI breakdown per cluster
st.subheader("OpenStreetMap POI density per cluster")
st.caption("Bus stops are excluded because they follow the road network and dominate all clusters.")

poi_cats = [
    c.replace("_per_cell", "")
    for c in pois.columns
    if c.endswith("_per_cell") and c not in ("poi_per_cell", "transport_per_cell")
]

fig_poi = go.Figure()
for _, row in merged.iterrows():
    cid = int(row["cluster"])
    prow = pois[pois["cluster"] == cid].iloc[0]
    fig_poi.add_trace(
        go.Bar(
            name=f"C{cid}: {row['label']}",
            x=[c.replace("_", " ").capitalize() for c in poi_cats],
            y=[float(prow[f"{c}_per_cell"]) for c in poi_cats],
            marker_color=cluster_hex(cid),
        )
    )

fig_poi.update_layout(
    barmode="group",
    yaxis_title="POIs per H3 cell",
    template=plotly_template(),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=60, b=40, l=40, r=20),
)
st.plotly_chart(
    fig_poi,
    alt="OpenStreetMap POI density per H3 cell across behaviour clusters",
)
