"""Cluster profiles, normalized feature comparison, POI breakdown, and cell comparator."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.load import (
    OUTPUTS_DIR,
    load_cell_annotations,
    load_cluster_pois,
    load_cluster_profiles,
    load_hourly_profile,
)
from utils.theme import cluster_hex, plotly_template

profiles = load_cluster_profiles(OUTPUTS_DIR)
pois = load_cluster_pois(OUTPUTS_DIR)
cells = load_cell_annotations(OUTPUTS_DIR)
hourly = load_hourly_profile(OUTPUTS_DIR)

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

# 2. Normalized feature comparison (Radar / Grouped Bar toggle)
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

chart_style = st.segmented_control(
    "Chart type",
    options=["Radar chart", "Grouped bar chart"],
    default="Radar chart",
    key="profile_chart_style",
)

fig_comp = go.Figure()
cats = list(compare_features.values())
keys = list(compare_features.keys())

for _, row in norm_df.iterrows():
    cid = int(row["cluster"])
    vals = [float(row[k]) for k in keys]
    if chart_style == "Grouped bar chart":
        fig_comp.add_trace(
            go.Bar(
                name=f"C{cid}: {row['label']}",
                x=cats,
                y=vals,
                marker_color=cluster_hex(cid),
            )
        )
    else:
        fig_comp.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=cats + [cats[0]],
                fill="toself",
                name=f"C{cid}: {row['label']}",
                line_color=cluster_hex(cid),
                opacity=0.75,
            )
        )

if chart_style == "Grouped bar chart":
    fig_comp.update_layout(
        barmode="group",
        yaxis_title="Normalised score (0 = lowest cluster, 1 = highest)",
        template=plotly_template(),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=60, b=40, l=40, r=20),
    )
else:
    fig_comp.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        template=plotly_template(),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="left", x=0),
        margin=dict(t=40, b=90, l=40, r=40),
    )

st.plotly_chart(
    fig_comp,
    alt="Normalized comparison of speed, dwell, volume, time-of-week, and POI density across clusters",
)

# 3. POI breakdown per cluster
st.subheader("OpenStreetMap POI density per cluster")
include_transport = st.toggle(
    "Include transport POIs (bus stops follow the road network and dominate all clusters)",
    value=False,
    key="poi_include_transport",
)

poi_cats = [
    c.replace("_per_cell", "")
    for c in pois.columns
    if c.endswith("_per_cell") and c != "poi_per_cell"
]
if not include_transport:
    poi_cats = [c for c in poi_cats if c != "transport"]

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

with st.container(horizontal=True):
    st.download_button(
        "Download cluster profiles (CSV)",
        data=profiles.to_csv(index=False).encode("utf-8"),
        file_name="cluster_profiles.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    st.download_button(
        "Download cluster POIs (CSV)",
        data=pois.to_csv(index=False).encode("utf-8"),
        file_name="cluster_pois.csv",
        mime="text/csv",
        icon=":material/download:",
    )

# 4. Extra: Compare two cells side by side
st.divider()
st.subheader("Compare two H3 cells side by side")
st.caption(
    "Compare measured cell features directly; hourly curves below show each cell's "
    "assigned behaviour cluster mean profile from `cluster_hourly_profile.parquet`."
)

cell_ids = sorted(cells["h3_cell"].tolist())
sel_a, sel_b = st.columns(2)
with sel_a:
    cell_a_id = st.selectbox("Cell A", options=cell_ids, index=0, key="compare_cell_a")
with sel_b:
    default_b = min(1, len(cell_ids) - 1)
    cell_b_id = st.selectbox("Cell B", options=cell_ids, index=default_b, key="compare_cell_b")

row_a = cells[cells["h3_cell"] == cell_a_id].iloc[0]
row_b = cells[cells["h3_cell"] == cell_b_id].iloc[0]

comp_table = pd.DataFrame(
    {
        "Metric": [
            "Behaviour cluster",
            "Rhythm cluster",
            "Mean speed (km/h)",
            "Slow-step share (<12 km/h)",
            "Distinct visits",
            "Peak hour",
            "Spatial anomaly z",
            "Temporal anomaly z",
            "Anomaly drivers",
        ],
        f"Cell A ({cell_a_id})": [
            row_a["cluster_name"],
            f"R{int(row_a['rhythm_cluster'])}: {row_a['rhythm_name']}",
            f"{row_a['speed_mean']:.1f}",
            f"{row_a['slow_share']:.1%}",
            f"{int(row_a['n_visits']):,}",
            f"{int(row_a['peak_hour']):02d}:00",
            f"{row_a['anomaly_z_spatial']:.2f}",
            f"{row_a['anomaly_z_temporal']:.2f}",
            row_a["drivers"] or "—",
        ],
        f"Cell B ({cell_b_id})": [
            row_b["cluster_name"],
            f"R{int(row_b['rhythm_cluster'])}: {row_b['rhythm_name']}",
            f"{row_b['speed_mean']:.1f}",
            f"{row_b['slow_share']:.1%}",
            f"{int(row_b['n_visits']):,}",
            f"{int(row_b['peak_hour']):02d}:00",
            f"{row_b['anomaly_z_spatial']:.2f}",
            f"{row_b['anomaly_z_temporal']:.2f}",
            row_b["drivers"] or "—",
        ],
    }
)
st.dataframe(comp_table, hide_index=True, alt="Side-by-side comparison of two H3 cells")

hours = list(range(168))
fig_cell_how = go.Figure()
for label, r in [(f"Cell A ({cell_a_id})", row_a), (f"Cell B ({cell_b_id})", row_b)]:
    cid = int(r["cluster"])
    if cid in hourly.index:
        vals = hourly.loc[cid, hours].to_numpy(dtype=float)
        fig_cell_how.add_trace(
            go.Scatter(
                x=hours,
                y=vals,
                mode="lines",
                name=f"{label} · C{cid}",
                line=dict(width=2, color=cluster_hex(cid)),
            )
        )

fig_cell_how.update_layout(
    xaxis_title="Hour of week (0 = Mon 00:00)",
    yaxis_title="Cluster mean pings / cell / hour",
    template=plotly_template(),
    margin=dict(t=30, b=40, l=40, r=20),
)
st.plotly_chart(
    fig_cell_how,
    alt="168-hour cluster activity profiles for the two selected cells",
)

