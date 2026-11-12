"""Anomalies page: flagged H3 cells (click to highlight on map) and flagged calendar days."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.load import (
    KNOWN_HOLIDAYS,
    OUTPUTS_DIR,
    load_anomalies,
    load_day_anomalies,
    load_json_output,
)
from utils.theme import plotly_template, render_taxi_caveat

anom = load_anomalies(OUTPUTS_DIR)
days = load_day_anomalies(OUTPUTS_DIR)
meta = load_json_output("interpret_meta.json", OUTPUTS_DIR)

st.info(
    "**Anomalies are rule-based candidates for inspection, not proven incidents.** "
    f"Cell rule: `{meta.get('anomaly_rules', {}).get('spatial', '')}`. "
    f"Day rule: `{meta.get('anomaly_rules', {}).get('daily', '')}`.",
    icon=":material/warning:",
)
render_taxi_caveat(compact=True)

# --- Part 1: Flagged cells ---
st.subheader("Flagged H3 cells")
f1, f2 = st.columns([1.2, 1.8], vertical_alignment="bottom")
with f1:
    selected_types = st.pills(
        "Anomaly type",
        options=["behaviour", "rhythm", "both"],
        default=["behaviour", "rhythm", "both"],
        selection_mode="multi",
        key="anom_type_filter",
    )
with f2:
    cluster_opts = sorted(anom["cluster"].unique().tolist())
    selected_clusters = st.multiselect(
        "Filter by behaviour cluster",
        options=cluster_opts,
        default=cluster_opts,
        format_func=lambda c: f"Cluster {c}",
        key="anom_cluster_filter",
    )

filtered_anom = anom[
    anom["anomaly_type"].isin(selected_types) & anom["cluster"].isin(selected_clusters)
].reset_index(drop=True)

display_cols = [
    "h3_cell",
    "cluster_name",
    "anomaly_type",
    "anomaly_z_spatial",
    "anomaly_z_temporal",
    "speed_mean",
    "n_visits",
    "drivers",
]

st.caption("Select any row below to highlight that hexagon on the Cluster map.")
event = st.dataframe(
    filtered_anom[display_cols],
    column_config={
        "h3_cell": st.column_config.TextColumn("H3 cell"),
        "cluster_name": st.column_config.TextColumn("Behaviour cluster"),
        "anomaly_type": st.column_config.TextColumn("Type"),
        "anomaly_z_spatial": st.column_config.NumberColumn("Spatial z", format="%.2f"),
        "anomaly_z_temporal": st.column_config.NumberColumn("Temporal z", format="%.2f"),
        "speed_mean": st.column_config.NumberColumn("Mean speed", format="%.1f km/h"),
        "n_visits": st.column_config.NumberColumn("Visits", format="%d"),
        "drivers": st.column_config.TextColumn("Drivers"),
    },
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="anom_table",
    alt="Sortable and filterable table of flagged anomalous H3 cells",
)

selected_rows = event.selection.rows if hasattr(event, "selection") else []
if selected_rows:
    chosen = filtered_anom.iloc[selected_rows[0]]
    chosen_id = str(chosen["h3_cell"])
    st.session_state["selected_h3_cell"] = chosen_id
    with st.container(horizontal=True, vertical_alignment="center"):
        st.success(f"Selected cell `{chosen_id}` ({chosen['anomaly_type']}): {chosen['drivers']}")
        if st.button("Open on cluster map", icon=":material/map:", type="primary"):
            st.switch_page("pages/2_cluster_map.py")

st.download_button(
    "Download filtered cell anomalies (CSV)",
    data=filtered_anom.to_csv(index=False).encode("utf-8"),
    file_name="fleetsense_cell_anomalies.csv",
    mime="text/csv",
    icon=":material/download:",
)

# --- Part 2: Calendar / Daily anomalies ---
st.divider()
st.subheader("Daily volume & speed anomalies")

signal = st.segmented_control(
    "Z-score signal",
    options=["Volume z-score (quasi-Poisson)", "Speed robust z-score"],
    default="Volume z-score (quasi-Poisson)",
    key="day_anom_signal",
)

y_col = "volume_z" if signal == "Volume z-score (quasi-Poisson)" else "speed_robust_z"
flag_col = "is_anomaly_volume" if signal == "Volume z-score (quasi-Poisson)" else "is_anomaly_speed"

fig_days = go.Figure()
normal_days = days[~days[flag_col]]
flagged_days = days[days[flag_col]]

fig_days.add_trace(
    go.Scatter(
        x=normal_days["date_dt"],
        y=normal_days[y_col],
        mode="lines+markers",
        name="Normal day",
        line=dict(color="#8b949e", width=1.2),
        marker=dict(size=4, color="#8b949e"),
        customdata=normal_days[["n_trips", "median_speed", "day_of_week"]],
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b> (%{customdata[2]})<br>"
            "z: %{y:.2f}<br>Trips: %{customdata[0]} · Median speed: %{customdata[1]} km/h<extra></extra>"
        ),
    )
)

fig_days.add_trace(
    go.Scatter(
        x=flagged_days["date_dt"],
        y=flagged_days[y_col],
        mode="markers",
        name="Flagged day (|z| ≥ 2.5)",
        marker=dict(size=10, color="#DC267F", symbol="diamond"),
        customdata=flagged_days[["n_trips", "median_speed", "anomaly_kind", "holiday"]],
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b> %{customdata[3]}<br>"
            "Flag: %{customdata[2]} · z: %{y:.2f}<br>"
            "Trips: %{customdata[0]} · Median speed: %{customdata[1]} km/h<extra></extra>"
        ),
    )
)

fig_days.add_hline(y=2.5, line_dash="dash", line_color="#FE6100", annotation_text="+2.5")
fig_days.add_hline(y=-2.5, line_dash="dash", line_color="#FE6100", annotation_text="-2.5")

# Annotate holidays present in the dataset
for h_date, h_name in KNOWN_HOLIDAYS.items():
    match = days[days["date"] == h_date]
    if not match.empty:
        r = match.iloc[0]
        fig_days.add_annotation(
            x=r["date_dt"],
            y=float(r[y_col]),
            text=h_name,
            showarrow=True,
            arrowhead=2,
            ay=-28 if float(r[y_col]) >= 0 else 28,
            font=dict(size=11),
        )

fig_days.update_layout(
    xaxis_title="Date",
    yaxis_title="Dispersion-corrected volume z" if y_col == "volume_z" else "Speed robust z",
    template=plotly_template(),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(t=50, b=40, l=40, r=20),
    height=400,
)
st.plotly_chart(
    fig_days,
    alt="Time series of daily z-scores with flagged anomalies and Portuguese holidays annotated",
)

show_only_flagged_days = st.toggle("Show flagged days only", value=True, key="day_anom_only")
day_table = days[days["is_anomaly"]] if show_only_flagged_days else days
st.dataframe(
    day_table[
        [
            "date",
            "day_of_week",
            "holiday",
            "n_trips",
            "median_speed",
            "volume_z",
            "speed_robust_z",
            "anomaly_kind",
            "note",
        ]
    ],
    column_config={
        "date": st.column_config.TextColumn("Date"),
        "day_of_week": st.column_config.TextColumn("Weekday"),
        "holiday": st.column_config.TextColumn("Holiday"),
        "n_trips": st.column_config.NumberColumn("Sampled trips", format="%d"),
        "median_speed": st.column_config.NumberColumn("Median speed", format="%.2f km/h"),
        "volume_z": st.column_config.NumberColumn("Volume z", format="%.2f"),
        "speed_robust_z": st.column_config.NumberColumn("Speed z", format="%.2f"),
        "anomaly_kind": st.column_config.TextColumn("Flagged signal"),
        "note": st.column_config.TextColumn("Note"),
    },
    hide_index=True,
    alt="Table of daily volume and speed anomaly checks",
)

st.download_button(
    "Download day anomalies (CSV)",
    data=day_table.drop(columns=["date_dt"]).to_csv(index=False).encode("utf-8"),
    file_name="fleetsense_day_anomalies.csv",
    mime="text/csv",
    icon=":material/download:",
)

