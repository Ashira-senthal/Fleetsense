"""Weekly rhythm explorer: 168-hour profiles and behaviour x rhythm crosstab."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.load import (
    OUTPUTS_DIR,
    load_cluster_profiles,
    load_hourly_profile,
    load_rhythm_crosstab,
    load_rhythm_names,
)
from utils.theme import cluster_hex, plotly_template, rhythm_hex

hourly = load_hourly_profile(OUTPUTS_DIR)
profiles = load_cluster_profiles(OUTPUTS_DIR)
rhythms = load_rhythm_names(OUTPUTS_DIR)
crosstab = load_rhythm_crosstab(OUTPUTS_DIR)

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOUR_LABELS = [f"{DAYS[h // 24]} {h % 24:02d}:00" for h in range(168)]

cluster_names = {int(r["cluster"]): f"C{int(r['cluster'])}: {r['label']}" for _, r in profiles.iterrows()}

# Controls
ctrl1, ctrl2, ctrl3 = st.columns([1.4, 1.8, 1], vertical_alignment="bottom")
with ctrl1:
    day_window = st.segmented_control(
        "Time window",
        options=["Full week (168h)", "Weekdays (Mon–Fri)", "Weekend (Sat–Sun)"],
        default="Full week (168h)",
        key="rhythm_day_window",
    )
    if day_window is None:
        day_window = "Full week (168h)"

with ctrl2:
    selected_clusters = st.pills(
        "Clusters to display",
        options=sorted(cluster_names.keys()),
        default=sorted(cluster_names.keys()),
        selection_mode="multi",
        format_func=lambda c: f"C{c}",
        key="rhythm_clusters_select",
    )

with ctrl3:
    normalise_curves = st.toggle(
        "Normalise each curve (share of peak)",
        value=True,
        help="Cluster 1 averages ~175 pings/cell/h vs ~5 for others; normalising reveals shape differences.",
        key="rhythm_normalise",
    )

if day_window == "Weekdays (Mon–Fri)":
    hour_range = list(range(0, 120))
elif day_window == "Weekend (Sat–Sun)":
    hour_range = list(range(120, 168))
else:
    hour_range = list(range(0, 168))

if not selected_clusters:
    st.info("Select at least one cluster above to plot its weekly profile.")
else:
    view_mode = st.segmented_control(
        "Display as",
        options=["Line chart", "Heatmap"],
        default="Line chart",
        key="rhythm_view_mode",
    )

    if view_mode == "Heatmap":
        z_rows = []
        y_labels = []
        for cid in sorted(selected_clusters):
            vals = hourly.loc[cid, hour_range].to_numpy(dtype=float)
            if normalise_curves and vals.max() > 0:
                vals = vals / vals.max()
            z_rows.append(vals)
            y_labels.append(cluster_names[cid])

        fig_hm = go.Figure(
            data=go.Heatmap(
                z=z_rows,
                x=[HOUR_LABELS[h] for h in hour_range],
                y=y_labels,
                colorscale="Viridis",
                colorbar_title="Share of peak" if normalise_curves else "Pings/cell/h",
            )
        )
        fig_hm.update_layout(
            xaxis_title="Day and hour",
            template=plotly_template(),
            margin=dict(t=30, b=50, l=40, r=20),
            height=340,
        )
        st.plotly_chart(
            fig_hm,
            alt="Heatmap of hourly taxi activity by behaviour cluster across the selected week window",
        )
    else:
        fig_line = go.Figure()
        for cid in sorted(selected_clusters):
            vals = hourly.loc[cid, hour_range].to_numpy(dtype=float)
            if normalise_curves and vals.max() > 0:
                vals = vals / vals.max()
            fig_line.add_trace(
                go.Scatter(
                    x=[HOUR_LABELS[h] for h in hour_range],
                    y=vals,
                    mode="lines",
                    name=cluster_names[cid],
                    line=dict(color=cluster_hex(cid), width=2.2),
                )
            )
        fig_line.update_layout(
            xaxis_title="Day and hour",
            yaxis_title="Relative activity (peak = 1.0)" if normalise_curves else "Mean pings / cell / hour",
            template=plotly_template(),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            margin=dict(t=60, b=50, l=40, r=20),
            height=400,
        )
        st.plotly_chart(
            fig_line,
            alt="Hourly taxi activity lines by behaviour cluster across the selected week window",
        )

# Rhythm cluster cards from TS-K-Means
st.subheader("Weekly rhythm clusters (TS-K-Means, separate label set)")
r_cols = st.columns(2)
for idx, (_, rrow) in enumerate(rhythms.iterrows()):
    rid = int(rrow["rhythm_cluster"])
    with r_cols[idx % 2].container(border=True):
        st.markdown(
            f"<span style='display:inline-block;width:12px;height:12px;"
            f"background:{rhythm_hex(rid)};border-radius:2px;margin-right:6px;'></span>"
            f"**Rhythm {rid} — {rrow['rhythm_name']}**",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Peak hour", f"{int(rrow['peak_hour']):02d}:00")
        c2.metric("Weekend share", f"{rrow['weekend_share']:.1%}")
        c3.metric("Weekly activity", f"{rrow['mean_total_activity']:,.0f}")

# Behaviour x Rhythm crosstab heatmap
st.subheader("Behaviour × Rhythm crosstab")
st.caption(
    "Cell counts crossing K-Means behaviour clusters (rows) with TS-K-Means rhythm clusters "
    "(columns). Cluster 1 sits 100% in Rhythm 0, while Cluster 2 splits across all four rhythms."
)

rhythm_cols = [c for c in crosstab.columns if c != "kmeans_cluster"]
ct_matrix = crosstab.set_index("kmeans_cluster")[rhythm_cols]
row_labels = [cluster_names.get(int(cid), f"C{cid}") for cid in ct_matrix.index]
col_labels = [
    f"R{int(c)}: {rhythms.set_index('rhythm_cluster').loc[int(c), 'rhythm_name'][:28]}…"
    for c in rhythm_cols
]

fig_ct = px.imshow(
    ct_matrix.to_numpy(),
    x=col_labels,
    y=row_labels,
    text_auto=True,
    color_continuous_scale="Blues",
    labels=dict(x="Weekly rhythm cluster", y="Behaviour cluster", color="Cells"),
)
fig_ct.update_layout(
    template=plotly_template(),
    margin=dict(t=20, b=40, l=20, r=20),
    height=340,
)
st.plotly_chart(
    fig_ct,
    alt="Annotated heatmap crossing behaviour clusters with weekly rhythm clusters",
)

with st.container(horizontal=True):
    st.download_button(
        "Download crosstab (CSV)",
        data=crosstab.to_csv(index=False).encode("utf-8"),
        file_name="cluster_rhythm_crosstab.csv",
        mime="text/csv",
        icon=":material/download:",
    )
    st.download_button(
        "Download rhythm cluster names (CSV)",
        data=rhythms.to_csv(index=False).encode("utf-8"),
        file_name="cluster_rhythm_names.csv",
        mime="text/csv",
        icon=":material/download:",
    )

