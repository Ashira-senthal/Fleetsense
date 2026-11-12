"""Main page: interactive H3 hexagon cluster map."""

# from __future__ import annotations

import math

import streamlit as st

from utils.hover_map import render_hover_map
from utils.load import (
    OUTPUTS_DIR,
    load_cell_annotations,
    load_cluster_profiles,
    load_rhythm_names,
)
from utils.theme import (
    cluster_hex,
    cluster_rgb,
    continuous_rgb,
    is_dark_theme,
    rhythm_hex,
    rhythm_rgb,
)

cells = load_cell_annotations(OUTPUTS_DIR)
profiles = load_cluster_profiles(OUTPUTS_DIR)
rhythms = load_rhythm_names(OUTPUTS_DIR)

cluster_short = {int(r["cluster"]): str(r["label"]) for _, r in profiles.iterrows()}
rhythm_labels = {
    int(r["rhythm_cluster"]): f"R{int(r['rhythm_cluster'])}: {r['rhythm_name']}"
    for _, r in rhythms.iterrows()
}

with st.container(border=True):
    # Compact inline toolbar directly above the map
    top_c1, top_c2 = st.columns([2.4, 0.4], vertical_alignment="bottom")
    with top_c1:
        colour_options = ["Behaviour cluster", "Rhythm cluster", "Anomaly score", "Visits"]
        colour_by = st.segmented_control(
            "Colour hexagons by",
            options=colour_options,
            default="Behaviour cluster",
            label_visibility="collapsed",
            key="map_colour_by",
        )
        if colour_by is None:
            colour_by = "Behaviour cluster"

    with top_c2:
        anomalies_only = st.toggle(
            "Show flagged cells",
            value=False,
            key="map_anomalies_only",
        )

    cluster_ids_sorted = sorted(cluster_short.keys())
    pill_css_rules = [
        f"""
        .st-key-map_filter_clusters button:nth-of-type({idx})::before {{
            content: "";
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background-color: {cluster_hex(cid)};
            margin-right: 8px;
            flex-shrink: 0;
        }}
        .st-key-map_filter_clusters button:nth-of-type({idx})[aria-pressed="true"],
        .st-key-map_filter_clusters button:nth-of-type({idx})[data-testid="stBaseButton-pillsActive"] {{
            border-color: {cluster_hex(cid)} !important;
            background-color: {cluster_hex(cid)}22 !important;
        }}
        """
        for idx, cid in enumerate(cluster_ids_sorted, start=1)
    ]
    st.markdown(f"<style>{''.join(pill_css_rules)}</style>", unsafe_allow_html=True)

    selected_clusters = st.pills(
        "Filter behaviour clusters",
        options=cluster_ids_sorted,
        default=cluster_ids_sorted,
        selection_mode="multi",
        format_func=lambda c: cluster_short[c],
        label_visibility="collapsed",
        key="map_filter_clusters",
    )

# Apply filters
filtered = cells[cells["cluster"].isin(selected_clusters)].copy()

if anomalies_only:
    filtered = filtered[filtered["is_anomaly"]].copy()

filtered["slow_pct"] = (filtered["slow_share"] * 100).round(1)
filtered["speed_mean_r"] = filtered["speed_mean"].round(1)
filtered["anomaly_score_r"] = filtered["anomaly_score"].round(2)
filtered["dwell_r"] = filtered["cluster_dwell_s_per_visit"].round(0)
filtered["poi_r"] = filtered["cluster_poi_per_cell"].round(1)

highlight_id = st.session_state.get("selected_h3_cell")

if filtered.empty:
    st.warning("No H3 cells match the current filters. Broaden one or more filters.")
else:
    if colour_by == "Behaviour cluster":
        filtered["fill_color"] = [cluster_rgb(c, alpha=210) for c in filtered["cluster"]]
    elif colour_by == "Rhythm cluster":
        filtered["fill_color"] = [rhythm_rgb(r, alpha=210) for r in filtered["rhythm_cluster"]]
    elif colour_by == "Anomaly score":
        vmin = float(filtered["anomaly_score"].min())
        vmax = float(filtered["anomaly_score"].max())
        filtered["fill_color"] = [
            continuous_rgb(v, vmin, vmax, alpha=215) for v in filtered["anomaly_score"]
        ]
    else:
        log_v = [math.log1p(v) for v in filtered["n_visits"]]
        vmin, vmax = min(log_v), max(log_v)
        filtered["fill_color"] = [continuous_rgb(v, vmin, vmax, alpha=215) for v in log_v]

    if highlight_id and highlight_id in set(filtered["h3_cell"]):
        focus_row = filtered[filtered["h3_cell"] == highlight_id].iloc[0]
        view_lat = float(focus_row["lat_centroid"])
        view_lon = float(focus_row["lon_centroid"])
        view_zoom = 12.6
    else:
        view_lat = float(filtered["lat_centroid"].mean())
        view_lon = float(filtered["lon_centroid"].mean())
        view_zoom = 11.2

    def _row_to_payload(row) -> dict:
        return {
            "h3_cell": str(row["h3_cell"]),
            "cluster_name": str(row["cluster_name"]),
            "rhythm_label": f"R{int(row['rhythm_cluster'])} · {row['rhythm_name']}",
            "speed_mean_r": float(row["speed_mean_r"]),
            "slow_pct": float(row["slow_pct"]),
            "n_visits": int(row["n_visits"]),
            "peak_hour_str": f"{int(row['peak_hour']):02d}:00",
            "dwell_r": float(row["cluster_dwell_s_per_visit"]),
            "poi_r": float(row["cluster_poi_per_cell"]),
            "anomaly_type": str(row["anomaly_type"]),
            "fill_color": row["fill_color"],
        }

    cells_payload = [_row_to_payload(row) for _, row in filtered.iterrows()]
    default_payload = _row_to_payload(filtered.iloc[0])

    def _sync_selected_cell() -> None:
        comp_state = st.session_state.get("fleetsense_hover_map")
        if isinstance(comp_state, dict) and "selected_h3_cell" in comp_state:
            st.session_state["selected_h3_cell"] = comp_state["selected_h3_cell"]

    render_hover_map(
        cells_data=cells_payload,
        default_cell=default_payload,
        selected_h3_cell=highlight_id if highlight_id in set(filtered["h3_cell"]) else None,
        view_lat=view_lat,
        view_lon=view_lon,
        view_zoom=view_zoom,
        is_dark=is_dark_theme(),
        key="fleetsense_hover_map",
        on_selected_h3_cell_change=_sync_selected_cell,
    )

_, legend_col = st.columns([2.15, 1], gap="medium")
with legend_col:
    # Compact legend card below detail
    with st.container(border=True):
        st.markdown(f"**Legend** · {len(filtered)} / {len(cells)} cells")
        if colour_by == "Behaviour cluster":
            counts = filtered["cluster"].value_counts().to_dict() if not filtered.empty else {}
            for cid in sorted(cluster_short.keys()):
                hex_c = cluster_hex(cid)
                st.markdown(
                    f"<span style='display:inline-block;width:10px;height:10px;"
                    f"background:{hex_c};border-radius:50%;margin-right:6px;'></span>"
                    f"**C{cid}** ({counts.get(cid, 0)}): {cluster_short[cid]}",
                    unsafe_allow_html=True,
                )
        elif colour_by == "Rhythm cluster":
            counts = filtered["rhythm_cluster"].value_counts().to_dict() if not filtered.empty else {}
            for rid in sorted(rhythm_labels.keys()):
                hex_c = rhythm_hex(rid)
                st.markdown(
                    f"<span style='display:inline-block;width:10px;height:10px;"
                    f"background:{hex_c};border-radius:50%;margin-right:6px;'></span>"
                    f"**R{rid}** ({counts.get(rid, 0)}): {rhythm_labels[rid].split(': ', 1)[1]}",
                    unsafe_allow_html=True,
                )
        else:
            st.caption(f"Viridis scale by **{colour_by}** (purple = low, yellow = high).")

        if not filtered.empty:
            export_cols = [
                "h3_cell",
                "cluster",
                "cluster_name",
                "rhythm_cluster",
                "rhythm_name",
                "anomaly_type",
                "is_anomaly",
                "anomaly_z_spatial",
                "anomaly_z_temporal",
                "drivers",
                "lat_centroid",
                "lon_centroid",
                "peak_hour",
                "speed_mean",
                "slow_share",
                "n_visits",
            ]
            st.download_button(
                "Export filtered cells (CSV)",
                data=filtered[export_cols].to_csv(index=False).encode("utf-8"),
                file_name="fleetsense_filtered_cells.csv",
                mime="text/csv",
                icon=":material/download:",
                width="stretch",
            )
