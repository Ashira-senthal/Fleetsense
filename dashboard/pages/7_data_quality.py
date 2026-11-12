"""Data quality, cleaning fixture report, Delhi GTFS audit, and privacy/limitations."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.load import OUTPUTS_DIR, load_json_output, load_privacy_doc

cleaning = load_json_output("cleaning_report.json", OUTPUTS_DIR)
audit = load_json_output("data_audit.json", OUTPUTS_DIR)
privacy_md = load_privacy_doc()

# --- Section 1: Cleaning report ---
st.subheader("Cleaning rules & synthetic regression report")
st.warning(
    f"**Provenance note:** `outputs/cleaning_report.json` has `synthetic={cleaning.get('synthetic')}` "
    f"(`data_source={cleaning.get('data_source')}`). It is a labelled regression test fixture "
    "used to score cleaning recall on injected defects; the Porto taxi pipeline applied the same "
    "100 km/h corroborated-jump and bounding-box rules.",
    icon=":material/info:",
)

rep = cleaning.get("report", {})
thresholds = rep.get("thresholds", {})
bbox = thresholds.get("bbox", {})

c1, c2, c3, c4 = st.columns(4)
c1.metric("Max speed threshold", f"{thresholds.get('max_speed_kmh', 0):.0f} km/h", border=True, height="stretch")
c2.metric("Bbox margin", f"{bbox.get('margin_deg', 0):.2f}°", border=True, height="stretch")
c3.metric("Fixture rows in", f"{rep.get('rows_in', 0):,}", border=True, height="stretch")
c4.metric("Fixture removed", f"{rep.get('pct_removed', 0):.2f}%", border=True, height="stretch")

removal_df = pd.DataFrame(
    [
        {"Reason": "Exact duplicate pings", "Rows removed": rep.get("dropped_exact_duplicate_pings", 0)},
        {"Reason": "Conflicting timestamps", "Rows removed": rep.get("dropped_conflicting_timestamps", 0)},
        {"Reason": "Outside bounding box", "Rows removed": rep.get("dropped_outside_bbox", 0)},
        {"Reason": "Corroborated impossible jumps", "Rows removed": rep.get("dropped_impossible_jumps", 0)},
        {"Reason": "Missing required fields", "Rows removed": rep.get("dropped_missing", 0)},
        {"Reason": "Non-positive timestep", "Rows removed": rep.get("dropped_nonpositive_timestep", 0)},
    ]
)

col_rem, col_def = st.columns(2)
with col_rem:
    st.markdown("**Fixture removals by rule**")
    st.dataframe(removal_df, hide_index=True, alt="Rows removed by cleaning rule on the synthetic fixture")

with col_def:
    st.markdown("**Injected-defect scoring**")
    scoring = cleaning.get("known_defect_scoring", {})
    score_rows = []
    for name, vals in scoring.items():
        score_rows.append(
            {
                "Defect type": name.replace("defect_", ""),
                "Injected": vals.get("injected", 0),
                "Recall": vals.get("recall", None),
                "Notes": vals.get("note", f"survived: {vals.get('survived_cleaning', vals.get('duplicate_keys_remaining_in_clean', '-'))}"),
            }
        )
    st.dataframe(
        pd.DataFrame(score_rows),
        column_config={"Recall": st.column_config.NumberColumn(format="%.3f")},
        hide_index=True,
        alt="Recall scores against injected defects in the synthetic cleaning fixture",
    )

# --- Section 2: Data audit (Delhi GTFS fallback) ---
st.divider()
st.subheader("Stage 1 data audit — failed Delhi GTFS attempt")
st.info(
    f"**Dataset audited:** {audit.get('label', 'Delhi OTD')} (`synthetic={audit.get('synthetic')}`, "
    f"audited `{audit.get('audited_at_utc', '')}`). "
    "The full official feed host refused connections and the Transitland mirror was empty, "
    "so the pipeline fell back to Porto taxi GPS (UCI 339) per the pre-agreed fallback rule.",
    icon=":material/info:",
)

integrity = audit.get("integrity", {})
a1, a2, a3, a4 = st.columns(4)
a1.metric("Routes", f"{integrity.get('trips_per_route', {}).get('n_routes', 0):,}", border=True, height="stretch")
a2.metric("Routes with 1 trip", f"{integrity.get('trips_per_route', {}).get('routes_with_one_trip', 0):,}", border=True, height="stretch")
a3.metric("Stops", f"{audit.get('stops', {}).get('n_stops', 0):,}", border=True, height="stretch")
a4.metric("Arrival/departure times", "Absent" if not integrity.get("has_arrival_departure_times") else "Present", border=True, height="stretch")

st.markdown("**Audit findings (`outputs/data_audit.json`)**")
for finding in audit.get("findings", []):
    st.markdown(f"- {finding}")

tables = audit.get("tables", {})
if tables:
    table_summary = pd.DataFrame(
        [
            {
                "GTFS table": tname,
                "Rows": tinfo.get("rows", 0),
                "Columns": tinfo.get("n_columns", 0),
                "Duplicate rows": tinfo.get("duplicate_rows", 0),
            }
            for tname, tinfo in tables.items()
        ]
    )
    st.dataframe(table_summary, hide_index=True, alt="Summary of tables in the audited Delhi trimmed GTFS feed")

# --- Section 3: Privacy and limitations ---
st.divider()
st.subheader("Privacy & limitations (`docs/PRIVACY_AND_LIMITATIONS.md`)")
if privacy_md:
    with st.container(border=True):
        st.markdown(privacy_md)
else:
    st.warning("`docs/PRIVACY_AND_LIMITATIONS.md` was not found in the repository.")

