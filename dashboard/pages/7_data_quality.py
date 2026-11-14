"""Data quality, cleaning rules, fallback note, and privacy/limitations."""

from __future__ import annotations

import streamlit as st

from utils.load import OUTPUTS_DIR, load_json_output, load_privacy_doc

cleaning = load_json_output("cleaning_report.json", OUTPUTS_DIR)
privacy_md = load_privacy_doc()

# --- Section 1: Cleaning rules ---
st.subheader("Cleaning rules")
rep = cleaning.get("report", {})
thresholds = rep.get("thresholds", {})
bbox = thresholds.get("bbox", {})

c1, c2 = st.columns(2)
c1.metric(
    "Max speed threshold",
    f"{thresholds.get('max_speed_kmh', 0):.0f} km/h",
    border=True,
    height="stretch",
)
c2.metric(
    "Bounding-box margin",
    f"{bbox.get('margin_deg', 0):.2f}°",
    border=True,
    height="stretch",
)
st.caption(
    "Corroborated-jump rule: consecutive pings implying speeds above the threshold are "
    "dropped only when corroborated by surrounding points, across up to "
    f"{thresholds.get('max_jump_passes', 5)} passes."
)

# --- Section 2: Dataset fallback note ---
st.divider()
st.info(
    "The project first targeted Delhi bus GTFS, the feed host refused connections and the "
    "mirror was empty, so it fell back to Porto taxi GPS; details are in `outputs/data_audit.json`.",
    icon=":material/info:",
)

# --- Section 3: Privacy and limitations ---
st.divider()
st.subheader("Privacy & limitations (`docs/PRIVACY_AND_LIMITATIONS.md`)")
if privacy_md:
    with st.container(border=True):
        st.markdown(privacy_md)
else:
    st.warning("`docs/PRIVACY_AND_LIMITATIONS.md` was not found in the repository.")
