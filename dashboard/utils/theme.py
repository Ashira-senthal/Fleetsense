"""Shared colourblind-safe palettes, theme helpers, and recurring UI caveats."""

from __future__ import annotations

import streamlit as st

# Colourblind-safe IBM / Okabe-Ito inspired palette for the 4 behaviour clusters.
BEHAVIOUR_HEX: dict[int, str] = {
    0: "#648FFF",  # Blue — Slow low-traffic pockets (weekend-heavy)
    1: "#FE6100",  # Orange — Busy mid-speed corridors (all-day, hospitality-dense)
    2: "#DC267F",  # Magenta — Severe stop-and-go pockets (all-day)
    3: "#009E73",  # Bluish-green — Quiet free-flowing fringe (weekend-heavy)
}

# Distinct colourblind-safe palette for the 4 TS-K-Means weekly rhythm clusters.
RHYTHM_HEX: dict[int, str] = {
    0: "#0072B2",  # Deep blue
    1: "#E69F00",  # Amber
    2: "#785EF0",  # Violet
    3: "#CC79A7",  # Rose
}

FALLBACK_HEX = "#6E7781"

TAXI_CAVEAT = (
    "**Scheduled GTFS, not live GPS:** these are Delhi NCR bus schedules "
    "(DIMTS / DTC). Speeds and dwell reflect scheduled stop-to-stop timings, "
    "not real-time traffic delays or passenger counts."
)


def hex_to_rgb(hex_color: str, alpha: int = 215) -> list[int]:
    """Convert a #RRGGBB hex string into an [R, G, B, A] list for pydeck."""
    h = hex_color.lstrip("#")
    return [int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha]


def cluster_hex(cluster_id: int) -> str:
    """Return consistent colourblind-safe hex colour for a behaviour cluster id."""
    return BEHAVIOUR_HEX.get(int(cluster_id), FALLBACK_HEX)


def cluster_rgb(cluster_id: int, alpha: int = 210) -> list[int]:
    """Return RGBA list for a behaviour cluster id."""
    return hex_to_rgb(cluster_hex(cluster_id), alpha=alpha)


def rhythm_hex(rhythm_id: int) -> str:
    """Return consistent colourblind-safe hex colour for a rhythm cluster id."""
    return RHYTHM_HEX.get(int(rhythm_id), FALLBACK_HEX)


def rhythm_rgb(rhythm_id: int, alpha: int = 210) -> list[int]:
    """Return RGBA list for a rhythm cluster id."""
    return hex_to_rgb(rhythm_hex(rhythm_id), alpha=alpha)


def continuous_rgb(value: float, vmin: float, vmax: float, alpha: int = 210) -> list[int]:
    """Map a scalar value onto a colourblind-safe Viridis-like 4-stop ramp."""
    if vmax <= vmin:
        t = 0.5
    else:
        t = max(0.0, min(1.0, (float(value) - vmin) / (vmax - vmin)))
    # Viridis-inspired colourblind-safe stops: dark purple -> teal -> green -> yellow
    stops = [
        (68, 1, 84),
        (49, 104, 142),
        (53, 183, 121),
        (253, 231, 37),
    ]
    scaled = t * (len(stops) - 1)
    idx = min(int(scaled), len(stops) - 2)
    frac = scaled - idx
    r0, g0, b0 = stops[idx]
    r1, g1, b1 = stops[idx + 1]
    return [
        int(r0 + (r1 - r0) * frac),
        int(g0 + (g1 - g0) * frac),
        int(b0 + (b1 - b0) * frac),
        alpha,
    ]


def is_dark_theme() -> bool:
    """Detect whether Streamlit's active UI theme is dark."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def plotly_template() -> str:
    """Return Plotly template name matching the active Streamlit theme."""
    return "plotly_dark" if is_dark_theme() else "plotly_white"


def render_taxi_caveat(compact: bool = False) -> None:
    """Render the mandatory data-limitation notice."""
    if compact:
        st.caption(
            ":material/directions_bus: **Scheduled GTFS, not live GPS:** no real-time delays "
            "or passenger counts. Cluster names are hypotheses, not ground truth."
        )
    else:
        st.warning(TAXI_CAVEAT, icon=":material/directions_bus:")


def render_jargon_popover() -> None:
    """Render a plain-language glossary popover explaining H3, DBCV, and ARI."""
    with st.popover("Glossary: H3, DBCV, ARI", icon=":material/help:"):
        st.markdown(
            """
            * **H3 (resolution 8):** Uber's hexagonal spatial grid. Each cell covers
              roughly **0.74 km²** (~460 m edge), avoiding arbitrary rectangular grid artifacts.
            * **DBCV (Density-Based Clustering Validation):** Measures cluster quality for
              density-based methods (DBSCAN/HDBSCAN) without assuming spherical shapes
              (`-1` to `+1`; higher is better).
            * **ARI (Adjusted Rand Index):** Measures how consistently two clusterings
              agree across bootstrap resamples (`0` = random, `1` = identical).
            * **Robust z-score:** `(value − median) / (1.4826 × MAD)`. Uses median and MAD
              instead of mean and SD so extreme outliers do not distort the baseline.
            """
        )

