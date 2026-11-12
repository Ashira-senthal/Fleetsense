"""Custom Component v2 (CCv2) for the interactive H3 map with live hover cell details."""

from __future__ import annotations

from collections.abc import Callable

import streamlit as st

_HTML = """
<div class="fs-layout">
  <div class="fs-map-card">
    <div id="fs-map" class="fs-map-canvas"></div>
    <div id="fs-loading" class="fs-loading">Loading interactive map…</div>
  </div>
  <div class="fs-detail-card">
    <div class="fs-detail-header">
      <h3 class="fs-cell-title">Cell <code id="fs-cell-id" class="fs-cell-code"></code></h3>
      <div id="fs-cell-status" class="fs-cell-status"></div>
    </div>
    <table class="fs-kv-table">
      <tbody id="fs-kv-body"></tbody>
    </table>
    <button id="fs-clear-btn" type="button" class="fs-clear-btn" style="display:none;">
      Clear cell selection
    </button>
  </div>
</div>
"""

_CSS = """
.fs-layout {
  display: grid;
  grid-template-columns: 2.15fr 1fr;
  gap: 1rem;
  align-items: start;
  width: 100%;
  font-family: var(--st-font, system-ui, -apple-system, sans-serif);
  color: var(--st-text-color, #fafafa);
}

@media (max-width: 900px) {
  .fs-layout {
    grid-template-columns: 1fr;
  }
}

.fs-map-card {
  position: relative;
  height: 540px;
  border-radius: var(--st-base-radius, 0.5rem);
  overflow: hidden;
  border: 1px solid var(--st-border-color, rgba(250, 250, 250, 0.15));
  background: var(--st-secondary-background-color, #111827);
}

.fs-map-canvas {
  width: 100%;
  height: 100%;
}

.fs-loading {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 0.9rem;
  opacity: 0.75;
  pointer-events: none;
}

.fs-detail-card {
  background: var(--st-background-color, transparent);
  border: 1px solid var(--st-border-color, rgba(250, 250, 250, 0.15));
  border-radius: var(--st-base-radius, 0.5rem);
  padding: 1rem 1.1rem;
  box-sizing: border-box;
}

.fs-detail-header {
  margin-bottom: 0.65rem;
}

.fs-cell-title {
  margin: 0 0 0.25rem 0;
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--st-heading-color, var(--st-text-color, #fafafa));
  font-family: var(--st-heading-font, var(--st-font, sans-serif));
  display: flex;
  align-items: center;
  gap: 0.45rem;
  flex-wrap: wrap;
}

.fs-cell-code {
  font-family: var(--st-code-font, monospace);
  font-size: 0.95rem;
  background: var(--st-code-background-color, rgba(128, 128, 128, 0.16));
  color: var(--st-code-text-color, #4ade80);
  padding: 0.15rem 0.45rem;
  border-radius: 4px;
}

.fs-cell-status {
  font-size: 0.8rem;
  opacity: 0.72;
  line-height: 1.35;
}

.fs-kv-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.875rem;
}

.fs-kv-table tr {
  border-top: 1px solid var(--st-border-color-light, rgba(128, 128, 128, 0.18));
}

.fs-kv-table tr:first-child {
  border-top: none;
}

.fs-kv-table td {
  padding: 0.48rem 0.25rem;
  vertical-align: top;
}

.fs-kv-key {
  opacity: 0.72;
  width: 44%;
  padding-right: 0.6rem;
}

.fs-kv-val {
  font-weight: 500;
  word-break: break-word;
}

.fs-drivers-box {
  margin-top: 0.75rem;
  padding: 0.65rem 0.8rem;
  border-radius: var(--st-base-radius, 0.5rem);
  background: rgba(234, 179, 8, 0.14);
  border: 1px solid rgba(234, 179, 8, 0.35);
  font-size: 0.83rem;
  line-height: 1.4;
  display: flex;
  gap: 0.5rem;
  align-items: flex-start;
}

.fs-drivers-icon {
  flex-shrink: 0;
}

.fs-clear-btn {
  margin-top: 0.75rem;
  width: 100%;
  padding: 0.45rem 0.75rem;
  border-radius: var(--st-button-radius, 0.5rem);
  border: 1px solid var(--st-widget-border-color, rgba(128, 128, 128, 0.35));
  background: var(--st-secondary-background-color, transparent);
  color: var(--st-text-color, inherit);
  font-family: inherit;
  font-size: 0.85rem;
  cursor: pointer;
}

.fs-clear-btn:hover {
  border-color: var(--st-primary-color, #ff4b4b);
}
"""

_JS = """
const loadScript = (src) =>
  new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) {
      if (existing.dataset.loaded === "1") {
        resolve();
        return;
      }
      existing.addEventListener("load", () => resolve(), { once: true });
      existing.addEventListener("error", reject, { once: true });
      return;
    }
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.onload = () => {
      s.dataset.loaded = "1";
      resolve();
    };
    s.onerror = reject;
    document.head.appendChild(s);
  });

const ensureLibs = () => {
  if (!window.__fsLibsPromise) {
    window.__fsLibsPromise = (async () => {
      await loadScript("https://unpkg.com/h3-js@3.7.2/dist/h3-js.umd.js");
      await loadScript("https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js");
      await loadScript("https://unpkg.com/deck.gl@8.9.35/dist.min.js");
    })();
  }
  return window.__fsLibsPromise;
};

const ensureMapLibreCss = (parentElement) => {
  const id = "fs-maplibre-css";
  if (parentElement.querySelector(`#${id}`)) return;
  const link = document.createElement("link");
  link.id = id;
  link.rel = "stylesheet";
  link.href = "https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css";
  parentElement.appendChild(link);
};

const escapeHtml = (str) =>
  String(str ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

export default function (component) {
  const { data, parentElement, setStateValue } = component;
  if (!data) return;

  ensureMapLibreCss(parentElement);

  const mapEl = parentElement.querySelector("#fs-map");
  const loadingEl = parentElement.querySelector("#fs-loading");
  const cellIdEl = parentElement.querySelector("#fs-cell-id");
  const statusEl = parentElement.querySelector("#fs-cell-status");
  const kvBodyEl = parentElement.querySelector("#fs-kv-body");
  const clearBtnEl = parentElement.querySelector("#fs-clear-btn");

  if (!mapEl || !kvBodyEl) return;

  if (!parentElement.__fsState) {
    parentElement.__fsState = {
      deck: null,
      hoveredId: null,
      pinnedId: data.selected_h3_cell || null,
      lastTheme: null,
      cellsById: new Map(),
      latestData: data,
    };
  }

  const state = parentElement.__fsState;
  state.latestData = data;
  state.pinnedId = data.selected_h3_cell || null;
  state.cellsById = new Map((data.cells || []).map((c) => [c.h3_cell, c]));

  const renderDetailCard = (cell, mode) => {
    if (!cell) return;
    cellIdEl.textContent = cell.cell_num != null ? `#${cell.cell_num}` : (cell.h3_cell || "");

    if (mode === "hover") {
      statusEl.textContent = state.pinnedId
        ? "Hovering preview · Click hexagon to pin"
        : "Hovering · Click hexagon to pin";
    } else if (mode === "pinned") {
      statusEl.textContent = "Pinned cell · Hover any hexagon to preview";
    } else {
      statusEl.textContent =
        "Showing sample cell. Hover or click any hexagon on the map, or use Filters & search.";
    }

    const rows = [
      ["Behaviour cluster", cell.cluster_name],
      ["Rhythm", cell.rhythm_label],
      ["Mean speed", `${Number(cell.speed_mean_r).toFixed(1)} km/h`],
      ["Slow steps (<12 km/h)", `${Number(cell.slow_pct).toFixed(0)}%`],
      ["Visits (sample)", Number(cell.n_visits).toLocaleString()],
      ["Peak hour", cell.peak_hour_str],
      ["Dwell per visit (cluster)", `${Number(cell.dwell_r).toFixed(1)} s`],
      ["POIs per cell (cluster)", `${Number(cell.poi_r).toFixed(1)}`],
      ["Flag", cell.anomaly_type],
    ];
    if (state.latestData.use_how) {
      rows.push([
        `Activity (${state.latestData.how_label})`,
        `${Number(cell.cluster_hour_pings).toFixed(1)} pings/cell/h`,
      ]);
    }

    kvBodyEl.innerHTML = rows
      .map(
        ([k, v]) =>
          `<tr><td class="fs-kv-key">${escapeHtml(k)}</td><td class="fs-kv-val">${escapeHtml(v)}</td></tr>`
      )
      .join("");

    clearBtnEl.style.display = state.pinnedId ? "block" : "none";
  };

  const getActiveCellAndMode = () => {
    if (state.hoveredId && state.cellsById.has(state.hoveredId)) {
      return [state.cellsById.get(state.hoveredId), "hover"];
    }
    if (state.pinnedId && state.cellsById.has(state.pinnedId)) {
      return [state.cellsById.get(state.pinnedId), "pinned"];
    }
    return [state.latestData.default_cell, "sample"];
  };

  const buildLayer = () => {
    const d = state.latestData;
    const highlightId = state.hoveredId || state.pinnedId;
    return new window.deck.H3HexagonLayer({
      id: "fleetsense-h3-hexagons",
      data: d.cells || [],
      pickable: true,
      stroked: true,
      filled: true,
      extruded: false,
      getHexagon: (obj) => obj.h3_cell,
      getFillColor: (obj) => obj.fill_color,
      getLineColor: (obj) =>
        obj.h3_cell === highlightId ? [255, 255, 255, 255] : [15, 23, 42, 110],
      getLineWidth: (obj) => (obj.h3_cell === highlightId ? 45 : 8),
      lineWidthMinPixels: 1,
      updateTriggers: {
        getFillColor: [d.cells],
        getLineColor: [highlightId],
        getLineWidth: [highlightId],
      },
      onHover: (info) => {
        const obj = info && info.object ? info.object : null;
        const nextHover = obj ? obj.h3_cell : null;

        if (nextHover !== state.hoveredId) {
          state.hoveredId = nextHover;
          const [activeCell, mode] = getActiveCellAndMode();
          renderDetailCard(activeCell, mode);
          if (state.deck) {
            state.deck.setProps({ layers: [buildLayer()] });
          }
        }
      },
      onClick: (info) => {
        if (!info || !info.object) return;
        const clickedId = info.object.h3_cell;
        state.pinnedId = clickedId;
        const [activeCell, mode] = getActiveCellAndMode();
        renderDetailCard(activeCell, mode);
        if (state.deck) {
          state.deck.setProps({ layers: [buildLayer()] });
        }
        setStateValue("selected_h3_cell", clickedId);
      },
    });
  };

  clearBtnEl.onclick = () => {
    state.pinnedId = null;
    const [activeCell, mode] = getActiveCellAndMode();
    renderDetailCard(activeCell, mode);
    if (state.deck) {
      state.deck.setProps({ layers: [buildLayer()] });
    }
    setStateValue("selected_h3_cell", null);
  };

  const [initialCell, initialMode] = getActiveCellAndMode();
  renderDetailCard(initialCell, initialMode);

  ensureLibs()
    .then(() => {
      if (loadingEl) loadingEl.style.display = "none";
      const mapStyle = data.is_dark
        ? "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
        : "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";

      if (!state.deck || state.lastTheme !== data.is_dark) {
        if (state.deck) {
          state.deck.finalize();
          state.deck = null;
        }
        state.lastTheme = data.is_dark;
        state.deck = new window.deck.DeckGL({
          container: mapEl,
          map: window.maplibregl,
          mapStyle,
          initialViewState: {
            longitude: data.view_lon,
            latitude: data.view_lat,
            zoom: data.view_zoom,
            pitch: 0,
            bearing: 0,
          },
          controller: true,
          getTooltip: () => null,
          layers: [buildLayer()],
        });
      } else {
        state.deck.setProps({ layers: [buildLayer()] });
      }
    })
    .catch((err) => {
      if (loadingEl) {
        loadingEl.textContent = `Could not load interactive map: ${err?.message || err}`;
      }
    });
}
"""

_HOVER_MAP = st.components.v2.component(
    "fleetsense_hover_map",
    html=_HTML,
    css=_CSS,
    js=_JS,
)


def render_hover_map(
    *,
    cells_data: list[dict],
    default_cell: dict,
    selected_h3_cell: str | None,
    view_lat: float,
    view_lon: float,
    view_zoom: float,
    is_dark: bool,
    use_how: bool,
    how_label: str,
    key: str = "fleetsense_hover_map",
    on_selected_h3_cell_change: Callable[[], None] | None = None,
):
    """Render the Deck.gl H3 map and live-hover Cell detail card."""
    if on_selected_h3_cell_change is None:
        on_selected_h3_cell_change = lambda: None

    return _HOVER_MAP(
        data={
            "cells": cells_data,
            "default_cell": default_cell,
            "selected_h3_cell": selected_h3_cell,
            "view_lat": view_lat,
            "view_lon": view_lon,
            "view_zoom": view_zoom,
            "is_dark": is_dark,
            "use_how": use_how,
            "how_label": how_label,
        },
        key=key,
        width="stretch",
        on_selected_h3_cell_change=on_selected_h3_cell_change,
    )

