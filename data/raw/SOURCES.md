# Data sources and provenance
---

## 1. Delhi Bus Transit Data (Enhanced for ML) — `data/raw/delhi_kaggle/` (ACTIVE)

| Field | Value |
|---|---|
| **Dataset** | Delhi Bus Transit Data (Enhanced for ML) |
| **Publisher / Creator** | Adarsh Dubey |
| **Download Source** | [Kaggle (`delhi-bus-transit-data-enhanced-for-ml`)](https://www.kaggle.com/datasets/adarshdubey001/delhi-bus-transit-data-enhanced-for-ml) |
| **Raw File** | `data/raw/delhi_kaggle/full_bus_schedule.csv` (~800 MB) |
| **Transit Agencies** | Delhi Integrated Multi-Modal Transit System (DIMTS) and Delhi Transport Corporation (DTC) |
| **Geographic Scope** | National Capital Region (NCR), Delhi, India |
| **Content** | Scheduled stop-to-stop bus timetable sequences, coordinates (`latitude`, `longitude`), arrival/departure timestamps, trip IDs, route IDs, and vehicle schedules across Delhi NCR routes |
| **`synthetic`** | **False** — authoritative public transit schedule and route network data |
| **Sample Used** | Systematic deterministic sample of **30,000 trips** (`random_state=42`) |
| **Processed Pings** | **1,202,552 trajectory points** after physical jump and bounding-box cleaning |
| **Spatial Aggregation** | Uber H3 resolution-8 hexagonal grid (~0.74 km² per cell, ~460 m edge); **1,155 cells** analysed (with ≥ 5 trip visits) |
| **Key Limitation** | **Scheduled GTFS bus timetable, not live GPS:** Speeds and dwell times reflect scheduled stop-to-stop transit timings rather than real-time traffic jams or on-board passenger counts. Clusters describe transit service patterns and corridor intensity. |

---

## 2. OpenStreetMap Contextual POIs — `data/raw/osm/` (ACTIVE)

| Field | Value |
|---|---|
| **Dataset** | OpenStreetMap (OSM) Points of Interest (POIs) |
| **Provider** | OpenStreetMap contributors via `osmnx` / Overpass API |
| **Local Cache** | `data/raw/osm/delhi_osm_pois.parquet` |
| **Licence** | Open Database License (ODbL) — attribution required |
| **Usage** | Spatial join of amenities (education, sustenance, healthcare, commercial, civic institutions) within each H3 hexagon to characterize urban land use per cluster and assist in automated cluster naming. |

---

## 3. Reference & Audit Artefacts (Historical Context)

* **Synthetic GPS Cleaning Fixture (`src/synthetic.py`):**
  * Used as a labelled regression unit test to validate recall of GPS jump filters, deduplication, and out-of-bounds cleaning rules (`outputs/cleaning_report.json`).
* **Initial Delhi OTD Static GTFS Trimmed Audit:**
  * Preserved in `outputs/data_audit.json` to record the initial schema audit of public Delhi static transit GTFS feeds.
