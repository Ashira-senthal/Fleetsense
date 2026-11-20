# FleetSense — Urban Mobility Pattern Clustering (Delhi Bus Transit)

Unsupervised discovery of recurring urban mobility patterns from public transit trajectories across Delhi NCR: spatial hotspots, temporal regimes, corridor dynamics, and the locations and times that break the pattern. No ground-truth labels are used — the clustering is discovered strictly from the data, then systematically interpreted, validated, and mapped.

---

## 1. What the pipeline does

| Stage | Script / Notebook | Question answered |
|---|---|---|
| 1 | `01_audit.ipynb` | What is in the raw feed? (route coverage, stop sequences, coordinate bounds, missing values) |
| 2 | `02_cleaning.ipynb` / `src/cleaning.py` | Remove impossible GPS jumps (>100 km/h), duplicates, and out-of-bounds coordinates with corroborated step filtering |
| 3 | `03_features.ipynb` / `src/features.py` | Aggregate trajectories into Uber H3 resolution-8 hexagons (~0.74 km²); compute volume, speed, dwell, and cyclic time features |
| 4 | `04_reduce.ipynb` / `src/reduce.py` | PCA for dimensional compression (PC1–PC5 as clustering input); 2D projections for visual sanity checks |
| 5 | `05_clustering.ipynb` / `src/clustering.py` | Six clustering algorithms evaluated with systematic parameter selection (K-Means, Ward, GMM, DBSCAN, HDBSCAN, TS-K-Means) |
| 6 | `06_evaluate.ipynb` / `src/evaluate.py` | Internal metrics (Silhouette, Davies–Bouldin, Calinski–Harabasz, DBCV) and bootstrap stability (ARI across resamples) |
| 7 | `07_interpret.ipynb` / `src/interpret.py` | Automated cluster naming from z-scores, OpenStreetMap POI context, spatial/temporal anomaly detection, and limitations |
| 8 | `src/schema.py` | Strict `outputs/` contract: declares schemas, validates data types and cross-table consistency, and writes `manifest.json` + `SCHEMA.md` |

The end-to-end process is automated by `src/run_delhi_pipeline.py`.

---

## 2. Dataset & Provenance

**Data source:** Delhi Bus Transit Data (Enhanced for ML) — Kaggle dataset by Adarsh Dubey (`delhi-bus-transit-data-enhanced-for-ml`), covering Delhi Integrated Multi-Modal Transit System (DIMTS) and Delhi Transport Corporation (DTC) bus networks across the National Capital Region (NCR).

* **Raw input:** `data/raw/delhi_kaggle/full_bus_schedule.csv` (~800 MB).
* **Trips sampled:** 30,000 representative trips sampled deterministically (`random_state=42`).
* **Trajectory pings after cleaning:** 1,202,552 valid point records.
* **Spatial indexing:** Uber H3 resolution-8 hexagonal grid (~0.74 km² per cell, ~460 m edge).
* **Cells analysed:** **1,155 cells** with ≥ 5 distinct trip visits across Delhi NCR.

> **Key dataset limitation:**
> These data represent **scheduled public bus transit timetables**, not live GPS vehicle tracking with real-time delay incidents. Speeds and dwell times reflect scheduled stop-to-stop transit timings and road segments rather than live congestion jams or on-board passenger counts. Cluster names describe transit service intensity and corridor profiles.

### Cleaning Rules & Justification

1. **100 km/h speed threshold:** Consecutive transit pings exceeding 100 km/h are physically impossible for city buses and represent coordinate or timing glitches.
2. **Delhi NCR bounding box:** Stops and coordinates bounded within latitudes 28.36°N–28.92°N and longitudes 76.79°E–77.46°E (with a 0.05° safety envelope).
3. **Corroborated jump filter:** A coordinate jump is only pruned if both the entry step and exit step violate physical velocity thresholds, preserving genuine high-frequency movements.

---

## 3. Results

### Behaviour Clusters — K-Means (k = 4 on PC1–PC5)

K-Means on the top 5 principal components was selected as the champion model. It achieved **100% cell coverage** across all 1,155 cells, high bootstrap stability (**ARI = 0.850 ± 0.132**), a compact Davies–Bouldin index (**0.658**), and the highest Calinski–Harabasz score (**2,745.15**). HDBSCAN achieved a higher raw silhouette (0.513) only by discarding **21.3% of cells as noise** (78.7% coverage).

| Cluster | Generated Name | Cells | Share | Mean Speed | Stop & Go CV | Dwell / Visit | Mean Visits / Cell | POI Context |
|---|---|---|---|---|---|---|---|---|
| **C0** | Slow low-traffic pockets (all-day) | 157 | 13.6% | 15.0 km/h | 0.001 | 0.0 s | 23.3 | Local access |
| **C1** | Busy mid-speed corridors (all-day, civic-dense) | 386 | 33.4% | 15.0 km/h | 0.007 | 0.0 s | 1,419.9 | Civic-dense (2.7× median POIs) |
| **C2** | Quiet free-flowing fringe (weekend-heavy) | 279 | 24.2% | 15.0 km/h | 0.004 | 0.0 s | 111.3 | Outer radial connections |
| **C3** | Slow low-traffic pockets (weekend-heavy) | 333 | 28.8% | 15.0 km/h | 0.007 | 0.0 s | 414.2 | Mixed arterial feeders |

*Cluster names are systematically generated using measured rank scores (volume index × congestion index quadrant + temporal modifier + OpenStreetMap POI over-representation).*

### Weekly Rhythms — TS-K-Means on 168-Hour Profiles (k = 4)

Weekly temporal profiles (168 hourly bins from Monday 00:00 to Sunday 23:00) were clustered separately using TimeSeries K-Means with Dynamic Time Warping (DTW):

| Rhythm | Name | Peak Hour | Peak Type | Weekend Share | Activity Tier |
|---|---|---|---|---|---|
| **R0** | weekday-only, midday twin-peak | 13:00 | Twin peak | 11.8% | Low-volume feeder |
| **R1** | all-week, evening single-peak | 16:00 | Single peak | 28.4% | High-volume trunk corridor |
| **R2** | all-week, midday twin-peak | 15:00 | Twin peak | 28.5% | Mid-volume connector |
| **R3** | all-week, morning twin-peak | 07:00 | Twin peak | 26.9% | Low-volume morning commuter |

Keeping behaviour clusters and rhythm clusters on independent axes enables cross-tabulation: high-volume corridor cells (C1) concentrate heavily in evening-peaked all-week rhythms (R1), while secondary pockets split across morning and midday schedules.

### Anomaly Detection

Anomalies are detected using robust distance thresholds rather than arbitrary cuts:
* **Spatial anomalies:** Cells in the top 2% of Euclidean distance to their assigned cluster centroid in PC space with robust $z \ge 2.0$.
* **Temporal anomalies:** Cells in the top 2% of $L_2$ distance to their cluster's mean 168-hour activity curve with robust $z \ge 2.0$.
* **Flagged cells:** **39 cells** flagged in `outputs/anomalies.csv`, each annotated with specific `drivers` (e.g. extreme visit volume, unusual speed variance).
* **Daily anomalies:** 14 calendar days evaluated in `outputs/day_anomalies.csv` using quasi-Poisson volume $z$-scores and speed robust $z$-scores.

---

## 4. Repository Layout

```
Fleetsense/
  data/
    raw/
      delhi_kaggle/      full_bus_schedule.csv (raw input)
      osm/               cached OpenStreetMap POI queries (.parquet)
      SOURCES.md         data provenance, URLs, and terms
    processed/
      features_meta.json pipeline metadata summary
  notebooks/             01_audit … 07_interpret (exploratory & reference stages)
  src/
    run_delhi_pipeline.py  automated pipeline execution (Stages 2–8)
    cleaning.py          bounding box, speed filters, jump corroboration
    features.py          H3 spatial binning, trip metrics, cyclic time encoding
    reduce.py            PCA fitting and variance analysis
    clustering.py        six clustering algorithms + parameter sweeps
    evaluate.py          internal validation metrics (Silhouette, DB, CH, DBCV, ARI)
    interpret.py         POI profiling, cluster naming, anomaly scoring
    schema.py            data contract validation and manifest generator
  outputs/               the complete contract read by the dashboard (see §6)
  dashboard/             Streamlit web application
    app.py               main navigation entrypoint with KPI banner
    pages/
      2_cluster_map.py   interactive Deck.gl H3 map with hover detail card
      3_cluster_profiles.py cluster metric cards, normalised bars, POI density
      4_weekly_rhythms.py  168-hour temporal line curves and crosstab heatmap
      5_anomalies.py     flagged cells table and daily z-score inspection
      6_model_comparison.py 6-algorithm validation comparison & key figures
      7_data_quality.py  cleaning thresholds and privacy documentation
    utils/
      hover_map.py       Streamlit Custom Component v2 (Deck.gl H3 layer)
      load.py            cached data loaders and schema validators
      theme.py           colourblind-safe palettes and styling
  docs/
    PRIVACY_AND_LIMITATIONS.md  generated privacy & data limitations document
  tests/
    test_manifest_and_dashboard.py pytest test suite
  requirements.txt       pinned project dependencies
  README.md              project documentation
```

---

## 5. Setup and Run

### Environment Setup

Python 3.12 or 3.13 is recommended.

```bash
# Create and activate virtual environment
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

To re-run the full unsupervised clustering pipeline from raw data to `outputs/`:

```bash
python -m src.run_delhi_pipeline
```

### Validate Outputs Contract

```bash
python -m src.schema
```

### Run Automated Tests

```bash
python -m pytest tests/test_manifest_and_dashboard.py -v
```

---

## 6. The `outputs/` Data Contract

The Streamlit dashboard reads **only** files in `outputs/` and never re-runs clustering at runtime. Every file and column is validated against `outputs/manifest.json`:

| File | Type | Description |
|---|---|---|
| `cell_annotations.parquet` | Parquet | **Primary join table**: 1,155 H3 cells with cluster assignments, rhythm labels, centroid coordinates, and anomaly scores |
| `cluster_profiles.csv` | CSV | 4 behaviour clusters: cell counts, speed, dwell, time-of-week shares, and descriptions |
| `cluster_pois.csv` | CSV | OpenStreetMap POI category counts and densities per cluster |
| `cluster_hourly_profile.parquet`| Parquet | 4 clusters × 168 hours of mean ping activity per cell |
| `cluster_rhythm_names.csv` | CSV | 4 TS-K-Means weekly rhythm cluster names and peak metrics |
| `cluster_rhythm_crosstab.csv` | CSV | 4 × 4 matrix crossing behaviour clusters with rhythm clusters |
| `anomalies.csv` | CSV | 39 rule-flagged outlier cells with contributing feature drivers |
| `day_anomalies.csv` | CSV | Daily trip volume, median speed, and robust $z$-scores across calendar dates |
| `cluster_comparison.csv` | CSV | Benchmark evaluation metrics across all six algorithms |
| `cluster_comparison.json` | JSON | Structured metric summaries, bootstrap configuration, and run metadata |
| `interpret_meta.json` | JSON | Stage 7 configuration, naming rules, and anomaly thresholds |
| `cleaning_report.json` | JSON | Data cleaning rules, speed cutoffs, and bounding-box coordinates |
| `data_audit.json` | JSON | Source integrity audit findings and table schemas |
| `manifest.json` | JSON | Machine-readable contract manifest with file hashes, schemas, and row counts |
| `SCHEMA.md` | Markdown | Formatted documentation of all output tables and column definitions |

---

## 7. Running the Streamlit Dashboard

Launch the interactive dashboard:

```bash
streamlit run dashboard/app.py
```

### Dashboard Pages

1. **Map (Main screen):** High-performance Deck.gl H3 hexagon map layered over carto base tiles. Features colour-by switching (Behaviour cluster, Rhythm cluster, Anomaly score, Visits), multi-select cluster pills, flagged-cell highlighting, and a live-updating **Cell Detail** card that reacts immediately to cursor hovering without slow server re-renders.
2. **Clusters:** Summary cards for each cluster with measured speed, slow-step share, dwell time, and POI density, paired with normalised feature comparisons and OSM category breakdowns.
3. **Weekly Rhythm:** 168-hour temporal line curves per cluster with normalisation toggles and an annotated Behaviour × Rhythm crosstab heatmap.
4. **Anomalies:** Sortable table of flagged outlier cells with driver details and direct "Open on cluster map" cross-navigation, alongside daily z-score tracking.
5. **Method:** Six-algorithm benchmark table (Silhouette, Davies–Bouldin, Calinski–Harabasz, DBCV, Coverage, Bootstrap ARI) highlighting the selection justification for K-Means.
6. **Data Quality:** Cleaning thresholds, bounding-box parameters, provenance notes, and rendered privacy documentation.

---

## 8. Reproducibility & Governance

* **Determinism:** `random_state = 42` is fixed across sampling, PCA decomposition, K-Means, GMM, bootstrap iterations, and TS-K-Means.
* **No ground-truth assumption:** Every cluster label is explicitly framed as an empirical hypothesis paired with its measured statistical evidence.
* **Privacy by design:** Analysis operates strictly on aggregate H3 resolution-8 cells (~0.74 km²). No individual vehicle journeys or personally identifiable traces are published in `outputs/`.
