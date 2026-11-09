# FleetSense — Urban Mobility Pattern Clustering

Unsupervised discovery of recurring mobility patterns from vehicle GPS: spatial hotspots,
temporal regimes, stop-and-go behaviour, and the days and places that break the pattern.
No labels are used — the grouping is found first, then interpreted and named.

**Status:** Tier 1 complete. Every result below was produced by the notebooks in
`notebooks/`, on real GPS data, and the exported files are validated by
`python -m src.schema`.

---

## 1. What the pipeline does

| Stage | Notebook | Question answered |
|---|---|---|
| 1 | `01_audit.ipynb` | What is actually in the feed? (missing values, duplicates, time zones, coordinate ranges) |
| 2 | `02_cleaning.ipynb` | Remove impossible GPS jumps, duplicates and out-of-area points, with every threshold justified and scored against known injected defects |
| 3 | `03_features.ipynb` | Aggregate pings into H3 cells and trips; cyclic time encoding; standardise |
| 4 | `04_reduce.ipynb` | PCA for compression (clustering input); UMAP for pictures only |
| 5 | `05_clustering.ipynb` | Six algorithms, six defensible parameter choices |
| 6 | `06_evaluate.ipynb` | Silhouette / Davies–Bouldin / Calinski–Harabasz / DBCV + bootstrap stability → winner |
| 7 | `07_interpret.ipynb` | Profiles, OSM POI context, names, anomalies, limitations and privacy |
| 8 | `src/schema.py` | The `outputs/` contract: declared, validated, documented |

---

## 2. Data, and the honest version of how it was chosen

**Used: Porto taxi trajectories** — UCI ML Repository dataset 339 (ECML/PKDD 2015),
CC BY 4.0, 15-second sampling.

The project started on **Delhi bus/GTFS data**. The portal pages responded, but the file
host they link to refused connections, so the download could not be completed; the
Transitland mirror of the same feed was empty. Per the pre-agreed fallback rule, the
pipeline moved to Porto taxi GPS rather than stall. The Delhi attempt is preserved in
`outputs/data_audit.json` so the decision is auditable, not folklore.

> **Limitation, repeated wherever results appear: these are taxis, not buses.**
> No fixed stops, no timetable, no schedule delay. "Dwell" means below walking pace
> (kerbside waiting, parking, traffic), not boarding passengers. Cluster names describe
> taxi movement, not passenger demand. `src/porto.py` and `src/cleaning.py` work unchanged
> on GTFS + GPS feeds, so the bus version is a data swap, not a rewrite.

Measured volumes (from `data/processed/features_meta.json`):

| Quantity | Value |
|---|---|
| Rows scanned in source | 1 710 670 |
| Trips sampled (every ~28th, seeded) | 58 706 |
| Trips with a usable trajectory | 58 686 |
| GPS pings | 2 927 093 |
| Pings removed by cleaning | 2.563 % (29 967 outside bbox, 45 045 impossible jumps) |
| Pings after cleaning | 2 852 081 |
| H3 res-8 cells touched | 976 |
| Cells analysed (≥ 5 visits) | **602** |

Cleaning thresholds, and why: **100 km/h** between consecutive pings (a city vehicle
cannot exceed it; consumer GPS noise at 15–30 s spacing is ~10 m, an order of magnitude
below a real move), an **area envelope from the 0.5–99.5 percentiles plus a 0.05° margin**,
and a **corroboration rule** for jumps: a ping is a spike only if both the step into it and
the step out of it are impossible. A naive "drop every fast step" filter deleted innocent
neighbours — measured on the labelled fixture: 3 953 removals for 2 156 injected defects.
After cleaning, 24 246 steps still exceed 100 km/h and cannot be attributed to either
endpoint; they are flagged and excluded from speed features rather than silently deleted.

---

## 3. Results

### Behaviour clusters — K-Means, k = 4, on PC1–PC5

Chosen because it led on Davies–Bouldin (1.435) and Calinski–Harabasz (182.19), tied for
best silhouette among full-coverage methods (0.244) and was by far the most stable
(bootstrap ARI **0.851 ± 0.052**, worst run 0.758, vs 0.414–0.695 for the others). HDBSCAN
reported a higher silhouette (0.333) but labelled only 39.7 % of cells; DBSCAN's positive
DBCV (0.183) is what DBCV is for, at 62.5 % coverage over 41 spatial blobs.

| C | Name (generated) | Cells | Mean speed | Slow steps | Speed CV | Dwell/visit | POI/cell |
|---|---|---|---|---|---|---|---|
| 0 | Slow low-traffic pockets (weekend-heavy) | 121 (20 %) | 34.3 km/h | 24 % | 0.66 | 57 s | 16.7 |
| 1 | Busy mid-speed corridors (all-day, hospitality-dense) | 164 (27 %) | 34.0 km/h | 29 % | 0.75 | 22 s | 50.6 |
| 2 | Severe stop-and-go pockets (all-day) | 130 (22 %) | 25.3 km/h | 41 % | 0.95 | 127 s | 22.6 |
| 3 | Quiet free-flowing fringe (weekend-heavy) | 187 (31 %) | 64.8 km/h | 3 % | 0.29 | 7.5 s | 9.3 |

Cluster 1 is the core by behaviour alone: 65–87× the visits per cell of the other three
clusters and 18.2 hospitality POIs per cell against a cluster median of 3.2 (5.6×). Names are built from measured
ranks (volume × congestion quadrant + dominant time window + POI contrast), and every name
ships with the numbers that produced it in `outputs/cluster_profiles.csv`.

### Weekly rhythms — TS-K-Means on 168-hour profiles

Four rhythm clusters, named by weekend share, peak shape and volume: all-week morning
twin-peak (high-volume), weekend-weighted midday twin-peak, all-week morning single-peak,
weekday-only morning twin-peak. The cross-tab is the interesting part: **cluster 1 sits
entirely in the high-volume rhythm (164/164)**, while cluster 2 splits 88/12/13/17 — 17 cells
behave like the stop-and-go cluster *only on weekdays*. Behaviour and rhythm are separate
questions, so they were kept as separate label sets.

### Anomalies

| Kind | Rule | Flagged |
|---|---|---|
| Behaviour (cell far from its cluster in PC space) | top 2 % **and** robust z ≥ 2 | 10 cells |
| Rhythm (cell used at the wrong times for its cluster) | top 2 % **and** robust z ≥ 2 | 4 cells |
| Both | — | 3 cells |
| Day (volume / speed) | quasi-Poisson z, dispersion-corrected, ≥ 30 sampled trips | 8 + 4 days |

17 of 602 cells (2.8 %), each with the features that drove the flag (`drivers`). Day flags
include 15 Aug 2013, 25 Dec 2013, 1 Jan 2014 and 4 Mar 2014 (Carnival Tuesday) — a real
validation that the detector fires on holidays rather than noise. Over-dispersion was
measured (φ = 3.21), not assumed: the raw Poisson test flagged 50 of 359 days.

---

## 4. Repository layout

```
Fleetsense/
  data/
    raw/            SOURCES.md + downloaded inputs (git-ignored)
    processed/      intermediate parquet + features_meta.json (git-ignored)
  notebooks/        01_audit … 07_interpret  (executed, with outputs)
  src/
    io.py           paths, GTFS loading, frame utilities
    audit.py        stage 1 audit
    cleaning.py     haversine, speed/jump rules, bounding box
    synthetic.py    labelled test fixture (synthetic: true) used to score cleaning
    porto.py        Porto sample loader (nested zip handling, UTC timestamps)
    features.py     H3 cell / trip / hour-of-week features
    reduce.py       PCA + UMAP
    clustering.py   six algorithms and their parameter selection
    evaluate.py     internal metrics, DBCV, bootstrap ARI
    interpret.py    profiles, OSM POIs, naming, anomalies, limitations doc
    schema.py       outputs contract: declare, validate, emit manifest + SCHEMA.md
  outputs/          the files the dashboard reads (see §6)
  docs/             PRIVACY_AND_LIMITATIONS.md (generated by the pipeline)
  requirements.txt, requirements-optional.txt, README.md
```

---

## 5. Setup and run

Tested with **Python 3.12** in a project virtual environment.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# run the pipeline in order (each notebook is self-contained and re-runnable)
jupyter lab            # or:
jupyter nbconvert --to notebook --execute --inplace notebooks/01_audit.ipynb
# … repeat for 02 … 07

# validate the outputs contract and regenerate manifest.json + SCHEMA.md
python -m src.schema
```

Runtime: most notebooks finish in seconds. `06_evaluate` takes ~73 s (DTW bootstrap
refits). `07_interpret` takes ~5 s once OSM POIs are cached in `data/raw/osm/`, and up to
~10 minutes on a cold cache, because it makes nine polite Overpass queries.

Optional Tier 2 / dashboard dependencies are listed but **not installed**, in
`requirements-optional.txt`.

---

## 6. The `outputs/` contract

The web app reads **only** this directory and never re-runs clustering.

| File | Content |
|---|---|
| `cluster_profiles.csv` | one row per cluster: size, centroid, feature means, name, description |
| `cluster_pois.csv` | OSM POI totals and per-cell densities per cluster |
| `cluster_hourly_profile.parquet` | cluster × 168-hour mean pings per cell (weekly chart) |
| `cluster_rhythm_names.csv`, `cluster_rhythm_crosstab.csv` | rhythm labels and behaviour × rhythm counts |
| `cell_annotations.parquet` | **the join table**: 602 cells with cluster, name, rhythm, anomaly scores/flags, coordinates |
| `anomalies.csv` | the 17 flagged cells with drivers |
| `day_anomalies.csv` | per-day counts, z-scores, flags, direction |
| `cluster_comparison.csv` / `.json` | the six-algorithm metric table |
| `interpret_meta.json`, `cleaning_report.json`, `data_audit.json` | provenance (the last two are earlier-stage artefacts, labelled per file in the manifest) |
| `figures/*.png` | 22 stage figures |
| `manifest.json` | machine-readable schema + observed dtypes, row counts, sizes, provenance |
| `SCHEMA.md` | the same schema as human-readable tables |

`manifest.json` and `SCHEMA.md` are generated by `src/schema.py`, which also **validates**
the directory: every declared column, dtype family, key uniqueness, and cross-file
consistency (cluster ids agree across files, `share_cells` sums to 1, `anomalies.csv`
contains exactly the flagged rows, `is_anomaly` agrees with `anomaly_type`, a speed flag
always has a direction). A renamed column fails the build instead of emptying a chart.

---

## 7. Reproducibility

* `random_state = 42` everywhere (sampling, K-Means, GMM, bootstrap, UMAP).
* Library versions pinned to what was actually installed and run (`requirements.txt`).
* Every threshold is a named constant with its justification in the docstring: 100 km/h
  jump speed, 5 km/h dwell, 12 km/h slow step, H3 res 8, ≥ 5 visits per cell, k = 4,
  ε and `min_samples` for DBSCAN, `min_cluster_size` for HDBSCAN, top 2 % + z ≥ 2 for
  anomalies, φ-corrected z ≥ 2.5 for days.
* The Porto sample is seeded and systematic, so a re-run reproduces the same 58 706 trips.

---

## 8. Limitations and privacy

Full text: `docs/PRIVACY_AND_LIMITATIONS.md` (generated by `src/interpret.py`, so it cannot
drift from the numbers). The short version:

* **Taxis, not buses**; no stops, timetable or schedule delay. No passenger counts.
* **No ground truth.** Every cluster name is a hypothesis printed next to its evidence.
* **Exposure bias.** A quiet street may be quiet because no taxi went there.
* **Seasonality collapsed** into one 168-hour week; one year averaged, not modelled.
* **Sampling, not census** — absolute counts are not city totals; comparisons are.
* **Anomalies are rule-based candidates**, not proven incidents. The rule is in the
  metadata; change it and the list changes.
* **Privacy:** inputs are pseudonymised vehicle trajectories; outputs are aggregates on
  ~0.74 km² hexagons, never individual tracks. No re-identification is attempted. If Tier 2
  is ever built it is **count-only** (pretrained person detector, no faces, no tracking
  across frames, aggregated counts only) and only for public footage.

---

## 9. Not built here

* **Tier 2 footfall (optional).** The module boundary is planned but not implemented or
  run: no model weights were downloaded and no public video was processed, so there are no
  footfall numbers in this repository and none are claimed. `requirements-optional.txt`
  lists the dependencies, unpinned on purpose (pinning untested versions would be a claim).
* **The Streamlit dashboard.** Built separately; it reads `outputs/` only, using
  `manifest.json` to check what it is reading.
