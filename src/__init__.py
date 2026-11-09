"""FleetSense - unsupervised urban mobility pattern clustering from bus GPS + GTFS.

Modules
-------
io        : loaders for GTFS static feeds and GPS point tables
audit     : data quality audit (Stage 1)
cleaning  : GPS jump removal, deduplication, bounding box, road snapping (Stage 2)
features  : stop / hex-cell / trip / temporal feature engineering (Stage 3)
reduce    : PCA compression + UMAP projection for plots (Stage 4)
clustering: KMeans, DBSCAN, HDBSCAN, GMM, Ward, time-series KMeans (Stage 5)
evaluate  : internal metrics + stability via ARI (Stage 6)
"""
