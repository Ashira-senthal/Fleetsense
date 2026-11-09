"""Stage 5 - the six clustering algorithms, each with its own input space.

WHY SIX
-------
They answer different questions, and no single objective is "correct" without labels:

| Algorithm | Input space | Assumption it makes | Question it answers |
|---|---|---|---|
| K-Means | PCA scores (5-D) | clusters are roughly spherical, similar size | "which cells behave alike on a few summary axes?" |
| Ward (hierarchical) | PCA scores (5-D) | nested grouping, compact clusters | "what is the hierarchy, and does a flat cut agree?" |
| GMM (EM) | PCA scores (5-D) | clusters are Gaussian, may overlap | "which behaviour regimes overlap, and how confidently?" |
| DBSCAN | raw lat/lon, **haversine** | dense regions separated by sparse space | "where are dense spatial groups?" (geography, not behaviour) |
| HDBSCAN | PCA scores (5-D) | clusters of varying density, noise allowed | "which cells are typical, which are outliers?" |
| Time-series K-Means + DTW | 168-hour profiles | clusters differ in *shape* of the week | "which weekly rhythms repeat?" |

MATHS IN ONE LINE EACH
----------------------
K-Means minimises within-cluster sum of squares (inertia)

    J = sum_k sum_{x in C_k} || x - mu_k ||^2

via Lloyd's iteration: assign to nearest centroid, then set mu_k = mean(C_k). It converges
to a *local* optimum, so n_init > 1 restarts are used.

DBSCAN: a point is a **core point** if at least `min_samples` points lie within `eps`
(here measured with haversine, so eps is a real distance in km). Clusters grow from core
points through density-reachability; points that are neither core nor reachable are noise
(label -1). It needs no k but is very sensitive to eps.

HDBSCAN: builds a hierarchy over all density thresholds, keeps the clusters that persist
longest (excess of mass), so it does not need eps and can find clusters of different density.

GMM / EM: maximise the likelihood of a mixture

    p(x) = sum_k pi_k N(x | mu_k, Sigma_k)

alternating E-step (posterior responsibilities) and M-step (weighted mean/covariance
updates). k is chosen by BIC = -2 log L + p log n, which penalises free parameters.

Ward: merges the pair of clusters that increases within-cluster variance least; a flat cut
at k gives a partition comparable to K-Means but obtained greedily.

Time-series K-Means with DTW: same objective as K-Means, but the distance is dynamic time
warping, which allows the *timing* of a peak to shift. Euclidean on a 168-length profile
would punish a peak at 08:00 vs 09:00 as heavily as a missing peak.

EVERY ALGORITHM gets random_state where randomness exists (K-Means init, GMM init, HDBSCAN
tie-breaking), so results are reproducible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.mixture import GaussianMixture

from src.cleaning import EARTH_RADIUS_KM

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    """Silhouette needs >= 2 clusters and < n distinct labels; noise counts as a label."""
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= len(labels):
        return None
    return float(silhouette_score(X, labels))


def cluster_sizes(labels: np.ndarray) -> dict:
    vals, counts = np.unique(labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


# ---------------------------------------------------------------------------
# 1. K-Means
# ---------------------------------------------------------------------------
def kmeans_scan(X: np.ndarray, ks=range(2, 9), n_init: int = 20,
                random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, dict[int, KMeans]]:
    """Fit K-Means for each k and record inertia (for the elbow) and silhouette.

    n_init=20 because Lloyd's algorithm is local: with one restart a bad initialisation
    can report a higher inertia for larger k, which makes the elbow meaningless.
    """
    rows, models = [], {}
    for k in ks:
        km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = km.fit_predict(X)
        rows.append({"k": k, "inertia": float(km.inertia_),
                     "silhouette": _safe_silhouette(X, labels),
                     "calinski_harabasz": float(calinski_harabasz_score(X, labels)) if len(set(labels)) > 1 else None})
        models[k] = km
    return pd.DataFrame(rows), models


# ---------------------------------------------------------------------------
# 2. Ward hierarchical
# ---------------------------------------------------------------------------
def ward_scan(X: np.ndarray, ks=range(2, 9)) -> pd.DataFrame:
    """Flat cuts of a Ward hierarchy, scored the same way as K-Means."""
    rows = []
    for k in ks:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
        rows.append({"k": k, "silhouette": _safe_silhouette(X, labels),
                     "calinski_harabasz": float(calinski_harabasz_score(X, labels))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Gaussian Mixture (EM)
# ---------------------------------------------------------------------------
def gmm_scan(X: np.ndarray, ks=range(2, 9), covariance_type: str = "full",
             n_init: int = 10, random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, dict[int, GaussianMixture]]:
    """Fit GMMs and record BIC/AIC.

    BIC = -2 log L + n_params * log(n). Lower is better. reg_covar adds a small value to
    covariance diagonals: without it, a component that collapses onto a few points gets a
    singular covariance and the likelihood goes to infinity.
    """
    rows, models = [], {}
    for k in ks:
        gm = GaussianMixture(n_components=k, covariance_type=covariance_type,
                             reg_covar=1e-4, n_init=n_init, random_state=random_state)
        gm.fit(X)
        labels = gm.predict(X)
        rows.append({"k": k, "bic": float(gm.bic(X)), "aic": float(gm.aic(X)),
                     "silhouette": _safe_silhouette(X, labels)})
        models[k] = gm
    return pd.DataFrame(rows), models


# ---------------------------------------------------------------------------
# 4. DBSCAN with haversine on raw coordinates
# ---------------------------------------------------------------------------
def k_distance_curve(lat: np.ndarray, lon: np.ndarray, k: int = 5) -> np.ndarray:
    """Distance (km) to the k-th nearest neighbour, sorted ascending.

    The classic DBSCAN eps heuristic: the curve has a knee where dense areas end. Reading
    eps off that knee is principled; guessing eps in km is not.
    """
    from sklearn.neighbors import BallTree

    pts = np.radians(np.column_stack([lat, lon]))          # haversine metric wants radians
    tree = BallTree(pts, metric="haversine")
    dist, _ = tree.query(pts, k=k + 1)                     # first neighbour is itself
    return np.sort(dist[:, k] * EARTH_RADIUS_KM)


def dbscan_haversine(lat: np.ndarray, lon: np.ndarray, eps_km: float, min_samples: int = 5) -> np.ndarray:
    """DBSCAN in *kilometres* using haversine.

    eps must be converted to radians because sklearn's haversine metric expects radian
    coordinates: eps_rad = eps_km / R. Using degrees here would silently produce a
    completely different neighbourhood size at this latitude.
    """
    pts = np.radians(np.column_stack([lat, lon]))
    model = DBSCAN(eps=eps_km / EARTH_RADIUS_KM, min_samples=min_samples, metric="haversine")
    return model.fit_predict(pts)


def dbscan_scan(lat: np.ndarray, lon: np.ndarray, eps_grid_km=(0.4, 0.6, 0.8, 1.0, 1.5, 2.0),
                min_samples_grid=(3, 5, 8)) -> pd.DataFrame:
    """Grid over eps and min_samples, recording cluster counts and noise share.

    There is no silhouette here: DBSCAN's noise label is not a cluster, and including it
    flatters or destroys the score depending on how much noise there is. Choose by noise
    share (a map where 60% of cells is 'noise' is useless) plus visual coherence.
    """
    rows = []
    for eps in eps_grid_km:
        for ms in min_samples_grid:
            labels = dbscan_haversine(lat, lon, eps, ms)
            sizes = cluster_sizes(labels)
            n_clusters = len([k for k in sizes if k != -1])
            rows.append({"eps_km": eps, "min_samples": ms, "n_clusters": n_clusters,
                         "noise": sizes.get(-1, 0), "noise_share": round(sizes.get(-1, 0) / len(labels), 3),
                         "largest_cluster": max([v for k, v in sizes.items() if k != -1], default=0)})
    return pd.DataFrame(rows).sort_values(["n_clusters", "noise_share"], ascending=[False, True])


# ---------------------------------------------------------------------------
# 5. HDBSCAN
# ---------------------------------------------------------------------------
def hdbscan_run(X: np.ndarray, min_cluster_size: int = 20, min_samples: int | None = None,
                metric: str = "euclidean") -> tuple[np.ndarray, dict]:
    """HDBSCAN on the PCA space; returns (labels, diagnostics).

    min_cluster_size is the smallest group worth calling a cluster. It is expressed in
    cells, so it encodes a judgement: 20 cells at resolution 8 is roughly 15 km2.
    `relative_validity_` is the library's DBCV-like score (higher = better separated
    density clusters); it is used to choose min_cluster_size, then reported properly in
    Stage 6.
    """
    import hdbscan

    model = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                            metric=metric, cluster_selection_method="eom",
                            # gen_min_span_tree is required for relative_validity_ (DBCV-like)
                            gen_min_span_tree=True, prediction_data=True, core_dist_n_jobs=1)
    labels = model.fit_predict(X)
    diag = {"n_clusters": int(labels.max() + 1),
            "noise": int((labels == -1).sum()),
            "relative_validity": float(getattr(model, "relative_validity_", np.nan)),
            "probabilities_mean": float(model.probabilities_.mean())}
    return labels, diag


def hdbscan_scan(X: np.ndarray, sizes=(5, 10, 20, 30),
                 min_samples_grid=(None, 3, 5)) -> pd.DataFrame:
    """Scan min_cluster_size x min_samples.

    min_samples defaults to min_cluster_size in the library, which makes HDBSCAN very
    conservative on a small n; scanning both is the honest way to show whether "no
    clusters" is a real property of the data or a side effect of one default.
    """
    rows = []
    for s in sizes:
        for ms in min_samples_grid:
            labels, diag = hdbscan_run(X, min_cluster_size=s, min_samples=ms)
            rows.append({"min_cluster_size": s, "min_samples": ms, **diag})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Time-series K-Means with DTW
# ---------------------------------------------------------------------------
def ts_kmeans(profiles: pd.DataFrame, k: int, metric: str = "dtw",
              max_iter: int = 10, random_state: int = RANDOM_STATE) -> tuple[np.ndarray, np.ndarray]:
    """Cluster weekly profiles by shape. Returns (labels, cluster centroids).

    DTW cost is quadratic in series length, so a Sakoe-Chiba band of 12 hours is used:
    it forbids matching 02:00 with 20:00, which is right for a weekly rhythm and cuts the
    cost by an order of magnitude.
    """
    from tslearn.clustering import TimeSeriesKMeans

    X = profiles.to_numpy(dtype=float)[:, :, None]     # tslearn wants (n, length, 1)
    model = TimeSeriesKMeans(n_clusters=k, metric=metric, max_iter=max_iter,
                             random_state=random_state,
                             metric_params={"global_constraint": "sakoe_chiba", "sakoe_chiba_radius": 12})
    labels = model.fit_predict(X)
    return labels, model.cluster_centers_[:, :, 0]


def dtw_silhouette(profiles: pd.DataFrame, labels: np.ndarray, sample_size: int = 300,
                   random_state: int = RANDOM_STATE) -> float | None:
    """Silhouette with a DTW distance matrix, on a subsample (the matrix is O(n^2) DTW).

    Needed because scoring DTW clusters with Euclidean silhouette measures the wrong thing:
    a peak shifted by one hour would be penalised even though DTW considered the shapes
    identical.
    """
    from tslearn.metrics import cdist_dtw

    uniq = np.unique(labels)
    if len(uniq) < 3:
        return None
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(labels), size=min(sample_size, len(labels)), replace=False)
    X = profiles.to_numpy(dtype=float)[idx][:, :, None]
    D = cdist_dtw(X, global_constraint="sakoe_chiba", sakoe_chiba_radius=12)
    return float(silhouette_score(D, labels[idx], metric="precomputed"))
