"""Stage 6 - evaluation of unsupervised clusterings.

Three questions, in increasing order of honesty:

1. **Is the partition internally coherent?** Silhouette, Davies-Bouldin,
   Calinski-Harabasz, and for density methods DBCV.
2. **Does it survive resampling?** Re-run on bootstrap subsamples and compare the two
   label sets with the Adjusted Rand Index. A partition that changes every time you
   drop 20% of the cells is not a finding, it is a coincidence.
3. **Is it worth reporting?** Coverage (how much of the city is even labelled),
   interpretability, and whether the metric was measured in the space the algorithm used.

METRICS, WITH THE ACTUAL FORMULAS
---------------------------------
Silhouette, per point i (with a(i) = mean distance to its own cluster, b(i) = mean
distance to the nearest other cluster):

    s(i) = (b(i) - a(i)) / max(a(i), b(i))  in [-1, 1], higher = better separated

Davies-Bouldin (lower better), with S_i = mean distance of cluster i's points to its
centroid and M_ij = distance between centroids i and j:

    DB = (1/k) * sum_i  max_{j != i} (S_i + S_j) / M_ij

Calinski-Harabasz (higher better), with B_k between-cluster and W_k within-cluster
dispersion:

    CH = [tr(B_k) / (k-1)] / [tr(W_k) / (n-k)]

DBCV (density-based validity, higher better; used for DBSCAN/HDBSCAN) compares, per
cluster, the sparsest internal edge with the densest separating edge:

    DBCV = sum_i (|C_i|/n) * (min_j DSPC(i,j) - dens_max(C_i)) / max(min_j DSPC(i,j), dens_max(C_i))

THREE RULES THIS MODULE ENFORCES
--------------------------------
* **Noise is not a cluster.** Cells labelled -1 are excluded from every metric and
  reported separately as `noise_share`. Including noise inflates separation when it is
  small and destroys the score when it is large - either way the number means nothing.
* **Score in the space the algorithm used.** DBSCAN clustered kilometres, not PCs, so its
  silhouette uses a haversine distance matrix. Numbers from different spaces are not
  comparable; a common-space (PCA) silhouette is reported alongside so they can be.
* **Davies-Bouldin and Calinski-Harabasz need a feature space.** They are undefined for a
  precomputed DTW distance matrix and for lat/lon with haversine, so they are left empty
  rather than faked with Euclidean-on-degrees.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

from src.cleaning import EARTH_RADIUS_KM

RANDOM_STATE = 42


def drop_noise(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (boolean mask of clustered points, labels without noise)."""
    labels = np.asarray(labels)
    mask = labels != -1
    return mask, labels[mask]


def score_partition(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean",
                    precomputed: bool = False) -> dict:
    """Silhouette / Davies-Bouldin / Calinski-Harabasz in one call.

    Returns None for metrics that are undefined in this space instead of a misleading
    number. `precomputed=True` means X is a square distance matrix.
    """
    mask, clean = drop_noise(labels)
    out = {
        "n_clusters": int(len(np.unique(clean))) if clean.size else 0,
        "coverage": round(float(mask.mean()), 4),
        "noise_share": round(float(1 - mask.mean()), 4),
    }
    if out["n_clusters"] < 2:
        out.update({"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None})
        return out

    Xm = X[mask][:, mask] if precomputed else X[mask]
    out["silhouette"] = round(float(silhouette_score(Xm, clean, metric=metric)), 4)

    if precomputed:
        out["davies_bouldin"] = None      # needs a feature space, not distances
        out["calinski_harabasz"] = None
    else:
        out["davies_bouldin"] = round(float(davies_bouldin_score(Xm, clean)), 4)
        out["calinski_harabasz"] = round(float(calinski_harabasz_score(Xm, clean)), 4)
    return out


def haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Full pairwise haversine distance matrix in km (radians -> chord-free formula)."""
    from sklearn.metrics import pairwise_distances

    pts = np.radians(np.column_stack([lat, lon]))
    return pairwise_distances(pts, metric="haversine") * EARTH_RADIUS_KM


def dbcv(X: np.ndarray, labels: np.ndarray, metric: str = "euclidean",
         precomputed: bool = False, d: int | None = None,
         min_cluster_size: int = 3) -> float | None:
    """DBCV via the hdbscan library; ignores noise points.

    `d` is NOT the distance matrix - it is the *dimensionality* of the implicit space,
    which the library needs for its all-points core distance. For precomputed matrices
    you must supply it (2 for a lat/lon plane). Getting this wrong raises a broadcast
    error rather than a clear message, which is why it is explicit here.
    """
    import hdbscan

    mask, clean = drop_noise(labels)

    # DBCV's core-distance formula divides by (cluster size - 1), so singleton clusters
    # make it undefined (a real case: DBSCAN produced 41 clusters, several of 1 cell).
    # Those clusters are dropped for this metric only, and the count is logged by the caller.
    ids, counts = np.unique(clean, return_counts=True)
    keep_ids = ids[counts >= min_cluster_size]
    if len(keep_ids) < 2:
        return None
    keep_mask = np.isin(clean, keep_ids)
    mask = np.where(mask)[0][keep_mask]
    clean = clean[keep_mask]
    clean = np.unique(clean, return_inverse=True)[1]   # relabel 0..k-1 contiguously

    try:
        if precomputed:
            sub = np.ascontiguousarray(X[mask][:, mask], dtype=np.float64)
            val = hdbscan.validity.validity_index(sub, clean, metric="precomputed", d=d or 2)
        else:
            val = hdbscan.validity.validity_index(X[mask].astype(np.float64), clean, metric=metric)
        return round(float(val), 4)
    except Exception:
        # Degenerate inputs (e.g. a cluster of one point) raise inside the library.
        return None


def bootstrap_stability(fit_fn, n_obs: int, n_boot: int = 10, frac: float = 0.8,
                        random_state: int = RANDOM_STATE) -> dict:
    """Re-fit on disjoint random subsamples and compare labels on the overlap with ARI.

    `fit_fn(idx)` must return a label array aligned to `idx`.

    ARI (Adjusted Rand Index) is 1 for identical partitions and ~0 for chance agreement:

        ARI = (RI - E[RI]) / (max RI - E[RI])

    Comparing on the *intersection* is what makes this fair: the two fits never see
    exactly the same rows, so agreement can only come from the structure, not the rows.
    Noise labels are compared too (as their own category), because a method that produces
    a different set of outliers each time is not stable either.
    """
    rng = np.random.default_rng(random_state)
    scores = []
    for _ in range(n_boot):
        a = rng.choice(n_obs, size=int(frac * n_obs), replace=False)
        b = rng.choice(n_obs, size=int(frac * n_obs), replace=False)
        common = np.intersect1d(a, b)
        if len(common) < 20:
            continue
        la = np.asarray(fit_fn(a))
        lb = np.asarray(fit_fn(b))
        pos_a = {v: i for i, v in enumerate(a)}
        pos_b = {v: i for i, v in enumerate(b)}
        ia = [pos_a[c] for c in common]
        ib = [pos_b[c] for c in common]
        scores.append(float(adjusted_rand_score(la[ia], lb[ib])))

    return {
        "n_boot": len(scores),
        "ari_mean": round(float(np.mean(scores)), 4) if scores else None,
        "ari_sd": round(float(np.std(scores)), 4) if scores else None,
        "ari_min": round(float(np.min(scores)), 4) if scores else None,
    }


def compare_table(rows: list[dict]) -> pd.DataFrame:
    """Assemble the comparison table with a documented column order."""
    cols = ["algorithm", "space", "n_clusters", "coverage", "noise_share", "silhouette",
            "silhouette_pca_space", "davies_bouldin", "calinski_harabasz", "dbcv",
            "ari_mean", "ari_sd", "ari_min", "ari_boot_n"]
    df = pd.DataFrame(rows)
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df[cols]
