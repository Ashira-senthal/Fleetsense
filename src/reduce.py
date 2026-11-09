"""Stage 4 - dimensionality reduction.

Two jobs, deliberately kept apart:

1. **PCA = compression, used for clustering.** 14 correlated features contain fewer
   than 14 independent directions. Feeding all of them to K-Means double-counts the
   shared variance (speed_mean and speed_p95 are almost the same feature twice), which
   silently raises the weight of whichever behaviour has more redundant columns.

2. **UMAP = a picture, never an input to clustering.** UMAP distorts global distances
   and its axes are not interpretable; two points close on a UMAP plot are "similar",
   but the *distance* between clusters on that plot means nothing. Clustering on a UMAP
   embedding invents clusters that do not exist in the feature space.

MATHS OF PCA (why eigenvectors of the covariance matrix)
--------------------------------------------------------
Let X be the n x p standardised matrix (already z-scored in Stage 3). Its sample
covariance is

    C = (1/(n-1)) * X^T X

which, because every column has mean 0 and sd 1, *is* the correlation matrix: PCA on
standardised data is correlation-based PCA, and off-diagonal entries tell you which
features are redundant.

We look for a unit direction v maximising the projected variance:

    maximise  v^T C v   subject to  ||v|| = 1

With a Lagrange multiplier, the stationarity condition is C v = lambda v: the solutions
are the eigenpairs of C. Sorting eigenvalues descending, the k-th component is

    z_k = X v_k,        explained variance ratio = lambda_k / sum_j lambda_j

and because v_k are orthogonal, the scores z_k are uncorrelated. Total variance is
conserved: sum_k lambda_k = p, so keeping k components retains
(sum_{j<=k} lambda_j)/p of the information measured in variance - and drops the rest,
which is exactly the "compression" part.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness

RANDOM_STATE = 42


def correlation_matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Pearson correlation between features - the evidence that PCA is warranted."""
    return frame[cols].corr()


def fit_pca(frame: pd.DataFrame, cols: list[str], n_components: int | None = None,
            random_state: int = RANDOM_STATE) -> dict:
    """Fit PCA on already-standardised columns.

    Returns a dict with: model, scores (DataFrame, one row per observation),
    explained variance table, and the loadings table (feature x component).
    """
    X = frame[cols].to_numpy(dtype=float)
    model = PCA(n_components=n_components, random_state=random_state)
    scores = model.fit_transform(X)

    score_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
    scores_df = pd.DataFrame(scores, columns=score_cols, index=frame.index)
    scores_df.insert(0, "h3_cell", frame["h3_cell"].to_numpy() if "h3_cell" in frame else np.arange(len(frame)))

    evr = pd.DataFrame({
        "component": score_cols,
        "eigenvalue": model.explained_variance_,
        "explained_variance_ratio": model.explained_variance_ratio_,
        "cumulative_ratio": np.cumsum(model.explained_variance_ratio_),
    })

    loadings = pd.DataFrame(model.components_.T, index=cols, columns=score_cols)
    return {"model": model, "scores": scores_df, "explained_variance": evr, "loadings": loadings}


def components_for_variance(evr: pd.DataFrame, threshold: float = 0.80) -> int:
    """Smallest number of components reaching `threshold` cumulative variance."""
    hit = evr[evr["cumulative_ratio"] >= threshold]
    if hit.empty:
        return int(evr["component"].str.removeprefix("PC").astype(int).max())
    return int(hit["component"].iloc[0].removeprefix("PC"))


def top_loadings(loadings: pd.DataFrame, component: str, k: int = 4) -> list[tuple[str, float]]:
    """Highest-|loading| features of one component, for human interpretation."""
    s = loadings[component].sort_values(key=np.abs, ascending=False).head(k)
    return [(str(idx), float(val)) for idx, val in s.items()]


def fit_umap(scores: pd.DataFrame, n_pcs: int = 10, n_neighbors: int = 15,
             min_dist: float = 0.1, random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, dict]:
    """Project the first `n_pcs` PCA scores to 2-D with UMAP, FOR PLOTTING ONLY.

    Parameters and what they actually control:
      * n_neighbors : small (5-15) emphasises local structure; large (50+) emphasises
        global layout but blurs small clusters. 15 is a middle default.
      * min_dist    : how tightly points may pack. 0.1 keeps clusters visible as blobs
        instead of collapsing them into a hairball.
      * metric      : 'euclidean' is correct here because the input is *standardised
        PCA scores*. Haversine would be wrong: these are not coordinates.
    Running UMAP on 10 PCs (rather than 14 correlated features) is the documented
    recommendation: it removes noise dimensions and is much faster.
    """
    import umap  # local import: UMAP is only needed by this function

    pc_cols = [f"PC{i+1}" for i in range(n_pcs)]
    X = scores[pc_cols].to_numpy(dtype=float)

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="euclidean",
                        random_state=random_state, n_components=2)
    xy = reducer.fit_transform(X)

    coords = pd.DataFrame({"h3_cell": scores["h3_cell"].to_numpy(),
                           "umap_x": xy[:, 0], "umap_y": xy[:, 1]})
    params = {"n_pcs": n_pcs, "n_neighbors": n_neighbors, "min_dist": min_dist,
              "metric": "euclidean", "random_state": random_state}
    return coords, params


def neighbourhood_trust(X: np.ndarray, embeddings: dict[str, np.ndarray],
                        n_neighbors: int = 10) -> pd.DataFrame:
    """Trustworthiness: are a point's nearest neighbours in 2-D also its neighbours in
    the full feature space?

    Score in [0, 1]; higher is better, and it is comparable across embeddings of the
    same data. This is the quantitative answer to "can I believe the picture?" - a
    pretty plot with low trustworthiness is decoration.
    """
    rows = []
    for name, emb in embeddings.items():
        rows.append({"embedding": name, "dims": emb.shape[1],
                     "trustworthiness": round(float(trustworthiness(X, emb, n_neighbors=n_neighbors)), 4)})
    return pd.DataFrame(rows).sort_values("trustworthiness", ascending=False)
