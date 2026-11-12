"""Model comparison page: six-algorithm evaluation and pipeline figure gallery."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.load import (
    OUTPUTS_DIR,
    list_figures,
    load_cluster_comparison,
    load_json_output,
    load_manifest,
)
from utils.theme import plotly_template, render_jargon_popover

comp = load_cluster_comparison(OUTPUTS_DIR)
comp_meta = load_json_output("cluster_comparison.json", OUTPUTS_DIR)
manifest = load_manifest(OUTPUTS_DIR)
figures = list_figures(OUTPUTS_DIR, manifest)

FIGURE_CAPTIONS: dict[str, str] = {
    "stage1_stop_coverage.png": "Stage 1: Spatial extent of the initial Delhi OTD trimmed GTFS audit (retained for auditability).",
    "stage2_removed_points.png": "Stage 2: GPS points removed by cleaning rules on the synthetic test fixture.",
    "stage2_speed_before_after.png": "Stage 2: Speed distribution before vs after the 100 km/h jump filter.",
    "stage3_cell_activity.png": "Stage 3: H3 resolution-8 cell visit distribution and ≥5-visit support threshold.",
    "stage3_cyclic_encoding.png": "Stage 3: Sine/cosine cyclic encoding of hour-of-week and day-of-week.",
    "stage3_scaling_effect.png": "Stage 3: Feature scaling effect prior to PCA.",
    "stage3_weekly_profile.png": "Stage 3: Aggregate 168-hour weekly activity curve.",
    "stage4_correlation.png": "Stage 4: Feature correlation matrix across cell-level mobility metrics.",
    "stage4_loadings.png": "Stage 4: Principal component loadings (PC1–PC5).",
    "stage4_pca_umap.png": "Stage 4: PCA and UMAP 2D projections (UMAP used for visualisation only).",
    "stage4_scree.png": "Stage 4: PCA explained variance scree plot.",
    "stage5_cluster_maps.png": "Stage 5: Spatial comparison of candidate clustering algorithms.",
    "stage5_dbscan_selection.png": "Stage 5: Spatial DBSCAN epsilon / min_samples selection.",
    "stage5_dtw_profiles.png": "Stage 5: TS-K-Means DTW 168-hour rhythm prototypes.",
    "stage5_gmm_bic.png": "Stage 5: Gaussian Mixture Model BIC curve across k.",
    "stage5_kmeans_selection.png": "Stage 5: K-Means elbow (inertia) and silhouette across k.",
    "stage6_agreement.png": "Stage 6: Cross-algorithm agreement (Adjusted Rand Index matrix).",
    "stage6_comparison.png": "Stage 6: Internal validity and bootstrap stability across all six algorithms.",
    "stage6_k_selection.png": "Stage 6: Final k=4 justification across inertia, BIC, and silhouette.",
    "stage7_anomalies.png": "Stage 7: Flagged spatial/temporal anomaly cells and robust-z distributions.",
    "stage7_poi.png": "Stage 7: OpenStreetMap POI density per H3 cell by cluster (bus stops excluded).",
    "stage7_profiles.png": "Stage 7: Cluster feature z-scores and 168-hour activity curves.",
}

with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown(
        "**Chosen model:** `kmeans` (k = 4 on PC1–PC5) · "
        "**Temporal axis:** `tskmeans` (k = 4 on 168-h profiles)"
    )
    render_jargon_popover()

st.success(
    "**Why K-Means was chosen:** K-Means achieved the highest bootstrap stability "
    "(**ARI = 0.851 ± 0.052**, worst resample 0.758), covered **100%** of the 602 analysed cells, "
    "and led on Davies–Bouldin (**1.435**, lower is better) and Calinski–Harabasz (**182.19**). "
    "HDBSCAN reported a higher silhouette (**0.333** vs 0.244) only by discarding **60.3%** of cells "
    "as noise (39.7% coverage).",
    icon=":material/check_circle:",
)

st.subheader("Six-algorithm evaluation metrics")

display_comp = comp.copy()
display_comp["selected"] = display_comp["algorithm"].eq("kmeans")


def _highlight_kmeans(row: pd.Series) -> list[str]:
    if row["algorithm"] == "kmeans":
        return ["font-weight: 700; background-color: rgba(100, 143, 255, 0.16)"] * len(row)
    return [""] * len(row)


styled_comp = display_comp[
    [
        "selected",
        "algorithm",
        "space",
        "n_clusters",
        "coverage",
        "silhouette",
        "silhouette_pca_space",
        "davies_bouldin",
        "calinski_harabasz",
        "dbcv",
        "ari_mean",
        "ari_sd",
    ]
].style.apply(_highlight_kmeans, axis=1)

st.dataframe(
    styled_comp,
    column_config={
        "selected": st.column_config.CheckboxColumn("Chosen"),
        "algorithm": st.column_config.TextColumn("Algorithm"),
        "space": st.column_config.TextColumn("Feature space"),
        "n_clusters": st.column_config.NumberColumn("Clusters", format="%d"),
        "coverage": st.column_config.NumberColumn("Coverage", format="percent"),
        "silhouette": st.column_config.NumberColumn("Silhouette (own)", format="%.3f"),
        "silhouette_pca_space": st.column_config.NumberColumn("Silhouette (PC1–5)", format="%.3f"),
        "davies_bouldin": st.column_config.NumberColumn("Davies–Bouldin ↓", format="%.3f"),
        "calinski_harabasz": st.column_config.NumberColumn("Calinski–Harabasz ↑", format="%.1f"),
        "dbcv": st.column_config.NumberColumn("DBCV ↑", format="%.3f"),
        "ari_mean": st.column_config.NumberColumn("Bootstrap ARI ↑", format="%.3f"),
        "ari_sd": st.column_config.NumberColumn("ARI SD", format="%.3f"),
    },
    hide_index=True,
    alt="Evaluation metrics and bootstrap stability across six clustering algorithms",
)

# Stability vs Coverage chart
fig_eval = go.Figure()
for _, row in comp.iterrows():
    is_winner = row["algorithm"] == "kmeans"
    fig_eval.add_trace(
        go.Bar(
            name=row["algorithm"],
            x=[row["algorithm"]],
            y=[float(row["ari_mean"])],
            error_y=dict(type="data", array=[float(row["ari_sd"])], visible=True),
            marker_color="#648FFF" if is_winner else "#8b949e",
            text=[f"Cov {row['coverage']:.0%}"],
            textposition="outside",
            showlegend=False,
        )
    )

fig_eval.update_layout(
    title="Bootstrap stability (mean ARI ± 1 SD) with cell coverage",
    yaxis_title="Bootstrap Adjusted Rand Index (ARI)",
    yaxis_range=[0, 1.05],
    template=plotly_template(),
    margin=dict(t=50, b=40, l=40, r=20),
    height=360,
)
st.plotly_chart(
    fig_eval,
    alt="Bar chart comparing bootstrap ARI stability and coverage across the six algorithms",
)

with st.expander("Metric notes from cluster_comparison.json"):
    for k, v in comp_meta.get("notes", {}).items():
        st.markdown(f"- **{k}:** {v}")

st.download_button(
    "Download model comparison table (CSV)",
    data=comp.to_csv(index=False).encode("utf-8"),
    file_name="cluster_comparison.csv",
    mime="text/csv",
    icon=":material/download:",
)

# Pipeline figure gallery
st.divider()
st.subheader("Pipeline figures gallery")

if not figures:
    st.info("No PNG figures were found in `outputs/figures/`.", icon=":material/image:")
else:
    stages = sorted({p.name.split("_")[0] for p in figures})
    selected_stages = st.pills(
        "Filter by pipeline stage",
        options=stages,
        default=stages,
        selection_mode="multi",
        key="gallery_stage_filter",
    )
    visible_figs = [p for p in figures if p.name.split("_")[0] in selected_stages]
    st.caption(f"Showing {len(visible_figs)} of {len(figures)} figures from `outputs/figures/`.")

    for i in range(0, len(visible_figs), 2):
        cols = st.columns(2)
        for col, fig_path in zip(cols, visible_figs[i : i + 2]):
            caption = FIGURE_CAPTIONS.get(fig_path.name, fig_path.name.replace("_", " "))
            with col.container(border=True):
                st.image(
                    str(fig_path),
                    caption=f"{fig_path.name} — {caption}",
                    alt=caption,
                    width="stretch",
                )

