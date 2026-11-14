"""Model comparison page: six-algorithm evaluation and key figures."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from utils.load import (
    FIGURES_DIR,
    OUTPUTS_DIR,
    get_overview_kpis,
    load_cluster_comparison,
    load_cluster_profiles,
    load_rhythm_names,
)
from utils.theme import plotly_template, render_jargon_popover

comp = load_cluster_comparison(OUTPUTS_DIR)
kpis = get_overview_kpis(OUTPUTS_DIR)
n_cells = kpis["cells_analysed"]

km_rows = comp[comp["algorithm"] == "kmeans"]
ts_rows = comp[comp["algorithm"] == "tskmeans"]
hd_rows = comp[comp["algorithm"] == "hdbscan"]

k_beh = int(km_rows.iloc[0]["n_clusters"]) if not km_rows.empty else len(load_cluster_profiles(OUTPUTS_DIR))
k_rhy = int(ts_rows.iloc[0]["n_clusters"]) if not ts_rows.empty else len(load_rhythm_names(OUTPUTS_DIR))

with st.container(horizontal=True, vertical_alignment="center"):
    st.markdown(
        f"**Chosen model:** `kmeans` (k={k_beh} on PC1–PC5) · "
        f"**Temporal axis:** `tskmeans` (k={k_rhy} on 168-h profiles)"
    )
    render_jargon_popover()

if not km_rows.empty:
    km = km_rows.iloc[0]
    why_parts = [
        f"**Why K-Means was chosen:** K-Means achieved strong bootstrap stability "
        f"(**ARI = {km['ari_mean']:.3f} ± {km['ari_sd']:.3f}**, worst resample {km['ari_min']:.3f}), "
        f"covered **{km['coverage']:.0%}** of the {n_cells:,} analysed cells, "
        f"with Davies–Bouldin **{km['davies_bouldin']:.3f}** (lower is better) "
        f"and Calinski–Harabasz **{km['calinski_harabasz']:.2f}**."
    ]
    if not hd_rows.empty:
        hd = hd_rows.iloc[0]
        discarded = 1.0 - float(hd["coverage"])
        why_parts.append(
            f" HDBSCAN reported a higher silhouette (**{hd['silhouette']:.3f}** vs {km['silhouette']:.3f}) "
            f"only by discarding **{discarded:.1%}** of cells as noise ({hd['coverage']:.1%} coverage)."
        )
    st.success("".join(why_parts), icon=":material/check_circle:")

st.subheader("Six-algorithm evaluation metrics")

display_comp = comp.copy()
display_comp["selected"] = display_comp["algorithm"].eq("kmeans")

st.dataframe(
    display_comp[
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
    ],
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

# Key figures (at most two, if present on disk)
key_figures = [
    ("stage6_comparison.png", "Internal validity and bootstrap stability across the six algorithms."),
    ("stage4_pca_umap.png", "PCA and UMAP 2D projections (UMAP used for visualisation only)."),
]
existing_figs = [(FIGURES_DIR / fname, cap) for fname, cap in key_figures if (FIGURES_DIR / fname).exists()]
if existing_figs:
    st.divider()
    st.subheader("Key figures")
    cols = st.columns(len(existing_figs))
    for col, (fig_path, caption) in zip(cols, existing_figs):
        with col.container(border=True):
            st.image(str(fig_path), caption=caption, alt=caption, width="stretch")
