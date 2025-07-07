#!/usr/bin/env python3
import argparse
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch
from matplotlib.patches import Ellipse
import itertools

# helpers

# log10(x+1)
def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    return np.log10(df + 1.0)

# z-score (mean-centre, unit variance)
def autoscale(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)

def confidence_ellipse(x, y, ax, n_std=2.0, **kwargs):
    if len(x) != len(y):
        raise ValueError("x and y must be the same length")
    
    if len(x) == 2:
        # Special case: only 2 points
        x0, x1 = x
        y0, y1 = y
        mean_x = (x0 + x1) / 2
        mean_y = (y0 + y1) / 2
        dx = x1 - x0
        dy = y1 - y0
        width = n_std * np.hypot(dx, dy)
        height = 2  # hardcoded height
        theta = np.degrees(np.arctan2(dy, dx))
        ellipse = Ellipse((mean_x, mean_y), width, height, angle=theta, **kwargs)
        ax.add_patch(ellipse)
        return ellipse
    
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(vals)
    ellipse = Ellipse((np.mean(x), np.mean(y)), width, height, angle=theta, **kwargs)
    ax.add_patch(ellipse)
    return ellipse

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PCA + clustering of peak matrix")
    parser.add_argument("--matrix", required=True, help="CSV matrix from process_peak_tables.py")
    parser.add_argument("--out-dir", default="results", help="directory for outputs")
    parser.add_argument("--presence", type=float, default=0.0, help="retain peaks present (area>0) in ≥ fraction of samples (0-1)")
    parser.add_argument("--log", action="store_true", help="apply log10(x+1) transform")
    parser.add_argument("--autoscale", action="store_true", help="mean-centre & unit variance")
    parser.add_argument("--components", type=int, default=3, help="number of PCA components")
    parser.add_argument("--k-clusters", type=int, default=2, help="number of clusters for grouping scores plot")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load matrix
    X = pd.read_csv(args.matrix, index_col=0)
    print(f"Loaded matrix shape={X.shape}")

    # filter by presence
    if args.presence > 0:
        keep = (X > 0).sum(axis=0) >= args.presence * len(X)
        X = X.loc[:, keep]
        print(f"After presence filter ({args.presence}): shape={X.shape}")

    # log transform
    if args.log:
        X = log_transform(X)
        print("Applied log10(x+1)")

    # autoscale
    if args.autoscale:
        X = autoscale(X)
        print("Applied autoscaling")

    # PCA
    pca = PCA(n_components=args.components, random_state=1)
    scores = pca.fit_transform(X)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(args.components)],
    )
    scores_df = pd.DataFrame(
        scores,
        index=X.index,
        columns=[f"PC{i+1}" for i in range(args.components)],
    )

    scores_path = out_dir / "pca_scores.csv"
    loadings_path = out_dir / "pca_loadings.csv"
    scores_df.to_csv(scores_path)
    loadings.to_csv(loadings_path)
    print("Saved:", scores_path, loadings_path)

    # PCA scores plot with group-based styling
    # derive cluster groups from hierarchical clustering
    Z = sch.linkage(X.values, metric="euclidean", method="ward")
    groups = sch.fcluster(Z, t=args.k_clusters, criterion="maxclust")
    # export sample-to-cluster mapping for downstream analyses
    pd.DataFrame({'sample': X.index, 'cluster': groups}).to_csv(
        out_dir / 'sample_clusters.csv', index=False
    )
    # colors and markers
    palette = sns.color_palette("tab10", args.k_clusters)
    markers = itertools.cycle(['o', 's', '^', 'd', 'v', '<', '>', 'p', 'h'])

    fig, ax = plt.subplots(figsize=(7, 6))
    # hide top/right spines; keep left/bottom at border
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_position(('outward', 0))
    ax.spines['bottom'].set_position(('outward', 0))
    # central crosshair lines
    ax.axhline(0, color='gray', linewidth=0.8, zorder=0)
    ax.axvline(0, color='gray', linewidth=0.8, zorder=0)
    # ticks on border
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.tick_params(direction='out', labelsize=9)
    # plot per cluster
    for i in range(1, args.k_clusters + 1):
        idx = groups == i
        color = palette[i - 1]
        marker = next(markers)
        ax.scatter(scores_df.loc[idx, 'PC1'], scores_df.loc[idx, 'PC2'], s=70, facecolor=color, edgecolor='k', marker=marker, label=f"Cluster {i}")
        confidence_ellipse(scores_df.loc[idx, 'PC1'], scores_df.loc[idx, 'PC2'], ax, edgecolor=color, facecolor='none', linestyle='--', linewidth=1.2)

    # overall ellipse
    confidence_ellipse(scores_df['PC1'], scores_df['PC2'], ax, edgecolor='0.5', facecolor='none', linestyle='-', linewidth=1.5)

    # annotate points
    for txt in scores_df.index:
        x, y = scores_df.loc[txt, ['PC1', 'PC2']]
        # offset all labels to the right of the point
        dx = 0.5
        dy = 0.0
        ax.text(x + dx, y + dy, txt, fontsize=9)

    # labels and legend
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f} %)")
    # move X label to lower-right corner
    ax.xaxis.set_label_coords(0.95, -0.05)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f} %)")
    # move Y label to upper-left corner
    ax.yaxis.set_label_coords(-0.05, 0.95)
    ax.set_title("PCA scores plot")
    # place cluster legend on outer border to keep central area clean
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), frameon=False)
    plt.tight_layout()
    pca_plot = out_dir / "pca_scores_plot.png"
    plt.savefig(pca_plot, dpi=300)
    plt.close()

    print("Saved:", pca_plot)


    # additional PCA biplot figure with loadings arrows
    fig2, ax2 = plt.subplots(figsize=(7, 6))
    markers2 = itertools.cycle(['o', 's', '^', 'd', 'v', '<', '>', 'p', 'h'])
    # compute top 10 variable loadings & scale for arrows
    top10 = loadings.abs().sum(axis=1).nlargest(10).index
    x_max, x_min = scores_df['PC1'].max(), scores_df['PC1'].min()
    y_max, y_min = scores_df['PC2'].max(), scores_df['PC2'].min()
    scale = 0.5 * max(x_max - x_min, y_max - y_min)
    # reuse same styling for scores
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_position(('outward', 0))
    ax2.spines['bottom'].set_position(('outward', 0))
    ax2.axhline(0, color='gray', linewidth=0.8, zorder=0)
    ax2.axvline(0, color='gray', linewidth=0.8, zorder=0)
    ax2.xaxis.set_ticks_position('bottom')
    ax2.yaxis.set_ticks_position('left')
    ax2.tick_params(direction='out', labelsize=9)
    # axis arrows
    arrow_fmt = dict(arrowstyle='->', color='k', mutation_scale=10)
    ax2.annotate('', xy=(1, 0), xytext=(0, 0), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrow_fmt)
    ax2.annotate('', xy=(0, 1), xytext=(0, 0), xycoords='axes fraction', textcoords='axes fraction', arrowprops=arrow_fmt)
    # plot scores by cluster
    for i in range(1, args.k_clusters + 1):
        idx = groups == i
        color = palette[i - 1]
        marker = next(markers2)
        ax2.scatter(scores_df.loc[idx, 'PC1'], scores_df.loc[idx, 'PC2'], s=70, facecolor=color, edgecolor='k', marker=marker)
    # overall ellipse
    confidence_ellipse(scores_df['PC1'], scores_df['PC2'], ax2, edgecolor='0.5', facecolor='none', linestyle='-', linewidth=1.5)

    # loadings arrows (scaled by √eigenvalues)
    for var in top10:
        dx = loadings.loc[var, 'PC1'] * np.sqrt(pca.explained_variance_[0]) * scale
        dy = loadings.loc[var, 'PC2'] * np.sqrt(pca.explained_variance_[1]) * scale
        ax2.arrow(0, 0, dx, dy, color='red', alpha=0.8, width=0.005, length_includes_head=True)
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f} %)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f} %)")

    ax2.set_title("PCA biplot")
    # annotate sample points to the right of markers
    for txt in scores_df.index:
        x, y = scores_df.loc[txt, ['PC1', 'PC2']]
        ax2.text(x + 0.5, y, txt, fontsize=9)
    # cluster legend on outer border
    ax2.legend(frameon=False, loc='upper right', bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    biplot_file = out_dir / "pca_biplot.png"
    plt.savefig(biplot_file, dpi=300)
    plt.close()
    print("Saved:", biplot_file)

    # hierarchical cluster heatmap (Ward, Euclidean)
    cg = sns.clustermap(X.T, metric="euclidean", method="ward", cmap="viridis", figsize=(8, 8))
    cg.ax_heatmap.set_xlabel("Sample Type")
    cg.ax_heatmap.set_ylabel("Retention Time")
    plt.setp(cg.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
    plt.setp(cg.ax_heatmap.get_yticklabels(), rotation=0)
    heatmap_path = out_dir / "heatmap_cluster.png"
    cg.savefig(heatmap_path, dpi=300)
    plt.close()
    print("Saved:", heatmap_path)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
