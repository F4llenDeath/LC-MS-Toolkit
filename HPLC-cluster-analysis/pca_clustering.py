#!/usr/bin/env python3
import argparse
import pathlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# helpers

# log10(x+1)  
def log_transform(df: pd.DataFrame) -> pd.DataFrame:
    return np.log10(df + 1.0)

# z-score (mean-centre, unit variance)
def autoscale(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std(ddof=0)

def main() -> None:
    parser = argparse.ArgumentParser(description="PCA + clustering of peak matrix")
    parser.add_argument("--matrix", required=True, help="CSV matrix from process_peak_tables.py")
    parser.add_argument("--out-dir", default="results", help="directory for outputs")
    parser.add_argument("--presence", type=float, default=0.0,
                        help="retain peaks present (area>0) in ≥ fraction of samples (0-1)")
    parser.add_argument("--log", action="store_true", help="apply log10(x+1) transform")
    parser.add_argument("--autoscale", action="store_true", help="mean-centre & unit variance")
    parser.add_argument("--components", type=int, default=3, help="number of PCA components")
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

    # scatter plot PC1 vs PC2
    plt.figure(figsize=(6, 5))
    sns.scatterplot(
        x=scores_df["PC1"],
        y=scores_df["PC2"],
        s=60,
        color="dodgerblue",
        edgecolor="k",
    )
    for txt, (x, y) in scores_df[["PC1", "PC2"]].iterrows():
        plt.text(x, y, txt, fontsize=8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f} %)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f} %)")
    plt.title("PCA score plot")
    plt.tight_layout()
    pca_plot = out_dir / "pca_scores_plot.png"
    plt.savefig(pca_plot, dpi=300)
    plt.close()
    print("Saved:", pca_plot)

    # hierarchical cluster heatmap (Ward, Euclidean)
    sns.clustermap(X, metric="euclidean", method="ward", cmap="viridis", figsize=(8, 8))
    heatmap_path = out_dir / "heatmap_cluster.png"
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print("Saved:", heatmap_path)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
