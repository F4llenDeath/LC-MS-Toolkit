#!/usr/bin/env python3
import argparse
import pathlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score


def vip_scores(pls, X, Y):
    T = pls.x_scores_
    W = pls.x_weights_
    Q = pls.y_loadings_
    p, h = W.shape
    SSY = np.array([(T[:, comp] * Q[comp]).T.dot(T[:, comp] * Q[comp]) for comp in range(h)])
    Wnorm2 = np.sum(W**2, axis=0)
    vip = np.zeros((p,))
    for i in range(p):
        sum_term = np.sum(SSY * (W[i, :]**2) / Wnorm2)
        vip[i] = np.sqrt(p * sum_term / np.sum(SSY))
    return vip

def main():
    parser = argparse.ArgumentParser(description="Exploratory PLS-DA with VIP scoring")
    parser.add_argument("--matrix", required=True, help="CSV data matrix (rows=samples, cols=features)")
    parser.add_argument("--map-file", required=True, help="CSV file mapping samples to clusters")
    parser.add_argument("--out-dir", default="results", help="Directory for outputs")
    parser.add_argument("--n-pred", type=int, default=1, help="Number of predictive components")
    parser.add_argument("--top-n", type=int, default=20, help="Number of top features to plot")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load data
    data = pd.read_csv(args.matrix, index_col=0)
    X = data.values
    # load sample-to-cluster mapping
    map_df = pd.read_csv(args.map_file).set_index('sample')['cluster']
    y = data.index.to_series().map(map_df).values

    # z-score
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)

    # leave-one-out cv to fit PLS model
    loo = LeaveOneOut()
    preds = np.zeros_like(y, dtype=float)
    for train, test in loo.split(Xz):
        pls = PLSRegression(n_components=args.n_pred)
        pls.fit(Xz[train], y[train])
        preds[test] = pls.predict(Xz[test]).ravel()
    print(f"LOO R2: {r2_score(y, preds):.3f}")

    # final model on full data
    pls_final = PLSRegression(n_components=args.n_pred)
    pls_final.fit(Xz, y)

    # compute VIP
    vip = vip_scores(pls_final, Xz, y)
    vip_series = pd.Series(vip, index=data.columns, name="VIP_score")
    vip_out = out_dir / "plda_vip_scores.csv"
    vip_series.sort_values(ascending=False).to_csv(vip_out)

    print(f"Saved VIP scores: {vip_out}")

    # plot top n VIP scores as bar chart
    import matplotlib.pyplot as plt
    top_n_vip = vip_series.sort_values(ascending=False).head(args.top_n)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(top_n_vip.index.astype(str), top_n_vip.values, color='skyblue', edgecolor='k')
    ax.set_xlabel("Features")
    ax.set_ylabel("VIP Score")
    ax.set_title("Top VIP Scores")
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    vip_bar = out_dir / "plda_vip_bar.png"
    plt.savefig(vip_bar, dpi=300)
    plt.close()
    print(f"Saved VIP bar chart: {vip_bar}")

if __name__ == "__main__":
    main()
