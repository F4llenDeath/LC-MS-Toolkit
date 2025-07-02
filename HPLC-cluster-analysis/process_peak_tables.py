#!/usr/bin/env python3
import argparse
import glob
import pathlib
import io
from typing import Dict, List
import numpy as np
import pandas as pd

# txt → DataFrame (R.Time, Area)
# Extract the [Peak Table] block from a LabSolutions TXT export and return a DataFrame with numeric R.Time and Area columns
def read_peak_table(txt_path: str) -> pd.DataFrame:

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as fh:
        lines = fh.readlines()

    # locate the start of the Peak Table
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("[Peak Table"))
    except StopIteration:
        raise ValueError(f"No '[Peak Table]' block found in {txt_path!s}")

    # header line is two rows below the block title
    header_line = start + 2
    peak_block = "".join(lines[header_line:])

    df = pd.read_csv(
        io.StringIO(peak_block),
        sep="\t",
        engine="python",
    )

    # enforce numeric and keep relevant columns
    df = df.loc[
        pd.to_numeric(df["R.Time"], errors="coerce").notna(), ["R.Time", "Area"]
    ]
    df["R.Time"] = df["R.Time"].astype(float)
    df["Area"] = pd.to_numeric(df["Area"], errors="coerce").fillna(0)

    return df.reset_index(drop=True)

# Build feature list (consensus peaks)
# Cluster retention times so that RTs within <tol> minutes form one feature. 
# Returns the centroid (mean RT) of each cluster.
def define_features(retention_times: List[float], tol: float = 0.15) -> np.ndarray:

    rts = np.sort(np.asarray(retention_times))
    if rts.size == 0:
        return np.array([])

    groups = [[rts[0]]]
    centroids = [rts[0]]

    for rt in rts[1:]:

        # if the current RT is within `tol` minutes of the last cluster’s centroid
        if rt - centroids[-1] > tol:
            # it belongs to that cluster
            # append to the group and update the centroid to the mean of that group.
            groups.append([rt])
            centroids.append(rt)
        else:
            # otherwise it starts a new cluster
            # create a new group and centroid.
            groups[-1].append(rt)
            centroids[-1] = float(np.mean(groups[-1]))

    return np.asarray(centroids)

# Assemble sample × feature matrix 
# Return DataFrame (rows: samples, cols: features) with peak areas
def build_matrix(
    sample_tables: Dict[str, pd.DataFrame],
    features: np.ndarray,   # list of consensus peak centroids (RTs)
    tol: float = 0.15,
) -> pd.DataFrame:
    cols = [f"{rt:.2f}" for rt in features]
    data = []

    for sample, df in sample_tables.items():
        row = []
        for f_rt in features:
            # For each feature RT it searches df for a peak whose RT is within ± tol minutes of that centroid
            match = df.loc[(df["R.Time"] - f_rt).abs() <= tol, "Area"]
            row.append(float(match.iloc[0]) if not match.empty else 0.0)
        data.append(row)

    return pd.DataFrame(data, index=list(sample_tables.keys()), columns=cols)

# CLI interface
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a peak-area matrix from LabSolutions TXT exports."
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing *.txt peak-table files",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=0.15,
        help="Retention-time tolerance (minutes) when grouping peaks",
    )
    parser.add_argument(
        "--out-prefix",
        default="peak_matrix",
        help="Prefix for output CSV files",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Also write total-area-normalised matrix (ppm)",
    )

    args = parser.parse_args()

    txt_files = glob.glob(str(pathlib.Path(args.input_dir) / "*.txt"))
    if not txt_files:
        raise SystemExit(f"No TXT files found in {args.input_dir!s}")

    # per-sample tables & collect RTs
    sample_tables, all_rts = {}, []
    for fpath in txt_files:
        sample = pathlib.Path(fpath).stem  
        df = read_peak_table(fpath)
        sample_tables[sample] = df
        all_rts.extend(df["R.Time"].to_numpy())

    # consensus features
    features = define_features(all_rts, args.tol)
    print(f"Defined {len(features)} consensus peaks (tol = {args.tol} min)")

    # matrix assembly
    matrix = build_matrix(sample_tables, features, args.tol)
    out_raw = f"{args.out_prefix}_raw.csv"
    pathlib.Path(out_raw).parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(out_raw)
    print("Wrote raw matrix →", out_raw)

    # optional total-area normalisation
    if args.norm:
        norm_mat = matrix.div(matrix.sum(axis=1), axis=0) * 1e6  # ppm
        out_norm = f"{args.out_prefix}_norm_ppm.csv"
        norm_mat.to_csv(out_norm)
        print("Wrote normalised matrix →", out_norm)


if __name__ == "__main__":
    main()
