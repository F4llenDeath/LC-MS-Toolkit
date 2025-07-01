#!/usr/bin/env python3
import os
import sys
import pandas as pd

def main():
    input_files = ['metabolites.csv', 'natural_products.csv']
    dfs = []

    for fname in input_files:

        # Load and validate each input file
        if not os.path.isfile(fname):
            sys.exit(f"Input file not found: {fname}")

        # Read only the necessary columns from the CSV
        df = pd.read_csv(
            fname,
            usecols=['Compound_CID', 'Compound', 'Source_Chemical', 'Source_Chemical_URL']
        )
        dfs.append(df)

    # Concatenate both DataFrames into a single one
    combined = pd.concat(dfs, ignore_index=True)
    # Remove duplicate entries based on Compound_CID, keeping the first occurrence
    combined = combined.drop_duplicates(subset=['Compound_CID'], keep='first')

    output_file = 'pubchem_combined.csv'
    combined.to_csv(output_file, index=False)
    print(f"Merged CSV written to: {output_file}")

if __name__ == "__main__":
    main()