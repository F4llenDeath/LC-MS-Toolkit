#!/usr/bin/env python3
import argparse
import pandas as pd


def merge_final_csvs(tcmip_csv, pubchem_csv, output_csv):
    # Load TCMIP and PubChem data
    df_t = pd.read_csv(tcmip_csv, dtype=str)
    df_p = pd.read_csv(pubchem_csv, dtype=str)

    # Rename TCMIP columns to align with PubChem
    df_t = df_t.rename(
        columns={
            'External Link to PubChem': 'Compound_CID',
            'Ingredient Name in English': 'Compound',
            'Molecular Formula': 'Formula',
            'Molecular Weight': 'Molecular_Weight',
        }
    )

    # Ensure CID columns are comparable
    df_t['Compound_CID'] = df_t['Compound_CID'].astype(str)
    df_p['Compound_CID'] = df_p['Compound_CID'].astype(str)

    # Merge on Compound_CID, preferring TCMIP data first, then PubChem
    merged = pd.merge(
        df_t,
        df_p,
        on='Compound_CID',
        how='outer',
        suffixes=('_tcmip', '_pubchem'),
    )

    # Combine primary fields: Compound, Formula, Molecular_Weight
    merged['Compound'] = merged['Compound_tcmip'].combine_first(merged.get('Compound_pubchem'))
    merged['Formula'] = merged['Formula_tcmip'].combine_first(merged.get('Formula_pubchem'))
    merged['Molecular_Weight'] = merged['Molecular_Weight_tcmip'].combine_first(
        merged.get('Molecular_Weight_pubchem')
    )

    # Drop intermediate columns
    drop_cols = [
        'Compound_tcmip', 'Compound_pubchem',
        'Formula_tcmip', 'Formula_pubchem',
        'Molecular_Weight_tcmip', 'Molecular_Weight_pubchem',
    ]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    # Reorder columns
    primary_cols = ['Compound_CID', 'Compound', 'Formula', 'Molecular_Weight']
    rest_cols = [c for c in merged.columns if c not in primary_cols]
    merged = merged[primary_cols + rest_cols]

    merged.to_csv(output_csv, index=False)


def main():
    parser = argparse.ArgumentParser(
        description='Merge final_tcmip.csv and final_pubchem.csv into a combined CSV.'
    )
    parser.add_argument(
        'tcmip_csv', help='Path to final_tcmip.csv (TCMIP data)'
    )
    parser.add_argument(
        'pubchem_csv', help='Path to final_pubchem.csv (PubChem data)'
    )
    parser.add_argument(
        'output_csv', help='Path for the merged output CSV'
    )
    args = parser.parse_args()
    merge_final_csvs(args.tcmip_csv, args.pubchem_csv, args.output_csv)


if __name__ == '__main__':
    main()