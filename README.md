# LC-MS Toolkit

This repository covers some LC-MS related toolkit for a small-molecule study on herbs
1. Web scraping of phytochemical references
2. Automated processing of raw HPLC/LC-MS chromatograms

## 1. Reference-Database Scraping (`web-crawler/`)

| Folder | Purpose | Main scripts |
| ------ | ------- | ------------ |
| `tcmip/`  | Parse ETCM/TCMIP herb component lists. | `tcmip_txt_to_csv.py` |
| `pubchem/`| Merge PubChem *Metabolites* + *Natural Products* and scrape mol-weight / formula from external DB pages. | `merge_pubchem_csv.py`, `scrape_mw_formula.py`, `pug_rest_api.py` |

Work in progress for more supported databases

## 2. Chromatogram Processing (`HPLC-cluster-analysis/`)

### `process_peak_tables.py` 
converts Shimadzu LabSolutions TXT exports into a tidy sample × feature matrix.

```
python HPLC-cluster-analysis/process_peak_tables.py \
    --input-dir raw-data-dir \
    --out-prefix \
    --norm               # optional: total-area normalisation (ppm)
    --tol 0.15           # RT tolerance in minutes (default 0.15)
```

Outputs:

* `peak_matrix_raw.csv`   — unnormalised peak areas  
* `peak_matrix_norm_ppm.csv` (if `--norm`)   — total-area-normalised  

The script:

1. Reads every TXT in `--input-dir`
2. Extracts `[Peak Table]` blocks (columns: `R.Time`, `Area`)
3. Clusters retention times within ± `tol` min to build consensus features
4. Returns a DataFrame (rows = samples, cols = consensus RTs) and writes CSV

### `pca_clustering.py`

Light-weight PCA + hierarchical clustering utility for the peak-area matrices created by `process_peak_tables.py`.

```
python analysis/pca_clustering.py \
        --matrix matrix-path \
        --out-dir \
        --presence 0.5     # keep peaks present in ≥50 % samples
        --log              # log10(x+1) transform
        --autoscale        # mean-centre & unit variance
```

Outputs: 
* `pca_scores.csv`
* `pca_loadings.csv`
* `pca_scores_plot.png`
* `heatmap_cluster.png`

## 3. Dependencies

* Python ≥ 3.9  
* **Core analysis**: 
    - `numpy`
    - `pandas`  
    - `scikit-learn`
* **Web scraping**: 
    - `requests`
    - `lxml`
    - `selenium`
    - `requests`
    - a local Chrome/Chromium + ChromeDriver

```
pip install -r requirements.txt        
```