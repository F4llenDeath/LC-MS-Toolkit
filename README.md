# LC-MS Toolkit

This repository covers some LC-MS related toolkit for a small-molecule study on herbs
1. Web scraping of phytochemical references
2. Automated processing of raw HPLC/LC-MS chromatograms

## 1. Reference-Database Scraping (`web-crawler/`)

| Folder | Purpose | Main scripts |
| ------ | ------- | ------------ |
| `tcmip/`  | Parse ETCM/TCMIP herb component lists. | `tcmip_txt_to_csv.py` |
| `pubchem/`| Merge PubChem *Metabolites* + *Natural Products* and fetch compound properties via PubChem PUG-REST. | `merge_pubchem_csv.py`, `pug_rest_api.py`  (legacy Selenium scraper in `pubchem/legacy/`) |

### `pug_rest_api.py`  
retrieves PubChem [compound properties](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest#section=Compound-Property-Tables) in bulk and merges them into a CSV.

```
python web-crawler/pubchem/pug_rest_api.py \
    --input  compound-list-dir \
    --id-column cid-column-title \
    --cid                       # identifiers are CIDs (omit for names) \
    --props expected-properties \
    --output dir \
    --cache  pubchem_cache.csv \
    --auto-delete-cache
```

Outputs:

* `pubchem_with_props.csv` — original columns + requested properties  
  (e.g. Formula, MW, ExactMass, …)

The script:

1. Reads the input CSV.  
2. Splits identifiers into ≤100-ID batches and POSTs to PubChem PUG-REST.  
3. Retrieves the specified property list (default Formula & MW).  
4. Optionally caches results and deletes the cache when finished.  
5. Writes a merged CSV with one row per compound.

Work in progress for more supported databases

## 2. Chromatogram Processing (`cluster-analysis/`)

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
        --matrix matrix-file \
        --out-dir \
        --presence 0.5     # keep peaks present in ≥50 % samples
        --log              # log10(x+1) transform
        --autoscale        # mean-centre & unit variance
        --k-clusters       # number of clusters for grouping scores plot
```

Outputs: 
* `pca_scores.csv`
* `pca_loadings.csv`
* `pca_scores_plot.png`
* `pca_biplot.png`
* `heatmap_cluster.png`
* `sample_clusters.csv`

### `opls_da.py`
Exploratory PLS-DA with leave-one-out cross-validation and VIP scoring.

```
python cluster-analysis/opls_da.py \
    --matrix matrix-file \
    --map-file sample_clusters.csv \
    --out-dir  \
    --n-pred 1 \
    --top-n 20
```

Outputs:
* `plda_vip_scores.csv` 
* `plda_vip_bar.png` 

Work in progress for more cluster analysis techniques

## 3. Dependencies

* Python ≥ 3.9  
* **Core analysis**: 
    - `numpy`
    - `pandas`  
    - `scikit-learn`
* **Web scraping**: 
    - `requests`
    - `lxml`
    - `selenium`   *(only needed for legacy scraper)*
    - a local Chrome/Chromium + ChromeDriver

```
pip install -r requirements.txt        
```

## Acknowledgements

I sincerely thank [Prof. Lina Chen](https://english.njmu.edu.cn/_t965/2019/0129/c11686a142685/page.htm) and her lab at [NJMU](https://www.njmu.edu.cn) for generously providing the raw chemical data used in this project.