# LCMS_data_scrape

A data scraping project for LC-MS analysis reference database on major Chinese medicine datasets.

Currently support:
1. Basic information of herb chemical components from [ECTM](http://www.tcmip.cn/ETCM)
    1. temip.cn seems to have anti-scraping implemented, If you've manually downloaded component info (the `tableExport*.txt` files) into the `component info` directory, you can aggregate fields into a single CSV with `tcmip_txt_to_csv.py`
2. Information of herb chemical components from [ncbi](https://pubchem.ncbi.nlm.nih.gov/taxonomy/94219#section=Natural-Products)
    1. Metabolite and natural products list are directly downloaded from the site. `merge_pubchem_csv.py` merges these two lists.
    2. After generating `pubchem/pubchem_raw.csv`, you can scrape the molecular weight and formula for each chemical URL (handling KNApSAcK, NPASS, and Wikidata layouts) with `scrape_mw_formula.py`

## Prerequisites

- Google Chrome browser installed.
- ChromeDriver executable installed and available on PATH.
- python libraries installed:
    - pandas
    - selenium