# LCMS_data_scrape

A data scraping project for LC-MS analysis reference database on major Chinese medicine encyclopedia.

Currently working on:
1. Basic information of herb chemical components from [ECTM](http://www.tcmip.cn/ETCM)
    1. temip.cn seems to have anti-scraping implemented, If you've manually downloaded component info (the `tableExport*.txt` files) into the `component info` directory, you can aggregate fields into a single CSV with tcmip_txt_to_csv.py

## Prerequisites

- Google Chrome browser installed.
- ChromeDriver executable installed and available on PATH.
- Selenium installed