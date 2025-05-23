#!/usr/bin/env python3
import sys
import time
import argparse
import shutil
from urllib.parse import urlparse

import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import NoSuchElementException, TimeoutException

PAGE_LOAD_TIMEOUT = 30
DEFAULT_PAUSE = 1.0

def setup_driver(headless=False):
    chromedriver_path = shutil.which("chromedriver")
    if not chromedriver_path:
        sys.exit("ERROR: chromedriver executable not found in PATH.")
    options = webdriver.ChromeOptions()
    options.page_load_strategy = 'eager'
    if headless:
        options.add_argument('--headless=new')
        options.add_argument('--disable-gpu')
    service = Service(chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)
    # driver.set_page_load_timeout(PAGE_LOAD_TIMEOUT)   //npass websites are really slow
    return driver


def get_text_label_in_table(driver, label):
    """
    Find a table row where the first cell matches `label` exactly,
    and return the text of the second cell, or None if not found.
    """
    try:
        row = driver.find_element(
            By.XPATH,
            f"//table//tr[normalize-space(.//th[1] | .//td[1])='{label}']"
        )
        return row.find_element(By.XPATH, './td[1]').text.strip()
    except NoSuchElementException:
        return None


def parse_knapsack(driver):
    formula = get_text_label_in_table(driver, 'Formula')
    weight = get_text_label_in_table(driver, 'Mw')
    if weight is None:
        weight = get_text_label_in_table(driver, 'Molecular weight')
    return weight, formula


def parse_npass(driver):
    try:
        formula = driver.find_element(
            By.XPATH,
            "//dt[contains(normalize-space(),'Molecular Formula')]/following-sibling::dd[1]"
        ).text.strip()
    except NoSuchElementException:
        formula = None
    try:
        weight = driver.find_element(
            By.XPATH,
            "//dt[contains(normalize-space(),'Molecular Weight')]/following-sibling::dd[1]"
        ).text.strip()
    except NoSuchElementException:
        weight = None

    # NPASS often uses a <table class="table_with_border">…</table> for Mw;
    # if the dt/dd lookup failed or returned '0', fall back to grabbing from the table.
    if not weight or weight == '0':
        try:
            weight = driver.find_element(
                By.XPATH,
                "//table[contains(@class,'table_with_border')]"
                "//tr[td[1][contains(normalize-space(.),'Molecular Weight')]]/td[2]"
            ).text.strip()
        except NoSuchElementException:
            weight = None

    return weight, formula


def parse_wikidata(entity_id):
    import urllib.request, json

    api_url = (
        'https://www.wikidata.org/w/api.php'
        '?action=wbgetentities&ids=%s&props=claims&format=json' % entity_id
    )
    try:
        with urllib.request.urlopen(api_url, timeout=PAGE_LOAD_TIMEOUT) as f:
            data = json.load(f)
        claims = data['entities'][entity_id]['claims']
        formula = None
        weight = None
        if 'P274' in claims:
            formula = claims['P274'][0]['mainsnak']['datavalue']['value']
        if 'P2067' in claims:
            weight = claims['P2067'][0]['mainsnak']['datavalue']['value']
            # Wikidata returns a dict {'amount': '+<value>', 'unit': ...}; extract the numeric amount
            if isinstance(weight, dict):
                raw_amount = weight.get('amount')
                weight = raw_amount.lstrip('+') if raw_amount is not None else None
        return weight, formula
    except Exception as e:
        print(f"WARNING: failed to fetch Wikidata {entity_id}: {e}", file=sys.stderr)
        return None, None


def dispatch_parse(driver, url):
    hostname = urlparse(url).hostname or ''
    if 'knapsackfamily.com' in hostname:
        driver.get(url)
        return parse_knapsack(driver)
    if 'bidd.group' in hostname:
        driver.get(url)
        return parse_npass(driver)
    if 'wikidata.org' in hostname:
        entity_id = url.rstrip('/').rsplit('/', 1)[-1]
        return parse_wikidata(entity_id)
    print(f"WARNING: no parser available for {url}", file=sys.stderr)
    return None, None


def main(input_csv, output_csv, pause, headless):
    df = pd.read_csv(input_csv)
    df['Molecular_Weight'] = None
    df['Formula'] = None

    driver = setup_driver(headless=headless)
    try:
        for idx, row in df.iterrows():
            url = row.get('Source_Chemical_URL')
            if not isinstance(url, str) or not url.startswith('http'):
                continue

            weight, formula = dispatch_parse(driver, url)
            df.at[idx, 'Molecular_Weight'] = weight
            df.at[idx, 'Formula'] = formula
            print(f"[{idx+1}/{len(df)}] {url} -> Mw={weight}, Formula={formula}")
            time.sleep(pause)
    finally:
        driver.quit()

    df.to_csv(output_csv, index=False)
    print(f"Results written to {output_csv}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Scrape molecular weight and formula for pubchem chemicals.'
    )
    p.add_argument('input_csv', help='Input CSV (final_pubchem.csv)')
    p.add_argument('output_csv', help='Output CSV with Mw and Formula')
    p.add_argument('--headless', action='store_true', help='Run Chrome in headless mode')
    p.add_argument('--pause', type=float, default=DEFAULT_PAUSE,
                   help='Seconds to pause between requests')
    args = p.parse_args()
    main(args.input_csv, args.output_csv, args.pause, args.headless)