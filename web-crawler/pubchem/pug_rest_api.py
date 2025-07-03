#!/usr/bin/env python3
"""
python web-crawler/pubchem/pug_rest_api.py \
    --input results/crawler/metabolites.csv \
    --id-column name \
    --output results/crawler/metabolites_with_pubchem.csv \
    --props MolecularFormula,MolecularWeight \
    --batch-size 100 \
    --sleep 0.2 \
    --cache pubchem_props_cache.csv
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import requests
import pandas as pd

# low-level HTTP helper
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "LCMS_toolkit/1.0"})

def pug_request(namespace: str, ids: List[str], props: str, retries: int = 3) -> Dict[str, Dict[str, str]]:
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/{namespace}/property/{props}/JSON"
    payload = {namespace: ids, "property": props}
    backoff = 1.0

    for attempt in range(retries):
        try:
            r = SESSION.post(url, data=payload, timeout=30)
            r.raise_for_status()
            data = r.json()["PropertyTable"]["Properties"]
            key_field = "CID" if namespace == "cid" else "Name"
            return {str(item[key_field]): item for item in data}
        except Exception as exc:
            if attempt == retries - 1:
                raise
            time.sleep(backoff)
            backoff *= 2
            continue

# caching utilities
from pandas.errors import EmptyDataError

def load_cache(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    try:
        df = pd.read_csv(path)
    except EmptyDataError:
        return {}
    return {str(row["identifier"]): row.drop("identifier").to_dict()
            for _, row in df.iterrows()}

def save_cache(cache: Dict[str, Dict[str, str]], path: Path) -> None:
    rows = [{"identifier": k, **v} for k, v in cache.items()]
    pd.DataFrame(rows).to_csv(path, index=False)

# main routine
def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch PubChem properties via PUG-REST")
    ap.add_argument("--input", required=True, help="Input CSV file")
    ap.add_argument("--id-column", default="name", help="Column containing identifiers")
    ap.add_argument("--cid", action="store_true", help="Treat identifiers as numeric CIDs")
    ap.add_argument("--output", help="Output CSV (default: <input>_with_pubchem.csv)")
    ap.add_argument("--props", default="MolecularFormula,MolecularWeight",
                    help="Comma-separated property list")
    ap.add_argument("--batch-size", type=int, default=100, help="IDs per request (max 100)")
    ap.add_argument("--sleep", type=float, default=0.2, help="Pause between requests (s)")
    ap.add_argument("--cache", help="Optional CSV cache of previous calls")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path.with_name(in_path.stem + "_with_pubchem.csv")
    cache_path = Path(args.cache) if args.cache else None

    df = pd.read_csv(in_path)
    if args.id_column not in df.columns:
        sys.exit(f"Column {args.id_column!r} not found in {in_path}")

    ids = df[args.id_column].astype(str).tolist()
    namespace = "cid" if args.cid else "name"

    # load cache if present
    cache: Dict[str, Dict[str, str]] = {}
    if cache_path:
        cache = load_cache(cache_path)

    # figure out which IDs still need querying
    to_query = [i for i in ids if i not in cache]
    print(f"{len(ids)} total IDs  /  {len(to_query)} to query (cached {len(ids)-len(to_query)})")

    # batch loop
    for i in range(0, len(to_query), args.batch_size):
        batch = to_query[i : i + args.batch_size]
        print(f"Fetching batch {i // args.batch_size + 1}  (size {len(batch)}) ...", end="", flush=True)
        try:
            props_dict = pug_request(namespace, batch, args.props)
            cache.update(props_dict)
            print(" done.")
        except Exception as exc:
            print(f" failed ({exc}).")
        time.sleep(args.sleep)

    # save cache
    if cache_path:
        save_cache(cache, cache_path)

    # add columns back to DataFrame
    prop_names = args.props.split(",")
    for prop in prop_names:
        df[prop] = df[args.id_column].astype(str).map(
            lambda x: cache.get(x, {}).get(prop, "")
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Saved", out_path)

if __name__ == "__main__":
    main()
