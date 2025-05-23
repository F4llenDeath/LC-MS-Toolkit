#!/usr/bin/env python3
import csv
import glob
import os

def find_component_folder():
    for name in os.listdir('.'):
        if os.path.isdir(name) and name.lower().startswith('tcmip_component_info'):
            return name
    raise FileNotFoundError("Could not find a directory starting with 'component info'")

def extract_index(path):
    base = os.path.basename(path)
    stem = base[len('tableExport'):-len('.txt')]
    if stem.startswith('-'):
        try:
            return int(stem[1:])
        except ValueError:
            pass
    return 0

def parse_fields(filepath, fields):
    data = {}
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            key = row[0].strip().strip('"')
            if key in fields:
                data[key] = row[1].strip()
                if all(field in data for field in fields):
                    break
    return data

def main():
    folder = find_component_folder()
    pattern = os.path.join(folder, 'tableExport*.txt')
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No export files found in '{folder}' matching tableExport*.txt")
    files.sort(key=extract_index)

    fields = ['Ingredient Name in English', 'Molecular Formula', 'Molecular Weight']

    output_csv = 'components.csv'
    with open(output_csv, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fields)
        writer.writeheader()
        for filepath in files:
            row = parse_fields(filepath, fields)
            writer.writerow({field: row.get(field, '') for field in fields})

    print(f"Wrote {len(files)} records to '{output_csv}'")

if __name__ == '__main__':
    main()