import re
import sys
from pathlib import Path

import pyarrow.parquet as pq

URL_RE = re.compile(r"https?://[^\s<>\[\]{}\"'`]+", re.UNICODE)

def get_text_columns(table):
    cols = []
    for name in table.column_names:
        if table[name].type == "string":
            cols.append(name)
    return cols


def scan(path):
    pf = pq.ParquetFile(path)

    rows = 0
    empty = 0
    urls = 0
    suspicious_urls = 0
    chars = 0

    for batch in pf.iter_batches(batch_size=4096):
        table = batch

        for name in get_text_columns(table):
            for value in table[name].to_pylist():
                if not value:
                    empty += 1
                    continue

                text = str(value)
                rows += 1
                chars += len(text)

                found = URL_RE.findall(text)
                urls += len(found)

                for url in found:
                    if url[:1] in "([{'\"`~;$\\|:":
                        suspicious_urls += 1
                    elif url[-1:] in ")]}>;,'\"`":
                        suspicious_urls += 1

    return {
        "rows": rows,
        "empty": empty,
        "urls": urls,
        "suspicious_urls": suspicious_urls,
        "chars": chars,
    }


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python compare_wiki_parquet.py ORIGINAL CLEANED")
        sys.exit(1)

    original = Path(sys.argv[1])
    cleaned = Path(sys.argv[2])

    if not original.exists():
        print(f"ERROR: missing original: {original}")
        sys.exit(1)

    if not cleaned.exists():
        print(f"ERROR: missing cleaned: {cleaned}")
        sys.exit(1)

    print("[+] Scanning original...")
    a = scan(original)

    print("[+] Scanning cleaned...")
    b = scan(cleaned)

    print()
    print("=" * 60)
    print("WIKI PARQUET QUALITY CHECK")
    print("=" * 60)

    print(f"{'Metric':25} {'Original':15} {'Cleaned':15}")
    print("-" * 60)

    for key in a:
        print(f"{key:25} {a[key]:15,} {b[key]:15,}")

    print("-" * 60)

    if a["chars"]:
        reduction = (a["chars"] - b["chars"]) / a["chars"] * 100
    else:
        reduction = 0.0

    print(f"Character change: {reduction:.4f}%")

    if a["rows"] == b["rows"]:
        print("[OK] Text-row count preserved")
    else:
        print("[WARNING] Text-row count changed")

    if b["suspicious_urls"] <= a["suspicious_urls"]:
        print("[OK] Suspicious URLs did not increase")
    else:
        print("[WARNING] Suspicious URLs increased")

    print("=" * 60)


if __name__ == "__main__":
    main()
