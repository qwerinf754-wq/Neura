import pyarrow.parquet as pq
from collections import Counter

PATH = "data/raw/wiki_ar_00006.cleaned.parquet"

table = pq.read_table(PATH)
rows = table.to_pylist()

short = []
patterns = Counter()

for row in rows:
    text = row.get("text") or ""
    title = row.get("title") or ""

    if len(text) >= 100:
        continue

    short.append(row)

    if "(توضيح)" in title:
        patterns["disambiguation_title"] += 1
    elif "قد يعني" in text or "قد تشير" in text or "قد يقصد" in text:
        patterns["disambiguation_text"] += 1
    elif "صفحات توضيح" in text:
        patterns["disambiguation_category"] += 1
    elif "انظر أيضا" in text:
        patterns["see_also"] += 1
    else:
        patterns["other_short"] += 1

print("=" * 70)
print("WIKIPEDIA SHORT PAGE ANALYSIS")
print("=" * 70)
print(f"Total rows       : {len(rows):,}")
print(f"Short <100 chars : {len(short):,}")
print()

for key, value in patterns.most_common():
    print(f"{key:24}: {value:,}")

print()
print("=" * 70)
print("SAMPLE OTHER SHORT PAGES")
print("=" * 70)

count = 0
for row in short:
    title = row.get("title") or ""
    text = row.get("text") or ""

    if "(توضيح)" in title:
        continue

    if "قد يعني" in text or "قد تشير" in text or "قد يقصد" in text:
        continue

    print(f"ID    : {row.get('id')!r}")
    print(f"TITLE : {title!r}")
    print(f"TEXT  : {text!r}")
    print("-" * 70)

    count += 1
    if count >= 30:
        break

print("=" * 70)
