import os
import re
from collections import Counter

import pyarrow.parquet as pq

PATH = "data/raw/wiki_ar_00006.cleaned_no_disambiguation.parquet"

print("=" * 70)
print("NEURA WIKI AR QUALITY AUDIT")
print("=" * 70)
print(f"Input: {PATH}")

if not os.path.exists(PATH):
    raise FileNotFoundError(PATH)

pf = pq.ParquetFile(PATH)

print(f"Rows: {pf.metadata.num_rows:,}")
print(f"Row groups: {pf.num_row_groups}")
print()

# Counters
rows = 0
empty_id = 0
empty_title = 0
empty_text = 0
short_text = 0
very_short_text = 0

redirect_count = 0
category_like_count = 0
list_like_count = 0
html_like_count = 0
reference_only_count = 0

id_counter = Counter()
title_counter = Counter()

lengths = []

examples = {
    "empty_text": [],
    "very_short": [],
    "redirect": [],
    "category_like": [],
    "list_like": [],
    "html_like": [],
    "reference_only": [],
}

redirect_re = re.compile(
    r"^\s*(?:تحويلة|إعادة توجيه|#تحويل|#REDIRECT)\b",
    re.IGNORECASE | re.UNICODE,
)

category_re = re.compile(
    r"^\s*(?:تصنيف|صفحات تصنيف|قائمة تصفح|فهرس)\b",
    re.UNICODE,
)

html_re = re.compile(
    r"<(?:ref|nowiki|math|table|div|span|br)\b",
    re.IGNORECASE | re.UNICODE,
)

reference_re = re.compile(
    r"^\s*(?:مراجع|المراجع|مصادر|المصادر)\s*$",
    re.MULTILINE | re.UNICODE,
)

LIST_MARKERS = (
    "\n* ",
    "\n# ",
    "\n- ",
    "\n• ",
)

def add_example(kind, row):
    if len(examples[kind]) < 10:
        examples[kind].append({
            "id": row["id"],
            "title": row["title"],
            "text": row["text"],
        })

# اقرأ البيانات على دفعات بدل تحويلها كلها إلى Python objects دفعة واحدة
for batch in pf.iter_batches(
    batch_size=5000,
    columns=["id", "title", "text"],
):
    ids = batch.column("id").to_pylist()
    titles = batch.column("title").to_pylist()
    texts = batch.column("text").to_pylist()

    for row_id, title, text in zip(ids, titles, texts):
        rows += 1

        row_id = str(row_id or "").strip()
        title = str(title or "").strip()
        text = str(text or "").strip()

        if not row_id:
            empty_id += 1
        else:
            id_counter[row_id] += 1

        if not title:
            empty_title += 1
        else:
            title_counter[title] += 1

        if not text:
            empty_text += 1
            add_example(
                "empty_text",
                {"id": row_id, "title": title, "text": text},
            )

        length = len(text)
        lengths.append(length)

        if length < 100:
            short_text += 1

        if length < 30:
            very_short_text += 1
            add_example(
                "very_short",
                {"id": row_id, "title": title, "text": text},
            )

        if redirect_re.search(text):
            redirect_count += 1
            add_example(
                "redirect",
                {"id": row_id, "title": title, "text": text},
            )

        if category_re.search(text):
            category_like_count += 1
            add_example(
                "category_like",
                {"id": row_id, "title": title, "text": text},
            )

        marker_count = sum(text.count(m) for m in LIST_MARKERS)

        if marker_count >= 10:
            list_like_count += 1
            add_example(
                "list_like",
                {"id": row_id, "title": title, "text": text},
            )

        if html_re.search(text):
            html_like_count += 1
            add_example(
                "html_like",
                {"id": row_id, "title": title, "text": text},
            )

        if length < 250 and reference_re.search(text):
            reference_only_count += 1
            add_example(
                "reference_only",
                {"id": row_id, "title": title, "text": text},
            )

print()
print("=" * 70)
print("BASIC INTEGRITY")
print("=" * 70)

print(f"Rows processed             : {rows:,}")
print(f"Empty IDs                  : {empty_id:,}")

duplicate_ids = {
    k: v for k, v in id_counter.items()
    if v > 1
}

print(f"Duplicate IDs              : {len(duplicate_ids):,}")
print(f"Empty titles               : {empty_title:,}")
print(f"Empty text                 : {empty_text:,}")

print()
print("=" * 70)
print("TEXT LENGTH")
print("=" * 70)

if lengths:
    lengths_sorted = sorted(lengths)

    def percentile(p):
        index = int((len(lengths_sorted) - 1) * p)
        return lengths_sorted[index]

    print(f"Minimum                    : {lengths_sorted[0]:,}")
    print(f"Maximum                    : {lengths_sorted[-1]:,}")
    print(f"Average                    : {sum(lengths) / len(lengths):,.1f}")
    print(f"Median                     : {percentile(0.50):,}")
    print(f"P90                        : {percentile(0.90):,}")
    print(f"P95                        : {percentile(0.95):,}")
    print(f"P99                        : {percentile(0.99):,}")

print()
print(f"Text < 100 chars          : {short_text:,}")
print(f"Text < 30 chars           : {very_short_text:,}")

print()
print("=" * 70)
print("STRUCTURAL SIGNALS")
print("=" * 70)

print(f"Redirect-like              : {redirect_count:,}")
print(f"Category/index-like        : {category_like_count:,}")
print(f"List-heavy                 : {list_like_count:,}")
print(f"HTML/markup-like            : {html_like_count:,}")
print(f"Reference-only short        : {reference_only_count:,}")

duplicate_titles = {
    k: v for k, v in title_counter.items()
    if v > 1
}

print()
print("=" * 70)
print("DUPLICATE TITLE SUMMARY")
print("=" * 70)

print(f"Titles appearing >1 time   : {len(duplicate_titles):,}")

for title, count in sorted(
    duplicate_titles.items(),
    key=lambda x: x[1],
    reverse=True,
)[:30]:
    print(f"{count:>6}  {title!r}")

def print_examples(title, kind):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    if not examples[kind]:
        print("NONE")
        return

    for i, row in enumerate(examples[kind], 1):
        print()
        print(f"#{i}")
        print(f"ID    : {row['id']!r}")
        print(f"TITLE : {row['title']!r}")
        print(f"TEXT  : {repr(row['text'][:700])}")

print_examples("EMPTY TEXT EXAMPLES", "empty_text")
print_examples("VERY SHORT TEXT EXAMPLES", "very_short")
print_examples("REDIRECT-LIKE EXAMPLES", "redirect")
print_examples("CATEGORY/INDEX-LIKE EXAMPLES", "category_like")
print_examples("LIST-HEAVY EXAMPLES", "list_like")
print_examples("HTML/MARKUP-LIKE EXAMPLES", "html_like")
print_examples("REFERENCE-ONLY EXAMPLES", "reference_only")

print()
print("=" * 70)
print("FINAL AUDIT STATUS")
print("=" * 70)

if empty_id == 0 and not duplicate_ids:
    print("ID INTEGRITY: PASS")
else:
    print("ID INTEGRITY: REVIEW")

if empty_title == 0:
    print("TITLE INTEGRITY: PASS")
else:
    print("TITLE INTEGRITY: REVIEW")

if empty_text == 0:
    print("TEXT PRESENCE: PASS")
else:
    print("TEXT PRESENCE: REVIEW")

print()
print("IMPORTANT:")
print("NO FILE WAS MODIFIED.")
print("NO ROW WAS DELETED.")
print("This audit is diagnostic only.")
print("=" * 70)
