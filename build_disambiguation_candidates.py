import pyarrow.parquet as pq
from pathlib import Path
import re

INPUT = Path("data/raw/wiki_ar_00006.cleaned.parquet")

AUTO_IDS = Path("data/raw/wiki_ar_00006.disambiguation_auto_ids.txt")
REVIEW_IDS = Path("data/raw/wiki_ar_00006.disambiguation_review_ids.txt")

AUTO_EXAMPLES = Path("data/raw/wiki_ar_00006.disambiguation_auto_examples.txt")
REVIEW_EXAMPLES = Path("data/raw/wiki_ar_00006.disambiguation_review_examples.txt")

START_PATTERNS = [
    r"^\s*قد\s+يقصد\s+من",
    r"^\s*قد\s+يقصد\s+ب",
    r"^\s*قد\s+يقصد\s+بـ",
    r"^\s*قد\s+تشير\s+",
    r"^\s*قد\s+يشير\s+",
    r"^\s*قد\s+تعني\s+",
    r"^\s*قد\s+يعني\s+",
    r"^\s*اسم\s+قد\s+يشير\s+إلى",
    r"^\s*يحمل\s+الأشخاص\s+الآتيون\s+اسم",
]

START_RE = re.compile(
    "|".join(START_PATTERNS),
    re.UNICODE,
)

CATEGORY_RE = re.compile(
    r"صفحات\s+توضيح",
    re.UNICODE,
)

LIST_MARKERS = [
    "\nأماكن",
    "\nأشخاص",
    "\nأفلام",
    "\nكتب",
    "\nأعمال",
    "\nشخصيات",
    "\nمواقع",
    "\nأندية",
    "\nمدن",
    "\nقرى",
    "\nبلديات",
    "\nأحداث",
    "\nأخرى",
]

def normalize_start(text):
    text = text.replace("\xa0", " ")
    return text.lstrip()[:700]

def classify(row):
    title = row.get("title") or ""
    text = row.get("text") or ""

    title_disamb = "(توضيح)" in title

    start = normalize_start(text)
    start_disamb = bool(START_RE.search(start))

    category = bool(CATEGORY_RE.search(text))

    list_count = sum(marker in text for marker in LIST_MARKERS)
    structured_list = list_count >= 2

    # حذف تلقائي فقط عند وجود دليل قوي.
    auto = (
        title_disamb
        or
        (start_disamb and (category or structured_list))
    )

    # مراجعة يدوية: افتتاحية توضيح بدون دليل إضافي.
    review = (
        not auto
        and start_disamb
    )

    return auto, review, title_disamb, start_disamb, category, structured_list


table = pq.read_table(INPUT)

auto_ids = []
review_ids = []

auto_examples = []
review_examples = []

for row in table.to_pylist():
    auto, review, title, start, category, structured = classify(row)

    row_id = row.get("id")

    if auto:
        auto_ids.append(row_id)

        if len(auto_examples) < 100:
            auto_examples.append(
                (
                    row_id,
                    row.get("title"),
                    row.get("text") or "",
                    title,
                    start,
                    category,
                    structured,
                )
            )

    elif review:
        review_ids.append(row_id)

        if len(review_examples) < 100:
            review_examples.append(
                (
                    row_id,
                    row.get("title"),
                    row.get("text") or "",
                    title,
                    start,
                    category,
                    structured,
                )
            )


# ---------------------------------------------------------
# حفظ IDs فقط
# ---------------------------------------------------------

AUTO_IDS.write_text(
    "\n".join(str(x) for x in auto_ids) + "\n",
    encoding="utf-8",
)

REVIEW_IDS.write_text(
    "\n".join(str(x) for x in review_ids) + "\n",
    encoding="utf-8",
)


def write_examples(path, examples, label):
    with path.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write(label + "\n")
        f.write("=" * 80 + "\n")

        for i, item in enumerate(examples, 1):
            (
                row_id,
                title_text,
                text,
                has_title,
                has_start,
                has_category,
                has_structured,
            ) = item

            f.write("\n")
            f.write(f"#{i}\n")
            f.write(f"ID         : {row_id!r}\n")
            f.write(f"TITLE      : {title_text!r}\n")
            f.write(f"title      : {has_title}\n")
            f.write(f"start_text : {has_start}\n")
            f.write(f"category   : {has_category}\n")
            f.write(f"structured : {has_structured}\n")
            f.write(f"TEXT       : {text[:1000]!r}\n")


write_examples(
    AUTO_EXAMPLES,
    auto_examples,
    "AUTO-REMOVE CANDIDATE EXAMPLES",
)

write_examples(
    REVIEW_EXAMPLES,
    review_examples,
    "MANUAL-REVIEW CANDIDATE EXAMPLES",
)


print("=" * 70)
print("DISAMBIGUATION CANDIDATE BUILD")
print("=" * 70)

print(f"Input rows          : {table.num_rows:,}")
print(f"Auto-remove         : {len(auto_ids):,}")
print(f"Manual review       : {len(review_ids):,}")
print(f"Remaining           : {table.num_rows - len(auto_ids):,}")

print()
print("Files created:")
print(f"  {AUTO_IDS}")
print(f"  {REVIEW_IDS}")
print(f"  {AUTO_EXAMPLES}")
print(f"  {REVIEW_EXAMPLES}")

print()
print("IMPORTANT:")
print("NO PARQUET FILE WAS MODIFIED.")
print("=" * 70)
