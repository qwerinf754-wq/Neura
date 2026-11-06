import pyarrow.parquet as pq
from collections import Counter
import re

PATH = "data/raw/wiki_ar_00006.cleaned.parquet"

table = pq.read_table(PATH)
rows = table.to_pylist()

stats = Counter()
examples = []
candidate_ids = []

# أنماط افتتاحية قوية لصفحات التوضيح
START_PATTERNS = [
    r"^\s*قد\s+يقصد\s+من",
    r"^\s*قد\s+يقصد\s+ب",
    r"^\s*قد\s+يقصد\s+بـ",
    r"^\s*قد\s+تشير\s+",
    r"^\s*قد\s+يشير\s+",
    r"^\s*قد\s+تعني\s+",
    r"^\s*قد\s+يعني\s+",
    r"^\s*اسم\s+قد\s+يشير\s+إلى",
]

START_RE = re.compile(
    "|".join(START_PATTERNS),
    re.UNICODE,
)

CATEGORY_RE = re.compile(
    r"صفحات\s+توضيح",
    re.UNICODE,
)

# مؤشرات قائمة توضيح حقيقية
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
]

def normalized_start(text):
    text = text.replace("\xa0", " ")
    text = text.lstrip()
    return text[:500]

def classify(row):
    title = row.get("title") or ""
    text = row.get("text") or ""

    title_disamb = "(توضيح)" in title

    start = normalized_start(text)
    start_disamb = bool(START_RE.search(start))

    category = bool(CATEGORY_RE.search(text))

    list_count = sum(marker in text for marker in LIST_MARKERS)
    strong_list = list_count >= 2

    reasons = []

    if title_disamb:
        reasons.append("title")

    if start_disamb:
        reasons.append("start_text")

    if category:
        reasons.append("category")

    if strong_list:
        reasons.append("structured_list")

    # ---------------------------------------------------------
    # قرار محافظ:
    #
    # 1. عنوان (توضيح) = مرشح قوي جدًا
    # 2. افتتاحية توضيح + قائمة/تصنيف = مرشح قوي
    # 3. افتتاحية توضيح وحدها = لا نحذف تلقائيًا
    # 4. عبارة "قد يشير" في منتصف مقال = ليست توضيحًا
    # ---------------------------------------------------------

    if title_disamb:
        decision = True
        confidence = "HIGH"

    elif start_disamb and (category or strong_list):
        decision = True
        confidence = "HIGH"

    elif start_disamb:
        decision = True
        confidence = "MEDIUM"

    else:
        decision = False
        confidence = "NONE"

    return decision, confidence, reasons


for i, row in enumerate(rows):
    decision, confidence, reasons = classify(row)

    if decision:
        stats["candidate"] += 1
        stats[f"confidence_{confidence}"] += 1

        for reason in reasons:
            stats[f"reason_{reason}"] += 1

        candidate_ids.append(row.get("id"))

        if len(examples) < 50:
            examples.append((i, row, confidence, reasons))


print("=" * 70)
print("WIKIPEDIA CONSERVATIVE DISAMBIGUATION ANALYSIS")
print("=" * 70)

print(f"Total rows          : {len(rows):,}")
print(f"Candidates          : {stats['candidate']:,}")
print()

print("Confidence:")
print(f"  HIGH              : {stats['confidence_HIGH']:,}")
print(f"  MEDIUM            : {stats['confidence_MEDIUM']:,}")

print()
print("Reasons:")
print(f"  title '(توضيح)'   : {stats['reason_title']:,}")
print(f"  start text        : {stats['reason_start_text']:,}")
print(f"  category          : {stats['reason_category']:,}")
print(f"  structured list   : {stats['reason_structured_list']:,}")

print()
print("=" * 70)
print("CANDIDATE EXAMPLES")
print("=" * 70)

for i, row, confidence, reasons in examples:
    print()
    print(f"ROW        : {i}")
    print(f"ID         : {row.get('id')!r}")
    print(f"TITLE      : {row.get('title')!r}")
    print(f"CONFIDENCE : {confidence}")
    print(f"REASONS    : {', '.join(reasons)}")
    print(f"TEXT       : {repr((row.get('text') or '')[:600])}")

print()
print("=" * 70)
print("IMPORTANT")
print("=" * 70)
print("NO FILE WAS MODIFIED.")
print(f"Candidate IDs: {len(candidate_ids):,}")
print("=" * 70)
