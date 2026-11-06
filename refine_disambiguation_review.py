import pyarrow.parquet as pq
from pathlib import Path
import re

PATH = "data/raw/wiki_ar_00006.cleaned.parquet"

REVIEW_IDS = Path(
    "data/raw/wiki_ar_00006.disambiguation_review_ids.txt"
)

OUT_AUTO = Path(
    "data/raw/wiki_ar_00006.disambiguation_refined_auto_ids.txt"
)

OUT_KEEP = Path(
    "data/raw/wiki_ar_00006.disambiguation_refined_keep_ids.txt"
)

OUT_EXAMPLES = Path(
    "data/raw/wiki_ar_00006.disambiguation_refined_examples.txt"
)


# ---------------------------------------------------------
# مؤشرات قوية على أن الصفحة قائمة توضيح حقيقية
# ---------------------------------------------------------

LIST_INTRO_RE = re.compile(
    r"^\s*"
    r"(قد\s+يقصد\s+من|"
    r"قد\s+يقصد\s+ب|"
    r"قد\s+يقصد\s+بـ|"
    r"قد\s+تشير|"
    r"قد\s+يشير|"
    r"قد\s+تعني|"
    r"قد\s+يعني)"
    r".*"
    r"(إلى|:)\s*$",
    re.UNICODE,
)


SECTION_RE = re.compile(
    r"\n\s*(?:"
    r"أماكن|"
    r"أشخاص|"
    r"أفلام|"
    r"كتب|"
    r"أعمال|"
    r"شخصيات|"
    r"مواقع|"
    r"أندية|"
    r"مدن|"
    r"قرى|"
    r"بلديات|"
    r"شركات|"
    r"استخدامات\s+اخرى|"
    r"استخدامات\s+أخرى|"
    r"العلوم|"
    r"تعليم|"
    r"برمجة"
    r")\s*(?:\n|$)",
    re.UNICODE,
)


def normalize(text):
    return (
        (text or "")
        .replace("\xa0", " ")
        .replace("\r\n", "\n")
        .strip()
    )


def count_list_like_lines(text):
    """
    تقدير عدد العناصر التي تبدو كعناصر توضيح.
    """
    lines = [
        x.strip()
        for x in text.splitlines()
        if x.strip()
    ]

    count = 0

    for line in lines:
        # أسطر قصيرة نسبيًا تنتهي بمعلومة/تعريف
        if 5 <= len(line) <= 180:
            if (
                ":" in line
                or "،" in line
                or "،" in line
            ):
                count += 1

    return count


def classify(text):
    text = normalize(text)

    if not text:
        return "KEEP", "empty"

    first_lines = "\n".join(text.splitlines()[:3])

    intro = bool(LIST_INTRO_RE.match(first_lines))

    sections = len(SECTION_RE.findall(text))

    list_lines = count_list_like_lines(text)

    # -----------------------------------------------------
    # توضيح واضح:
    #
    # مقدمة توضيح + أكثر من عنصر
    # -----------------------------------------------------

    if intro and (sections >= 1 or list_lines >= 2):
        return "AUTO_REMOVE", "disambiguation_list"

    # -----------------------------------------------------
    # إذا كانت المقدمة توضيحية لكن النص طويل ومترابط
    # نحتفظ بها للمراجعة/التدريب
    # -----------------------------------------------------

    if intro:
        if len(text) > 700:
            return "KEEP", "long_expository_article"

        return "REVIEW", "ambiguous_intro"

    return "KEEP", "normal_article"


table = pq.read_table(PATH)
rows = table.to_pylist()

review_ids = {
    line.strip()
    for line in REVIEW_IDS.read_text(encoding="utf-8").splitlines()
    if line.strip()
}


auto_ids = []
keep_ids = []
review_ids_final = []

examples = []


for row in rows:
    row_id = str(row.get("id") or "").strip()

    if row_id not in review_ids:
        continue

    text = row.get("text") or ""
    title = row.get("title") or ""

    decision, reason = classify(text)

    if decision == "AUTO_REMOVE":
        auto_ids.append(row_id)

    elif decision == "KEEP":
        keep_ids.append(row_id)

    else:
        review_ids_final.append(row_id)

    examples.append(
        {
            "id": row_id,
            "title": title,
            "decision": decision,
            "reason": reason,
            "text": normalize(text)[:1000],
        }
    )


OUT_AUTO.write_text(
    "\n".join(auto_ids) + ("\n" if auto_ids else ""),
    encoding="utf-8",
)

OUT_KEEP.write_text(
    "\n".join(keep_ids) + ("\n" if keep_ids else ""),
    encoding="utf-8",
)

Path(
    "data/raw/wiki_ar_00006.disambiguation_refined_review_ids.txt"
).write_text(
    "\n".join(review_ids_final)
    + ("\n" if review_ids_final else ""),
    encoding="utf-8",
)


with OUT_EXAMPLES.open("w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("REFINED DISAMBIGUATION REVIEW\n")
    f.write("=" * 80 + "\n")

    for i, ex in enumerate(examples, 1):
        f.write(f"\n#{i}\n")
        f.write(f"ID         : {ex['id']!r}\n")
        f.write(f"TITLE      : {ex['title']!r}\n")
        f.write(f"DECISION   : {ex['decision']}\n")
        f.write(f"REASON     : {ex['reason']}\n")
        f.write(f"TEXT       : {ex['text']!r}\n")


print("=" * 70)
print("REFINED DISAMBIGUATION REVIEW")
print("=" * 70)

print(f"Original review       : {len(review_ids):,}")
print(f"Refined auto-remove   : {len(auto_ids):,}")
print(f"Refined keep          : {len(keep_ids):,}")
print(f"Still needs review    : {len(review_ids_final):,}")

print()
print("Files created:")
print(f"  {OUT_AUTO}")
print(f"  {OUT_KEEP}")
print("  data/raw/wiki_ar_00006.disambiguation_refined_review_ids.txt")
print(f"  {OUT_EXAMPLES}")

print()
print("IMPORTANT:")
print("NO PARQUET FILE WAS MODIFIED.")
print("=" * 70)
