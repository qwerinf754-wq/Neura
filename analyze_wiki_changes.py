import pyarrow.parquet as pq
import re

ORIGINAL = "data/raw/wiki_ar_00006.parquet"
CLEANED = "data/raw/wiki_ar_00006.cleaned.parquet"

N = 2000

orig = pq.read_table(ORIGINAL).slice(0, N)
clean = pq.read_table(CLEANED).slice(0, N)

url_re = re.compile(r"https?://\S+")

changed = 0
url_only = 0
other = 0

for col in orig.column_names:
    if orig[col].type != clean[col].type:
        continue

    a = orig[col].to_pylist()
    b = clean[col].to_pylist()

    for i, (before, after) in enumerate(zip(a, b)):
        if before == after:
            continue

        changed += 1

        before_s = "" if before is None else str(before)
        after_s = "" if after is None else str(after)

        # إزالة الروابط من المقارنة لمعرفة هل التغيير
        # محصور في محيط URL
        before_no_url = url_re.sub("URL", before_s)
        after_no_url = url_re.sub("URL", after_s)

        if before_no_url == after_no_url:
            url_only += 1
        else:
            other += 1

            if other <= 20:
                print()
                print("=" * 80)
                print("NON-URL CHANGE")
                print(f"COLUMN: {col}")
                print(f"ROW: {i}")
                print("-" * 80)
                print("BEFORE:", repr(before_s[:500]))
                print("AFTER :", repr(after_s[:500]))

print()
print("=" * 80)
print(f"Changed values : {changed:,}")
print(f"URL-only       : {url_only:,}")
print(f"Other changes  : {other:,}")
print("=" * 80)
