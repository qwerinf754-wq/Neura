import pyarrow.parquet as pq
import difflib

ORIGINAL = "data/raw/wiki_ar_00006.parquet"
CLEANED = "data/raw/wiki_ar_00006.cleaned.parquet"

ROW = 610
COLUMN = "text"

a = pq.read_table(ORIGINAL, columns=[COLUMN]).column(COLUMN)[ROW].as_py()
b = pq.read_table(CLEANED, columns=[COLUMN]).column(COLUMN)[ROW].as_py()

if a == b:
    print("NO CHANGE")
    raise SystemExit

print("=" * 80)
print("EXACT CHANGE LOCATION")
print("=" * 80)
print(f"Column: {COLUMN}")
print(f"Row:    {ROW}")
print(f"Before length: {len(a):,}")
print(f"After length : {len(b):,}")
print()

matcher = difflib.SequenceMatcher(None, a, b)

changes = 0

for tag, i1, i2, j1, j2 in matcher.get_opcodes():
    if tag == "equal":
        continue

    changes += 1

    print("-" * 80)
    print(f"CHANGE #{changes}")
    print(f"Type: {tag}")
    print(f"BEFORE positions: {i1}:{i2}")
    print(f"AFTER  positions: {j1}:{j2}")

    before = a[i1:i2]
    after = b[j1:j2]

    print("BEFORE:", repr(before))
    print("AFTER :", repr(after))

    # Context around the modification
    context_before_start = max(0, i1 - 100)
    context_before_end = min(len(a), i2 + 100)

    context_after_start = max(0, j1 - 100)
    context_after_end = min(len(b), j2 + 100)

    print()
    print("BEFORE CONTEXT:")
    print(repr(a[context_before_start:context_before_end]))

    print()
    print("AFTER CONTEXT:")
    print(repr(b[context_after_start:context_after_end]))

print()
print("=" * 80)
print(f"Total change regions: {changes}")
print("=" * 80)
