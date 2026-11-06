import pyarrow.parquet as pq

ORIGINAL = "data/raw/wiki_ar_00006.parquet"
CLEANED = "data/raw/wiki_ar_00006.cleaned.parquet"

SAMPLE_SIZE = 2000
MAX_SHOW = 20

orig = pq.read_table(
    ORIGINAL,
    columns=None
).slice(0, SAMPLE_SIZE)

clean = pq.read_table(
    CLEANED,
    columns=None
).slice(0, SAMPLE_SIZE)

changed = 0

for col in orig.column_names:
    if orig[col].type != clean[col].type:
        print(f"[WARNING] Type changed: {col}")
        continue

    a = orig[col].to_pylist()
    b = clean[col].to_pylist()

    for i, (before, after) in enumerate(zip(a, b)):
        if before != after:
            changed += 1

            if changed <= MAX_SHOW:
                print()
                print("=" * 80)
                print(f"COLUMN: {col}")
                print(f"ROW: {i}")
                print("-" * 80)
                print("BEFORE:")
                print(repr(before))
                print("-" * 80)
                print("AFTER:")
                print(repr(after))

print()
print("=" * 80)
print(f"CHANGED VALUES IN FIRST {SAMPLE_SIZE:,} ROWS: {changed:,}")
print("=" * 80)
