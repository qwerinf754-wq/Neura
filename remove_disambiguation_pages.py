import os
import pyarrow as pa
import pyarrow.parquet as pq

INPUT = "data/raw/wiki_ar_00006.cleaned.parquet"
IDS_FILE = "data/raw/wiki_ar_00006.disambiguation_final_ids.txt"
OUTPUT = "data/raw/wiki_ar_00006.cleaned_no_disambiguation.parquet"

REMOVE_IDS = {
    line.strip()
    for line in open(IDS_FILE, encoding="utf-8")
    if line.strip()
}

print("=" * 70)
print("WIKIPEDIA DISAMBIGUATION REMOVAL")
print("=" * 70)

print(f"Input             : {INPUT}")
print(f"Output            : {OUTPUT}")
print(f"Removal IDs       : {len(REMOVE_IDS):,}")

if os.path.exists(OUTPUT):
    raise FileExistsError(
        f"Output already exists: {OUTPUT}\n"
        "Refusing to overwrite it."
    )

table = pq.read_table(INPUT)

ids = table["id"].to_pylist()

keep_mask = [
    str(x).strip() not in REMOVE_IDS
    for x in ids
]

removed = sum(not x for x in keep_mask)
remaining = sum(keep_mask)

print()
print(f"Input rows        : {table.num_rows:,}")
print(f"Rows to remove    : {removed:,}")
print(f"Rows remaining    : {remaining:,}")

if removed != len(REMOVE_IDS):
    raise RuntimeError(
        f"Safety check failed: expected to remove {len(REMOVE_IDS):,}, "
        f"but found {removed:,}."
    )

expected_remaining = table.num_rows - len(REMOVE_IDS)

if remaining != expected_remaining:
    raise RuntimeError(
        f"Safety check failed: expected {expected_remaining:,} "
        f"remaining rows, got {remaining:,}."
    )

# إنشاء النسخة المنظفة فقط
cleaned = table.filter(pa.array(keep_mask))

pq.write_table(
    cleaned,
    OUTPUT,
    compression="zstd",
)

print()
print("=" * 70)
print("REMOVAL COMPLETE")
print("=" * 70)
print(f"Output rows       : {cleaned.num_rows:,}")
print(f"Removed rows      : {removed:,}")
print(f"Expected rows     : {expected_remaining:,}")
print(f"Output file       : {OUTPUT}")
print()
print("ORIGINAL PARQUET WAS NOT MODIFIED.")
print("=" * 70)
