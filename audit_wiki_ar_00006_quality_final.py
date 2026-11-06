from pathlib import Path
import pyarrow.parquet as pq

INPUT = Path(
    "data/raw/wiki_ar_00006.cleaned_quality.parquet"
)

REMOVED_ID = "9266385"
BATCH_SIZE = 4096

print("=" * 70)
print("NEURA WIKI AR FINAL QUALITY VERIFICATION")
print("=" * 70)
print(f"Input: {INPUT}")
print()

if not INPUT.exists():
    raise FileNotFoundError(f"File not found: {INPUT}")

pf = pq.ParquetFile(INPUT)

expected_rows = 170772

print("PARQUET METADATA")
print("=" * 70)
print(f"Rows       : {pf.metadata.num_rows}")
print(f"Row groups : {pf.num_row_groups}")
print(f"Columns    : {pf.schema_arrow.names}")

if pf.metadata.num_rows != expected_rows:
    raise RuntimeError(
        f"Unexpected row count: {pf.metadata.num_rows}"
    )

print()
print("STREAMING ID CHECK")
print("=" * 70)

seen = set()
duplicate_count = 0
empty_ids = 0
removed_present = 0
processed = 0

for batch in pf.iter_batches(
    batch_size=BATCH_SIZE,
    columns=["id"]
):
    ids = batch.column("id").to_pylist()

    for value in ids:
        value = "" if value is None else str(value).strip()

        if not value:
            empty_ids += 1
            continue

        if value == REMOVED_ID:
            removed_present += 1

        if value in seen:
            duplicate_count += 1
        else:
            seen.add(value)

        processed += 1

    if processed % 20000 < len(ids):
        print(
            f"Processed: {processed:,} / {expected_rows:,}",
            flush=True
        )

print()
print("=" * 70)
print("ID INTEGRITY")
print("=" * 70)
print(f"Rows processed       : {processed}")
print(f"Unique IDs           : {len(seen)}")
print(f"Empty IDs            : {empty_ids}")
print(f"Duplicate IDs        : {duplicate_count}")
print(f"Removed ID present   : {removed_present}")

if (
    processed != expected_rows
    or len(seen) != expected_rows
    or empty_ids != 0
    or duplicate_count != 0
    or removed_present != 0
):
    print()
    print("STATUS: VERIFICATION FAILED")
    raise SystemExit(1)

print()
print("=" * 70)
print("FINAL QUALITY CLEANUP VERIFIED")
print("=" * 70)
print("STATUS: PASS")
print()
print("The quality-removal output is structurally valid.")
print("ID 9266385 is absent.")
print("Original parquet remains untouched.")
print("=" * 70)
