from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.compute as pc


INPUT = Path("data/raw/wiki_ar_00006.cleaned_no_disambiguation.parquet")
REMOVE_IDS_FILE = Path("data/raw/wiki_ar_00006.quality_remove_ids.txt")
OUTPUT = Path("data/raw/wiki_ar_00006.cleaned_quality.parquet")


print("=" * 70)
print("NEURA WIKI AR QUALITY CANDIDATE REMOVAL")
print("=" * 70)
print(f"Input : {INPUT}")
print(f"Output: {OUTPUT}")
print()


# ------------------------------------------------------------
# 1. Validate input
# ------------------------------------------------------------

if not INPUT.exists():
    raise FileNotFoundError(f"Input parquet not found: {INPUT}")

if not REMOVE_IDS_FILE.exists():
    raise FileNotFoundError(
        f"Removal ID file not found: {REMOVE_IDS_FILE}"
    )


# ------------------------------------------------------------
# 2. Read removal IDs
# ------------------------------------------------------------

remove_ids = [
    line.strip()
    for line in REMOVE_IDS_FILE.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

remove_ids = list(dict.fromkeys(remove_ids))

print("Removal IDs:")
for rid in remove_ids:
    print(f"  {rid}")

print()
print(f"Unique removal IDs: {len(remove_ids)}")


if not remove_ids:
    raise RuntimeError("Removal ID list is empty. Refusing to modify/create output.")


# ------------------------------------------------------------
# 3. Read parquet
# ------------------------------------------------------------

table = pq.read_table(INPUT)

original_rows = table.num_rows

print()
print("=" * 70)
print("INPUT VERIFICATION")
print("=" * 70)
print(f"Input rows : {original_rows}")
print(f"Columns    : {table.column_names}")


if "id" not in table.column_names:
    raise RuntimeError("Parquet does not contain required 'id' column.")


# ------------------------------------------------------------
# 4. Verify every removal ID exists exactly once
# ------------------------------------------------------------

id_column = table["id"]

print()
print("=" * 70)
print("REMOVAL ID VERIFICATION")
print("=" * 70)

found_counts = {}

for rid in remove_ids:
    mask = pc.equal(id_column, pa.scalar(rid))
    count = int(pc.sum(pc.cast(mask, pa.int64())).as_py())
    found_counts[rid] = count

    print(f"ID {rid}: found {count} time(s)")

    if count != 1:
        raise RuntimeError(
            f"SAFETY CHECK FAILED: ID {rid} occurs {count} times. "
            "Expected exactly 1. No output will be written."
        )


# ------------------------------------------------------------
# 5. Create removal mask
# ------------------------------------------------------------

remove_mask = pc.is_in(
    id_column,
    value_set=pa.array(remove_ids, type=pa.string()),
)

remove_count = int(
    pc.sum(pc.cast(remove_mask, pa.int64())).as_py()
)

expected_rows = original_rows - len(remove_ids)

print()
print("=" * 70)
print("REMOVAL PLAN")
print("=" * 70)
print(f"Original rows : {original_rows}")
print(f"Rows to remove: {remove_count}")
print(f"Expected rows : {expected_rows}")


if remove_count != len(remove_ids):
    raise RuntimeError(
        "SAFETY CHECK FAILED: number of matched rows does not equal "
        "number of removal IDs."
    )


# ------------------------------------------------------------
# 6. Keep everything except removal IDs
# ------------------------------------------------------------

keep_mask = pc.invert(remove_mask)

cleaned = table.filter(keep_mask)

actual_rows = cleaned.num_rows

print()
print("=" * 70)
print("OUTPUT VERIFICATION")
print("=" * 70)
print(f"Output rows   : {actual_rows}")
print(f"Expected rows : {expected_rows}")


if actual_rows != expected_rows:
    raise RuntimeError(
        "SAFETY CHECK FAILED: output row count is incorrect."
    )


# ------------------------------------------------------------
# 7. Verify removed IDs are actually gone
# ------------------------------------------------------------

remaining_ids = cleaned["id"]

still_present = []

for rid in remove_ids:
    mask = pc.equal(remaining_ids, pa.scalar(rid))
    count = int(pc.sum(pc.cast(mask, pa.int64())).as_py())

    if count != 0:
        still_present.append((rid, count))


if still_present:
    raise RuntimeError(
        f"SAFETY CHECK FAILED: removed IDs still present: {still_present}"
    )


# ------------------------------------------------------------
# 8. Write NEW parquet
# ------------------------------------------------------------

if OUTPUT.exists():
    raise FileExistsError(
        f"Output already exists: {OUTPUT}\n"
        "Refusing to overwrite it automatically."
    )

pq.write_table(cleaned, OUTPUT)


# ------------------------------------------------------------
# 9. Final read-back verification
# ------------------------------------------------------------

verification = pq.read_table(OUTPUT)

verification_rows = verification.num_rows

if verification_rows != expected_rows:
    raise RuntimeError(
        "FINAL VERIFICATION FAILED: output parquet row count mismatch."
    )

verification_ids = verification["id"]

for rid in remove_ids:
    mask = pc.equal(verification_ids, pa.scalar(rid))
    count = int(pc.sum(pc.cast(mask, pa.int64())).as_py())

    if count != 0:
        raise RuntimeError(
            f"FINAL VERIFICATION FAILED: ID {rid} still exists."
        )


# ------------------------------------------------------------
# 10. Final status
# ------------------------------------------------------------

print()
print("=" * 70)
print("QUALITY REMOVAL COMPLETE")
print("=" * 70)
print(f"Original rows          : {original_rows}")
print(f"Removed rows           : {remove_count}")
print(f"Output rows            : {verification_rows}")
print(f"Expected output rows  : {expected_rows}")
print(f"Removed IDs remaining : 0")
print(f"Output file            : {OUTPUT}")
print()
print("ORIGINAL PARQUET WAS NOT MODIFIED.")
print("=" * 70)
