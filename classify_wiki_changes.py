import pyarrow.parquet as pq
import re
from collections import Counter

ORIGINAL = "data/raw/wiki_ar_00006.parquet"
CLEANED = "data/raw/wiki_ar_00006.cleaned.parquet"

BATCH_SIZE = 4096

URL_RE = re.compile(r"https?://\S+")

stats = Counter()
examples = {}

orig_pf = pq.ParquetFile(ORIGINAL)
clean_pf = pq.ParquetFile(CLEANED)

if orig_pf.metadata.num_rows != clean_pf.metadata.num_rows:
    raise RuntimeError("Row count mismatch")

if orig_pf.metadata.num_columns != clean_pf.metadata.num_columns:
    raise RuntimeError("Column count mismatch")

total_rows = orig_pf.metadata.num_rows
rows_inspected = 0
changed_rows = set()


def remove_punct(s):
    punctuation = set("""()[]{}'"`~;$\\|:,""")
    return "".join(c for c in s if c not in punctuation)


def save_example(kind, col, row, before, after):
    examples.setdefault(
        kind,
        (col, row, before, after)
    )


orig_batches = orig_pf.iter_batches(
    batch_size=BATCH_SIZE
)

clean_batches = clean_pf.iter_batches(
    batch_size=BATCH_SIZE
)

for orig_batch, clean_batch in zip(
    orig_batches,
    clean_batches
):
    orig_table = orig_batch
    clean_table = clean_batch

    if orig_table.num_rows != clean_table.num_rows:
        raise RuntimeError(
            "Batch row count mismatch"
        )

    for col in orig_table.column_names:

        if col not in clean_table.column_names:
            raise RuntimeError(
                f"Missing column in cleaned file: {col}"
            )

        if orig_table[col].type != clean_table[col].type:
            continue

        before_values = orig_table[col].to_pylist()
        after_values = clean_table[col].to_pylist()

        for local_i, (before, after) in enumerate(
            zip(before_values, after_values)
        ):
            row = rows_inspected + local_i

            if before == after:
                continue

            changed_rows.add(row)

            before = "" if before is None else str(before)
            after = "" if after is None else str(after)

            # URL-only change
            if URL_RE.sub("URL", before) == URL_RE.sub(
                "URL", after
            ):
                stats["url_only"] += 1
                if "http://" in before or "https://" in before or "http://" in after or "https://" in after:
                    save_example(
                        "url_only",
                        col,
                        row,
                        before,
                        after,
                    )
                continue

            # Whitespace-only change
            if " ".join(before.split()) == " ".join(
                after.split()
            ):
                stats["whitespace_only"] += 1
                save_example(
                    "whitespace_only",
                    col,
                    row,
                    before,
                    after,
                )
                continue

            # Quote/bracket punctuation only
            if remove_punct(before) == remove_punct(after):
                stats["punctuation_only"] += 1
                save_example(
                    "punctuation_only",
                    col,
                    row,
                    before,
                    after,
                )
                continue

            stats["other"] += 1
            save_example(
                "other",
                col,
                row,
                before,
                after,
            )

    rows_inspected += orig_table.num_rows

    print(
        f"\r[+] Rows inspected: "
        f"{rows_inspected:,}/{total_rows:,}",
        end="",
        flush=True,
    )

print()
print("=" * 70)
print("WIKI FULL CHANGE CLASSIFICATION")
print("=" * 70)

print(f"Rows inspected : {rows_inspected:,}")
print(f"Changed rows   : {len(changed_rows):,}")
print()

for kind in (
    "url_only",
    "whitespace_only",
    "punctuation_only",
    "other",
):
    print(
        f"{kind:22}: "
        f"{stats[kind]:,}"
    )

print()
print("=" * 70)
print("EXAMPLES")
print("=" * 70)

for kind, item in examples.items():
    col, row, before, after = item

    print()
    print(f"[{kind}]")
    print(f"COLUMN: {col}")
    print(f"ROW:    {row}")
    print(
        "BEFORE:",
        repr(before[:700])
    )
    print(
        "AFTER :",
        repr(after[:700])
    )

print()
print("=" * 70)
