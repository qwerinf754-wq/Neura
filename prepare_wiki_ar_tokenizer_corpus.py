from pathlib import Path
import re
import pyarrow.parquet as pq


INPUT = Path(
    "data/raw/wiki_ar_00006.cleaned_quality.parquet"
)

OUTPUT = Path(
    "data/processed/wiki_ar_00006.tokenizer.txt"
)

BATCH_SIZE = 1024
EXPECTED_ROWS = 170772


print("=" * 70)
print("NEURA WIKI AR TOKENIZER CORPUS PREPARATION")
print("=" * 70)
print(f"Input : {INPUT}")
print(f"Output: {OUTPUT}")
print()

if not INPUT.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

pf = pq.ParquetFile(INPUT)

if pf.metadata.num_rows != EXPECTED_ROWS:
    raise RuntimeError(
        f"Unexpected input rows: {pf.metadata.num_rows}"
    )

print(f"Input rows : {pf.metadata.num_rows}")
print(f"Row groups : {pf.num_row_groups}")
print()

processed = 0
written = 0
skipped = 0


def normalize_text(value):
    """
    Normalize a Wikipedia field into a single logical line.

    This prevents internal blank lines inside articles from being
    confused with document separators during verification.
    """
    if value is None:
        return ""

    value = str(value)

    # Normalize all newline variants.
    value = value.replace("\r\n", "\n")
    value = value.replace("\r", "\n")

    # Collapse whitespace/newlines into single spaces.
    value = re.sub(r"\s+", " ", value)

    return value.strip()


with OUTPUT.open(
    "w",
    encoding="utf-8",
    newline="\n",
) as out:

    for batch in pf.iter_batches(
        batch_size=BATCH_SIZE,
        columns=["title", "text"],
    ):
        titles = batch.column("title").to_pylist()
        texts = batch.column("text").to_pylist()

        for title, text in zip(titles, texts):

            title = normalize_text(title)
            text = normalize_text(text)

            processed += 1

            if not title and not text:
                skipped += 1
                continue

            # One logical document:
            # title
            # text
            #
            # followed by exactly one blank separator line.

            if title:
                out.write(title)
                out.write("\n")

            if text:
                out.write(text)
                out.write("\n")

            out.write("\n")

            written += 1

        if processed % 10000 < len(titles):
            print(
                f"Processed: {processed:,} / {EXPECTED_ROWS:,}",
                flush=True,
            )


print()
print("=" * 70)
print("CORPUS PREPARATION COMPLETE")
print("=" * 70)
print(f"Rows processed : {processed:,}")
print(f"Documents      : {written:,}")
print(f"Skipped        : {skipped:,}")
print(f"Output         : {OUTPUT}")
print()

if processed != EXPECTED_ROWS:
    raise RuntimeError(
        f"Processed {processed} rows, expected {EXPECTED_ROWS}"
    )

if written != EXPECTED_ROWS - skipped:
    raise RuntimeError(
        "Written document count is inconsistent."
    )

if not OUTPUT.exists():
    raise RuntimeError("Output corpus was not created.")

size = OUTPUT.stat().st_size

print(f"Corpus size    : {size:,} bytes")

if size == 0:
    raise RuntimeError("Corpus is empty.")

print()
print("STATUS: CORPUS READY")
print("=" * 70)
