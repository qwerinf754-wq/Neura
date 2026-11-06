import re
import sys
from pathlib import Path

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    print("ERROR: pyarrow is not installed")
    sys.exit(1)


# URL مع الحفاظ على النص المحيط به.
URL_RE = re.compile(
    r"https?://[^\s<>\[\]{}\"'`]+",
    re.UNICODE,
)


# رموز معروفة ظهرت كـ artifacts قبل روابط Wikipedia.
LEADING_ARTIFACTS = "([{'\"`~;$\\|:"

# علامات يمكن أن تكون artifacts في نهاية الرابط.
TRAILING_ARTIFACTS = ".,!?;:)]}>'\"`"


def clean_url(url: str) -> str:
    """
    تنظيف الرابط نفسه فقط.
    لا يلمس أي نص خارج الرابط.
    """

    url = url.lstrip(LEADING_ARTIFACTS)
    url = url.rstrip(TRAILING_ARTIFACTS)

    # إصلاح escape شائع دون تغيير النص المحيط.
    url = url.replace("\\/", "/")

    return url


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return text

    def replace_url(match):
        original = match.group(0)
        cleaned = clean_url(original)

        return cleaned

    # IMPORTANT:
    # لا توجد هنا أي عمليات على المسافات أو الأسطر أو Unicode.
    return URL_RE.sub(replace_url, text)


def clean_value(value):
    if isinstance(value, str):
        return clean_text(value)

    if isinstance(value, list):
        return [clean_value(x) for x in value]

    if isinstance(value, dict):
        return {
            k: clean_value(v)
            for k, v in value.items()
        }

    return value


def process_file(src: Path, dst: Path):
    print(f"[+] Input : {src}")
    print(f"[+] Output: {dst}")

    pf = pq.ParquetFile(src)

    writer = None
    rows = 0

    try:
        for batch in pf.iter_batches(batch_size=1024):
            table = pa.Table.from_batches([batch])

            columns = {}

            for name in table.column_names:
                values = table[name].to_pylist()

                values = [
                    clean_value(v)
                    for v in values
                ]

                columns[name] = pa.array(values)

            cleaned = pa.table(columns)

            if writer is None:
                writer = pq.ParquetWriter(
                    dst,
                    cleaned.schema,
                    compression="zstd",
                )

            writer.write_table(cleaned)

            rows += cleaned.num_rows

            print(
                f"\r[+] Rows cleaned: {rows}",
                end="",
                flush=True,
            )

    finally:
        if writer is not None:
            writer.close()

    print()
    print("[+] Cleaning complete.")


def main():
    if len(sys.argv) != 2:
        print("Usage:")
        print(
            "  python clean_wiki_parquet.py "
            "<file.parquet>"
        )
        sys.exit(1)

    src = Path(
        sys.argv[1]
    ).expanduser().resolve()

    if not src.exists():
        print(f"ERROR: file not found: {src}")
        sys.exit(1)

    if src.suffix.lower() != ".parquet":
        print("ERROR: input must be a .parquet file")
        sys.exit(1)

    dst = src.with_name(
        src.stem + ".cleaned.parquet"
    )

    if dst.exists():
        print("ERROR: output already exists:")
        print(dst)
        print(
            "Delete it manually if you want "
            "to regenerate it."
        )
        sys.exit(1)

    process_file(src, dst)

    # Validation
    print("[+] Validating output...")

    out = pq.ParquetFile(dst)

    print(
        f"[+] Output rows   : "
        f"{out.metadata.num_rows}"
    )

    print(
        f"[+] Output columns: "
        f"{out.metadata.num_columns}"
    )

    print(
        f"[+] Output size   : "
        f"{dst.stat().st_size:,} bytes"
    )

    print("[+] Parquet validation: OK")
    print()
    print(f"DONE: {dst}")


if __name__ == "__main__":
    main()
