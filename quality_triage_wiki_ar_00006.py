from pathlib import Path
import re
import pyarrow.parquet as pq


INPUT = Path("data/raw/wiki_ar_00006.cleaned_no_disambiguation.parquet")
OUT_DIR = Path("data/raw")

SAFE_IDS = OUT_DIR / "wiki_ar_00006.quality_safe_ids.txt"
REVIEW_IDS = OUT_DIR / "wiki_ar_00006.quality_review_ids.txt"
REMOVE_IDS = OUT_DIR / "wiki_ar_00006.quality_remove_ids.txt"
EXAMPLES = OUT_DIR / "wiki_ar_00006.quality_examples.txt"


def normalize(text):
    return re.sub(r"\s+", " ", text or "").strip()


def classify(row):
    row_id = str(row.get("id") or "").strip()
    title = normalize(row.get("title"))
    text = row.get("text") or ""
    clean = normalize(text)

    signals = []

    if not row_id:
        return "REMOVE", ["empty_id"]

    if not title:
        return "REMOVE", ["empty_title"]

    if not clean:
        return "REMOVE", ["empty_text"]

    if re.match(
        r"^(?:#REDIRECT|تحويل|إعادة توجيه)\b",
        clean,
        re.IGNORECASE,
    ):
        signals.append("redirect")

    if title.startswith(("تصنيف:", "فهرس:")):
        signals.append("category_or_index_title")

    for pattern in (
        r"<ref\b",
        r"</ref>",
        r"<span\b",
        r"</span>",
        r"<div\b",
        r"</div>",
        r"<table\b",
        r"</table>",
        r"\[\[.*?\]\]",
    ):
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL):
            signals.append("markup")
            break

    length = len(clean)

    if length < 30:
        signals.append("very_short")
    elif length < 100:
        signals.append("short")

    if length < 1200:
        lines = [
            normalize(x)
            for x in text.splitlines()
            if normalize(x)
        ]

        reference_markers = {
            "مراجع",
            "المراجع",
            "مصادر",
            "المصادر",
            "وصلات خارجية",
            "انظر أيضًا",
            "انظر ايضا",
        }

        marker_count = sum(
            1 for line in lines
            if line in reference_markers
        )

        if marker_count and len(lines) <= 8:
            signals.append("reference_heavy_short")

    # Conservative policy.
    if "empty_id" in signals:
        return "REMOVE", signals

    if "empty_title" in signals:
        return "REMOVE", signals

    if "empty_text" in signals:
        return "REMOVE", signals

    if "redirect" in signals and length < 300:
        return "REMOVE", signals

    suspicious = {
        "category_or_index_title",
        "markup",
        "very_short",
        "short",
        "reference_heavy_short",
        "redirect",
    }

    if suspicious.intersection(signals):
        return "REVIEW", signals

    return "SAFE", signals


def main():
    print("=" * 70)
    print("NEURA WIKI AR QUALITY TRIAGE")
    print("=" * 70)
    print("Input:", INPUT)

    pf = pq.ParquetFile(INPUT)

    print("Rows:", pf.metadata.num_rows)
    print("Row groups:", pf.num_row_groups)
    print()

    safe = []
    review = []
    remove = []

    examples = {
        "REVIEW": [],
        "REMOVE": [],
    }

    processed = 0

    # Small batches are intentional for Termux/mobile RAM.
    for batch in pf.iter_batches(
        batch_size=256,
        columns=["id", "title", "text"],
    ):
        rows = batch.to_pylist()

        for row in rows:
            decision, signals = classify(row)
            row_id = str(row.get("id") or "").strip()

            if decision == "SAFE":
                safe.append(row_id)

            elif decision == "REVIEW":
                review.append(row_id)

                if len(examples["REVIEW"]) < 50:
                    examples["REVIEW"].append({
                        "id": row_id,
                        "title": row.get("title") or "",
                        "text": (row.get("text") or "")[:1200],
                        "signals": signals,
                    })

            else:
                remove.append(row_id)

                if len(examples["REMOVE"]) < 50:
                    examples["REMOVE"].append({
                        "id": row_id,
                        "title": row.get("title") or "",
                        "text": (row.get("text") or "")[:1200],
                        "signals": signals,
                    })

        processed += len(rows)

        if processed % 10000 < 256:
            print(f"Processed: {processed:,} / {pf.metadata.num_rows:,}")

    safe = sorted(set(safe))
    review = sorted(set(review))
    remove = sorted(set(remove))

    SAFE_IDS.write_text(
        "\n".join(safe) + ("\n" if safe else ""),
        encoding="utf-8",
    )

    REVIEW_IDS.write_text(
        "\n".join(review) + ("\n" if review else ""),
        encoding="utf-8",
    )

    REMOVE_IDS.write_text(
        "\n".join(remove) + ("\n" if remove else ""),
        encoding="utf-8",
    )

    with EXAMPLES.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("NEURA WIKI AR QUALITY TRIAGE EXAMPLES\n")
        f.write("=" * 80 + "\n\n")

        for decision in ("REMOVE", "REVIEW"):
            f.write("=" * 80 + "\n")
            f.write(decision + "\n")
            f.write("=" * 80 + "\n\n")

            for i, item in enumerate(examples[decision], 1):
                f.write(f"#{i}\n")
                f.write(f"ID      : {item['id']!r}\n")
                f.write(f"TITLE   : {item['title']!r}\n")
                f.write(f"SIGNALS : {', '.join(item['signals'])}\n")
                f.write(f"TEXT    : {item['text']!r}\n\n")

    print()
    print("=" * 70)
    print("TRIAGE SUMMARY")
    print("=" * 70)
    print("Total rows :", pf.metadata.num_rows)
    print("SAFE       :", len(safe))
    print("REVIEW     :", len(review))
    print("REMOVE     :", len(remove))

    print()
    print("Files created:")
    print(" ", SAFE_IDS)
    print(" ", REVIEW_IDS)
    print(" ", REMOVE_IDS)
    print(" ", EXAMPLES)

    print()
    print("IMPORTANT:")
    print("NO PARQUET FILE WAS MODIFIED.")
    print("=" * 70)


if __name__ == "__main__":
    main()
