from pathlib import Path


CORPUS = Path(
    "data/processed/wiki_ar_00006.tokenizer.txt"
)

EXPECTED_DOCS = 170772


print("=" * 70)
print("NEURA WIKI AR TOKENIZER CORPUS VERIFICATION")
print("=" * 70)
print(f"Corpus: {CORPUS}")
print()

if not CORPUS.exists():
    raise FileNotFoundError(
        f"Corpus not found: {CORPUS}"
    )

size = CORPUS.stat().st_size

print("FILE")
print("=" * 70)
print(f"Size      : {size:,} bytes")
print(f"Size MB   : {size / (1024 * 1024):.2f}")
print()

if size == 0:
    raise RuntimeError("Corpus is empty.")


print("STREAMING CHECK")
print("=" * 70)

documents = 0
nonempty_lines = 0
empty_lines = 0
characters = 0
arabic_chars = 0

# A document consists of:
#
# title
# text
# <blank line>
#
# Because preparation normalizes title/text to one line each,
# exactly one blank line terminates every document.


document_lines = []

with CORPUS.open(
    "r",
    encoding="utf-8",
    errors="strict",
) as f:

    for line in f:

        characters += len(line)

        stripped = line.strip()

        if stripped:

            nonempty_lines += 1
            document_lines.append(stripped)

            for ch in stripped:
                code = ord(ch)

                if (
                    0x0600 <= code <= 0x06FF
                    or 0x0750 <= code <= 0x077F
                    or 0x08A0 <= code <= 0x08FF
                ):
                    arabic_chars += 1

        else:

            empty_lines += 1

            if document_lines:
                documents += 1
                document_lines.clear()

        if nonempty_lines % 1000000 == 0:
            print(
                f"Lines processed: {nonempty_lines:,}",
                flush=True,
            )


# Detect a final document that wasn't followed by a blank line.
if document_lines:
    documents += 1


print()
print("=" * 70)
print("CORPUS STATISTICS")
print("=" * 70)

print(f"Documents detected : {documents:,}")
print(f"Non-empty lines    : {nonempty_lines:,}")
print(f"Empty lines        : {empty_lines:,}")
print(f"Characters         : {characters:,}")
print(f"Arabic characters  : {arabic_chars:,}")

arabic_ratio = (
    arabic_chars / characters
    if characters
    else 0
)

print(f"Arabic ratio       : {arabic_ratio:.2%}")


print()
print("=" * 70)
print("SAMPLE")
print("=" * 70)

with CORPUS.open(
    "r",
    encoding="utf-8",
    errors="strict",
) as f:

    shown = 0

    for line in f:

        if line.strip():

            print(line.strip()[:500])
            shown += 1

            if shown >= 5:
                break


print()
print("=" * 70)
print("FINAL CORPUS STATUS")
print("=" * 70)

if (
    size > 0
    and documents == EXPECTED_DOCS
    and arabic_chars > 0
):

    print("STATUS: CORPUS VERIFIED")

else:

    print("STATUS: CORPUS VERIFICATION FAILED")

    print()
    print("Expected documents :", EXPECTED_DOCS)
    print("Detected documents :", documents)

    raise SystemExit(1)


print("=" * 70)
