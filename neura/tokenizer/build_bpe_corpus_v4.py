from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]

WIKI_PATH = ROOT / "data/processed/wiki_ar_00006.tokenizer.txt"
SFT_DIR = ROOT / "data/sft"

OUT_DIR = ROOT / "data/processed/neura_arabic_bpe_32k_v4"
CORPUS_PATH = OUT_DIR / "corpus.txt"
MANIFEST_PATH = OUT_DIR / "manifest.json"

# Android-safe deterministic sample.
WIKI_MAX_LINES = 150_000
SEED = 42

# Keep very short Wikipedia fragments out of the tokenizer corpus.
MIN_WIKI_CHARS = 20


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def iter_sft_texts() -> Iterable[str]:
    """
    Extract only conversational text from SFT JSONL files.

    We intentionally do not feed raw JSON syntax into the tokenizer.
    """
    for path in sorted(SFT_DIR.glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line_number, raw in enumerate(f, 1):
                raw = raw.strip()

                if not raw:
                    continue

                try:
                    item = json.loads(raw)
                except json.JSONDecodeError:
                    print(
                        f"[WARN] invalid JSON: {path}:{line_number}"
                    )
                    continue

                messages = item.get("messages")

                if not isinstance(messages, list):
                    continue

                for message in messages:
                    if not isinstance(message, dict):
                        continue

                    content = message.get("content")

                    if not isinstance(content, str):
                        continue

                    content = content.strip()

                    if content:
                        yield content


def select_wikipedia_lines(
    path: Path,
    max_lines: int,
    seed: int,
) -> list[str]:
    """
    Deterministic reservoir sampling.

    This avoids loading the 529 MB Wikipedia file into RAM.
    """
    rng = random.Random(seed)

    reservoir: list[str] = []
    seen = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            text = raw.strip()

            if len(text) < MIN_WIKI_CHARS:
                continue

            seen += 1

            if len(reservoir) < max_lines:
                reservoir.append(text)
                continue

            index = rng.randrange(seen)

            if index < max_lines:
                reservoir[index] = text

    return reservoir


def main() -> None:
    if not WIKI_PATH.exists():
        raise FileNotFoundError(WIKI_PATH)

    if not SFT_DIR.exists():
        raise FileNotFoundError(SFT_DIR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("===== Neura BPE v4 Corpus Builder =====")
    print(f"Wikipedia: {WIKI_PATH}")
    print(f"SFT dir:   {SFT_DIR}")
    print(f"Output:    {CORPUS_PATH}")
    print()

    print(
        f"[1/4] Sampling up to {WIKI_MAX_LINES:,} "
        "Wikipedia lines..."
    )

    wiki_lines = select_wikipedia_lines(
        WIKI_PATH,
        WIKI_MAX_LINES,
        SEED,
    )

    print(f"      Selected: {len(wiki_lines):,}")

    print()
    print("[2/4] Collecting SFT conversation text...")

    sft_texts = list(iter_sft_texts())

    print(f"      SFT text segments: {len(sft_texts):,}")

    print()
    print("[3/4] Writing corpus...")

    # Deterministic ordering:
    # Wikipedia first, then SFT.
    #
    # We intentionally do not deduplicate aggressively here.
    # Repetition frequency is useful to BPE for common language patterns.
    with CORPUS_PATH.open("w", encoding="utf-8", newline="\n") as out:
        for text in wiki_lines:
            out.write(text)
            out.write("\n")

        for text in sft_texts:
            out.write(text)
            out.write("\n")

    print(f"      Corpus: {CORPUS_PATH}")

    print()
    print("[4/4] Building manifest...")

    line_count = 0
    char_count = 0

    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line_count += 1
            char_count += len(line.rstrip("\n"))

    manifest = {
        "format": "neura-bpe-v4-corpus",
        "version": 1,
        "seed": SEED,
        "wikipedia": {
            "source": str(WIKI_PATH.relative_to(ROOT)),
            "source_sha256": sha256_file(WIKI_PATH),
            "sample_limit": WIKI_MAX_LINES,
            "selected_lines": len(wiki_lines),
        },
        "sft": {
            "source_dir": str(SFT_DIR.relative_to(ROOT)),
            "files": sorted(
                str(p.relative_to(ROOT))
                for p in SFT_DIR.glob("*.jsonl")
            ),
            "text_segments": len(sft_texts),
        },
        "corpus": {
            "path": str(CORPUS_PATH.relative_to(ROOT)),
            "lines": line_count,
            "characters": char_count,
            "sha256": sha256_file(CORPUS_PATH),
        },
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("===== COMPLETE =====")
    print(f"Corpus lines: {line_count:,}")
    print(f"Characters:   {char_count:,}")
    print(f"Corpus SHA256: {manifest['corpus']['sha256']}")
    print(f"Manifest:     {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
