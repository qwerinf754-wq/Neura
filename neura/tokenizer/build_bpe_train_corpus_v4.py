from __future__ import annotations

import json
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

WIKI_PATH = (
    ROOT / "data/processed/wiki_ar_00006.tokenizer.txt"
)

SFT_DIR = ROOT / "data/sft"

OUT_PATH = (
    ROOT
    / "data/processed/neura_arabic_bpe_32k_v4"
    / "train_corpus.txt"
)

WIKI_LINES = 30_000
SEED = 42
MIN_CHARS = 20


def sample_wikipedia() -> list[str]:
    rng = random.Random(SEED)

    reservoir: list[str] = []
    seen = 0

    with WIKI_PATH.open(
        "r",
        encoding="utf-8",
        errors="replace",
    ) as f:
        for raw in f:
            text = raw.strip()

            if len(text) < MIN_CHARS:
                continue

            seen += 1

            if len(reservoir) < WIKI_LINES:
                reservoir.append(text)
                continue

            index = rng.randrange(seen)

            if index < WIKI_LINES:
                reservoir[index] = text

    return reservoir


def collect_sft() -> list[str]:
    texts: list[str] = []

    for path in sorted(SFT_DIR.glob("*.jsonl")):
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line_number, raw in enumerate(f, 1):
                raw = raw.strip()

                if not raw:
                    continue

                try:
                    item = json.loads(raw)
                except json.JSONDecodeError as exc:
                    print(
                        f"[WARN] Invalid JSON: "
                        f"{path}:{line_number}: {exc}"
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
                        texts.append(content)

    return texts


def main() -> None:
    print("===== BUILD BPE V4 TRAIN CORPUS =====")

    print()
    print("Sampling Wikipedia...")
    wiki = sample_wikipedia()
    print(f"Wikipedia: {len(wiki):,}")

    print()
    print("Collecting SFT...")
    sft = collect_sft()
    print(f"SFT text segments: {len(sft):,}")

    if len(sft) == 0:
        raise RuntimeError(
            "No SFT text was collected."
        )

    OUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("Writing corpus...")

    with OUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as f:
        for text in wiki:
            f.write(text)
            f.write("\n")

        for text in sft:
            f.write(text)
            f.write("\n")

    lines = 0
    chars = 0

    with OUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            lines += 1
            chars += len(line.rstrip("\n"))

    print()
    print("===== COMPLETE =====")
    print(f"Output: {OUT_PATH}")
    print(f"Lines: {lines:,}")
    print(f"Characters: {chars:,}")


if __name__ == "__main__":
    main()
