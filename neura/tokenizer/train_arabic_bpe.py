"""
Neura Arabic Byte-Level BPE Trainer
===================================

Builds a production tokenizer artifact from the Arabic corpus
without loading the complete corpus into memory.

Input:
    data/processed/wiki_ar_00006.tokenizer.txt

Output:
    data/checkpoints/tokenizer/neura_arabic_bpe.json
    data/checkpoints/tokenizer/neura_arabic_bpe.checkpoint.json
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from neura.tokenizer.bpe import BPETrainer


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------

CORPUS_PATH = Path(
    "data/processed/wiki_ar_00006.tokenizer.txt"
)

OUTPUT_PATH = Path(
    "data/checkpoints/tokenizer/neura_arabic_bpe.json"
)

CHECKPOINT_PATH = Path(
    "data/checkpoints/tokenizer/neura_arabic_bpe.checkpoint.json"
)

TARGET_VOCAB_SIZE = 32768

MIN_FREQUENCY = 2

# Maximum number of non-empty lines used during the first tokenizer
# build. This keeps the initial run safe on Android/Termux.
MAX_LINES = 500_000

CHECKPOINT_INTERVAL = 250


# ------------------------------------------------------------------
# Corpus sampling
# ------------------------------------------------------------------

def collect_lines(
    path: Path,
    max_lines: int,
) -> list[str]:

    if not path.exists():
        raise FileNotFoundError(path)

    lines: list[str] = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:

        for line in handle:

            line = line.strip()

            if not line:
                continue

            lines.append(line)

            if len(lines) >= max_lines:
                break

    if not lines:
        raise RuntimeError(
            "No usable corpus lines were found."
        )

    return lines


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("NEURA ARABIC BYTE-LEVEL BPE TRAINING")
    print("=" * 70)

    print("Corpus       :", CORPUS_PATH)
    print("Target vocab :", TARGET_VOCAB_SIZE)
    print("Min freq     :", MIN_FREQUENCY)
    print("Max lines    :", MAX_LINES)
    print()

    print("[1/4] Reading corpus sample...")

    lines = collect_lines(
        CORPUS_PATH,
        MAX_LINES,
    )

    print(
        "Lines loaded:",
        f"{len(lines):,}",
    )

    total_chars = sum(
        len(line)
        for line in lines
    )

    print(
        "Characters  :",
        f"{total_chars:,}",
    )

    print()

    # --------------------------------------------------------------
    # Trainer
    # --------------------------------------------------------------

    print("[2/4] Creating BPE trainer...")

    trainer = BPETrainer(
        vocab_size=TARGET_VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        checkpoint_path=CHECKPOINT_PATH,
    )

    # --------------------------------------------------------------
    # Resume if possible
    # --------------------------------------------------------------

    resumed = trainer.load_checkpoint()

    if resumed:
        print(
            "Existing BPE checkpoint found."
        )

        print(
            "Current vocab:",
            f"{len(trainer.vocab):,}",
        )

        print(
            "Current merges:",
            f"{len(trainer.merges):,}",
        )

    else:
        print(
            "Starting new BPE training."
        )

    print()

    # --------------------------------------------------------------
    # Training
    # --------------------------------------------------------------

    print("[3/4] Training BPE...")

    result = trainer.train_texts(
        lines,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )

    print()

    print(
        "BPE result:"
    )

    for key, value in result.items():

        print(
            f"  {key:16s}: {value:,}"
        )

    print()

    # --------------------------------------------------------------
    # Save final artifact
    # --------------------------------------------------------------

    print("[4/4] Saving tokenizer...")

    trainer.save(
        OUTPUT_PATH
    )

    print(
        "Tokenizer saved:",
        OUTPUT_PATH,
    )

    print(
        "Checkpoint saved:",
        CHECKPOINT_PATH,
    )

    print()
    print("BPE TRAINING COMPLETE")


if __name__ == "__main__":
    main()
