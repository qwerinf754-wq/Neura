from __future__ import annotations

import json
from pathlib import Path

from neura.tokenizer.bpe import BPETrainer


ROOT = Path(__file__).resolve().parents[1]

CORPUS_PATH = (
    ROOT / "data/processed/wiki_ar_00006.tokenizer.txt"
)

CHECKPOINT_PATH = (
    ROOT
    / "data/checkpoints/tokenizer/"
    "neura_arabic_bpe_32k_v3.checkpoint.json"
)

OUTPUT_PATH = (
    ROOT
    / "data/processed/neura_arabic_bpe_32k_v3/"
    "tokenizer.json"
)

TARGET_VOCAB_SIZE = 32768
MIN_FREQUENCY = 2
CHECKPOINT_INTERVAL = 250
MAX_LINES = 10_000


def collect_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)

    lines: list[str] = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="strict",
    ) as handle:
        for line in handle:
            text = line.rstrip("\r\n")

            if not text:
                continue

            lines.append(text)

            if len(lines) >= MAX_LINES:
                break

    if not lines:
        raise RuntimeError("No usable corpus lines found.")

    return lines


def verify_experimental_artifact() -> None:
    experimental = (
        ROOT
        / "data/processed/neura_arabic_bpe_32k/"
        "tokenizer.json"
    )

    if not experimental.exists():
        raise RuntimeError(
            "Experimental 2048 artifact is missing: "
            f"{experimental}"
        )

    data = json.loads(
        experimental.read_text(encoding="utf-8")
    )

    vocab = data.get("vocab", {})

    if len(vocab) != 2048:
        raise RuntimeError(
            "Experimental artifact changed unexpectedly. "
            f"Expected 2048, got {len(vocab)}."
        )

    print(
        "Experimental 2048 artifact: VERIFIED"
    )


def main() -> None:
    print("=" * 70)
    print("NEURA REAL ARABIC BYTE-LEVEL BPE 32K V3")
    print("=" * 70)

    print("Corpus       :", CORPUS_PATH)
    print("Checkpoint   :", CHECKPOINT_PATH)
    print("Output       :", OUTPUT_PATH)
    print("Target vocab :", TARGET_VOCAB_SIZE)
    print("Min freq     :", MIN_FREQUENCY)
    print("Max lines    :", f"{MAX_LINES:,}")
    print("Checkpoint   :", f"every {CHECKPOINT_INTERVAL} merges")
    print()

    verify_experimental_artifact()

    print()
    print("[1/4] Loading corpus...")

    lines = collect_lines(CORPUS_PATH)

    print(
        "Lines loaded:",
        f"{len(lines):,}",
    )

    print(
        "Characters  :",
        f"{sum(len(x) for x in lines):,}",
    )

    print()
    print("[2/4] Creating BPETrainer...")

    trainer = BPETrainer(
        vocab_size=TARGET_VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        checkpoint_path=CHECKPOINT_PATH,
    )

    print()
    print("[3/4] Checking checkpoint...")

    if CHECKPOINT_PATH.exists():
        print("RESUME: checkpoint found")
        resumed = trainer.load_checkpoint()

        if not resumed:
            raise RuntimeError(
                "Checkpoint exists but could not be loaded."
            )

        print(
            "Current vocab :",
            f"{len(trainer.vocab):,}",
        )
        print(
            "Current merges:",
            f"{len(trainer.merges):,}",
        )
    else:
        resumed = False
        print("RESUME: no checkpoint")
        print("Starting fresh 32K BPE training.")

    print()
    print("[4/4] Training...")
    print(
        "Checkpoint interval:",
        CHECKPOINT_INTERVAL,
    )
    print()

    result = trainer.train_texts(
        lines,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )

    print()
    print("Training result:")

    for key, value in result.items():
        print(f"  {key:18s}: {value:,}")

    print()
    print("Saving REAL 32K tokenizer artifact...")

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save(OUTPUT_PATH)

    print()
    print("=" * 70)
    print("REAL 32K BPE TRAINING COMPLETE")
    print("=" * 70)
    print("Artifact   :", OUTPUT_PATH)
    print("Checkpoint :", CHECKPOINT_PATH)
    print("=" * 70)


if __name__ == "__main__":
    main()
