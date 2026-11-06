from __future__ import annotations

import json
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CORPUS_PATH = (
    ROOT
    / "data/processed/neura_arabic_bpe_32k_v4/corpus.txt"
)

MANIFEST_PATH = (
    ROOT
    / "data/processed/neura_arabic_bpe_32k_v4/manifest.json"
)

OUTPUT_PATH = (
    ROOT
    / "data/processed/neura_arabic_bpe_32k_v4/tokenizer.json"
)

CHECKPOINT_PATH = (
    ROOT
    / "data/checkpoints/tokenizer/"
    "neura_arabic_bpe_32k_v4.checkpoint.json"
)

TARGET_VOCAB_SIZE = 32768
MIN_FREQUENCY = 2
CHECKPOINT_INTERVAL = 250


def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    return json.loads(
        MANIFEST_PATH.read_text(encoding="utf-8")
    )


def verify_corpus(manifest: dict) -> None:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS_PATH}"
        )

    expected_sha = (
        manifest.get("corpus", {})
        .get("sha256")
    )

    if not expected_sha:
        raise RuntimeError(
            "Manifest does not contain corpus SHA256."
        )

    print("Verifying corpus SHA256...")

    import hashlib

    digest = hashlib.sha256()

    with CORPUS_PATH.open(
        "rb"
    ) as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            digest.update(chunk)

    actual_sha = digest.hexdigest()

    if actual_sha != expected_sha:
        raise RuntimeError(
            "Corpus SHA256 mismatch.\n"
            f"Expected: {expected_sha}\n"
            f"Actual:   {actual_sha}"
        )

    print("Corpus SHA256: OK")


def load_training_texts() -> list[str]:
    """
    Load corpus line-by-line.

    BPETrainer currently requires a materialized list of
    training strings, so this is intentionally explicit.
    """
    print()
    print("Loading corpus into training sequences...")

    texts: list[str] = []

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8",
        errors="strict",
    ) as f:
        for index, line in enumerate(f, 1):
            text = line.rstrip("\r\n")

            if not text:
                continue

            texts.append(text)

            if index % 10_000 == 0:
                print(
                    f"\rLoaded {index:,} lines...",
                    end="",
                    flush=True,
                )

    print()
    print(f"Training sequences: {len(texts):,}")

    if not texts:
        raise RuntimeError(
            "Corpus is empty."
        )

    return texts


def main() -> int:
    started = time.time()

    print("=" * 60)
    print("Neura Arabic Byte-Level BPE v4")
    print("=" * 60)
    print(f"Corpus:     {CORPUS_PATH}")
    print(f"Output:     {OUTPUT_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Target:     {TARGET_VOCAB_SIZE:,}")
    print(f"Min freq:   {MIN_FREQUENCY}")
    print("=" * 60)

    manifest = load_manifest()

    print()
    print(
        "Manifest corpus lines:",
        f"{manifest['corpus']['lines']:,}",
    )
    print(
        "Manifest characters:",
        f"{manifest['corpus']['characters']:,}",
    )

    verify_corpus(manifest)

    # Import after validation so a bad corpus fails before
    # allocating BPE structures.
    from neura.tokenizer.bpe import BPETrainer

    texts = load_training_texts()

    print()
    print("Creating BPE trainer...")

    trainer = BPETrainer(
        vocab_size=TARGET_VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        checkpoint_path=CHECKPOINT_PATH,
    )

    print()
    print("Starting BPE training...")
    print(
        "This may take a significant amount of time on Android."
    )
    print()

    result = trainer.train_texts(
        texts,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )

    print()
    print("BPE training returned:")
    print(result)

    print()
    print("Saving v4 tokenizer artifact...")

    metadata = {
        "name": "Neura Arabic Byte-Level BPE",
        "version": "4.0",
        "language": "ar",
        "base_tokenizer": "byte-level-bpe",
        "corpus": str(
            CORPUS_PATH.relative_to(ROOT)
        ),
        "corpus_manifest": str(
            MANIFEST_PATH.relative_to(ROOT)
        ),
        "corpus_sha256": manifest["corpus"]["sha256"],
        "corpus_lines": manifest["corpus"]["lines"],
        "corpus_characters": manifest["corpus"]["characters"],
        "wikipedia_lines": manifest["wikipedia"][
            "selected_lines"
        ],
        "sft_text_segments": manifest["sft"][
            "text_segments"
        ],
        "target_vocab_size": TARGET_VOCAB_SIZE,
        "min_frequency": MIN_FREQUENCY,
        "seed": manifest["seed"],
    }

    trainer.save_artifact(
        OUTPUT_PATH,
        metadata=metadata,
    )

    elapsed = time.time() - started

    print()
    print("=" * 60)
    print("BPE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Artifact: {OUTPUT_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Elapsed: {elapsed:.2f} seconds")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
