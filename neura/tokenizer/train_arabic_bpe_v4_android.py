from __future__ import annotations

import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]

CORPUS_PATH = (
    ROOT
    / "data/processed/neura_arabic_bpe_32k_v4"
    / "train_corpus.txt"
)

MANIFEST_PATH = (
    ROOT
    / "data/processed/neura_arabic_bpe_32k_v4"
    / "manifest.json"
)

OUTPUT_PATH = (
    ROOT
    / "data/processed/neura_arabic_bpe_32k_v4"
    / "tokenizer.json"
)

CHECKPOINT_PATH = (
    ROOT
    / "data/checkpoints/tokenizer"
    / "neura_arabic_bpe_32k_v4.checkpoint.json"
)

TARGET_VOCAB_SIZE = 32768
MIN_FREQUENCY = 2
CHECKPOINT_INTERVAL = 25


def load_texts() -> list[str]:
    texts: list[str] = []

    print("Loading training corpus...")

    with CORPUS_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        for line in f:
            text = line.rstrip("\r\n")

            if text:
                texts.append(text)

    print(f"Training sequences: {len(texts):,}")

    if not texts:
        raise RuntimeError("Training corpus is empty.")

    return texts


def main() -> None:
    started = time.time()

    print("=" * 64)
    print("NEURA BYTE-LEVEL BPE V4 — ANDROID")
    print("=" * 64)

    if not CORPUS_PATH.exists():
        raise FileNotFoundError(CORPUS_PATH)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    print(f"Corpus:     {CORPUS_PATH}")
    print(f"Output:     {OUTPUT_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Target:     {TARGET_VOCAB_SIZE:,}")
    print(f"Min freq:   {MIN_FREQUENCY}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL}")
    print("=" * 64)

    texts = load_texts()

    # Import only after validating the corpus.
    from neura.tokenizer.bpe import BPETrainer

    print()
    print("Creating BPETrainer...")

    trainer = BPETrainer(
        vocab_size=TARGET_VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        checkpoint_path=CHECKPOINT_PATH,
    )

    # ------------------------------------------------------------
    # Safe checkpoint resume
    # ------------------------------------------------------------
    if CHECKPOINT_PATH.exists():
        print()
        print("Existing BPE checkpoint found.")
        print(f"Loading: {CHECKPOINT_PATH}")

        try:
            payload = trainer.load_checkpoint()

            print(
                "Checkpoint loaded:",
                f"format={payload.get('format')}",
                f"vocab={len(trainer.vocab):,}",
                f"merges={len(trainer.merges):,}",
            )

        except Exception as exc:
            raise RuntimeError(
                "Existing BPE checkpoint could not be loaded safely. "
                "Refusing to overwrite or restart automatically. "
                f"Reason: {exc}"
            ) from exc

    else:
        print()
        print("No existing BPE checkpoint.")
        print("Starting from a fresh trainer.")

    print()
    print("Starting training...")
    print()
    print(
        "IMPORTANT: keep Termux in the foreground "
        "while this runs."
    )

    result = trainer.train_texts(
        texts,
        checkpoint_interval=CHECKPOINT_INTERVAL,
    )

    print()
    print("Training result:")
    print(result)

    manifest = json.loads(
        MANIFEST_PATH.read_text(
            encoding="utf-8"
        )
    )

    metadata = {
        "name": "Neura Arabic Byte-Level BPE",
        "version": "4.0",
        "language": "ar",
        "base_tokenizer": "byte-level-bpe",
        "training_corpus": str(
            CORPUS_PATH.relative_to(ROOT)
        ),
        "source_corpus_manifest": str(
            MANIFEST_PATH.relative_to(ROOT)
        ),
        "source_corpus_sha256": manifest[
            "corpus"
        ]["sha256"],
        "training_corpus_lines": len(texts),
        "target_vocab_size": TARGET_VOCAB_SIZE,
        "min_frequency": MIN_FREQUENCY,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "seed": 42,
        "android_safe_training_subset": True,
    }

    print()
    print("Saving tokenizer artifact...")

    trainer.save_artifact(
        OUTPUT_PATH,
        metadata=metadata,
    )

    elapsed = time.time() - started

    print()
    print("=" * 64)
    print("BPE V4 COMPLETE")
    print("=" * 64)
    print(f"Artifact: {OUTPUT_PATH}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Elapsed: {elapsed:.1f}s")
    print("=" * 64)


if __name__ == "__main__":
    main()
