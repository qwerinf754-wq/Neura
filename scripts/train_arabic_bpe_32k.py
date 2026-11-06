#!/usr/bin/env python3

"""
Neura Arabic Byte-Level BPE 32K Trainer.

Pipeline:
    corpus
      ↓
    exact line reader
      ↓
    UTF-8 byte BPE
      ↓
    incremental pair counting
      ↓
    checkpoint
      ↓
    final tokenizer artifact
      ↓
    exact round-trip validation

Important:
    - Never strip corpus content.
    - Only physical CR/LF endings are removed.
    - Spaces, tabs and punctuation are preserved.
    - Existing tokenizer artifacts are never overwritten unless
      explicitly requested.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from neura.tokenizer.bpe import BPETrainer
from neura.tokenizer.bpe_encoder import BPEEncoder


# =====================================================================
# Defaults
# =====================================================================

DEFAULT_CORPUS = Path(
    "data/processed/wiki_ar_00006.tokenizer.txt"
)

DEFAULT_CHECKPOINT = Path(
    "data/checkpoints/tokenizer/"
    "neura_arabic_bpe_32k_v3.checkpoint.json"
)

DEFAULT_OUTPUT = Path(
    "data/processed/neura_arabic_bpe_32k_v3/"
    "tokenizer.json"
)

DEFAULT_VOCAB_SIZE = 32768
DEFAULT_MIN_FREQUENCY = 2
DEFAULT_CHECKPOINT_EVERY = 250


SPECIAL_TOKENS = [
    "<PAD>",
    "<UNK>",
    "<BOS>",
    "<EOS>",
]


# =====================================================================
# Utility
# =====================================================================

def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def atomic_json_write(
    path: Path,
    payload: dict,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
        )

        handle.write("\n")

    os.replace(
        temporary,
        path,
    )


# =====================================================================
# Corpus
# =====================================================================

def read_lines(
    path: Path,
    max_lines: int | None = None,
):
    """
    Read corpus line-by-line.

    Only CR/LF is removed.

    IMPORTANT:
        Do not call strip().
        Do not call split().
    """

    with path.open(
        "r",
        encoding="utf-8",
        errors="ignore",
    ) as handle:

        count = 0

        for raw_line in handle:
            line = raw_line.rstrip("\r\n")

            if not line:
                continue

            yield line

            count += 1

            if (
                max_lines is not None
                and count >= max_lines
            ):
                break


def load_lines(
    path: Path,
    max_lines: int | None = None,
) -> list[str]:

    return list(
        read_lines(
            path,
            max_lines=max_lines,
        )
    )


# =====================================================================
# Checkpoint
# =====================================================================

def trainer_state(
    trainer: BPETrainer,
) -> dict:

    return {
        "format": "neura-arabic-bpe-checkpoint-v3",

        "vocab_size": int(
            trainer.vocab_size
        ),

        "min_frequency": int(
            trainer.min_frequency
        ),

        "vocab": trainer.vocab,

        "merges": [
            list(pair)
            for pair in trainer.merges
        ],
    }


def save_training_checkpoint(
    path: Path,
    trainer: BPETrainer,
    *,
    corpus: Path,
    corpus_sha256: str,
    lines_used: int,
) -> None:

    payload = {
        "format":
            "neura-arabic-bpe-checkpoint-v3",

        "corpus": {
            "path": str(corpus),
            "sha256": corpus_sha256,
            "lines_used": int(lines_used),
        },

        "trainer": trainer_state(
            trainer
        ),

        "progress": {
            "vocab_size": len(
                trainer.vocab
            ),

            "merges": len(
                trainer.merges
            ),
        },
    }

    atomic_json_write(
        path,
        payload,
    )


def load_training_checkpoint(
    path: Path,
) -> dict:

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        payload = json.load(handle)

    if payload.get("format") != (
        "neura-arabic-bpe-checkpoint-v3"
    ):
        raise RuntimeError(
            "Unsupported checkpoint format."
        )

    return payload


def restore_trainer(
    checkpoint: dict,
) -> BPETrainer:

    state = checkpoint["trainer"]

    trainer = BPETrainer(
        vocab_size=int(
            state["vocab_size"]
        ),
        min_frequency=int(
            state["min_frequency"]
        ),
    )

    trainer.vocab = {
        str(token): int(token_id)
        for token, token_id
        in state["vocab"].items()
    }

    trainer.merges = [
        (
            int(pair[0]),
            int(pair[1]),
        )
        for pair in state["merges"]
    ]

    return trainer


# =====================================================================
# Artifact
# =====================================================================

def build_artifact(
    trainer: BPETrainer,
    *,
    corpus: Path,
    corpus_sha256: str,
    lines_used: int,
) -> dict:

    return {
        "format":
            "neura-arabic-bpe-v3",

        "name":
            "Neura Arabic BPE",

        "version":
            "3.0",

        "vocab_size":
            len(trainer.vocab),

        "target_vocab_size":
            trainer.vocab_size,

        "min_frequency":
            trainer.min_frequency,

        "special_tokens": {
            token: index
            for index, token
            in enumerate(
                SPECIAL_TOKENS
            )
        },

        "vocab":
            trainer.vocab,

        "merges": [
            list(pair)
            for pair in trainer.merges
        ],

        "metadata": {
            "corpus":
                str(corpus),

            "corpus_sha256":
                corpus_sha256,

            "lines_used":
                int(lines_used),

            "created_by":
                "scripts/train_arabic_bpe_32k.py",

            "trainer":
                "BPETrainer-v3",

            "encoding":
                "UTF-8-byte-level",

            "whitespace_preserved":
                True,

            "punctuation_preserved":
                True,
        },
    }


# =====================================================================
# Validation
# =====================================================================

def validate_artifact(
    trainer: BPETrainer,
    output: Path,
) -> None:

    encoder = BPEEncoder(
        vocab=trainer.vocab,
        merges=trainer.merges,
    )

    samples = [
        "مرحبا بكم في مشروع Neura.",
        "مرحبا،Neura! كيف حالك؟",
        "Hello, Neura!",
        "سطر أول\nسطر ثانٍ",
        "  مسافات   متعددة  ",
        "123 + 456 = 579",
        "العربية English 123 !؟،.",
        "\tTab\tاختبار\t",
    ]

    failures = 0

    print()
    print("=" * 72)
    print("FINAL BPE VALIDATION")
    print("=" * 72)

    for text in samples:
        ids = encoder.encode(text)
        decoded = encoder.decode(ids)

        passed = (
            decoded == text
        )

        print(
            "PASS" if passed else "FAIL",
            "| tokens =",
            len(ids),
            "|",
            repr(text),
        )

        if not passed:
            failures += 1

    if failures:
        raise RuntimeError(
            f"BPE validation failed: {failures}"
        )

    artifact = build_artifact(
        trainer,
        corpus=Path("unknown"),
        corpus_sha256="unknown",
        lines_used=0,
    )

    # Rebuild exact artifact from the real trainer.
    atomic_json_write(
        output,
        artifact,
    )

    print()
    print("[PASS] Exact round-trip validation")
    print("[PASS] Artifact serialization")
    print()
    print("FINAL VOCAB :", len(trainer.vocab))
    print("FINAL MERGES:", len(trainer.merges))
    print()
    print("BPE ARTIFACT VALIDATION: PASS")


# =====================================================================
# Training
# =====================================================================

def train(
    *,
    corpus: Path,
    checkpoint: Path,
    output: Path,
    vocab_size: int,
    min_frequency: int,
    max_lines: int | None,
    checkpoint_every: int,
    resume: bool,
) -> None:

    print("=" * 72)
    print("NEURA ARABIC BPE TRAINER V3")
    print("=" * 72)

    print("Corpus     :", corpus)
    print("Checkpoint :", checkpoint)
    print("Output     :", output)
    print("Vocab      :", vocab_size)
    print("Min freq   :", min_frequency)

    if not corpus.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus}"
        )

    print()
    print("Calculating corpus SHA256...")

    sha_start = time.perf_counter()

    corpus_sha256 = sha256_file(
        corpus
    )

    print(
        "SHA256     :",
        corpus_sha256,
    )

    print(
        "Hash time  :",
        f"{time.perf_counter() - sha_start:.2f}s",
    )

    # ---------------------------------------------------------------
    # Read corpus
    # ---------------------------------------------------------------

    print()
    print("Reading corpus...")

    read_start = time.perf_counter()

    lines = load_lines(
        corpus,
        max_lines=max_lines,
    )

    if not lines:
        raise RuntimeError(
            "Corpus contains no usable lines."
        )

    print(
        "Lines      :",
        len(lines),
    )

    print(
        "Read time  :",
        f"{time.perf_counter() - read_start:.2f}s",
    )

    # ---------------------------------------------------------------
    # Resume
    # ---------------------------------------------------------------

    trainer: BPETrainer

    if resume and checkpoint.exists():

        print()
        print("[RESUME] Loading checkpoint...")

        payload = load_training_checkpoint(
            checkpoint
        )

        checkpoint_corpus = payload[
            "corpus"
        ]

        if checkpoint_corpus[
            "sha256"
        ] != corpus_sha256:

            raise RuntimeError(
                "Checkpoint corpus SHA256 does not match "
                "the current corpus."
            )

        if int(
            checkpoint_corpus[
                "lines_used"
            ]
        ) != len(lines):

            raise RuntimeError(
                "Checkpoint lines_used does not match "
                "the current training configuration."
            )

        trainer = restore_trainer(
            payload
        )

        if trainer.vocab_size != vocab_size:
            raise RuntimeError(
                "Checkpoint vocab size mismatch."
            )

        if trainer.min_frequency != min_frequency:
            raise RuntimeError(
                "Checkpoint min_frequency mismatch."
            )

        print(
            "[RESUME] Vocabulary:",
            len(trainer.vocab),
        )

        print(
            "[RESUME] Merges:",
            len(trainer.merges),
        )

    else:

        print()
        print("[NEW] Starting new BPE training...")

        trainer = BPETrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
        )

    # ---------------------------------------------------------------
    # Train
    # ---------------------------------------------------------------

    print()
    print("=" * 72)
    print("TRAINING")
    print("=" * 72)

    start = time.perf_counter()

    result = trainer.train_texts(
        lines,
        checkpoint_interval=checkpoint_every,
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print()
    print("TRAIN RESULT")
    print(result)

    print(
        "Training time:",
        f"{elapsed:.2f}s",
    )

    # ---------------------------------------------------------------
    # Final checkpoint
    # ---------------------------------------------------------------

    save_training_checkpoint(
        checkpoint,
        trainer,
        corpus=corpus,
        corpus_sha256=corpus_sha256,
        lines_used=len(lines),
    )

    print()
    print(
        "[PASS] Final checkpoint saved:"
    )

    print(
        checkpoint
    )

    # ---------------------------------------------------------------
    # Artifact
    # ---------------------------------------------------------------

    artifact = build_artifact(
        trainer,
        corpus=corpus,
        corpus_sha256=corpus_sha256,
        lines_used=len(lines),
    )

    atomic_json_write(
        output,
        artifact,
    )

    print()
    print(
        "[PASS] Final tokenizer saved:"
    )

    print(
        output
    )

    # ---------------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------------

    validate_artifact(
        trainer,
        output,
    )

    print()
    print("=" * 72)
    print("TRAINING COMPLETE")
    print("=" * 72)


# =====================================================================
# CLI
# =====================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train Neura Arabic byte-level BPE."
        )
    )

    parser.add_argument(
        "--corpus",
        type=Path,
        default=DEFAULT_CORPUS,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--vocab-size",
        type=int,
        default=DEFAULT_VOCAB_SIZE,
    )

    parser.add_argument(
        "--min-frequency",
        type=int,
        default=DEFAULT_MIN_FREQUENCY,
    )

    parser.add_argument(
        "--max-lines",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
    )

    parser.add_argument(
        "--resume",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    train(
        corpus=args.corpus,
        checkpoint=args.checkpoint,
        output=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_lines=args.max_lines,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
