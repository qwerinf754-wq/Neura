"""
Neura BPE Smoke Test

Purpose:
    Validate BPE training on a small streaming sample before
    running expensive training on the full Arabic corpus.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from neura.tokenizer.bpe import BPETrainer
from neura.tokenizer.bpe_encoder import BPEEncoder
from neura.tokenizer.bpe_decoder import BPEDecoder


CORPUS = Path(
    "data/processed/wiki_ar_00006.tokenizer.txt"
)

OUTPUT = Path(
    "data/processed/tokenizer/smoke_test_bpe.json"
)

SAMPLE_LINES = 20_000
VOCAB_SIZE = 512


def read_sample(path: Path, limit: int):
    texts = []

    with path.open(
        "r",
        encoding="utf-8",
        errors="strict",
    ) as f:

        for line_no, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            texts.append(line)

            if len(texts) >= limit:
                break

    return texts


def main():

    print("=" * 70)
    print("NEURA BPE SMOKE TEST")
    print("=" * 70)

    if not CORPUS.exists():
        raise FileNotFoundError(
            f"Corpus not found: {CORPUS}"
        )

    corpus_size = CORPUS.stat().st_size

    print(f"Corpus       : {CORPUS}")
    print(f"Corpus bytes : {corpus_size:,}")
    print(f"Sample lines : {SAMPLE_LINES:,}")
    print(f"Vocab target : {VOCAB_SIZE:,}")
    print()

    print("=" * 70)
    print("READING SAMPLE")
    print("=" * 70)

    start = time.time()

    texts = read_sample(
        CORPUS,
        SAMPLE_LINES,
    )

    elapsed = time.time() - start

    if not texts:
        raise RuntimeError(
            "No training text was read"
        )

    print(f"Texts loaded : {len(texts):,}")
    print(f"Read time    : {elapsed:.2f}s")

    print()
    print("=" * 70)
    print("TRAINING")
    print("=" * 70)

    trainer = BPETrainer(
        vocab_size=VOCAB_SIZE,
    )

    start = time.time()

    trainer.train_texts(
        texts
    )

    elapsed = time.time() - start

    print(f"Training time : {elapsed:.2f}s")
    print(f"Vocabulary    : {len(trainer.vocab):,}")
    print(f"Merges        : {len(trainer.merges):,}")

    if len(trainer.vocab) < 4:
        raise RuntimeError(
            "Vocabulary is unexpectedly small"
        )

    if not trainer.merges:
        raise RuntimeError(
            "No BPE merges were learned"
        )

    print()
    print("=" * 70)
    print("SAVE")
    print("=" * 70)

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save(
        OUTPUT
    )

    if not OUTPUT.exists():
        raise RuntimeError(
            "Tokenizer artifact was not created"
        )

    print(f"Artifact : {OUTPUT}")
    print(f"Size     : {OUTPUT.stat().st_size:,} bytes")

    print()
    print("=" * 70)
    print("ENCODE / DECODE")
    print("=" * 70)

    encoder = BPEEncoder(
        trainer.vocab,
        trainer.merges,
    )

    id_to_token = {
        idx: token
        for token, idx in trainer.vocab.items()
    }

    decoder = BPEDecoder(
        id_to_token
    )

    samples = [
        "مرحبا",
        "مرحبا بالعالم",
        "الذكاء الاصطناعي العربي",
        "نموذج Neura للذكاء الاصطناعي",
        "اختبار UTF-8: العربية 123 !؟",
    ]

    for text in samples:

        ids = encoder.encode(text)

        decoded = decoder.decode(ids)

        print()
        print("INPUT  :", repr(text))
        print("TOKENS :", len(ids))
        print("OUTPUT :", repr(decoded))

        if decoded != text:
            raise RuntimeError(
                f"Round-trip failed: {text!r}"
            )

    print()
    print("=" * 70)
    print("ARTIFACT VALIDATION")
    print("=" * 70)

    with OUTPUT.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = json.load(f)

    if "vocab" not in data:
        raise RuntimeError(
            "Artifact has no vocabulary"
        )

    if "merges" not in data:
        raise RuntimeError(
            "Artifact has no merges"
        )

    if not data["vocab"]:
        raise RuntimeError(
            "Artifact vocabulary is empty"
        )

    print("JSON structure : PASS")
    print("Vocabulary     : PASS")
    print("Merges         : PASS")
    print("Round-trip     : PASS")

    print()
    print("=" * 70)
    print("STATUS: BPE SMOKE TEST PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
