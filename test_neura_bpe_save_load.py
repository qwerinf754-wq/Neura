"""
Neura BPE save/load integrity test.
"""

from pathlib import Path

from neura.tokenizer.bpe import BPETrainer
from neura.tokenizer.bpe_encoder import BPEEncoder
from neura.tokenizer.bpe_decoder import BPEDecoder


ARTIFACT = Path(
    "data/processed/tokenizer/test_bpe.json"
)

CHECKPOINT = Path(
    "data/checkpoints/tokenizer/test_bpe.json"
)


def build_encoder(trainer):
    return BPEEncoder(
        trainer.vocab,
        trainer.merges,
    )


def build_decoder(trainer):
    id_to_token = {
        idx: token
        for token, idx in trainer.vocab.items()
    }

    return BPEDecoder(id_to_token)


def main():

    print("=" * 70)
    print("NEURA BPE SAVE/LOAD INTEGRITY TEST")
    print("=" * 70)

    trainer = BPETrainer(
        vocab_size=300,
        min_frequency=1,
        checkpoint_path=CHECKPOINT,
    )

    trainer.train_texts(
        [
            "مرحبا بالعالم",
            "مرحبا Neura",
            "الذكاء الاصطناعي العربي",
            "Neura AI",
            "اختبار الحفظ وإعادة التحميل",
        ],
        checkpoint_interval=0,
    )

    print()
    print("TRAINED VOCAB :", len(trainer.vocab))
    print("TRAINED MERGES:", len(trainer.merges))

    ARTIFACT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trainer.save(ARTIFACT)

    if not ARTIFACT.exists():
        raise RuntimeError(
            "Artifact was not created"
        )

    print("ARTIFACT      :", ARTIFACT)
    print("ARTIFACT SIZE :", ARTIFACT.stat().st_size)

    encoder_before = build_encoder(trainer)
    decoder_before = build_decoder(trainer)

    samples = [
        "مرحبا",
        "مرحبا بالعالم",
        "الذكاء الاصطناعي العربي",
        "Neura AI",
        "اختبار UTF-8: العربية 123 !؟",
    ]

    encoded_before = {}

    for text in samples:
        ids = encoder_before.encode(text)
        decoded = decoder_before.decode(ids)

        if decoded != text:
            raise RuntimeError(
                f"Pre-save round-trip failed: {text!r}"
            )

        encoded_before[text] = ids

    print()
    print("PRE-SAVE ROUND-TRIP: PASS")

    loaded = BPETrainer(
        vocab_size=300,
        min_frequency=1,
        checkpoint_path=CHECKPOINT,
    )

    loaded_data = loaded.load_checkpoint()

    if not loaded_data:
        raise RuntimeError(
            "Checkpoint could not be loaded"
        )

    print("CHECKPOINT LOAD : PASS")

    # Verify exact learned state.
    if trainer.vocab != loaded.vocab:
        raise RuntimeError(
            "Vocabulary changed after checkpoint load"
        )

    if trainer.merges != loaded.merges:
        raise RuntimeError(
            "Merges changed after checkpoint load"
        )

    print("VOCAB INTEGRITY  : PASS")
    print("MERGE INTEGRITY  : PASS")

    encoder_after = build_encoder(loaded)
    decoder_after = build_decoder(loaded)

    for text in samples:

        ids = encoder_after.encode(text)
        decoded = decoder_after.decode(ids)

        if ids != encoded_before[text]:
            raise RuntimeError(
                f"Encoding changed after reload: {text!r}"
            )

        if decoded != text:
            raise RuntimeError(
                f"Post-load round-trip failed: {text!r}"
            )

    print("POST-LOAD ENCODE : PASS")
    print("POST-LOAD DECODE : PASS")

    print()
    print("=" * 70)
    print("STATUS: BPE SAVE/LOAD PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
