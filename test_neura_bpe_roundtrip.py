"""
Neura BPE round-trip test.
"""

from neura.tokenizer.bpe import BPETrainer
from neura.tokenizer.bpe_encoder import BPEEncoder
from neura.tokenizer.bpe_decoder import BPEDecoder


def main():

    trainer = BPETrainer(
        vocab_size=300,
        min_frequency=1,
    )

    trainer.train_texts(
        [
            "مرحبا بالعالم",
            "مرحبا Neura",
            "الذكاء الاصطناعي العربي",
            "Neura AI",
        ],
        checkpoint_interval=0,
    )

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
        "Neura AI",
        "اختبار UTF-8: العربية 123 !؟",
    ]

    print("=" * 70)
    print("NEURA BPE ROUND-TRIP TEST")
    print("=" * 70)

    for text in samples:

        ids = encoder.encode(text)

        decoded = decoder.decode(ids)

        print()
        print("INPUT  :", repr(text))
        print("TOKENS :", len(ids))
        print("OUTPUT :", repr(decoded))

        if decoded != text:
            raise RuntimeError(
                "ROUND-TRIP FAILED"
            )

    print()
    print("=" * 70)
    print("STATUS: BPE ROUND-TRIP PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
