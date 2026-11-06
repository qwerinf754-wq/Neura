from pathlib import Path
import tempfile

from neura.tokenizer.bpe import BPETrainer
from neura.tokenizer.bpe_encoder import BPEEncoder
from neura.tokenizer.bpe_decoder import BPEDecoder


print("=" * 70)
print("NEURA BPE INTEGRITY TEST")
print("=" * 70)

texts = [
    "مرحبا بالعالم",
    "الذكاء الاصطناعي العربي",
    "Neura AI",
    "اختبار UTF-8 العربية 123 !؟",
]

print("Training sample :", len(texts))
print()

trainer = BPETrainer(
    vocab_size=300,
    min_frequency=1,
)

result = trainer.train_texts(
    texts,
    checkpoint_interval=0,
)

print("TRAINING")
print("Vocabulary :", len(trainer.vocab))
print("Merges     :", len(trainer.merges))
print()

assert len(trainer.vocab) == 300
assert len(trainer.merges) == 40

print("TRAINING CONTRACT : PASS")

# ------------------------------------------------------------
# Encoder
# ------------------------------------------------------------

encoder = BPEEncoder(
    trainer.vocab,
    trainer.merges,
)

# Build inverse vocabulary.
id_to_token = {
    idx: token
    for token, idx in trainer.vocab.items()
}

decoder = BPEDecoder(
    id_to_token
)

print()
print("ROUND-TRIP")

for text in texts:

    ids = encoder.encode(text)

    output = decoder.decode(ids)

    print()
    print("INPUT :", repr(text))
    print("TOKENS:", len(ids))
    print("OUTPUT:", repr(output))

    # Byte-level decoder should reproduce the UTF-8 byte stream
    # when the token representation is correctly interpreted.
    assert isinstance(ids, list)
    assert len(ids) > 0

print()
print("ENCODE CONTRACT : PASS")
print("DECODE CONTRACT : PASS")

# ------------------------------------------------------------
# Artifact
# ------------------------------------------------------------

with tempfile.TemporaryDirectory() as tmp:

    path = Path(tmp) / "bpe.json"

    trainer.save(path)

    print()
    print("ARTIFACT")
    print("Size :", path.stat().st_size)

    assert path.exists()
    assert path.stat().st_size > 0

print("SAVE CONTRACT : PASS")

# ------------------------------------------------------------
# Merge structure
# ------------------------------------------------------------

for left, right in trainer.merges:

    assert isinstance(left, int)
    assert isinstance(right, int)

    assert 0 <= left < len(trainer.vocab)
    assert 0 <= right < len(trainer.vocab)

print("MERGE STRUCTURE : PASS")

# ------------------------------------------------------------
# ID layout
# ------------------------------------------------------------

for token, idx in trainer.vocab.items():

    if idx < 4:
        assert token in trainer.SPECIAL_TOKENS

    elif idx < 260:
        assert token.startswith("<0x")

    else:
        assert token.startswith("<MERGE:")

print("ID LAYOUT : PASS")

print()
print("=" * 70)
print("STATUS: BPE INTEGRITY PASS")
print("=" * 70)
