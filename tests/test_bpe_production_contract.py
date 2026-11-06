from __future__ import annotations

from pathlib import Path

from neura.tokenizer.neura_bpe_tokenizer import NeuraBPETokenizer
from neura.training.chat_sft_bpe_dataset import (
    ChatSFTBuilderBPE,
    validate_assistant_only_mask,
)


ARTIFACT = Path(
    "data/processed/neura_arabic_bpe_32k/tokenizer.json"
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


print("=" * 72)
print("NEURA BPE PRODUCTION CONTRACT TEST")
print("NO TRAINING")
print("=" * 72)

check(ARTIFACT.exists(), f"Artifact not found: {ARTIFACT}")

tokenizer = NeuraBPETokenizer(
    ARTIFACT,
    frozen=True,
)

print()
print("[PASS] Tokenizer loaded")
print("       Artifact:", ARTIFACT)
print("       Vocab size:", tokenizer.vocab_size)

check(tokenizer.vocab_size == 2048, "Unexpected vocabulary size")

check(
    tokenizer.vocab["<PAD>"] == 0,
    "<PAD> ID mismatch",
)
check(
    tokenizer.vocab["<UNK>"] == 1,
    "<UNK> ID mismatch",
)
check(
    tokenizer.vocab["<BOS>"] == 2,
    "<BOS> ID mismatch",
)
check(
    tokenizer.vocab["<EOS>"] == 3,
    "<EOS> ID mismatch",
)

print("[PASS] Special token IDs")

for value in range(256):
    token = f"<0x{value:02X}>"
    expected = 4 + value

    check(
        tokenizer.vocab.get(token) == expected,
        f"Byte token mismatch: {token}",
    )

print("[PASS] Raw byte token IDs 0..255")

texts = [
    "السلام عليكم",
    "مرحبا بالعالم",
    "Neura AI",
    "Hello world",
    "Neura AI - نظام ذكاء اصطناعي",
    "1234567890",
    "!@#$%^&*()",
    "العراق Iraq 2026",
    "هذا اختبار للنظام.",
]

print()
print("=" * 72)
print("ROUND TRIP TESTS")
print("=" * 72)

for text in texts:
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids)

    print()
    print("TEXT   :", repr(text))
    print("TOKENS :", len(ids))
    print("IDS    :", ids[:20], "..." if len(ids) > 20 else "")
    print("DECODE :", repr(decoded))

    check(
        decoded == text,
        f"Round-trip mismatch for {text!r}",
    )

    check(
        len(ids) > 0,
        "Encoder returned empty token sequence",
    )

    check(
        all(
            isinstance(token_id, int)
            for token_id in ids
        ),
        "Non-integer token ID detected",
    )

    check(
        all(
            0 <= token_id < tokenizer.vocab_size
            for token_id in ids
        ),
        "Token ID outside vocabulary range",
    )

print()
print("[PASS] All round-trip tests")

print()
print("=" * 72)
print("BOS / EOS TEST")
print("=" * 72)

text = "Neura نظام ذكي"

plain = tokenizer.encode(text)
with_specials = tokenizer.encode(
    text,
    add_bos=True,
    add_eos=True,
)

check(with_specials[0] == 2, "BOS missing")
check(with_specials[-1] == 3, "EOS missing")
check(
    with_specials[1:-1] == plain,
    "BOS/EOS changed content tokenization",
)

print("[PASS] BOS/EOS contract")

decoded_special = tokenizer.decode(
    with_specials,
    skip_special_tokens=True,
)

check(
    decoded_special == text,
    "Special-token round trip failed",
)

print("[PASS] Special-token decode")

print()
print("=" * 72)
print("CHAT SFT BPE TEST")
print("=" * 72)

builder = ChatSFTBuilderBPE(
    tokenizer=tokenizer,
    max_length=4096,
    add_bos=True,
    add_eos=False,
)

messages = [
    {
        "role": "system",
        "content": "أنت مساعد ذكي.",
    },
    {
        "role": "user",
        "content": "ما هو Neura؟",
    },
    {
        "role": "assistant",
        "content": "Neura هو نظام ذكاء اصطناعي.",
    },
]

example = builder.build(messages)

print("Input length :", len(example.input_ids))
print("Label length:", len(example.labels))
print("Mask length :", len(example.loss_mask))

check(
    len(example.input_ids)
    == len(example.labels)
    == len(example.loss_mask),
    "SFT lengths mismatch",
)

check(
    len(example.input_ids) > 0,
    "Empty SFT example",
)

check(
    any(x == 1 for x in example.loss_mask),
    "No assistant loss tokens",
)

validate_assistant_only_mask(example)

print("[PASS] Assistant-only loss mask")

assistant_positions = [
    i
    for i, value in enumerate(example.loss_mask)
    if value == 1
]

ignored_positions = [
    i
    for i, value in enumerate(example.loss_mask)
    if value == 0
]

print(
    "Assistant tokens:",
    len(assistant_positions),
)

print(
    "Ignored tokens:",
    len(ignored_positions),
)

print()
print("=" * 72)
print("FINAL RESULT")
print("=" * 72)
print("BPE PRODUCTION CONTRACT: PASS")
print("NO TRAINING WAS PERFORMED")
print("=" * 72)
