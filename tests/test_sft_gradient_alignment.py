import math
import sys

sys.path.insert(0, ".")

from neura.tokenizer.neura_bpe_tokenizer import NeuraBPETokenizer
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE, IGNORE_INDEX
from neura.training.loss import CrossEntropyLoss


TOKENIZER_PATH = "data/processed/neura_arabic_bpe_32k/tokenizer.json"


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def main():
    print("=" * 80)
    print("NEURA SFT GRADIENT ALIGNMENT CONTRACT")
    print("FULL SEQUENCE GRADIENT + ASSISTANT-ONLY MASK")
    print("=" * 80)

    print("\n[1/5] BPE tokenizer")

    tokenizer = NeuraBPETokenizer(TOKENIZER_PATH)

    print("[PASS] Tokenizer")
    print("Vocab:", tokenizer.vocab_size)

    print("\n[2/5] Build Chat SFT sample")

    builder = ChatSFTBuilderBPE(
        tokenizer=tokenizer,
        max_length=256,
        add_bos=True,
        add_eos=False,
    )

    messages = [
        {
            "role": "system",
            "content": "You are Neura.",
        },
        {
            "role": "user",
            "content": "What is AI?",
        },
        {
            "role": "assistant",
            "content": "AI is artificial intelligence.",
        },
    ]

    sample = builder.build(messages)

    input_ids = sample.input_ids
    labels = sample.labels
    loss_mask = sample.loss_mask

    print("[PASS] Chat SFT")
    print("Sequence:", len(input_ids))
    print("Assistant tokens:", sum(loss_mask))

    if len(input_ids) != len(labels):
        raise AssertionError("input_ids / labels length mismatch")

    print("\n[3/5] Construct synthetic logits")

    vocab_size = tokenizer.vocab_size

    logits = []

    for position, token_id in enumerate(input_ids):
        row = [0.0] * vocab_size

        # Deterministic finite logits.
        row[token_id] = 1.0 + (position % 7) * 0.01

        logits.append(row)

    if len(logits) != len(labels):
        raise AssertionError("logits / labels length mismatch")

    print("[PASS] Synthetic logits")
    print("Logit rows:", len(logits))
    print("Vocab width:", vocab_size)

    print("\n[4/5] Assistant-only masked loss")

    filtered_logits = []
    filtered_labels = []
    assistant_positions = []

    for position, (row, label) in enumerate(
        zip(logits, labels)
    ):
        if label == IGNORE_INDEX:
            continue

        filtered_logits.append(row)
        filtered_labels.append(label)
        assistant_positions.append(position)

    if not filtered_labels:
        raise AssertionError("No assistant targets")

    loss_fn = CrossEntropyLoss()

    loss = loss_fn.forward(
        filtered_logits,
        filtered_labels,
    )

    gradients = loss_fn.gradients()

    if len(gradients) != len(assistant_positions):
        raise AssertionError(
            "Filtered gradient count does not match assistant targets"
        )

    if not finite(loss):
        raise AssertionError("Loss is not finite")

    print("[PASS] Masked loss")
    print("Loss:", loss)
    print("Assistant gradient rows:", len(gradients))

    print("\n[5/5] Reconstruct FULL SEQUENCE gradient")

    full_gradients = []

    gradient_width = vocab_size

    gradient_index = 0

    for position, label in enumerate(labels):

        if label == IGNORE_INDEX:
            row = [0.0] * gradient_width
        else:
            row = gradients[gradient_index]
            gradient_index += 1

        if len(row) != gradient_width:
            raise AssertionError(
                f"Gradient width mismatch at position {position}"
            )

        if not all(finite(value) for value in row):
            raise AssertionError(
                f"Non-finite gradient at position {position}"
            )

        full_gradients.append(row)

    if gradient_index != len(gradients):
        raise AssertionError("Unused filtered gradients remain")

    if len(full_gradients) != len(labels):
        raise AssertionError(
            "Full gradient length does not match sequence length"
        )

    zero_rows = 0
    nonzero_rows = 0

    for position, (row, label) in enumerate(
        zip(full_gradients, labels)
    ):
        magnitude = sum(abs(x) for x in row)

        if label == IGNORE_INDEX:
            if magnitude != 0.0:
                raise AssertionError(
                    f"Masked position {position} has non-zero gradient"
                )
            zero_rows += 1
        else:
            if magnitude == 0.0:
                raise AssertionError(
                    f"Assistant position {position} has zero gradient"
                )
            nonzero_rows += 1

    print("[PASS] Full gradient reconstruction")
    print("Gradient rows:", len(full_gradients))
    print("Zero/masked rows:", zero_rows)
    print("Assistant rows:", nonzero_rows)

    print()
    print("=" * 80)
    print("===== SFT GRADIENT ALIGNMENT: PASS =====")
    print("=" * 80)
    print("FULL SEQUENCE GRADIENT: PASS")
    print("ASSISTANT-ONLY GRADIENT: PASS")
    print("MASKED ROWS ZERO: PASS")
    print("GRADIENTS FINITE: PASS")
    print("NO MODEL")
    print("NO OPTIMIZER")
    print("NO WEIGHT UPDATE")
    print("=" * 80)


if __name__ == "__main__":
    main()
