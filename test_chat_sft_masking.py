from neura.training.chat_dataset import (
    IGNORE_INDEX,
    ChatSFTDatasetBuilder,
)
from neura.tokenizer import TOKENIZER


messages = [
    {
        "role": "system",
        "content": "You are Neura.",
    },
    {
        "role": "user",
        "content": "Hello",
    },
    {
        "role": "assistant",
        "content": "Hello. I'm Neura.",
    },
    {
        "role": "user",
        "content": "What are you?",
    },
    {
        "role": "assistant",
        "content": "I am Neura.",
    },
]


builder = ChatSFTDatasetBuilder()

example = builder.build(messages)

print("=" * 60)
print("NEURA CHAT SFT MASK INSPECTION")
print("=" * 60)

print()

train_count = 0
ignored_count = 0

for index, (token_id, label) in enumerate(
    zip(
        example.input_ids,
        example.labels,
    )
):

    token = TOKENIZER.vocab.decode(
        token_id
    )

    if label == IGNORE_INDEX:
        status = "IGNORE"
        ignored_count += 1
    else:
        status = "TRAIN"
        train_count += 1

    print(
        f"{index:02d} "
        f"id={token_id:03d} "
        f"status={status:<6} "
        f"token={token!r}"
    )


print()
print("=" * 60)
print("COUNTS")
print("=" * 60)

print("train:", train_count)
print("ignored:", ignored_count)

assert train_count > 0
assert ignored_count > 0

# ---------------------------------------------------------
# CRITICAL CHECKS
# ---------------------------------------------------------

CONTROL_TOKENS = {
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<PAD>",
    "<BOS>",
    "<EOS>",
}

# ---------------------------------------------------------
# ROLE-AWARE CONTROL CHECKS
# ---------------------------------------------------------
#
# Control markers must not receive loss, except:
#
#   assistant <|end|>
#
# which is intentionally supervised so the model learns
# when to terminate an assistant response.
#

rendered = builder.template.render(messages)

assistant_eos_positions = []

cursor = 0

for message in messages:

    block = builder.template.render(
        [message]
    )

    start = rendered.find(
        block,
        cursor,
    )

    if start < 0:
        raise RuntimeError(
            "unable to locate message block during "
            "masking contract validation"
        )

    end = start + len(block)

    if message["role"] == "assistant":

        assistant_eos_start = (
            end
            - len(
                "\n"
                + builder.template.END
            )
        )

        # The EOS token is the final tokenizer token in
        # the assistant block before the trailing newline.
        prefix_ids = builder.tokenizer.encode(
            rendered[:assistant_eos_start]
        )

        eos_ids = builder.tokenizer.encode(
            builder.template.END
        )

        if len(eos_ids) != 1:
            raise RuntimeError(
                "expected assistant EOS marker to encode "
                "as exactly one token"
            )

        assistant_eos_positions.append(
            len(prefix_ids)
        )

    cursor = end

    if (
        cursor < len(rendered)
        and rendered[cursor] == "\n"
    ):
        cursor += 1


assistant_eos_positions = set(
    assistant_eos_positions
)

print()
print(
    "assistant EOS positions:",
    sorted(assistant_eos_positions),
)

for index, (token_id, label) in enumerate(
    zip(
        example.input_ids,
        example.labels,
    )
):

    token = TOKENIZER.vocab.decode(
        token_id
    )

    if token in CONTROL_TOKENS:

        assert label == IGNORE_INDEX, (
            "control token received training loss: "
            f"{token!r} at position {index}"
        )

    elif token == "<|end|>":

        if index in assistant_eos_positions:
            assert label != IGNORE_INDEX, (
                "assistant EOS did not receive "
                "training loss at position "
                f"{index}"
            )
        else:
            assert label == IGNORE_INDEX, (
                "non-assistant EOS received "
                "training loss at position "
                f"{index}"
            )

print()
print("=" * 60)
print("RESULT: CHAT SFT MASK INSPECTION PASS")
print("=" * 60)
