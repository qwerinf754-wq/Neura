from neura.training.chat_dataset import (
    IGNORE_INDEX,
    ChatSFTDatasetBuilder,
)


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

example = builder.build(
    messages
)

print("=" * 60)
print("NEURA CHAT SFT DATASET TEST")
print("=" * 60)

print("input length:", len(example.input_ids))
print("label length:", len(example.labels))

assert len(example.input_ids) == len(
    example.labels
)

train_positions = [
    index
    for index, label in enumerate(
        example.labels
    )
    if label != IGNORE_INDEX
]

ignored_positions = [
    index
    for index, label in enumerate(
        example.labels
    )
    if label == IGNORE_INDEX
]

assert train_positions
assert ignored_positions

print(
    "train positions:",
    train_positions,
)

print(
    "ignored positions:",
    ignored_positions,
)

print()
print("=" * 60)
print("RESULT: CHAT SFT DATASET PASS")
print("=" * 60)
