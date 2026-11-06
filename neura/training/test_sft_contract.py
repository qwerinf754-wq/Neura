from neura.chat import ChatTemplate
from neura.training.chat_dataset import (
    IGNORE_INDEX,
    ChatSFTDataset,
)
from neura.training.sft import SFTTrainer


print("=" * 60)
print("NEURA SFT CONTRACT")
print("=" * 60)


# ---------------------------------------------------------
# CHAT TEMPLATE
# ---------------------------------------------------------

messages = [
    {
        "role": "system",
        "content": "You are Neura.",
    },
    {
        "role": "user",
        "content": "Hello!",
    },
    {
        "role": "assistant",
        "content": "Hello! How can I help you?",
    },
]

template = ChatTemplate()

rendered = template.render(
    messages
)

assert "<|system|>" in rendered
assert "<|user|>" in rendered
assert "<|assistant|>" in rendered

print("Chat template rendering: PASS")


rendered, spans = template.assistant_spans(
    messages
)

assert len(spans) == 1
assert spans[0][0] < spans[0][1]

print("Assistant span detection: PASS")


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

dataset = ChatSFTDataset(
    [
        messages,
        [
            {
                "role": "user",
                "content": "What is Neura?",
            },
            {
                "role": "assistant",
                "content": "Neura is an AI model.",
            },
        ],
    ]
)

assert len(dataset) == 2

print("Dataset creation: PASS")


example = dataset[0]

assert len(example.input_ids) == len(
    example.labels
)

print("Input/label alignment: PASS")


assert any(
    label != IGNORE_INDEX
    for label in example.labels
)

print("Assistant labels present: PASS")


assert any(
    label == IGNORE_INDEX
    for label in example.labels
)

print("Non-assistant masking: PASS")


# ---------------------------------------------------------
# SECOND EXAMPLE ISOLATION
# ---------------------------------------------------------

example_b = dataset[1]

assert example.input_ids != example_b.input_ids

print("Example isolation: PASS")


# ---------------------------------------------------------
# SFT TRAINER
# ---------------------------------------------------------

trainer = SFTTrainer(
    model=None
)

result = trainer.train_step(
    [example]
)

assert result["status"] == "prepared"
assert result["assistant_tokens"] > 0
assert result["step"] == 1

print("SFT batch preparation: PASS")
print("Assistant token counting: PASS")
print("Training step contract: PASS")


print("=" * 60)
print("RESULT: SFT CONTRACT PASSED")
print("=" * 60)
