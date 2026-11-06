import json
import torch

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime
from neura.training.chat_sft_bpe_dataset import (
    ChatSFTBuilderBPE,
    IGNORE_INDEX,
)


CKPT = "checkpoints/chat_sft_v2_repaired/neura_chat_sft_best.pt"
DATA = "data/processed/neura_chat_sft_v2/train.jsonl"


def decode_token(tokenizer, token_id):
    try:
        return tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
        )
    except Exception as exc:
        return f"<decode-error:{exc}>"


def get_field(obj, name):
    """
    استخراج حقل من ChatSFTExample سواء كان:
    - dataclass
    - object عادي
    - dict
    """
    if isinstance(obj, dict):
        return obj[name]

    if hasattr(obj, name):
        return getattr(obj, name)

    raise AttributeError(
        f"ChatSFTExample has no field '{name}'. "
        f"Available attributes: {dir(obj)}"
    )


print("=== LOAD RUNTIME ===")

rt = NeuraSFTV2Runtime(
    CKPT,
    device="cpu",
)

print("checkpoint:", CKPT)
print("step:", rt.step)


print()
print("=== LOAD FIRST TRAIN EXAMPLE ===")

with open(DATA, "r", encoding="utf-8") as f:
    example = json.loads(f.readline())

print(
    json.dumps(
        example,
        ensure_ascii=False,
        indent=2,
    )
)


print()
print("=== BUILD DATASET ITEM ===")

builder = ChatSFTBuilderBPE(
    tokenizer=rt.tokenizer,
    max_length=512,
    add_bos=True,
    add_eos=True,
)

# build() expects the messages list.
item = builder.build(example["messages"])

print("item type:", type(item))
print("item class:", type(item).__name__)


print()
print("=== ITEM ATTRIBUTES ===")

for name in dir(item):
    if name.startswith("_"):
        continue

    try:
        value = getattr(item, name)
    except Exception:
        continue

    if callable(value):
        continue

    print(
        f"{name}: type={type(value).__name__}"
    )


print()
print("=== EXTRACT INPUT IDS / LABELS ===")

input_ids = get_field(
    item,
    "input_ids",
)

labels = get_field(
    item,
    "labels",
)

if isinstance(input_ids, torch.Tensor):
    input_ids = input_ids.tolist()

if isinstance(labels, torch.Tensor):
    labels = labels.tolist()

print("input_ids type:", type(input_ids).__name__)
print("labels type:", type(labels).__name__)

print("input length:", len(input_ids))
print("label length:", len(labels))

if len(input_ids) != len(labels):
    raise RuntimeError(
        "INPUT/LABEL LENGTH MISMATCH: "
        f"{len(input_ids)} != {len(labels)}"
    )


print()
print("=== ALL TOKENS ===")

for i, (input_id, label_id) in enumerate(
    zip(input_ids, labels)
):
    input_text = decode_token(
        rt.tokenizer,
        input_id,
    )

    if label_id == IGNORE_INDEX:
        label_text = "<IGNORE>"
        marker = " "
    else:
        label_text = decode_token(
            rt.tokenizer,
            label_id,
        )
        marker = "*"

    print(
        f"{marker} "
        f"pos={i:3d} "
        f"input_id={input_id:4d} "
        f"input={input_text!r:20s} "
        f"label_id={label_id:4d} "
        f"label={label_text!r}"
    )


print()
print("=== SUPERVISED LABELS ONLY ===")

supervised = []

for i, label_id in enumerate(labels):
    if label_id != IGNORE_INDEX:
        supervised.append(
            (
                i,
                input_ids[i],
                label_id,
            )
        )

print(
    "supervised token count:",
    len(supervised),
)

for pos, input_id, label_id in supervised:
    input_text = decode_token(
        rt.tokenizer,
        input_id,
    )

    label_text = decode_token(
        rt.tokenizer,
        label_id,
    )

    print(
        f"pos={pos:3d} "
        f"input_id={input_id:4d} "
        f"input={input_text!r:20s} "
        f"label_id={label_id:4d} "
        f"label={label_text!r}"
    )


print()
print("=== FIRST SUPERVISED LABEL ===")

if not supervised:
    raise RuntimeError(
        "NO SUPERVISED LABELS FOUND"
    )

first_pos, first_input_id, first_label_id = supervised[0]

first_input_text = decode_token(
    rt.tokenizer,
    first_input_id,
)

first_label_text = decode_token(
    rt.tokenizer,
    first_label_id,
)

print("position:", first_pos)
print("input id:", first_input_id)
print(
    "input text:",
    repr(first_input_text),
)
print("label id:", first_label_id)
print(
    "label text:",
    repr(first_label_text),
)


print()
print("=== CONTEXT AROUND FIRST SUPERVISED LABEL ===")

start = max(
    0,
    first_pos - 25,
)

end = min(
    len(input_ids),
    first_pos + 40,
)

for i in range(start, end):
    input_id = input_ids[i]
    label_id = labels[i]

    input_text = decode_token(
        rt.tokenizer,
        input_id,
    )

    if label_id == IGNORE_INDEX:
        label_text = "<IGNORE>"
        marker = ""
    else:
        label_text = decode_token(
            rt.tokenizer,
            label_id,
        )
        marker = "<<< SUPERVISED"

    print(
        f"{i:3d}: "
        f"input={input_id:4d} "
        f"{input_text!r:20s} "
        f"label={label_id:4d} "
        f"{label_text!r:20s} "
        f"{marker}"
    )


print()
print("=== FULL INPUT DECODE ===")

try:
    full_input = rt.tokenizer.decode(
        input_ids,
        skip_special_tokens=False,
    )
except Exception as exc:
    full_input = f"<decode-error:{exc}>"

print(repr(full_input))


print()
print("=== SUPERVISED TARGET DECODE ===")

target_ids = [
    label_id
    for label_id in labels
    if label_id != IGNORE_INDEX
]

try:
    target_text = rt.tokenizer.decode(
        target_ids,
        skip_special_tokens=False,
    )
except Exception as exc:
    target_text = f"<decode-error:{exc}>"

print(repr(target_text))


print()
print("=== EXPECTED ASSISTANT TEXT ===")

expected = "مرحباً! أنا Neura. كيف يمكنني مساعدتك؟"

print(
    "expected:",
    repr(expected),
)


print()
print("=== RESULT ===")

if first_label_text == "مر":
    print("FIRST LABEL CHECK: PASS")
    print(
        "The first supervised label is "
        "the expected assistant token."
    )
else:
    print("FIRST LABEL CHECK: FAIL")
    print(
        "Expected first supervised label:",
        repr("مر"),
    )
    print(
        "Actual first supervised label:",
        repr(first_label_text),
    )

print()
print("Done.")
