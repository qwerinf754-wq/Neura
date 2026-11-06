import json
import torch

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime
from neura.training.chat_sft_bpe_dataset import (
    ChatSFTBuilderBPE,
    IGNORE_INDEX,
)

CKPT = "checkpoints/chat_sft_v2_repaired/neura_chat_sft_best.pt"
DATA = "data/processed/neura_chat_sft_v2/train.jsonl"


def decode(tokenizer, token_id):
    try:
        return tokenizer.decode(
            [int(token_id)],
            skip_special_tokens=False,
        )
    except Exception:
        return f"<decode-error:{token_id}>"


rt = NeuraSFTV2Runtime(
    CKPT,
    device="cpu",
)

with open(DATA, "r", encoding="utf-8") as f:
    example = json.loads(f.readline())

builder = ChatSFTBuilderBPE(
    tokenizer=rt.tokenizer,
    max_length=512,
    add_bos=True,
    add_eos=True,
)

item = builder.build(example["messages"])

input_ids = item.input_ids
labels = item.labels

x = torch.tensor(
    [input_ids],
    dtype=torch.long,
)

with torch.no_grad():
    logits = rt.model(x)

print("=== SHAPES ===")
print("input:", tuple(x.shape))
print("logits:", tuple(logits.shape))

# Correct causal-LM alignment:
# logits[t-1] predicts label[t]
first_target = next(
    i
    for i, label in enumerate(labels)
    if label != IGNORE_INDEX
)

print()
print("=== FIRST TARGET ===")
print("target position:", first_target)
print("target id:", labels[first_target])
print(
    "target text:",
    repr(
        decode(
            rt.tokenizer,
            labels[first_target],
        )
    ),
)

prediction_position = first_target - 1

print()
print("=== PREDICTION POSITION ===")
print("logits position:", prediction_position)
print(
    "input token there:",
    input_ids[prediction_position],
)
print(
    "input text there:",
    repr(
        decode(
            rt.tokenizer,
            input_ids[prediction_position],
        )
    ),
)

row = logits[0, prediction_position]

top_values, top_ids = torch.topk(
    row,
    k=20,
)

print()
print("=== TOP 20 AT CORRECT CAUSAL POSITION ===")

for rank, (token_id, value) in enumerate(
    zip(
        top_ids.tolist(),
        top_values.tolist(),
    ),
    start=1,
):
    print(
        f"{rank:2d}. "
        f"id={token_id:4d} "
        f"logit={value: .6f} "
        f"text={decode(rt.tokenizer, token_id)!r}"
    )

target_id = labels[first_target]

target_logit = row[target_id].item()

rank = (
    int(
        (row > row[target_id])
        .sum()
        .item()
    )
    + 1
)

print()
print("=== TARGET SCORE ===")
print("target id:", target_id)
print("target text:", repr(decode(rt.tokenizer, target_id)))
print("target logit:", target_logit)
print("target rank:", rank)

print()
print("=== DIRECT INFERENCE POSITION ===")

# This is what the current generation code effectively uses:
generation_row = logits[0, -1]

gen_values, gen_ids = torch.topk(
    generation_row,
    k=10,
)

for rank, (token_id, value) in enumerate(
    zip(
        gen_ids.tolist(),
        gen_values.tolist(),
    ),
    start=1,
):
    print(
        f"{rank:2d}. "
        f"id={token_id:4d} "
        f"logit={value: .6f} "
        f"text={decode(rt.tokenizer, token_id)!r}"
    )

print()
print("Done.")
