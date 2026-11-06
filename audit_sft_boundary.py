import json

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE

CKPT = "checkpoints/chat_sft_v2_600/neura_chat_sft_best.pt"
TRAIN = "data/processed/neura_chat_sft_v2/train.jsonl"

rt = NeuraSFTV2Runtime(CKPT, device="cpu")
tok = rt.tokenizer

with open(TRAIN, "r", encoding="utf-8") as f:
    row = json.loads(next(x for x in f if x.strip()))

messages = row["messages"]

builder = ChatSFTBuilderBPE(
    tokenizer=tok,
    max_length=512,
    add_bos=True,
    add_eos=True,
)

example = builder.build(messages)

ids = example.input_ids
labels = example.labels
mask = example.loss_mask

print("=" * 80)
print("MESSAGES")
print("=" * 80)

for i, msg in enumerate(messages):
    print(i, msg["role"], repr(msg["content"]))

print()
print("=" * 80)
print("TOKEN STREAM")
print("=" * 80)

for i, (token_id, label, m) in enumerate(zip(ids, labels, mask)):
    token_text = tok.decode(
        [token_id],
        skip_special_tokens=False,
    )

    if label == -100:
        label_text = "IGNORE"
    else:
        label_text = repr(
            tok.decode(
                [label],
                skip_special_tokens=False,
            )
        )

    print(
        f"{i:4d} "
        f"input_id={token_id:4d} "
        f"input={token_text!r:<12} "
        f"label={label!s:<6} "
        f"label_text={label_text:<12} "
        f"mask={m}"
    )

print()
print("=" * 80)
print("FIRST SUPERVISED POSITIONS")
print("=" * 80)

shown = 0

for i, label in enumerate(labels):
    if label != -100:
        print(
            "label_pos=",
            i,
            "prediction_pos=",
            i - 1,
            "input=",
            repr(tok.decode([ids[i]], skip_special_tokens=False)),
            "label=",
            label,
            "label_text=",
            repr(tok.decode([label], skip_special_tokens=False)),
        )
        shown += 1

        if shown >= 15:
            break

print()
print("=" * 80)
print("SPECIAL IDS")
print("=" * 80)
print("BOS:", tok.BOS_ID)
print("EOS:", tok.EOS_ID)
print("PAD:", tok.PAD_ID)
print("UNK:", tok.UNK_ID)
