import json
import torch

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE

CKPT = "checkpoints/chat_sft_v2_600/neura_chat_sft_best.pt"
TRAIN = "data/processed/neura_chat_sft_v2/train.jsonl"

rt = NeuraSFTV2Runtime(CKPT, device="cpu")
model = rt.model
tok = rt.tokenizer
model.eval()

with open(TRAIN, "r", encoding="utf-8") as f:
    rows = [json.loads(x) for x in f if x.strip()]

builder = ChatSFTBuilderBPE(
    tokenizer=tok,
    max_length=512,
    add_bos=True,
    add_eos=True,
)

print("checkpoint:", CKPT)
print("train rows:", len(rows))

for i, row in enumerate(rows[:5]):
    messages = row['messages']
    example = builder.build(messages)

    ids = example.input_ids
    labels = example.labels

    # Find the first supervised label.
    first = None
    for j, label in enumerate(labels):
        if label != -100:
            first = j
            break

    if first is None:
        print(f"[{i}] NO ASSISTANT LABEL")
        continue

    # Because the loss is shifted:
    # logits[first-1] predicts labels[first].
    pred_pos = first - 1

    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)[0, pred_pos]

    target_id = labels[first]

    rank = int((logits > logits[target_id]).sum().item()) + 1
    top_values, top_ids = torch.topk(logits, k=10)

    print()
    print("=" * 70)
    print("ROW:", i)
    print("FIRST LABEL POSITION:", first)
    print("PREDICTION POSITION:", pred_pos)

    print(
        "EXPECTED FIRST TOKEN:",
        repr(tok.decode([target_id], skip_special_tokens=False))
    )
    print("EXPECTED ID:", target_id)
    print("EXPECTED LOGIT:", round(float(logits[target_id]), 5))
    print("EXPECTED RANK:", rank)

    print()
    print("TOP 10:")

    for r, (value, token_id) in enumerate(
        zip(top_values.tolist(), top_ids.tolist()), 1
    ):
        text = tok.decode(
            [token_id],
            skip_special_tokens=False,
        )

        print(
            f"{r:2d}. "
            f"id={token_id:4d} "
            f"logit={value:9.4f} "
            f"text={text!r}"
        )
