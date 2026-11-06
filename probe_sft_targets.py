import json
import math
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

first_losses = []
eos_losses = []

print("checkpoint:", CKPT)
print("examples:", len(rows))

for i, row in enumerate(rows):
    ex = builder.build(row["messages"])

    ids = ex.input_ids
    labels = ex.labels

    first = None
    for j, label in enumerate(labels):
        if label != -100:
            first = j
            break

    if first is None or first == 0:
        continue

    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)[0]

    # First assistant target.
    first_target = labels[first]
    first_logits = logits[first - 1]

    first_log_probs = torch.log_softmax(first_logits, dim=-1)
    first_loss = -float(first_log_probs[first_target])

    first_rank = (
        int((first_logits > first_logits[first_target]).sum().item())
        + 1
    )

    # Find EOS target position.
    eos_positions = [
        j for j, label in enumerate(labels)
        if label == tok.EOS_ID
    ]

    eos_loss = None
    eos_rank = None

    if eos_positions:
        eos_label_pos = eos_positions[-1]

        if eos_label_pos > 0:
            eos_logits = logits[eos_label_pos - 1]
            eos_log_probs = torch.log_softmax(eos_logits, dim=-1)
            eos_loss = -float(eos_log_probs[tok.EOS_ID])
            eos_rank = (
                int(
                    (
                        eos_logits > eos_logits[tok.EOS_ID]
                    ).sum().item()
                )
                + 1
            )

            eos_losses.append(eos_loss)

    first_losses.append(first_loss)

    print(
        f"{i:2d} "
        f"first_loss={first_loss:7.3f} "
        f"first_rank={first_rank:4d} "
        f"eos_loss={eos_loss if eos_loss is not None else 'NA':>7} "
        f"eos_rank={eos_rank if eos_rank is not None else 'NA':>4}"
    )

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    "mean first-token loss:",
    round(sum(first_losses) / len(first_losses), 4),
)

print(
    "mean EOS loss:",
    round(sum(eos_losses) / len(eos_losses), 4),
)

print(
    "first-token perplexity:",
    round(math.exp(sum(first_losses) / len(first_losses)), 3),
)

print(
    "EOS perplexity:",
    round(math.exp(sum(eos_losses) / len(eos_losses)), 3),
)
