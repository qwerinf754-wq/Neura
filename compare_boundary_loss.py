import json
import math
import torch

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE

TRAIN = "data/processed/neura_chat_sft_v2/train.jsonl"

CHECKPOINTS = [
    ("496", "checkpoints/chat_sft_v2_repaired/neura_chat_sft_best.pt"),
    ("569", "checkpoints/chat_sft_v2_600/neura_chat_sft_best.pt"),
    ("600", "checkpoints/chat_sft_v2_600/neura_chat_sft_last.pt"),
]

with open(TRAIN, "r", encoding="utf-8") as f:
    rows = [json.loads(x) for x in f if x.strip()]


def measure(ckpt):
    rt = NeuraSFTV2Runtime(ckpt, device="cpu")
    tok = rt.tokenizer
    model = rt.model
    model.eval()

    builder = ChatSFTBuilderBPE(
        tokenizer=tok,
        max_length=512,
        add_bos=True,
        add_eos=True,
    )

    first_losses = []
    eos_losses = []

    first_ranks = []
    eos_ranks = []

    for row in rows:
        ex = builder.build(row["messages"])

        ids = ex.input_ids
        labels = ex.labels

        first = next(
            (j for j, x in enumerate(labels) if x != -100),
            None,
        )

        if first is None or first == 0:
            continue

        x = torch.tensor([ids], dtype=torch.long)

        with torch.no_grad():
            logits = model(x)[0]

        # First assistant target.
        target = labels[first]
        z = logits[first - 1]

        lp = torch.log_softmax(z, dim=-1)
        loss = -float(lp[target])

        rank = int((z > z[target]).sum().item()) + 1

        first_losses.append(loss)
        first_ranks.append(rank)

        # EOS.
        eos_positions = [
            j for j, x in enumerate(labels)
            if x == tok.EOS_ID
        ]

        if eos_positions:
            pos = eos_positions[-1]

            if pos > 0:
                z = logits[pos - 1]
                lp = torch.log_softmax(z, dim=-1)

                loss = -float(lp[tok.EOS_ID])
                rank = int(
                    (z > z[tok.EOS_ID]).sum().item()
                ) + 1

                eos_losses.append(loss)
                eos_ranks.append(rank)

    return {
        "first_loss": sum(first_losses) / len(first_losses),
        "eos_loss": sum(eos_losses) / len(eos_losses),
        "first_ppl": math.exp(
            sum(first_losses) / len(first_losses)
        ),
        "eos_ppl": math.exp(
            sum(eos_losses) / len(eos_losses)
        ),
        "first_top1": sum(x == 1 for x in first_ranks),
        "eos_top1": sum(x == 1 for x in eos_ranks),
        "first_mean_rank": sum(first_ranks) / len(first_ranks),
        "eos_mean_rank": sum(eos_ranks) / len(eos_ranks),
    }


print("=" * 80)
print("FIRST-TOKEN vs EOS ACROSS CHECKPOINTS")
print("=" * 80)

for name, ckpt in CHECKPOINTS:
    r = measure(ckpt)

    print()
    print("CHECKPOINT:", name)
    print("first loss :", round(r["first_loss"], 4))
    print("EOS loss   :", round(r["eos_loss"], 4))
    print("first PPL  :", round(r["first_ppl"], 3))
    print("EOS PPL    :", round(r["eos_ppl"], 3))
    print("first top1 :", f'{r["first_top1"]}/40')
    print("EOS top1   :", f'{r["eos_top1"]}/40')
    print("first rank :", round(r["first_mean_rank"], 2))
    print("EOS rank   :", round(r["eos_mean_rank"], 2))
