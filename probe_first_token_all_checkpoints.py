import json
import torch

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE

CHECKPOINTS = [
    "checkpoints/chat_sft_v2_repaired/neura_chat_sft_best.pt",
    "checkpoints/chat_sft_v2_600/neura_chat_sft_best.pt",
    "checkpoints/chat_sft_v2_600/neura_chat_sft_last.pt",
]

TRAIN = "data/processed/neura_chat_sft_v2/train.jsonl"

with open(TRAIN, "r", encoding="utf-8") as f:
    rows = [json.loads(x) for x in f if x.strip()]


for ckpt in CHECKPOINTS:
    print()
    print("=" * 80)
    print("CHECKPOINT:", ckpt)

    rt = NeuraSFTV2Runtime(ckpt, device="cpu")
    model = rt.model
    tok = rt.tokenizer
    model.eval()

    builder = ChatSFTBuilderBPE(
        tokenizer=tok,
        max_length=512,
        add_bos=True,
        add_eos=True,
    )

    ranks = []
    top1 = 0
    eos_top1 = 0

    for i, row in enumerate(rows):
        example = builder.build(row["messages"])

        ids = example.input_ids
        labels = example.labels

        first = None
        for j, label in enumerate(labels):
            if label != -100:
                first = j
                break

        if first is None or first == 0:
            continue

        pred_pos = first - 1

        x = torch.tensor([ids], dtype=torch.long)

        with torch.no_grad():
            logits = model(x)[0, pred_pos]

        target_id = labels[first]

        rank = int(
            (logits > logits[target_id]).sum().item()
        ) + 1

        ranks.append(rank)

        top_id = int(torch.argmax(logits).item())

        if top_id == target_id:
            top1 += 1

        if top_id == tok.EOS_ID:
            eos_top1 += 1

    ranks_sorted = sorted(ranks)

    mean_rank = sum(ranks) / len(ranks)

    median_rank = ranks_sorted[len(ranks_sorted) // 2]

    print("examples:", len(ranks))
    print("top1:", f"{top1}/{len(ranks)}")
    print("top1_rate:", f"{top1 / len(ranks) * 100:.2f}%")
    print("EOS top1:", f"{eos_top1}/{len(ranks)}")
    print("EOS top1_rate:", f"{eos_top1 / len(ranks) * 100:.2f}%")
    print("mean rank:", round(mean_rank, 2))
    print("median rank:", median_rank)
    print("best rank:", min(ranks))
    print("worst rank:", max(ranks))
