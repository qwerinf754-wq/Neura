import json
import math
import torch

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE

CKPT = "checkpoints/chat_sft_v2_600/neura_chat_sft_last.pt"
TRAIN = "data/processed/neura_chat_sft_v2/train.jsonl"

rt = NeuraSFTV2Runtime(CKPT, device="cpu")
tok = rt.tokenizer
model = rt.model
model.eval()

builder = ChatSFTBuilderBPE(
    tokenizer=tok,
    max_length=512,
    add_bos=True,
    add_eos=True,
)

with open(TRAIN, "r", encoding="utf-8") as f:
    rows = [json.loads(x) for x in f if x.strip()]


# For every assistant message, collect its content token ids.
# We locate the exact assistant content span in the built sequence.
positions = {
    1: [],
    2: [],
    3: [],
    4: [],
    5: [],
    "rest": [],
}


def find_assistant_spans(ids, labels, messages):
    marker = tok.encode("<|assistant|>\n")
    spans = []

    search_from = 0

    for msg in messages:
        if msg["role"] != "assistant":
            continue

        content_ids = tok.encode(msg["content"])

        found = None

        for j in range(search_from, len(ids) - len(marker) - len(content_ids) + 1):
            if ids[j:j + len(marker)] != marker:
                continue

            k = j + len(marker)

            if ids[k:k + len(content_ids)] == content_ids:
                if k < len(labels) and labels[k] == content_ids[0]:
                    found = k
                    break

        if found is not None:
            spans.append((found, len(content_ids)))
            search_from = found + len(content_ids)

    return spans


for row in rows:
    ex = builder.build(row["messages"])

    ids = ex.input_ids
    labels = ex.labels

    x = torch.tensor([ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(x)[0]

    spans = find_assistant_spans(ids, labels, row["messages"])

    for start, length in spans:
        for offset in range(length):
            pos = start + offset

            if pos <= 0:
                continue

            target = labels[pos]

            if target == -100:
                continue

            z = logits[pos - 1]
            lp = torch.log_softmax(z, dim=-1)

            loss = -float(lp[target])

            if offset == 0:
                group = 1
            elif offset == 1:
                group = 2
            elif offset == 2:
                group = 3
            elif offset == 3:
                group = 4
            elif offset == 4:
                group = 5
            else:
                group = "rest"

            positions[group].append(loss)


print("=" * 70)
print("ASSISTANT POSITION LOSS")
print("=" * 70)
print("checkpoint:", CKPT)

for group in [1, 2, 3, 4, 5, "rest"]:
    values = positions[group]

    if not values:
        continue

    mean = sum(values) / len(values)

    print(
        f"position {group}: "
        f"count={len(values):4d} "
        f"loss={mean:8.4f} "
        f"ppl={math.exp(mean):10.3f}"
    )
