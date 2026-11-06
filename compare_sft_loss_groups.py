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

IGNORE_INDEX = -100


with open(TRAIN, "r", encoding="utf-8") as f:
    rows = [json.loads(x) for x in f if x.strip()]


def is_end_marker_position(ids, pos, end_ids):
    if pos + 1 > len(ids):
        return False

    # labels[pos] corresponds to input_ids[pos].
    start = pos - len(end_ids) + 1

    if start < 0:
        return False

    return ids[start:pos + 1] == end_ids


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

    end_ids = tok.encode("\n<|end|>\n")

    groups = {
        "first_content": [],
        "other_content": [],
        "end_marker": [],
        "eos": [],
    }

    counts = {k: 0 for k in groups}

    for row in rows:
        ex = builder.build(row["messages"])

        ids = ex.input_ids
        labels = ex.labels

        x = torch.tensor([ids], dtype=torch.long)

        with torch.no_grad():
            logits = model(x)[0]

        # Find the beginning of each assistant content span.
        assistant_first_positions = []

        pos = 0
        while pos < len(row["messages"]):
            msg = row["messages"][pos]

            if msg["role"] == "assistant":
                # Find the first supervised label belonging to this
                # assistant message by matching its encoded content.
                content_ids = tok.encode(msg["content"])

                # Search labels/input sequence for this content after
                # the assistant role marker.
                marker = tok.encode("<|assistant|>\n")

                found = None

                for j in range(len(ids) - len(marker) - len(content_ids) + 1):
                    if ids[j:j + len(marker)] != marker:
                        continue

                    k = j + len(marker)

                    if ids[k:k + len(content_ids)] == content_ids:
                        if labels[k] == content_ids[0]:
                            found = k
                            break

                if found is not None:
                    assistant_first_positions.append(found)

            pos += 1

        first_set = set(assistant_first_positions)

        for j, target in enumerate(labels):
            if target == IGNORE_INDEX:
                continue

            if j == 0:
                continue

            z = logits[j - 1]
            lp = torch.log_softmax(z, dim=-1)

            loss = -float(lp[target])

            if target == tok.EOS_ID:
                group = "eos"
            elif is_end_marker_position(ids, j, end_ids):
                group = "end_marker"
            elif j in first_set:
                group = "first_content"
            else:
                group = "other_content"

            groups[group].append(loss)
            counts[group] += 1

    result = {}

    for name, values in groups.items():
        if values:
            mean = sum(values) / len(values)
            result[name] = {
                "count": len(values),
                "loss": mean,
                "ppl": math.exp(mean),
            }
        else:
            result[name] = {
                "count": 0,
                "loss": float("nan"),
                "ppl": float("nan"),
            }

    return result


print("=" * 80)
print("SFT LOSS BY TARGET GROUP")
print("=" * 80)

for name, ckpt in CHECKPOINTS:
    r = measure(ckpt)

    print()
    print("CHECKPOINT:", name)

    for group in [
        "first_content",
        "other_content",
        "end_marker",
        "eos",
    ]:
        x = r[group]

        print(
            f"{group:15s} "
            f"count={x['count']:5d} "
            f"loss={x['loss']:8.4f} "
            f"ppl={x['ppl']:10.3f}"
        )
