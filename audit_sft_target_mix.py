import json
from collections import Counter

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE

CKPT = "checkpoints/chat_sft_v2_600/neura_chat_sft_best.pt"
TRAIN = "data/processed/neura_chat_sft_v2/train.jsonl"

rt = NeuraSFTV2Runtime(CKPT, device="cpu")
tok = rt.tokenizer

builder = ChatSFTBuilderBPE(
    tokenizer=tok,
    max_length=512,
    add_bos=True,
    add_eos=True,
)

with open(TRAIN, "r", encoding="utf-8") as f:
    rows = [json.loads(x) for x in f if x.strip()]

total_supervised = 0
total_content = 0
total_end = 0
total_eos = 0

target_counts = Counter()

for i, row in enumerate(rows):
    ex = builder.build(row["messages"])

    # Build the exact END marker ids.
    end_ids = tok.encode("\n<|end|>\n")

    # Locate assistant messages and count their content tokens.
    content_count = 0
    end_count = 0

    for msg in row["messages"]:
        if msg["role"] != "assistant":
            continue

        content_count += len(tok.encode(msg["content"]))
        end_count += len(end_ids)

    eos_count = sum(1 for x in ex.labels if x == tok.EOS_ID)

    supervised = sum(1 for x in ex.labels if x != -100)

    total_content += content_count
    total_end += end_count
    total_eos += eos_count
    total_supervised += supervised

    for x in ex.labels:
        if x != -100:
            target_counts[x] += 1

    print(
        f"{i:2d}: "
        f"content={content_count:3d} "
        f"end={end_count:2d} "
        f"eos={eos_count:1d} "
        f"supervised={supervised:3d}"
    )

print()
print("=" * 70)
print("TARGET MIX")
print("=" * 70)

print("examples:", len(rows))
print("content tokens:", total_content)
print("END marker tokens:", total_end)
print("EOS tokens:", total_eos)
print("total supervised:", total_supervised)

print()

for name, n in [
    ("content", total_content),
    ("END marker", total_end),
    ("EOS", total_eos),
]:
    print(
        f"{name:15s}: "
        f"{n:5d} "
        f"{100*n/total_supervised:6.2f}%"
    )

print()
print("Most frequent supervised target IDs:")

for token_id, count in target_counts.most_common(20):
    try:
        text = tok.decode([token_id])
    except Exception:
        text = "<partial-utf8>"

    print(
        f"id={token_id:4d} "
        f"count={count:4d} "
        f"text={text!r}"
    )
