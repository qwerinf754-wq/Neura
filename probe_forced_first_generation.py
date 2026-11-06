import json
import torch

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE

CKPT = "checkpoints/chat_sft_v2_600/neura_chat_sft_last.pt"
VAL = "data/processed/neura_chat_sft_v2/validation.jsonl"

rt = NeuraSFTV2Runtime(CKPT, device="cpu")
model = rt.model
tok = rt.tokenizer
model.eval()

builder = ChatSFTBuilderBPE(
    tokenizer=tok,
    max_length=512,
    add_bos=True,
    add_eos=True,
)

with open(VAL, "r", encoding="utf-8") as f:
    rows = [json.loads(x) for x in f if x.strip()]


def build_prompt(messages):
    parts = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts.append(
            f"<|{role}|>\n{content}\n<|end|>\n"
        )

    parts.append("<|assistant|>\n")

    return "".join(parts)


def find_first_assistant_target(messages):
    ex = builder.build(messages)

    for i, label in enumerate(ex.labels):
        if label != -100:
            return label

    return None


def greedy_from_ids(ids, max_new_tokens=30):
    generated = list(ids)

    for _ in range(max_new_tokens):
        x = torch.tensor(
            [generated],
            dtype=torch.long,
        )

        with torch.no_grad():
            logits = model(x)[0, -1]

        next_id = int(torch.argmax(logits).item())
        generated.append(next_id)

        if next_id == tok.EOS_ID:
            break

        # Decode generated suffix and stop at END marker.
        try:
            text = tok.decode(generated[len(ids):])
        except Exception:
            text = ""

        if "\n<|end|>\n" in text:
            break

    try:
        return tok.decode(generated[len(ids):])
    except Exception:
        return "<decode-error>"


print("=" * 80)
print("FORCED FIRST-TOKEN GENERATION")
print("=" * 80)
print("checkpoint:", CKPT)
print()

for i, row in enumerate(rows):
    messages = row["messages"]

    prompt = build_prompt(messages)

    prompt_ids = [tok.BOS_ID] + tok.encode(prompt)

    gold_first = find_first_assistant_target(messages)

    if gold_first is None:
        print(i, "NO TARGET")
        continue

    # Normal greedy generation.
    normal = greedy_from_ids(
        prompt_ids,
        max_new_tokens=30,
    )

    # Force the actual training first token.
    forced_ids = prompt_ids + [gold_first]

    forced = greedy_from_ids(
        forced_ids,
        max_new_tokens=30,
    )

    try:
        gold_first_text = tok.decode([gold_first])
    except Exception:
        gold_first_text = "<partial-utf8>"

    print("-" * 80)
    print("ROW:", i)
    print("USER:", messages[-1]["content"])
    print("GOLD FIRST:", repr(gold_first_text))
    print("NORMAL:", repr(normal))
    print("FORCED :", repr(forced))
