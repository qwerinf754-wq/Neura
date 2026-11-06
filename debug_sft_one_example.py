import json
from pathlib import Path

import torch
import torch.nn as nn

from neura.brain.sft_v2_inference import NeuraSFTV2Runtime
from neura.training.chat_sft_bpe_dataset import ChatSFTBuilderBPE


BASE_CKPT = "checkpoints/chat_sft_v2_repaired/neura_chat_sft_best.pt"
OUT = Path("checkpoints/debug_sft_overfit")
OUT.mkdir(parents=True, exist_ok=True)

TRAIN = "data/processed/neura_chat_sft_v2/train.jsonl"

device = torch.device("cpu")

# ------------------------------------------------------------
# Load exactly the first training example.
# ------------------------------------------------------------

with open(TRAIN, "r", encoding="utf-8") as f:
    row = json.loads(next(f))

messages = row["messages"]

# ------------------------------------------------------------
# Load model/tokenizer from an existing checkpoint.
# We do NOT modify the checkpoint on disk.
# ------------------------------------------------------------

rt = NeuraSFTV2Runtime(BASE_CKPT, device="cpu")

model = rt.model.to(device)
tok = rt.tokenizer

model.train()

builder = ChatSFTBuilderBPE(
    tokenizer=tok,
    max_length=512,
    add_bos=True,
    add_eos=True,
)

ex = builder.build(messages)

input_ids = torch.tensor(
    [ex.input_ids],
    dtype=torch.long,
    device=device,
)

labels = torch.tensor(
    [ex.labels],
    dtype=torch.long,
    device=device,
)

IGNORE_INDEX = -100

criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.0,
)

print("=" * 80)
print("ONE-EXAMPLE SFT OVERFIT")
print("=" * 80)

print("base checkpoint:", BASE_CKPT)
print("output dir:", OUT)
print("sequence length:", len(ex.input_ids))
print(
    "supervised tokens:",
    int((labels != IGNORE_INDEX).sum().item()),
)

print()
print("MESSAGES:")
print(json.dumps(messages, ensure_ascii=False, indent=2))

# ------------------------------------------------------------
# Train only on this ONE example.
# ------------------------------------------------------------

STEPS = 100

for step in range(1, STEPS + 1):
    optimizer.zero_grad(set_to_none=True)

    logits = model(input_ids)

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    loss = criterion(
        shifted_logits.reshape(-1, tok.vocab_size),
        shifted_labels.reshape(-1),
    )

    loss.backward()

    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        1.0,
    )

    optimizer.step()

    if step == 1 or step % 100 == 0:
        with torch.no_grad():
            pred = shifted_logits.argmax(dim=-1)
            active = shifted_labels != IGNORE_INDEX

            correct = (
                (pred == shifted_labels) & active
            ).sum().item()

            total = active.sum().item()

            acc = correct / total if total else 0.0

        print(
            f"step={step:4d} "
            f"loss={loss.item():.6f} "
            f"token_acc={acc:.4f}"
        )

# ------------------------------------------------------------
# Save debug checkpoint.
# ------------------------------------------------------------

debug_ckpt = OUT / "one_example_overfit.pt"

torch.save(
    {
        "model": model.state_dict(),
        "base_checkpoint": BASE_CKPT,
        "step": STEPS,
    },
    debug_ckpt,
)

print()
print("saved:", debug_ckpt)

# ------------------------------------------------------------
# Evaluate teacher-forced predictions.
# ------------------------------------------------------------

model.eval()

with torch.no_grad():
    logits = model(input_ids)

    shifted_logits = logits[:, :-1, :]
    shifted_labels = labels[:, 1:]

    pred = shifted_logits.argmax(dim=-1)

active = shifted_labels != IGNORE_INDEX

correct = (
    (pred == shifted_labels) & active
).sum().item()

total = active.sum().item()

print()
print("=" * 80)
print("FINAL TEACHER-FORCED RESULT")
print("=" * 80)

print("correct:", correct)
print("total:", total)
print("accuracy:", correct / total if total else 0.0)

# ------------------------------------------------------------
# Show the first assistant target.
# ------------------------------------------------------------

first = None

for i, x in enumerate(ex.labels):
    if x != IGNORE_INDEX:
        first = i
        break

if first is not None and first > 0:
    target = ex.labels[first]

    z = logits[0, first - 1]

    rank = (
        int((z > z[target]).sum().item())
        + 1
    )

    try:
        target_text = tok.decode([target])
    except Exception:
        target_text = "<partial-utf8>"

    print()
    print("FIRST ASSISTANT TARGET")
    print("position:", first)
    print("target id:", target)
    print("target text:", repr(target_text))
    print("rank:", rank)

# ------------------------------------------------------------
# Generation using the actual prompt.
# ------------------------------------------------------------

def build_prompt(messages):
    parts = []

    for msg in messages:
        parts.append(
            f"<|{msg['role']}|>\n"
            f"{msg['content']}\n"
            f"<|end|>\n"
        )

    parts.append("<|assistant|>\n")

    return "".join(parts)


prompt = build_prompt(messages[:2])

prompt_ids = [tok.BOS_ID] + tok.encode(prompt)

generated = []

for _ in range(80):
    ids = prompt_ids + generated

    x = torch.tensor(
        [ids],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        z = model(x)[0, -1]

    next_id = int(torch.argmax(z).item())

    generated.append(next_id)

    if next_id == tok.EOS_ID:
        break

    try:
        text = tok.decode(generated)
    except Exception:
        text = ""

    if "\n<|end|>\n" in text:
        break

try:
    generated_text = tok.decode(generated)
except Exception:
    generated_text = "<decode-error>"

print()
print("=" * 80)
print("GREEDY GENERATION")
print("=" * 80)

print("prompt:")
print(prompt)

print()
print("generated:")
print(repr(generated_text))
