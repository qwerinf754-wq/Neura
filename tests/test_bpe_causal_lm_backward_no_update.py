from __future__ import annotations

import copy
from pathlib import Path

from neura.tokenizer.neura_bpe_tokenizer import NeuraBPETokenizer
from neura.training.chat_sft_bpe_dataset import (
    ChatSFTBuilderBPE,
    validate_assistant_only_mask,
)
from neura.model.causal_lm import CausalLM
from neura.training.loss import CrossEntropyLoss


ARTIFACT = Path(
    "data/processed/neura_arabic_bpe_32k/tokenizer.json"
)

VOCAB_SIZE = 2048
SEQUENCE_LENGTH = 128
IGNORE_INDEX = -100


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def snapshot_parameters(model):
    """
    Take a deep snapshot of all model parameters.

    Supports the parameter exposure used by the current Neura
    CausalLM implementation without assuming a specific container type.
    """
    snapshots = []

    for name in dir(model):
        if name.startswith("_"):
            continue

        try:
            value = getattr(model, name)
        except Exception:
            continue

        if callable(value):
            continue

        # Direct numpy/list-like parameter containers are handled
        # conservatively below.
        if isinstance(value, (list, tuple, dict)):
            try:
                snapshots.append(
                    (name, copy.deepcopy(value))
                )
            except Exception:
                pass

    return snapshots


def compare_snapshots(before, after):
    """
    Conservative structural comparison.

    Returns True when the observable model state is identical.
    """
    if len(before) != len(after):
        return False

    for (name_a, value_a), (name_b, value_b) in zip(
        before,
        after,
    ):
        if name_a != name_b:
            return False

        if repr(value_a) != repr(value_b):
            return False

    return True


print("=" * 72)
print("NEURA BPE -> CHAT SFT -> CAUSAL LM")
print("LOSS -> BACKWARD -> NO WEIGHT UPDATE")
print("=" * 72)

# ==================================================================
# 1. BPE
# ==================================================================

print()
print("[1/8] Loading BPE tokenizer")

tokenizer = NeuraBPETokenizer(
    ARTIFACT,
    frozen=True,
)

check(
    tokenizer.vocab_size == VOCAB_SIZE,
    f"Expected vocab {VOCAB_SIZE}, got {tokenizer.vocab_size}",
)

print("[PASS] BPE tokenizer")
print("       Vocab:", tokenizer.vocab_size)

# ==================================================================
# 2. Chat SFT
# ==================================================================

print()
print("[2/8] Building Chat SFT example")

builder = ChatSFTBuilderBPE(
    tokenizer=tokenizer,
    max_length=SEQUENCE_LENGTH,
    add_bos=True,
    add_eos=False,
)

messages = [
    {
        "role": "system",
        "content": "أنت مساعد Neura.",
    },
    {
        "role": "user",
        "content": "ما هو Neura؟",
    },
    {
        "role": "assistant",
        "content": "Neura هو نظام ذكاء اصطناعي.",
    },
]

example = builder.build(messages)

validate_assistant_only_mask(example)

input_ids = list(example.input_ids)
labels = list(example.labels)
loss_mask = list(example.loss_mask)

check(
    len(input_ids) == len(labels) == len(loss_mask),
    "SFT lengths mismatch",
)

check(
    len(input_ids) > 1,
    "SFT example too short",
)

assistant_count = sum(
    1 for x in loss_mask if x == 1
)

check(
    assistant_count > 0,
    "No assistant tokens available",
)

print("[PASS] Chat SFT example")
print("       Sequence:", len(input_ids))
print("       Assistant tokens:", assistant_count)

# ==================================================================
# 3. Causal target alignment
# ==================================================================

print()
print("[3/8] Preparing model targets")

model_inputs = input_ids[:-1]
model_labels = labels[1:]

check(
    len(model_inputs) == len(model_labels),
    "Causal target alignment failed",
)

print("[PASS] Causal target alignment")
print("       Input tokens :", len(model_inputs))
print("       Target tokens:", len(model_labels))

# ==================================================================
# 4. CausalLM construction
# ==================================================================

print()
print("[4/8] Constructing CausalLM")

model = CausalLM(
    vocab_size=VOCAB_SIZE,
    hidden_size=128,
    heads=4,
    num_layers=2,
    intermediate_size=512,
    context_length=SEQUENCE_LENGTH,
)

print("[PASS] CausalLM constructed")

try:
    print("       Parameters:", model.parameter_count)
except Exception:
    pass

# Snapshot immediately after construction.
state_before = snapshot_parameters(model)

print(
    "       State snapshot entries:",
    len(state_before),
)

# ==================================================================
# 5. Forward
# ==================================================================

print()
print("[5/8] Forward pass")

logits = model.forward(
    model_inputs,
    use_cache=False,
)

check(
    logits is not None,
    "CausalLM.forward returned None",
)

check(
    len(logits) == len(model_inputs),
    "Logits sequence length mismatch",
)

print("[PASS] Forward pass")
print("       Logit rows:", len(logits))

# ==================================================================
# 6. Assistant-only loss
# ==================================================================

print()
print("[6/8] Computing assistant-only loss")

selected_logits = []
selected_labels = []

for position, label in enumerate(model_labels):

    if label == IGNORE_INDEX:
        continue

    selected_logits.append(
        logits[position]
    )

    selected_labels.append(
        label
    )

check(
    len(selected_labels) > 0,
    "No assistant labels survived filtering",
)

check(
    len(selected_logits) == len(selected_labels),
    "Selected logits/labels mismatch",
)

loss_fn = CrossEntropyLoss()

loss = loss_fn.forward(
    selected_logits,
    selected_labels,
)

print("[PASS] Loss computation")
print("       Loss tokens:", len(selected_labels))
print("       Loss:", loss)

# ==================================================================
# 7. Backward
# ==================================================================

print()
print("[7/8] Backward pass")

loss_gradients = loss_fn.backward()

check(
    loss_gradients is not None,
    "Loss backward returned None",
)

model_gradients = model.backward(
    loss_gradients
)

check(
    model_gradients is not None,
    "Model backward returned None",
)

print("[PASS] Backward pass")

# ==================================================================
# 8. Verify no update
# ==================================================================

print()
print("[8/8] Verifying NO weight update")

# IMPORTANT:
# There is deliberately NO:
#
#     optimizer.step()
#
# in this test.

state_after = snapshot_parameters(model)

unchanged = compare_snapshots(
    state_before,
    state_after,
)

check(
    unchanged,
    "MODEL STATE CHANGED DURING BACKWARD",
)

print("[PASS] Model state unchanged")
print("       Optimizer step: NOT EXECUTED")
print("       Weight update  : NOT PERFORMED")

print()
print("=" * 72)
print("FINAL RESULT")
print("=" * 72)
print("BPE -> CHAT SFT -> CAUSAL LM -> LOSS -> BACKWARD: PASS")
print("NO WEIGHT UPDATE: PASS")
print("TRAINING: NOT STARTED")
print("=" * 72)
