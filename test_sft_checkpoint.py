from pathlib import Path

import torch
from torch import nn

from neura.model.torch_model import NeuraTorchModel
from neura.training.sft_checkpoint import (
    SFTCheckpoint,
)


print("=" * 60)
print("NEURA SFT CHECKPOINT TEST")
print("=" * 60)


checkpoint_path = Path(
    "checkpoints/test_chat_sft.pt"
)

if checkpoint_path.exists():
    checkpoint_path.unlink()


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = NeuraTorchModel(
    vocab_size=24,
    hidden_size=64,
    layers=2,
    heads=4,
    context_length=64,
    dropout=0.0,
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
)


# ---------------------------------------------------------
# ONE REAL TRAINING STEP
# ---------------------------------------------------------

input_ids = torch.randint(
    0,
    24,
    (2, 12),
)

targets = torch.randint(
    0,
    24,
    (2, 12),
)

# Ignore approximately half of the sequence.
targets[:, :6] = -100


criterion = nn.CrossEntropyLoss(
    ignore_index=-100,
)


optimizer.zero_grad()

logits = model(
    input_ids
)

loss = criterion(
    logits.reshape(
        -1,
        24,
    ),
    targets.reshape(-1),
)

loss.backward()

optimizer.step()


loss_value = float(
    loss.detach()
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

checkpoint = SFTCheckpoint()

saved = checkpoint.save(
    checkpoint_path,
    model,
    optimizer,
    step=1,
    loss=loss_value,
    loss_history=[
        loss_value
    ],
    model_config={
        "vocab_size": 24,
        "hidden_size": 64,
        "layers": 2,
        "heads": 4,
        "context_length": 64,
    },
    sft_config={
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "gradient_clip": 1.0,
    },
)


assert saved.exists()

print(
    "checkpoint:",
    saved,
)

print(
    "checkpoint size:",
    saved.stat().st_size,
)


# ---------------------------------------------------------
# LOAD
# ---------------------------------------------------------

state = checkpoint.load_state(
    checkpoint_path,
)

assert state["version"] == 2
assert state["training"]["step"] == 1
assert state["masking"]["policy"] == (
    "assistant_only"
)
assert state["masking"]["ignore_index"] == -100


# ---------------------------------------------------------
# RESTORE INTO NEW MODEL
# ---------------------------------------------------------

restored_model = NeuraTorchModel(
    vocab_size=24,
    hidden_size=64,
    layers=2,
    heads=4,
    context_length=64,
    dropout=0.0,
)

restored_optimizer = torch.optim.AdamW(
    restored_model.parameters(),
    lr=1e-4,
)


restored = checkpoint.restore(
    checkpoint_path,
    restored_model,
    restored_optimizer,
)


assert restored["step"] == 1

assert (
    restored["masking"]["policy"]
    == "assistant_only"
)


# ---------------------------------------------------------
# PARAMETER VERIFICATION
# ---------------------------------------------------------

original_state = model.state_dict()
restored_state = restored_model.state_dict()

for key in original_state:

    assert torch.equal(
        original_state[key],
        restored_state[key],
    ), (
        "parameter mismatch: "
        f"{key}"
    )


# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------

checkpoint_path.unlink()


print()
print("=" * 60)
print("RESULT: SFT CHECKPOINT PASS")
print("=" * 60)
