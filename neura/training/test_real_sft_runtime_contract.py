from pathlib import Path
from tempfile import TemporaryDirectory

from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny

from neura.training.chat_dataset import (
    ChatSFTDataset,
    IGNORE_INDEX,
)
from neura.training.sft_runtime import (
    SFTRuntime,
)
from neura.training.sft_real_runtime import (
    RealSFTRuntime,
)


class ContractLoss:

    def forward(self, logits, labels):
        assert labels

        assert all(
            label != IGNORE_INDEX
            for label in labels
        )

        return float(len(labels))


class ContractOptimizer:

    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


print("=" * 60)
print("NEURA REAL SFT RUNTIME CONTRACT")
print("=" * 60)


# ---------------------------------------------------------
# DATA
# ---------------------------------------------------------

dataset = ChatSFTDataset(
    [
        [
            {
                "role": "system",
                "content": "You are Neura.",
            },
            {
                "role": "user",
                "content": "Hello.",
            },
            {
                "role": "assistant",
                "content": "Hello from Neura.",
            },
        ],
    ]
)

example = dataset[0]

assert any(
    label != IGNORE_INDEX
    for label in example.labels
)

assert any(
    label == IGNORE_INDEX
    for label in example.labels
)

print("SFT dataset masking: PASS")


# ---------------------------------------------------------
# REAL MODEL
# ---------------------------------------------------------

model = NeuraModel(
    config=NeuraTiny()
)

print("Real NeuraModel creation: PASS")


# ---------------------------------------------------------
# REAL SFT RUNTIME
# ---------------------------------------------------------

loss = ContractLoss()
optimizer = ContractOptimizer()

runtime = RealSFTRuntime(
    model=model,
    loss=loss,
    optimizer=optimizer,
)

assert isinstance(
    runtime.runtime,
    SFTRuntime,
)

print("Real SFT runtime creation: PASS")


# ---------------------------------------------------------
# TRAIN
# ---------------------------------------------------------

result = runtime.train_step(
    [example]
)

assert result["status"] == "trained"
assert result["step"] == 1
assert result["assistant_tokens"] > 0

print("Real model SFT forward: PASS")
print("Assistant-only loss path: PASS")
print("Real SFT training step: PASS")


assert optimizer.steps == 1

print("Optimizer integration: PASS")


assert runtime.step_count == 1
assert runtime.last_loss is not None

print("Runtime accounting: PASS")


# ---------------------------------------------------------
# CHECKPOINT
# ---------------------------------------------------------

with TemporaryDirectory() as directory:

    path = Path(directory) / "real_sft.json"

    runtime.save_checkpoint(
        path
    )

    assert path.exists()

    print("Real SFT checkpoint save: PASS")


    saved_step = runtime.step_count
    saved_loss = runtime.last_loss


    # Simulate restart.
    runtime.runtime.step_count = 0
    runtime.runtime.last_loss = None

    runtime.load_checkpoint(
        path
    )

    assert runtime.step_count == saved_step
    assert runtime.last_loss == saved_loss

    print("Real SFT checkpoint restore: PASS")


    # -----------------------------------------------------
    # Resume
    # -----------------------------------------------------

    result2 = runtime.train_step(
        [example]
    )

    assert result2["step"] == saved_step + 1
    assert runtime.step_count == saved_step + 1

    print("Real SFT resume: PASS")


print("=" * 60)
print("RESULT: REAL SFT RUNTIME CONTRACT PASSED")
print("=" * 60)
