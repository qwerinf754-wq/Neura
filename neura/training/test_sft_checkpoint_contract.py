from pathlib import Path
from tempfile import TemporaryDirectory

from neura.training.sft_checkpoint import (
    SFTCheckpoint,
)
from neura.training.sft_runtime import (
    SFTRuntime,
)


class FakeModel:

    def __init__(self):
        self.value = 10

    def state_dict(self):
        return {
            "value": self.value,
        }

    def load_state_dict(self, state):
        self.value = state["value"]


class FakeOptimizer:

    def __init__(self):
        self.step_count = 0

    def state_dict(self):
        return {
            "step_count": self.step_count,
        }

    def load_state_dict(self, state):
        self.step_count = state["step_count"]

    def step(self):
        self.step_count += 1


class FakeLoss:

    def forward(self, logits, labels):
        return float(len(labels))


print("=" * 60)
print("NEURA SFT CHECKPOINT CONTRACT")
print("=" * 60)


model = FakeModel()
optimizer = FakeOptimizer()
loss = FakeLoss()

runtime = SFTRuntime(
    model=model,
    loss=loss,
    optimizer=optimizer,
)

runtime.step_count = 7
runtime.last_loss = 1.25
model.value = 42
optimizer.step_count = 7

checkpoint = SFTCheckpoint(
    runtime
)

print("Checkpoint creation: PASS")


state = checkpoint.state()

assert state["version"] == 1
assert state["step_count"] == 7
assert state["last_loss"] == 1.25
assert state["model_state"]["value"] == 42
assert state["optimizer_state"]["step_count"] == 7

print("Checkpoint state capture: PASS")


with TemporaryDirectory() as directory:

    path = Path(directory) / "sft.json"

    checkpoint.save(path)

    assert path.exists()

    print("Checkpoint save: PASS")


    # -----------------------------------------------------
    # Simulate process restart / state loss
    # -----------------------------------------------------

    model.value = 0
    optimizer.step_count = 0

    runtime.step_count = 0
    runtime.last_loss = None

    print("Runtime reset: PASS")


    # -----------------------------------------------------
    # Restore
    # -----------------------------------------------------

    restored = checkpoint.load(path)

    assert restored["version"] == 1

    assert runtime.step_count == 7
    assert runtime.last_loss == 1.25

    print("Runtime metadata restore: PASS")


    assert model.value == 42

    print("Model state restore: PASS")


    assert optimizer.step_count == 7

    print("Optimizer state restore: PASS")


    # -----------------------------------------------------
    # Resume
    # -----------------------------------------------------

    runtime.step_count += 1

    assert runtime.step_count == 8

    print("Training resume accounting: PASS")


    checkpoint.save(path)

    checkpoint.load(path)

    assert runtime.step_count == 8

    print("Resumed checkpoint persistence: PASS")


print("=" * 60)
print("RESULT: SFT CHECKPOINT CONTRACT PASSED")
print("=" * 60)
