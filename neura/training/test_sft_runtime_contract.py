from neura.training.sft_runtime import SFTRuntime
from neura.training.loss import CrossEntropyLoss


print("=" * 60)
print("NEURA SFT RUNTIME CONTRACT")
print("=" * 60)


class Example:

    def __init__(self):
        self.input_ids = [4, 5, 6]
        self.labels = [
            -100,
            7,
            8,
        ]


class FakeModel:

    def __init__(self):
        self.forward_calls = 0
        self.backward_calls = 0
        self.last_backward = None

    def forward(
        self,
        sequence,
        use_cache=False,
    ):
        self.forward_calls += 1

        return [
            [0.0] * 16,
            [0.1 + i * 0.01 for i in range(16)],
            [0.2 + i * 0.01 for i in range(16)],
        ]

    def backward(self, gradients):
        self.backward_calls += 1
        self.last_backward = gradients
        return gradients


class FakeOptimizer:

    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


model = FakeModel()
loss = CrossEntropyLoss()
optimizer = FakeOptimizer()

runtime = SFTRuntime(
    model=model,
    loss=loss,
    optimizer=optimizer,
)

print("Runtime creation: PASS")


example = Example()

batch = runtime.prepare_batch(
    [example]
)

assert batch.assistant_token_count() == 2

print("Masked assistant-token batch: PASS")


logits = runtime.forward(
    batch
)

assert len(logits) == 1
assert len(logits[0]) == 3
assert all(
    len(row) == 16
    for row in logits[0]
)

print("Forward execution: PASS")


loss_value = runtime.compute_loss(
    logits,
    batch.labels,
)

assert loss_value > 0.0

print("Masked loss computation: PASS")


gradients = runtime.gradients()

assert gradients
assert len(gradients) == 2
assert all(
    len(gradient) == 16
    for gradient in gradients
)

print("Loss gradient extraction: PASS")


runtime.step(
    [example]
)

assert model.backward_calls == 1
assert model.last_backward is not None
assert model.last_backward == runtime.last_gradients

print("Backward receives loss gradients: PASS")


assert model.last_backward != logits[0]

print("Raw logits are not passed to backward: PASS")


assert optimizer.step_calls == 1

print("Optimizer step: PASS")


assert runtime.step_count == 1
assert runtime.last_loss is not None
assert runtime.last_gradients is not None

print("Runtime statistics: PASS")


print("=" * 60)
print("RESULT: SFT RUNTIME CONTRACT PASSED")
print("=" * 60)
