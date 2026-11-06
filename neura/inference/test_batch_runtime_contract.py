from neura.tokenizer import VOCAB, TOKENIZER
from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny

from neura.inference.request_manager import RequestManager
from neura.inference.scheduler import InferenceScheduler
from neura.inference.batch_runtime import (
    BatchRuntime,
    RuntimeStepResult,
)
from neura.inference.state import InferenceState


print("=" * 60)
print("NEURA BATCH RUNTIME STEP CONTRACT")
print("=" * 60)


# ---------------------------------------------------------
# Setup
# ---------------------------------------------------------

VOCAB.ensure_size(32768)
TOKENIZER.unfreeze()

manager = RequestManager()
scheduler = InferenceScheduler(manager)

model = NeuraModel(
    config=NeuraTiny()
)

runtime = BatchRuntime(
    model=model,
    scheduler=scheduler,
)

print("Model creation: PASS")
print("Batch runtime creation: PASS")


# ---------------------------------------------------------
# Requests
# ---------------------------------------------------------

prompt_a = TOKENIZER.encode(
    "Hello Neura"
)

prompt_b = TOKENIZER.encode(
    "Hello AI"
)

request_a = manager.create(
    prompt_tokens=prompt_a,
    max_tokens=3,
)

request_b = manager.create(
    prompt_tokens=prompt_b,
    max_tokens=3,
)

print("Request creation: PASS")


# ---------------------------------------------------------
# Start
# ---------------------------------------------------------

runtime.start(
    [
        request_a,
        request_b,
    ]
)

assert runtime.started is True

print("Runtime start: PASS")

assert runtime.active_size == 2

print("Active batch population: PASS")


# ---------------------------------------------------------
# First step
# ---------------------------------------------------------

step_1 = runtime.step()

assert len(step_1) == 2

assert all(
    isinstance(result, RuntimeStepResult)
    for result in step_1
)

print("First batch decode step: PASS")

assert len(request_a.generated_tokens) == 1
assert len(request_b.generated_tokens) == 1

print("Generated-state update: PASS")

assert request_a.generated_tokens is not request_b.generated_tokens

print("Generated-token isolation: PASS")


# ---------------------------------------------------------
# Position accounting
# ---------------------------------------------------------

assert request_a.position == (
    len(prompt_a) + 1
)

assert request_b.position == (
    len(prompt_b) + 1
)

print("Independent position accounting: PASS")


# ---------------------------------------------------------
# Token validity
# ---------------------------------------------------------

for request in (
    request_a,
    request_b,
):
    assert all(
        isinstance(token, int)
        and 0 <= token < VOCAB.capacity
        for token in request.generated_tokens
    )

print("Decoded token validity: PASS")


# ---------------------------------------------------------
# More steps
# ---------------------------------------------------------

steps = 1

while not runtime.completed:
    runtime.step()
    steps += 1

    assert steps <= 10

print("Multi-step batch execution: PASS")


# ---------------------------------------------------------
# Completion
# ---------------------------------------------------------

assert request_a.state == InferenceState.FINISHED
assert request_b.state == InferenceState.FINISHED

print("Batch completion lifecycle: PASS")

assert request_a.finish_reason in (
    "eos",
    "length",
)

assert request_b.finish_reason in (
    "eos",
    "length",
)

print("Finish reason accounting: PASS")


# ---------------------------------------------------------
# Prompt preservation
# ---------------------------------------------------------

assert request_a.prompt_tokens == prompt_a
assert request_b.prompt_tokens == prompt_b

print("Prompt preservation: PASS")


# ---------------------------------------------------------
# Scheduler cleanup
# ---------------------------------------------------------

assert scheduler.running_size() == 0
assert scheduler.queued_size() == 0
assert scheduler.finished_size() == 2

print("Scheduler lifecycle completion: PASS")


# ---------------------------------------------------------
# Cache cleanup
# ---------------------------------------------------------

assert model.cache_size() == 0

print("Final KV cache cleanup: PASS")


# ---------------------------------------------------------
# Second batch isolation
# ---------------------------------------------------------

request_c = manager.create(
    prompt_tokens=TOKENIZER.encode(
        "Second request"
    ),
    max_tokens=2,
)

runtime.clear()

runtime.start(
    [request_c]
)

assert runtime.active_size == 1

print("Second batch start isolation: PASS")

runtime.run()

assert request_c.state == InferenceState.FINISHED

print("Second batch completion: PASS")

assert model.cache_size() == 0

print("Second batch cache cleanup: PASS")


# ---------------------------------------------------------
# Final cleanup
# ---------------------------------------------------------

runtime.clear()

assert runtime.started is False
assert runtime.completed is False
assert runtime.active_size == 0
assert model.cache_size() == 0

print("Runtime final cleanup: PASS")


print("=" * 60)
print("RESULT: BATCH RUNTIME STEP CONTRACT PASSED")
print("=" * 60)
