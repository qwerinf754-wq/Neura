from neura.tokenizer import VOCAB, TOKENIZER
from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny

from neura.inference.request_manager import RequestManager
from neura.inference.scheduler import InferenceScheduler
from neura.inference.batch import (
    BatchExecutor,
    BatchResult,
)
from neura.inference.state import InferenceState


print("=" * 60)
print("NEURA BATCH EXECUTION CORE CONTRACT")
print("=" * 60)


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------

VOCAB.ensure_size(32768)
TOKENIZER.unfreeze()

manager = RequestManager()
scheduler = InferenceScheduler(manager)

model = NeuraModel(
    config=NeuraTiny()
)

executor = BatchExecutor(
    model=model,
    scheduler=scheduler,
)

print("Model creation: PASS")
print("Batch executor creation: PASS")


# ---------------------------------------------------------
# Requests
# ---------------------------------------------------------

tokens_a = TOKENIZER.encode("Hello Neura")
tokens_b = TOKENIZER.encode("Hello AI")

request_a = manager.create(
    prompt_tokens=tokens_a,
    max_tokens=3,
)

request_b = manager.create(
    prompt_tokens=tokens_b,
    max_tokens=3,
)

print("Request creation: PASS")


# ---------------------------------------------------------
# Scheduler import
# ---------------------------------------------------------

imported = scheduler.enqueue_pending()

assert imported == [
    request_a,
    request_b,
]

print("Batch request registration: PASS")

assert scheduler.queued_size() == 2

print("FIFO batch queue: PASS")


# ---------------------------------------------------------
# Execute
# ---------------------------------------------------------

result = executor.execute(
    [request_a, request_b]
)

assert isinstance(result, BatchResult)

print("Batch execution result: PASS")

assert result.size == 2

print("Batch size accounting: PASS")


# ---------------------------------------------------------
# Request A
# ---------------------------------------------------------

assert request_a.prompt_tokens == tokens_a

assert request_a.tokens[:len(tokens_a)] == tokens_a

assert all(
    isinstance(token, int)
    and 0 <= token < VOCAB.capacity
    for token in request_a.generated_tokens
)

print("Request A token validity: PASS")


# ---------------------------------------------------------
# Request B
# ---------------------------------------------------------

assert request_b.prompt_tokens == tokens_b

assert request_b.tokens[:len(tokens_b)] == tokens_b

assert all(
    isinstance(token, int)
    and 0 <= token < VOCAB.capacity
    for token in request_b.generated_tokens
)

print("Request B token validity: PASS")


# ---------------------------------------------------------
# Isolation
# ---------------------------------------------------------

assert request_a.generated_tokens is not request_b.generated_tokens

print("Generated-token isolation: PASS")

assert request_a.prompt_tokens == tokens_a
assert request_b.prompt_tokens == tokens_b

print("Prompt isolation: PASS")

assert request_a.tokens != request_b.tokens

print("Request token-state isolation: PASS")


# ---------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------

assert request_a.state == InferenceState.FINISHED
assert request_b.state == InferenceState.FINISHED

print("Request completion lifecycle: PASS")

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
# Scheduler lifecycle
# ---------------------------------------------------------

assert scheduler.running_size() == 0
assert scheduler.queued_size() == 0
assert scheduler.finished_size() == 2

print("Scheduler completion accounting: PASS")


# ---------------------------------------------------------
# Cache isolation / cleanup
# ---------------------------------------------------------

assert model.cache_size() == 0

print("Batch cache cleanup: PASS")


# ---------------------------------------------------------
# Second batch
# ---------------------------------------------------------

request_c = manager.create(
    prompt_tokens=TOKENIZER.encode("Second request"),
    max_tokens=2,
)

scheduler.enqueue_pending()

second_result = executor.execute(
    [request_c]
)

assert second_result.size == 1

assert request_c.state == InferenceState.FINISHED

assert model.cache_size() == 0

print("Second batch isolation: PASS")
print("Second batch cleanup: PASS")


# ---------------------------------------------------------
# Final
# ---------------------------------------------------------

print("=" * 60)
print("RESULT: BATCH EXECUTION CORE CONTRACT PASSED")
print("=" * 60)
