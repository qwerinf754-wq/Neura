from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny

from neura.inference.batch_builder import BatchBuilder
from neura.inference.batch_executor import (
    BatchExecutor,
    BatchExecutionError,
)
from neura.inference.request import RequestContext
from neura.inference.state import InferenceState
from neura.tokenizer import TOKENIZER, VOCAB


print("=" * 60)
print("NEURA BATCH EXECUTOR CONTRACT")
print("=" * 60)


# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------

VOCAB.ensure_size(32768)
TOKENIZER.unfreeze()

model = NeuraModel(
    config=NeuraTiny()
)

executor = BatchExecutor(model)

builder = BatchBuilder()


request_a = RequestContext(
    request_id="req-a",
    prompt_tokens=TOKENIZER.encode(
        "Hello Neura"
    ),
    max_tokens=4,
)

request_b = RequestContext(
    request_id="req-b",
    prompt_tokens=TOKENIZER.encode(
        "Hello AI"
    ),
    max_tokens=4,
)

request_c = RequestContext(
    request_id="req-c",
    prompt_tokens=TOKENIZER.encode(
        "Neura"
    ),
    max_tokens=2,
)


print("Model creation: PASS")
print("Executor creation: PASS")
print("Request creation: PASS")


# ---------------------------------------------------------
# PREFILL BATCH
# ---------------------------------------------------------

prefill = builder.build(
    [
        request_a,
        request_b,
        request_c,
    ],
    BatchBuilder.PREFILL,
)

results = executor.execute_prefill(
    prefill
)

assert len(results) == 3

assert [
    result.request_id
    for result in results
] == [
    "req-a",
    "req-b",
    "req-c",
]

print("Prefill batch execution: PASS")
print("Prefill result ordering: PASS")


# ---------------------------------------------------------
# PREFILL OUTPUT
# ---------------------------------------------------------

for result, request in zip(
    results,
    [
        request_a,
        request_b,
        request_c,
    ],
):
    assert len(result.logits) == len(
        request.prompt_tokens
    )

    assert len(result.logits[-1]) == 32768

print("Prefill logits shape: PASS")


# ---------------------------------------------------------
# STATE TRANSITION
# ---------------------------------------------------------

assert request_a.state == InferenceState.DECODING
assert request_b.state == InferenceState.DECODING
assert request_c.state == InferenceState.DECODING

print("Prefill state transition: PASS")


# ---------------------------------------------------------
# REQUEST ISOLATION
# ---------------------------------------------------------

assert request_a.generated_tokens == []
assert request_b.generated_tokens == []
assert request_c.generated_tokens == []

print("Initial generated-state isolation: PASS")


# ---------------------------------------------------------
# DECODE
# ---------------------------------------------------------

decode = builder.build(
    [
        request_a,
        request_b,
        request_c,
    ],
    BatchBuilder.DECODE,
)

decode_results = executor.execute_decode(
    decode,
    {
        "req-a": 5,
        "req-b": 4,
        "req-c": 5,
    },
)

assert len(decode_results) == 3

print("Decode batch execution: PASS")


# ---------------------------------------------------------
# DECODE OUTPUT
# ---------------------------------------------------------

for result in decode_results:

    assert len(result.logits) == 1
    assert len(result.logits[-1]) == 32768

print("Decode logits shape: PASS")


# ---------------------------------------------------------
# GENERATED TOKEN ISOLATION
# ---------------------------------------------------------

assert request_a.generated_tokens == [5]
assert request_b.generated_tokens == [4]
assert request_c.generated_tokens == [5]

print("Generated-token isolation: PASS")


# ---------------------------------------------------------
# POSITION ACCOUNTING
# ---------------------------------------------------------

assert request_a.position == (
    len(request_a.prompt_tokens) + 1
)

assert request_b.position == (
    len(request_b.prompt_tokens) + 1
)

assert request_c.position == (
    len(request_c.prompt_tokens) + 1
)

print("Position accounting: PASS")


# ---------------------------------------------------------
# CACHE CLEANUP
# ---------------------------------------------------------

assert model.cache_size() == 0

print("Batch cache cleanup: PASS")


# ---------------------------------------------------------
# SECOND BATCH ISOLATION
# ---------------------------------------------------------

request_d = RequestContext(
    request_id="req-d",
    prompt_tokens=TOKENIZER.encode(
        "Second request"
    ),
    max_tokens=2,
)

second_batch = builder.build(
    [request_d],
    BatchBuilder.PREFILL,
)

second_results = executor.execute_prefill(
    second_batch
)

assert len(second_results) == 1
assert request_d.state == InferenceState.DECODING

assert model.cache_size() == 0

print("Second-request isolation: PASS")


# ---------------------------------------------------------
# INVALID MODE
# ---------------------------------------------------------

try:
    executor.execute_prefill(
        decode
    )

    raise AssertionError(
        "Decode batch accepted as prefill."
    )

except Exception:
    pass

print("Invalid batch protection: PASS")


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

assert request_a.generated_tokens == [5]
assert request_b.generated_tokens == [4]
assert request_c.generated_tokens == [5]

assert model.cache_size() == 0

print("Final request isolation: PASS")
print("Final cache cleanup: PASS")

print("=" * 60)
print(
    "RESULT: BATCH EXECUTOR CONTRACT PASSED"
)
print("=" * 60)
