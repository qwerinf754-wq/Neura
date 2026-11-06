from neura.inference.batch_builder import (
    BatchBuilder,
    BatchError,
    DuplicateRequestError,
)
from neura.inference.request import RequestContext
from neura.inference.state import InferenceState


print("=" * 60)
print("NEURA BATCH BUILDER CONTRACT")
print("=" * 60)


builder = BatchBuilder()


# ---------------------------------------------------------
# REQUEST CREATION
# ---------------------------------------------------------

request_a = RequestContext(
    request_id="req-a",
    prompt_tokens=[4, 5],
    max_tokens=4,
)

request_b = RequestContext(
    request_id="req-b",
    prompt_tokens=[7, 8],
    max_tokens=4,
)

request_c = RequestContext(
    request_id="req-c",
    prompt_tokens=[9],
    max_tokens=2,
)

print("Request creation: PASS")


# ---------------------------------------------------------
# EMPTY BATCH
# ---------------------------------------------------------

empty = builder.build(
    [],
    BatchBuilder.PREFILL,
)

assert empty.size == 0
assert empty.request_ids == []
assert empty.slots == []

print("Empty batch: PASS")


# ---------------------------------------------------------
# PREFILL BATCH
# ---------------------------------------------------------

prefill = builder.build(
    [request_a, request_b, request_c],
    BatchBuilder.PREFILL,
)

assert prefill.mode == "prefill"
assert prefill.size == 3

assert prefill.request_ids == [
    "req-a",
    "req-b",
    "req-c",
]

print("Prefill batch creation: PASS")


# ---------------------------------------------------------
# SLOT ORDER
# ---------------------------------------------------------

assert prefill.slots == [0, 1, 2]

assert prefill.items[0].request is request_a
assert prefill.items[1].request is request_b
assert prefill.items[2].request is request_c

print("Stable batch slot ordering: PASS")


# ---------------------------------------------------------
# LOOKUP
# ---------------------------------------------------------

assert prefill.get("req-a").request is request_a
assert prefill.get("req-b").request is request_b
assert prefill.get("missing") is None

print("Batch request lookup: PASS")


# ---------------------------------------------------------
# REQUEST ISOLATION
# ---------------------------------------------------------

request_a.append_token(11)

assert request_a.generated_tokens == [11]
assert request_b.generated_tokens == []
assert request_c.generated_tokens == []

print("Request isolation: PASS")


# ---------------------------------------------------------
# DUPLICATE PROTECTION
# ---------------------------------------------------------

try:
    builder.build(
        [request_a, request_a],
        BatchBuilder.PREFILL,
    )
    raise AssertionError(
        "Duplicate request accepted."
    )
except DuplicateRequestError:
    pass

print("Duplicate request protection: PASS")


# ---------------------------------------------------------
# INVALID PREFILL STATE
# ---------------------------------------------------------

request_a.state = InferenceState.PREFILL

try:
    builder.build(
        [request_a],
        BatchBuilder.PREFILL,
    )
    raise AssertionError(
        "Invalid prefill state accepted."
    )
except BatchError:
    pass

print("Prefill state protection: PASS")


# ---------------------------------------------------------
# DECODE BATCH
# ---------------------------------------------------------

request_b.state = InferenceState.PREFILL
request_c.state = InferenceState.DECODING

decode = builder.build(
    [request_b, request_c],
    BatchBuilder.DECODE,
)

assert decode.mode == "decode"
assert decode.size == 2
assert decode.request_ids == [
    "req-b",
    "req-c",
]

print("Decode batch creation: PASS")


# ---------------------------------------------------------
# INVALID DECODE STATE
# ---------------------------------------------------------

new_request = RequestContext(
    request_id="req-new",
    prompt_tokens=[10],
)

try:
    builder.build(
        [new_request],
        BatchBuilder.DECODE,
    )
    raise AssertionError(
        "CREATED request accepted into decode batch."
    )
except BatchError:
    pass

print("Decode state protection: PASS")


# ---------------------------------------------------------
# MODE VALIDATION
# ---------------------------------------------------------

try:
    builder.build(
        [],
        "invalid",
    )
    raise AssertionError(
        "Invalid batch mode accepted."
    )
except ValueError:
    pass

print("Batch mode validation: PASS")


# ---------------------------------------------------------
# BATCH ID
# ---------------------------------------------------------

second_batch = builder.build(
    [],
    BatchBuilder.PREFILL,
)

assert prefill.batch_id != second_batch.batch_id

print("Unique batch IDs: PASS")


print("=" * 60)
print("RESULT: BATCH BUILDER CONTRACT PASSED")
print("=" * 60)
