from neura.tokenizer import VOCAB, TOKENIZER
from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny

from neura.inference.request import RequestContext
from neura.inference.state import InferenceState
from neura.inference.batch_executor import BatchExecutor


print("=" * 60)
print("NEURA REQUEST-SCOPED KV CONTRACT")
print("=" * 60)


VOCAB.ensure_size(32768)
TOKENIZER.unfreeze()


# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = NeuraModel(
    config=NeuraTiny()
)

executor = BatchExecutor(model)

print("Model / executor creation: PASS")


# ---------------------------------------------------------
# REQUESTS
# ---------------------------------------------------------

tokens_a = TOKENIZER.encode("Hello Neura")
tokens_b = TOKENIZER.encode("Hello AI")

request_a = RequestContext(
    request_id="scope-a",
    prompt_tokens=tokens_a,
    max_tokens=4,
)

request_b = RequestContext(
    request_id="scope-b",
    prompt_tokens=tokens_b,
    max_tokens=4,
)

executor.begin([
    request_a,
    request_b,
])

assert executor.active_size() == 2

print("Request registration: PASS")


# ---------------------------------------------------------
# PREFILL A
# ---------------------------------------------------------

executor.prefill(
    request_a.request_id
)

cache_a = executor.cache(
    request_a.request_id
)

size_a = cache_a.total_size()

assert size_a > 0

print("Request A prefill: PASS")
print("Request A KV population: PASS")


# ---------------------------------------------------------
# PREFILL B
# ---------------------------------------------------------

executor.prefill(
    request_b.request_id
)

cache_b = executor.cache(
    request_b.request_id
)

size_b = cache_b.total_size()

assert size_b > 0

print("Request B prefill: PASS")
print("Request B KV population: PASS")


# ---------------------------------------------------------
# CACHE IDENTITY
# ---------------------------------------------------------

assert cache_a is not cache_b

print("Independent KV objects: PASS")


assert cache_a.total_size() == size_a
assert cache_b.total_size() == size_b

print("Initial KV isolation: PASS")


# ---------------------------------------------------------
# A DECODE
# ---------------------------------------------------------

executor.decode(
    request_a.request_id,
    5,
)

assert executor.cache(
    request_a.request_id
).total_size() > size_a

assert (
    executor.cache(
        request_b.request_id
    ).total_size()
    == size_b
)

print("Request A incremental KV growth: PASS")
print("Request B KV isolation after A decode: PASS")


# ---------------------------------------------------------
# B DECODE
# ---------------------------------------------------------

executor.decode(
    request_b.request_id,
    4,
)

assert executor.cache(
    request_b.request_id
).total_size() > size_b

print("Request B incremental KV growth: PASS")


assert (
    executor.cache(
        request_a.request_id
    ).total_size()
    > size_a
)

print("Cross-request KV independence: PASS")


# ---------------------------------------------------------
# SECOND INTERLEAVED DECODE
# ---------------------------------------------------------

a_before = executor.cache(
    request_a.request_id
).total_size()

b_before = executor.cache(
    request_b.request_id
).total_size()


executor.decode(
    request_a.request_id,
    5,
)

assert (
    executor.cache(
        request_a.request_id
    ).total_size()
    > a_before
)

assert (
    executor.cache(
        request_b.request_id
    ).total_size()
    == b_before
)

print("Interleaved A decode isolation: PASS")


executor.decode(
    request_b.request_id,
    4,
)

assert (
    executor.cache(
        request_b.request_id
    ).total_size()
    > b_before
)

print("Interleaved B decode isolation: PASS")


# ---------------------------------------------------------
# REQUEST TOKEN ISOLATION
# ---------------------------------------------------------

assert request_a.generated_tokens == [5, 5]
assert request_b.generated_tokens == [4, 4]

print("Generated-token isolation: PASS")


# ---------------------------------------------------------
# CACHE CLEANUP
# ---------------------------------------------------------

executor.remove(
    request_a.request_id
)

assert executor.get(
    request_a.request_id
) is None

assert executor.cache(
    request_b.request_id
).total_size() > 0

print("Request A cache removal isolation: PASS")


executor.clear()

assert executor.active_size() == 0
assert executor.completed_size() == 0
assert model.cache_size() == 0

print("Global cache cleanup: PASS")


print("=" * 60)
print(
    "RESULT: REQUEST-SCOPED KV CONTRACT PASSED"
)
print("=" * 60)
