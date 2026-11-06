from neura.inference.request import RequestContext
from neura.inference.session import InferenceSession
from neura.inference.state import InferenceState


print("=" * 60)
print("NEURA REQUEST SESSION CONTRACT")
print("=" * 60)


request_a = RequestContext(
    request_id="request-a",
    prompt_tokens=[4, 5],
    max_tokens=4,
)

request_b = RequestContext(
    request_id="request-b",
    prompt_tokens=[7, 8, 9],
    max_tokens=2,
)


assert request_a.request_id != request_b.request_id
print("Request identity isolation: PASS")


assert request_a.generated_tokens == []
assert request_b.generated_tokens == []

print("Initial generated state: PASS")


session_a = InferenceSession(request_a)
session_b = InferenceSession(request_b)


session_a.start_prefill()
session_b.start_prefill()


assert request_a.state == InferenceState.PREFILL
assert request_b.state == InferenceState.PREFILL

print("Independent prefill state: PASS")


request_a.append_token(11)

assert request_a.generated_tokens == [11]
assert request_b.generated_tokens == []

print("Generated-token isolation: PASS")


session_a.start_decode()

assert request_a.state == InferenceState.DECODING
assert request_b.state == InferenceState.PREFILL

print("Independent decode state: PASS")


assert request_a.position == 3
assert request_b.position == 3

print("Position accounting: PASS")


request_a.append_token(12)

assert request_a.tokens == [4, 5, 11, 12]
assert request_b.tokens == [7, 8, 9]

print("Token composition isolation: PASS")


session_a.finish("length")

assert session_a.finished is True
assert request_a.state == InferenceState.FINISHED
assert request_a.finish_reason == "length"

assert session_b.finished is False
assert request_b.state == InferenceState.PREFILL

print("Session completion isolation: PASS")


session_b.fail()

assert session_b.finished is True
assert request_b.state == InferenceState.FAILED

print("Failure isolation: PASS")


print("=" * 60)
print("RESULT: REQUEST SESSION CONTRACT PASSED")
print("=" * 60)
