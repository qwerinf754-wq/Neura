from neura.inference.request import RequestContext
from neura.inference.request_manager import RequestManager
from neura.inference.state import InferenceState


print("=" * 60)
print("NEURA REQUEST MANAGER CONTRACT")
print("=" * 60)


manager = RequestManager()

assert manager.size() == 0
assert manager.pending_size() == 0

print("Initial manager state: PASS")


# ---------------------------------------------------------
# CREATE
# ---------------------------------------------------------

request_a = manager.create(
    prompt_tokens=[4, 5],
    max_tokens=4,
)

request_b = manager.create(
    prompt_tokens=[7, 8],
    max_tokens=2,
)

assert request_a.request_id != request_b.request_id

print("Unique request IDs: PASS")


assert manager.size() == 2
assert manager.pending_size() == 2

print("Request registration: PASS")


# ---------------------------------------------------------
# LOOKUP
# ---------------------------------------------------------

assert manager.get(
    request_a.request_id
) is request_a

assert manager.get(
    request_b.request_id
) is request_b

print("Request lookup: PASS")


# ---------------------------------------------------------
# FIFO
# ---------------------------------------------------------

first = manager.pop_pending()

assert first is request_a

print("FIFO pending order: PASS")


assert manager.pending_size() == 1

print("Pending queue accounting: PASS")


# ---------------------------------------------------------
# STATE TRANSITION
# ---------------------------------------------------------

first.state = InferenceState.PREFILL

assert manager.active_size() == 1

print("Active request tracking: PASS")


second = manager.pop_pending()

assert second is request_b

print("Second request dequeue: PASS")


second.state = InferenceState.DECODING

assert manager.active_size() == 2

print("Multiple active request tracking: PASS")


# ---------------------------------------------------------
# COMPLETION
# ---------------------------------------------------------

first.finish("eos")

assert manager.completed_size() == 1
assert manager.active_size() == 1

print("Completed request tracking: PASS")


second.finish("length")

assert manager.completed_size() == 2
assert manager.active_size() == 0

print("Completion accounting: PASS")


# ---------------------------------------------------------
# ISOLATION
# ---------------------------------------------------------

assert request_a.generated_tokens == []
assert request_b.generated_tokens == []

request_a.append_token(11)

assert request_a.generated_tokens == [11]
assert request_b.generated_tokens == []

print("Request state isolation: PASS")


# ---------------------------------------------------------
# DUPLICATE PROTECTION
# ---------------------------------------------------------

duplicate = RequestContext(
    request_id=request_a.request_id,
    prompt_tokens=[99],
)

try:
    manager.submit(duplicate)
    raise AssertionError(
        "Duplicate request ID accepted."
    )
except ValueError:
    pass

print("Duplicate request protection: PASS")


# ---------------------------------------------------------
# FAILED
# ---------------------------------------------------------

failed = manager.create(
    prompt_tokens=[10],
    max_tokens=1,
)

failed.state = InferenceState.FAILED

assert manager.failed_size() == 1

print("Failed request tracking: PASS")


# ---------------------------------------------------------
# REMOVAL
# ---------------------------------------------------------

removed = manager.remove(
    request_a.request_id
)

assert removed is request_a
assert manager.get(
    request_a.request_id
) is None

print("Request removal: PASS")


# ---------------------------------------------------------
# CLEANUP
# ---------------------------------------------------------

removed_count = manager.clear_completed()

assert removed_count == 1

assert manager.completed_size() == 0
assert manager.failed_size() == 1
assert manager.size() == 1

print("Completed-request cleanup: PASS")


# ---------------------------------------------------------
# MISSING REQUEST
# ---------------------------------------------------------

assert manager.get(
    "does-not-exist"
) is None

assert manager.remove(
    "does-not-exist"
) is None

print("Missing-request handling: PASS")


print("=" * 60)
print("RESULT: REQUEST MANAGER CONTRACT PASSED")
print("=" * 60)
