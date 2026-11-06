from neura.inference.request_manager import RequestManager
from neura.inference.scheduler import (
    InferenceScheduler,
    SchedulerStateError,
)
from neura.inference.state import InferenceState


print("=" * 60)
print("NEURA INFERENCE SCHEDULER CONTRACT")
print("=" * 60)


# ---------------------------------------------------------
# SETUP
# ---------------------------------------------------------

manager = RequestManager()
scheduler = InferenceScheduler(manager)

assert scheduler.queued_size() == 0
assert scheduler.running_size() == 0
assert scheduler.finished_size() == 0
assert scheduler.failed_size() == 0

print("Initial scheduler state: PASS")


# ---------------------------------------------------------
# REQUEST CREATION
# ---------------------------------------------------------

request_a = manager.create(
    prompt_tokens=[4, 5],
    max_tokens=4,
)

request_b = manager.create(
    prompt_tokens=[7, 8],
    max_tokens=3,
)

print("Request creation: PASS")


# ---------------------------------------------------------
# FIFO ENQUEUE
# ---------------------------------------------------------

scheduler.enqueue(request_a)
scheduler.enqueue(request_b)

assert scheduler.queued_size() == 2
assert scheduler.queued()[0] is request_a
assert scheduler.queued()[1] is request_b

print("FIFO queue ordering: PASS")


# ---------------------------------------------------------
# DUPLICATE PROTECTION
# ---------------------------------------------------------

try:
    scheduler.enqueue(request_a)
    raise AssertionError(
        "Duplicate enqueue accepted."
    )
except SchedulerStateError:
    pass

print("Duplicate enqueue protection: PASS")


# ---------------------------------------------------------
# FIRST SCHEDULE
# ---------------------------------------------------------

first = scheduler.schedule_next()

assert first is request_a
assert scheduler.running_size() == 1
assert scheduler.queued_size() == 1
assert scheduler.running()[0] is request_a

print("First request scheduling: PASS")


# ---------------------------------------------------------
# RUNNING DUPLICATE PROTECTION
# ---------------------------------------------------------

try:
    scheduler.enqueue(request_a)
    raise AssertionError(
        "Running request re-enqueued."
    )
except SchedulerStateError:
    pass

print("Running-request isolation: PASS")


# ---------------------------------------------------------
# SECOND REQUEST
# ---------------------------------------------------------

second = scheduler.schedule_next()

assert second is request_b
assert scheduler.running_size() == 2
assert scheduler.queued_size() == 0

print("Second request scheduling: PASS")
print("Multiple running request tracking: PASS")


# ---------------------------------------------------------
# FINISH FIRST
# ---------------------------------------------------------

first.finish("eos")
scheduler._running.pop(
    first.request_id,
    None,
)
scheduler._finished[first.request_id] = first

assert scheduler.running_size() == 1
assert scheduler.finished_size() == 1
assert first.state == InferenceState.FINISHED

print("Finished-request tracking: PASS")


# ---------------------------------------------------------
# FINISH SECOND THROUGH API
# ---------------------------------------------------------

scheduler.finish(
    second,
    "length",
)

assert scheduler.running_size() == 0
assert scheduler.finished_size() == 2
assert second.state == InferenceState.FINISHED

print("Scheduler finish lifecycle: PASS")


# ---------------------------------------------------------
# REQUEST ISOLATION
# ---------------------------------------------------------

assert first.generated_tokens == []
assert second.generated_tokens == []

first.append_token(11)

assert first.generated_tokens == [11]
assert second.generated_tokens == []

print("Request state isolation: PASS")


# ---------------------------------------------------------
# FAILURE PATH
# ---------------------------------------------------------

failed = manager.create(
    prompt_tokens=[10],
    max_tokens=1,
)

scheduler.enqueue(failed)

scheduled_failed = scheduler.schedule_next()

assert scheduled_failed is failed
assert scheduler.running_size() == 1

scheduler.fail(
    failed,
    RuntimeError("synthetic failure"),
)

assert scheduler.running_size() == 0
assert scheduler.failed_size() == 1
assert failed.state == InferenceState.FAILED

print("Failure lifecycle: PASS")


# ---------------------------------------------------------
# PENDING IMPORT
# ---------------------------------------------------------

manager2 = RequestManager()

pending_a = manager2.create(
    prompt_tokens=[4],
)

pending_b = manager2.create(
    prompt_tokens=[5],
)

scheduler2 = InferenceScheduler(
    manager2
)

imported = scheduler2.enqueue_pending()

assert len(imported) == 2
assert imported[0] is pending_a
assert imported[1] is pending_b
assert scheduler2.queued_size() == 2

print("Pending-request import: PASS")


# ---------------------------------------------------------
# IMPORT IDEMPOTENCY
# ---------------------------------------------------------

imported_again = scheduler2.enqueue_pending()

assert imported_again == []
assert scheduler2.queued_size() == 2

print("Pending import idempotency: PASS")


# ---------------------------------------------------------
# LOOKUP
# ---------------------------------------------------------

assert scheduler.get(
    first.request_id
) is first

assert scheduler.get(
    second.request_id
) is second

assert scheduler.get(
    failed.request_id
) is failed

assert scheduler.get(
    "does-not-exist"
) is None

print("Scheduler lookup: PASS")


print("=" * 60)
print("RESULT: INFERENCE SCHEDULER CONTRACT PASSED")
print("=" * 60)
