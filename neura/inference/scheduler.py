"""
Neura Inference Scheduler

First runtime scheduling layer.

Responsibilities:
- Accept registered requests from RequestManager.
- Preserve FIFO ordering.
- Move requests from queued -> running.
- Prevent duplicate execution.
- Track finished / failed requests.
- Keep scheduler bookkeeping isolated from model/KV state.

This version intentionally does NOT perform batching yet.
It establishes the scheduling contract required by BatchBuilder.
"""

from __future__ import annotations

from collections import deque

from .request import RequestContext
from .request_manager import RequestManager
from .state import InferenceState


class SchedulerError(RuntimeError):
    """Base scheduler error."""


class SchedulerStateError(SchedulerError):
    """Raised when a request violates scheduler lifecycle rules."""


class InferenceScheduler:

    def __init__(self, manager: RequestManager):
        if not isinstance(manager, RequestManager):
            raise TypeError(
                "manager must be RequestManager"
            )

        self.manager = manager

        self._queued = deque()
        self._queued_ids = set()

        self._running = {}
        self._finished = {}
        self._failed = {}
        self._failed_errors = {}

    # ---------------------------------------------------------
    # Queue registration
    # ---------------------------------------------------------

    def enqueue(self, request: RequestContext):
        if not isinstance(request, RequestContext):
            raise TypeError(
                "request must be RequestContext"
            )

        request_id = request.request_id

        if request_id in self._queued_ids:
            raise SchedulerStateError(
                f"Request already queued: {request_id}"
            )

        if request_id in self._running:
            raise SchedulerStateError(
                f"Request already running: {request_id}"
            )

        if request_id in self._finished:
            raise SchedulerStateError(
                f"Request already finished: {request_id}"
            )

        if request_id in self._failed:
            raise SchedulerStateError(
                f"Request already failed: {request_id}"
            )

        if self.manager.get(request_id) is not request:
            raise SchedulerStateError(
                "Request must be registered with RequestManager."
            )

        self._queued.append(request_id)
        self._queued_ids.add(request_id)

        return request

    # ---------------------------------------------------------
    # Queue discovery
    # ---------------------------------------------------------

    def enqueue_pending(self):
        """
        Import CREATED requests from RequestManager.

        Existing scheduler entries are never duplicated.
        """
        imported = []

        for request in self.manager.pending():
            request_id = request.request_id

            if (
                request_id in self._queued_ids
                or request_id in self._running
                or request_id in self._finished
                or request_id in self._failed
            ):
                continue

            self._queued.append(request_id)
            self._queued_ids.add(request_id)
            imported.append(request)

        return imported

    # ---------------------------------------------------------
    # Scheduling
    # ---------------------------------------------------------

    def schedule_next(self):
        """
        Pop exactly one request from the scheduler queue.

        The request is moved to the running set.
        """
        while self._queued:

            request_id = self._queued.popleft()
            self._queued_ids.discard(request_id)

            request = self.manager.get(request_id)

            if request is None:
                continue

            if request.state != InferenceState.CREATED:
                continue

            if request_id in self._running:
                raise SchedulerStateError(
                    f"Request already running: {request_id}"
                )

            self._running[request_id] = request

            return request

        return None

    # ---------------------------------------------------------
    # Completion
    # ---------------------------------------------------------

    def finish(
        self,
        request: RequestContext,
        reason="length",
    ):
        request_id = request.request_id

        active = self._running.pop(
            request_id,
            None,
        )

        if active is not request:
            raise SchedulerStateError(
                f"Request is not running: {request_id}"
            )

        if request.state != InferenceState.FINISHED:
            request.finish(reason)

        self._finished[request_id] = request

        return request

    # ---------------------------------------------------------
    # Failure
    # ---------------------------------------------------------

    def fail(
        self,
        request: RequestContext,
        error=None,
    ):
        request_id = request.request_id

        active = self._running.pop(
            request_id,
            None,
        )

        if active is not request:
            raise SchedulerStateError(
                f"Request is not running: {request_id}"
            )

        request.state = InferenceState.FAILED

        if error is not None:
            self._failed_errors[request_id] = error

        self._failed[request_id] = request

        return request

    # ---------------------------------------------------------
    # State views
    # ---------------------------------------------------------

    def queued(self):
        result = []

        for request_id in self._queued:
            request = self.manager.get(request_id)

            if request is not None:
                result.append(request)

        return result

    def running(self):
        return list(
            self._running.values()
        )

    def finished(self):
        return list(
            self._finished.values()
        )

    def failed(self):
        return list(
            self._failed.values()
        )

    def failure(self, request_id):
        return self._failed_errors.get(
            request_id
        )

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def queued_size(self):
        return len(self._queued_ids)

    def running_size(self):
        return len(self._running)

    def finished_size(self):
        return len(self._finished)

    def failed_size(self):
        return len(self._failed)

    def size(self):
        return (
            self.queued_size()
            + self.running_size()
            + self.finished_size()
            + self.failed_size()
        )

    # ---------------------------------------------------------
    # Lookup
    # ---------------------------------------------------------

    def get(self, request_id):
        if request_id in self._running:
            return self._running[request_id]

        if request_id in self._finished:
            return self._finished[request_id]

        if request_id in self._failed:
            return self._failed[request_id]

        request = self.manager.get(request_id)

        if request is not None:
            if request_id in self._queued_ids:
                return request

        return None


__all__ = [
    "InferenceScheduler",
    "SchedulerError",
    "SchedulerStateError",
]
