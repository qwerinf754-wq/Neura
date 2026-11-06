"""
Neura Request Manager

Owns request registration and lifecycle bookkeeping.

Responsibilities:
- Create request IDs
- Register requests
- Maintain FIFO pending queue
- Track active / completed / failed requests
- Prevent duplicate request IDs
- Preserve request isolation
- Remove finished requests
"""

from collections import deque
from itertools import count

from .request import RequestContext
from .state import InferenceState


class RequestManager:

    def __init__(self):
        self._requests = {}
        self._pending = deque()
        self._counter = count(1)

    # ---------------------------------------------------------
    # Request creation
    # ---------------------------------------------------------

    def create(
        self,
        prompt_tokens,
        max_tokens=20,
        temperature=1.0,
        top_k=None,
        repetition_penalty=None,
        eos_token=3,
    ):
        request_id = f"req-{next(self._counter):08d}"

        request = RequestContext(
            request_id=request_id,
            prompt_tokens=list(prompt_tokens),
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            eos_token=eos_token,
        )

        self.submit(request)

        return request

    # ---------------------------------------------------------
    # Registration
    # ---------------------------------------------------------

    def submit(self, request):
        if not isinstance(request, RequestContext):
            raise TypeError(
                "request must be RequestContext"
            )

        if request.request_id in self._requests:
            raise ValueError(
                f"Duplicate request ID: {request.request_id}"
            )

        self._requests[request.request_id] = request
        self._pending.append(request.request_id)

        return request

    # ---------------------------------------------------------
    # Lookup
    # ---------------------------------------------------------

    def get(self, request_id):
        return self._requests.get(request_id)

    # ---------------------------------------------------------
    # FIFO queue
    # ---------------------------------------------------------

    def pop_pending(self):
        while self._pending:

            request_id = self._pending.popleft()

            request = self._requests.get(
                request_id
            )

            if request is None:
                continue

            if request.state != InferenceState.CREATED:
                continue

            return request

        return None

    def pending(self):
        pending_requests = []

        for request_id in self._pending:

            request = self._requests.get(
                request_id
            )

            if request is None:
                continue

            if request.state != InferenceState.CREATED:
                continue

            pending_requests.append(request)

        return pending_requests

    # ---------------------------------------------------------
    # State views
    # ---------------------------------------------------------

    def active(self):
        return [
            request
            for request in self._requests.values()
            if request.state in (
                InferenceState.PREFILL,
                InferenceState.DECODING,
            )
        ]

    def completed(self):
        return [
            request
            for request in self._requests.values()
            if request.state == InferenceState.FINISHED
        ]

    def failed(self):
        return [
            request
            for request in self._requests.values()
            if request.state == InferenceState.FAILED
        ]

    # ---------------------------------------------------------
    # Removal
    # ---------------------------------------------------------

    def remove(self, request_id):
        request = self._requests.pop(
            request_id,
            None,
        )

        if request is None:
            return None

        self._pending = deque(
            rid
            for rid in self._pending
            if rid != request_id
        )

        return request

    # ---------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------

    def size(self):
        return len(self._requests)

    def pending_size(self):
        # IMPORTANT:
        # Pending means "still queued", not merely state == CREATED.
        return len(self.pending())

    def active_size(self):
        return len(self.active())

    def completed_size(self):
        return len(self.completed())

    def failed_size(self):
        return len(self.failed())

    def clear_completed(self):
        completed_ids = [
            request.request_id
            for request in self.completed()
        ]

        for request_id in completed_ids:
            self.remove(request_id)

        return len(completed_ids)


__all__ = [
    "RequestManager",
]
