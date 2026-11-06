"""
Neura Inference Batch
=====================

Runtime batch state and request isolation.
"""

from dataclasses import dataclass, field

from .request import RequestContext
from .state import InferenceState


class BatchError(RuntimeError):
    """Base batch runtime error."""


class BatchStateError(BatchError):
    """Raised when a batch operation violates its lifecycle."""


class DuplicateRequestError(BatchError):
    """Raised when a request is duplicated inside a batch."""


@dataclass(slots=True)
class BatchItem:
    request: RequestContext
    slot: int

    @property
    def request_id(self):
        return self.request.request_id


@dataclass(slots=True)
class Batch:
    batch_id: str
    mode: str
    items: list[BatchItem] = field(default_factory=list)

    CREATED = "created"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"

    state: str = CREATED

    @property
    def size(self):
        return len(self.items)

    @property
    def request_ids(self):
        return [
            item.request_id
            for item in self.items
        ]

    @property
    def slots(self):
        return [
            item.slot
            for item in self.items
        ]

    def get(self, request_id):
        for item in self.items:
            if item.request_id == request_id:
                return item
        return None

    def start(self):
        if self.state != self.CREATED:
            raise BatchStateError(
                "Batch can only start from CREATED state."
            )

        self.state = self.RUNNING
        return self

    def finish(self):
        if self.state != self.RUNNING:
            raise BatchStateError(
                "Batch can only finish from RUNNING state."
            )

        self.state = self.FINISHED
        return self

    def fail(self):
        if self.state not in (
            self.CREATED,
            self.RUNNING,
        ):
            raise BatchStateError(
                "Batch cannot fail from current state."
            )

        self.state = self.FAILED
        return self

    def is_terminal(self):
        return self.state in (
            self.FINISHED,
            self.FAILED,
        )


class BatchBuilder:

    PREFILL = "prefill"
    DECODE = "decode"

    def __init__(self, manager=None):
        self.manager = manager
        self._counter = 0

    def _next_batch_id(self):
        self._counter += 1
        return f"batch-{self._counter:08d}"

    def build(self, requests, mode):
        if mode not in (
            self.PREFILL,
            self.DECODE,
        ):
            raise ValueError(
                f"Unsupported batch mode: {mode}"
            )

        requests = list(requests)

        seen = set()
        items = []

        for slot, request in enumerate(requests):

            if not isinstance(
                request,
                RequestContext,
            ):
                raise TypeError(
                    "batch items must be RequestContext"
                )

            if request.request_id in seen:
                raise DuplicateRequestError(
                    f"Duplicate request: "
                    f"{request.request_id}"
                )

            seen.add(request.request_id)

            if mode == self.PREFILL:
                if request.state != (
                    InferenceState.CREATED
                ):
                    raise BatchError(
                        "Prefill request must be CREATED: "
                        f"{request.request_id}"
                    )

            elif mode == self.DECODE:
                if request.state not in (
                    InferenceState.PREFILL,
                    InferenceState.DECODING,
                ):
                    raise BatchError(
                        "Decode request must be active: "
                        f"{request.request_id}"
                    )

            items.append(
                BatchItem(
                    request=request,
                    slot=slot,
                )
            )

        return Batch(
            batch_id=self._next_batch_id(),
            mode=mode,
            items=items,
        )


InferenceBatch = Batch


__all__ = [
    "Batch",
    "InferenceBatch",
    "BatchItem",
    "BatchBuilder",
    "BatchError",
    "BatchStateError",
    "DuplicateRequestError",
]
