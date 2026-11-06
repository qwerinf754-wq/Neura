"""
Neura Batch Builder
===================

Builds isolated inference batches from scheduled requests.

Responsibilities:
- Preserve request ordering.
- Prevent duplicate requests in a batch.
- Separate prefill and decode work.
- Assign stable batch slots.
- Keep request state isolated.
- Provide deterministic batch metadata.
"""

from dataclasses import dataclass

from .request import RequestContext
from .state import InferenceState


class BatchError(RuntimeError):
    """Base exception for batch construction errors."""


class DuplicateRequestError(BatchError):
    """Raised when the same request is inserted twice."""


@dataclass(slots=True)
class BatchItem:
    request: RequestContext
    slot: int

    @property
    def request_id(self):
        return self.request.request_id


@dataclass(slots=True)
class InferenceBatch:
    batch_id: str
    mode: str
    items: list[BatchItem]

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


class BatchBuilder:

    PREFILL = "prefill"
    DECODE = "decode"

    def __init__(self):
        self._counter = 0

    def _next_batch_id(self):
        self._counter += 1
        return f"batch-{self._counter:08d}"

    def build(
        self,
        requests,
        mode,
    ):
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

        return InferenceBatch(
            batch_id=self._next_batch_id(),
            mode=mode,
            items=items,
        )


__all__ = [
    "BatchError",
    "DuplicateRequestError",
    "BatchItem",
    "InferenceBatch",
    "BatchBuilder",
]
