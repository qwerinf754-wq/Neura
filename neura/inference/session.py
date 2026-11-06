"""
Neura Inference Session

Represents the execution session of one request.
"""

from dataclasses import dataclass

from .request import RequestContext
from .state import InferenceState


@dataclass(slots=True)
class InferenceSession:

    context: RequestContext

    logits: list[list[float]] | None = None

    started: bool = False
    finished: bool = False

    def start_prefill(self):
        if self.started:
            raise RuntimeError(
                "Inference session already started."
            )

        self.started = True
        self.context.state = InferenceState.PREFILL

    def start_decode(self):
        if not self.started:
            raise RuntimeError(
                "Prefill must start before decode."
            )

        if self.finished:
            raise RuntimeError(
                "Inference session already finished."
            )

        self.context.state = InferenceState.DECODING

    def finish(self, reason: str):
        self.context.finish(reason)
        self.finished = True

    def fail(self):
        self.context.fail()
        self.finished = True

    @property
    def position(self):
        return self.context.position

    @property
    def tokens(self):
        return self.context.tokens


__all__ = ["InferenceSession"]
