"""
Neura Inference Request Context

Owns the immutable request configuration and the mutable
execution state belonging to one inference request.
"""

from dataclasses import dataclass, field
from .state import InferenceState


@dataclass(slots=True)
class RequestContext:
    request_id: str
    prompt_tokens: list[int]

    max_tokens: int = 20
    temperature: float = 1.0
    top_k: int | None = None
    repetition_penalty: float | None = None
    eos_token: int = 3

    generated_tokens: list[int] = field(
        default_factory=list
    )

    state: InferenceState = InferenceState.CREATED
    finish_reason: str | None = None

    @property
    def position(self):
        return (
            len(self.prompt_tokens)
            + len(self.generated_tokens)
        )

    @property
    def finished(self):
        return self.state == InferenceState.FINISHED

    @property
    def tokens(self):
        return (
            list(self.prompt_tokens)
            + list(self.generated_tokens)
        )

    def append_token(self, token: int):
        self.generated_tokens.append(token)

    def finish(self, reason: str):
        self.finish_reason = reason
        self.state = InferenceState.FINISHED

    def fail(self):
        self.state = InferenceState.FAILED


__all__ = ["RequestContext"]
