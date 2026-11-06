"""
Neura Inference Runtime State

Explicit lifecycle state for one inference request.
"""

from enum import Enum


class InferenceState(str, Enum):
    CREATED = "created"
    PREFILL = "prefill"
    DECODING = "decoding"
    FINISHED = "finished"
    FAILED = "failed"


__all__ = ["InferenceState"]
