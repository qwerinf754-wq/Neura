"""
Neura Real SFT Runtime

Connects the SFT runtime to the real Neura model/training components
without modifying the inference engine.
"""

from __future__ import annotations

from .sft_runtime import SFTRuntime
from .sft_checkpoint import SFTCheckpoint


class RealSFTRuntime:

    def __init__(
        self,
        model,
        loss,
        optimizer,
    ):
        self.runtime = SFTRuntime(
            model=model,
            loss=loss,
            optimizer=optimizer,
        )

        self.checkpoint = SFTCheckpoint(
            self.runtime
        )

    @property
    def model(self):
        return self.runtime.model

    @property
    def loss(self):
        return self.runtime.loss

    @property
    def optimizer(self):
        return self.runtime.optimizer

    @property
    def step_count(self):
        return self.runtime.step_count

    @property
    def last_loss(self):
        return self.runtime.last_loss

    def train_step(self, examples):
        return self.runtime.step(
            examples
        )

    def save_checkpoint(self, path):
        return self.checkpoint.save(
            path
        )

    def load_checkpoint(self, path):
        return self.checkpoint.load(
            path
        )


__all__ = ["RealSFTRuntime"]
