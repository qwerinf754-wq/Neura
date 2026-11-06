"""
Neura SFT Training Loop

Multi-batch supervised fine-tuning runtime with
checkpoint/resume bookkeeping.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class SFTTrainingState:
    epoch: int = 0
    step: int = 0
    optimizer_steps: int = 0
    examples_seen: int = 0
    loss_sum: float = 0.0

    @property
    def mean_loss(self):
        if self.step == 0:
            return 0.0
        return self.loss_sum / self.step


class SFTTrainer:

    def __init__(
        self,
        runtime,
        optimizer,
        checkpoint_manager=None,
    ):
        self.runtime = runtime
        self.optimizer = optimizer
        self.checkpoint_manager = checkpoint_manager
        self.state = SFTTrainingState()

    def train_batch(self, batch):
        """
        Train exactly one SFT batch through SFTRuntime.

        SFTRuntime owns the complete optimization step:

            forward
              -> masked loss
              -> backward
              -> gradient accumulation
              -> optimizer.step(PARAMETERS)

        The trainer only updates training-state bookkeeping.
        """

        result = self.runtime.step(batch)

        loss = float(result["loss"])

        self.state.step += 1
        self.state.optimizer_steps += 1

        # Prefer explicit example counting when the runtime input
        # is a normal iterable of SFT examples.
        try:
            examples_seen = len(batch)
        except TypeError:
            examples_seen = 1

        self.state.examples_seen += examples_seen
        self.state.loss_sum += loss

        return {
            "loss": loss,
            "step": self.state.step,
            "optimizer_steps": (
                self.state.optimizer_steps
            ),
        }

    def train_epoch(self, batches):
        self.state.epoch += 1

        results = []

        for batch in batches:
            results.append(
                self.train_batch(batch)
            )

        return results

    def checkpoint_state(self):
        return {
            "epoch": self.state.epoch,
            "step": self.state.step,
            "optimizer_steps": (
                self.state.optimizer_steps
            ),
            "examples_seen": (
                self.state.examples_seen
            ),
            "loss_sum": self.state.loss_sum,
        }

    def resume(self, state):
        required = {
            "epoch",
            "step",
            "optimizer_steps",
            "examples_seen",
            "loss_sum",
        }

        missing = required - set(state)

        if missing:
            raise ValueError(
                f"Invalid SFT state; missing: {sorted(missing)}"
            )

        self.state = SFTTrainingState(
            epoch=int(state["epoch"]),
            step=int(state["step"]),
            optimizer_steps=int(
                state["optimizer_steps"]
            ),
            examples_seen=int(
                state["examples_seen"]
            ),
            loss_sum=float(
                state["loss_sum"]
            ),
        )


__all__ = [
    "SFTTrainingState",
    "SFTTrainer",
]
