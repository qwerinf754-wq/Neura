"""
Neura Supervised Fine-Tuning Core

First contract-level implementation.

This module intentionally keeps the training step small and explicit.
The existing Neura model/trainer can be integrated without changing the
Inference Engine.
"""

from __future__ import annotations

from .chat_dataset import IGNORE_INDEX


class SFTBatch:

    def __init__(
        self,
        input_ids,
        labels,
    ):
        self.input_ids = input_ids
        self.labels = labels

        if len(input_ids) != len(labels):
            raise ValueError(
                "batch input/label mismatch"
            )

    def assistant_token_count(self):
        return sum(
            1
            for row in self.labels
            for label in row
            if label != IGNORE_INDEX
        )


class SFTTrainer:

    def __init__(
        self,
        model,
        optimizer=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.step_count = 0

    def prepare_batch(
        self,
        examples,
    ):
        if not examples:
            raise ValueError(
                "examples cannot be empty"
            )

        input_ids = [
            list(example.input_ids)
            for example in examples
        ]

        labels = [
            list(example.labels)
            for example in examples
        ]

        return SFTBatch(
            input_ids=input_ids,
            labels=labels,
        )

    def train_step(self, examples):
        batch = self.prepare_batch(
            examples
        )

        if batch.assistant_token_count() == 0:
            raise ValueError(
                "batch contains no assistant training tokens"
            )

        self.step_count += 1

        return {
            "step": self.step_count,
            "assistant_tokens":
                batch.assistant_token_count(),
            "status": "prepared",
        }


__all__ = [
    "SFTBatch",
    "SFTTrainer",
]
