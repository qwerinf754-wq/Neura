"""
Neura SFT Runtime

Execution contract:

ChatSFTDataset
    -> SFT batch
    -> model forward
    -> masked cross-entropy
    -> loss gradients
    -> model backward
    -> optimizer step
    -> runtime statistics

The runtime preserves the existing Neura training APIs while
making the forward/loss/backward contract explicit.
"""

from __future__ import annotations

from .chat_dataset import IGNORE_INDEX
from .sft import SFTBatch


class SFTRuntime:

    def __init__(
        self,
        model,
        loss,
        optimizer,
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer

        self.step_count = 0
        self.last_loss = None
        self.last_gradients = None

    # ---------------------------------------------------------
    # Batch preparation
    # ---------------------------------------------------------

    def prepare_batch(self, examples):

        if not examples:
            raise ValueError(
                "examples cannot be empty"
            )

        batch = SFTBatch(
            input_ids=[
                list(example.input_ids)
                for example in examples
            ],
            labels=[
                list(example.labels)
                for example in examples
            ],
        )

        if batch.assistant_token_count() == 0:
            raise ValueError(
                "batch contains no assistant training tokens"
            )

        return batch

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(self, batch):

        logits = []

        for sequence in batch.input_ids:

            sequence_logits = self.model.forward(
                sequence,
                use_cache=False,
            )

            logits.append(
                sequence_logits
            )

        return logits

    # ---------------------------------------------------------
    # Masked loss
    # ---------------------------------------------------------

    def compute_loss(
        self,
        logits,
        labels,
    ):
        filtered_logits = []
        filtered_labels = []

        for row_logits, row_labels in zip(
            logits,
            labels,
        ):

            if len(row_logits) != len(row_labels):
                raise ValueError(
                    "logits and labels sequence length mismatch"
                )

            for token_logits, label in zip(
                row_logits,
                row_labels,
            ):

                if label == IGNORE_INDEX:
                    continue

                filtered_logits.append(
                    token_logits
                )

                filtered_labels.append(
                    label
                )

        if not filtered_labels:
            raise ValueError(
                "no valid assistant targets"
            )

        if hasattr(self.loss, "forward"):

            return self.loss.forward(
                filtered_logits,
                filtered_labels,
            )

        if callable(self.loss):

            return self.loss(
                filtered_logits,
                filtered_labels,
            )

        raise TypeError(
            "loss must be callable or expose forward()"
        )

    # ---------------------------------------------------------
    # Gradient extraction
    # ---------------------------------------------------------

    def gradients(self):

        if hasattr(self.loss, "gradients"):

            gradients = self.loss.gradients()

            if gradients is None:
                raise RuntimeError(
                    "loss produced no gradients"
                )

            self.last_gradients = gradients

            return gradients

        raise TypeError(
            "loss must expose gradients()"
        )

    # ---------------------------------------------------------
    # Training step
    # ---------------------------------------------------------

    def step(self, examples):
        """
        Execute exactly one SFT optimizer step.

        Flow:

            examples
              -> prepare_batch
              -> forward
              -> assistant-only loss
              -> compressed loss gradients
              -> reconstruct full sequence gradients
              -> CausalLM.backward()
              -> accumulate parameter gradients
              -> exactly one optimizer.step(PARAMETERS)

        The model forward cache is LIFO, therefore examples are
        backpropagated in reverse order.
        """
        # Start every optimization step with clean gradients.
        # Gradients may accumulate across examples within this step,
        # but must never leak from the previous optimizer step.
        from neura.model.parameters import PARAMETERS
        self.optimizer.zero_grad(PARAMETERS)

        batch = self.prepare_batch(examples)

        logits = self.forward(batch)

        loss_value = self.compute_loss(
            logits,
            batch.labels,
        )

        compressed_gradients = self.gradients()

        if not compressed_gradients:
            raise RuntimeError(
                "loss gradients cannot be empty"
            )

        # --------------------------------------------------------
        # Reconstruct one full gradient sequence per example.
        #
        # CrossEntropyLoss receives only assistant rows, so its
        # gradient output is compressed. CausalLM.backward()
        # requires one vocabulary-sized row for every sequence
        # position.
        # --------------------------------------------------------

        full_gradients = []
        gradient_index = 0

        for sequence_logits, sequence_labels in zip(
            logits,
            batch.labels,
        ):
            if len(sequence_logits) != len(sequence_labels):
                raise ValueError(
                    "logits and labels sequence length mismatch"
                )

            if not sequence_logits:
                raise ValueError(
                    "empty sequence is not valid for SFT backward"
                )

            vocab_size = len(sequence_logits[0])

            sequence_gradient = []

            for token_logits, label in zip(
                sequence_logits,
                sequence_labels,
            ):
                if len(token_logits) != vocab_size:
                    raise ValueError(
                        "inconsistent vocabulary width in logits"
                    )

                if label == IGNORE_INDEX:
                    sequence_gradient.append(
                        [0.0 for _ in range(vocab_size)]
                    )
                    continue

                if gradient_index >= len(compressed_gradients):
                    raise RuntimeError(
                        "compressed gradient count is smaller "
                        "than the number of assistant targets"
                    )

                gradient_row = compressed_gradients[
                    gradient_index
                ]

                if len(gradient_row) != vocab_size:
                    raise ValueError(
                        "loss gradient vocabulary width mismatch"
                    )

                sequence_gradient.append(
                    list(gradient_row)
                )

                gradient_index += 1

            full_gradients.append(sequence_gradient)

        if gradient_index != len(compressed_gradients):
            raise RuntimeError(
                "unused compressed loss gradients remain: "
                f"{len(compressed_gradients) - gradient_index}"
            )

        # --------------------------------------------------------
        # CausalLM forward() pushes caches in example order.
        # CausalLM.backward() pops caches in reverse order.
        # --------------------------------------------------------

        if not hasattr(self.model, "backward"):
            raise TypeError(
                "model must expose backward()"
            )

        for sequence_gradient in reversed(full_gradients):
            self.model.backward(
                sequence_gradient
            )

        # --------------------------------------------------------
        # Exactly ONE optimizer update for the whole batch.
        # AdamW expects the PARAMETERS registry, not the gradient
        # list.
        # --------------------------------------------------------

        if self.optimizer is None:
            raise TypeError(
                "optimizer cannot be None for a training step"
            )

        if not hasattr(self.optimizer, "step"):
            raise TypeError(
                "optimizer must expose step()"
            )

        from neura.model.parameters import PARAMETERS

        self.optimizer.step(PARAMETERS)

        self.step_count += 1
        self.last_loss = loss_value

        return {
            "step": self.step_count,
            "loss": loss_value,
            "assistant_tokens": (
                batch.assistant_token_count()
            ),
            "gradient_count": len(compressed_gradients),
            "status": "trained",
        }


__all__ = ["SFTRuntime"]
