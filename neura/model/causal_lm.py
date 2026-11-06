"""
Neura Causal Language Model
===========================

Transformer + next-token prediction + causal language-model loss.

Inference KV cache is owned by the underlying SelfAttention layers.
Training/backward cache remains separate.
"""

from __future__ import annotations

import math

from .parameters import PARAMETERS
from .transformer import Transformer


class CausalLM:
    """Autoregressive causal language model."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        heads: int,
        num_layers: int,
        intermediate_size: int | None = None,
        context_length: int = 4096,
        rms_epsilon: float = 1e-6,
        tie_embeddings: bool = False,
    ):
        if vocab_size <= 0:
            raise ValueError(
                "vocab_size must be positive."
            )

        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive."
            )

        if heads <= 0:
            raise ValueError(
                "heads must be positive."
            )

        if num_layers <= 0:
            raise ValueError(
                "num_layers must be positive."
            )

        if hidden_size % heads != 0:
            raise ValueError(
                "hidden_size must be divisible by heads."
            )

        if context_length <= 0:
            raise ValueError(
                "context_length must be positive."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.context_length = context_length

        self.transformer = Transformer(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            heads=heads,
            num_layers=num_layers,
            intermediate_size=intermediate_size,
            context_length=context_length,
            rms_epsilon=rms_epsilon,
            tie_embeddings=tie_embeddings,
        )

        # Training/backward cache.
        #
        # This is deliberately separate from the inference KV cache
        # stored inside each SelfAttention layer.
        self._cache = []

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_input_ids(self, input_ids):
        if not isinstance(input_ids, list):
            raise TypeError(
                "input_ids must be a list."
            )

        if not input_ids:
            raise ValueError(
                "input_ids cannot be empty."
            )

        for token_id in input_ids:
            if not isinstance(token_id, int):
                raise TypeError(
                    "Token IDs must be integers."
                )

            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(
                    f"Token ID {token_id} is outside vocabulary."
                )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids,
        use_cache: bool = False,
        cache_position: int | None = None,
    ):
        """
        Forward pass.

        Args:
            input_ids:
                Token IDs.

            use_cache:
                Enable inference KV caching.

            cache_position:
                Absolute position of the first token.

                When omitted during cached inference, the current KV
                cache size is used automatically. This keeps the API
                convenient for incremental decoding while still
                allowing explicit positions.

        Returns:
            list[list[float]]
        """

        self._validate_input_ids(input_ids)

        if not isinstance(use_cache, bool):
            raise TypeError(
                "use_cache must be bool."
            )

        if cache_position is not None:
            if not isinstance(cache_position, int):
                raise TypeError(
                    "cache_position must be int or None."
                )

            if cache_position < 0:
                raise ValueError(
                    "cache_position must be non-negative."
                )

        # --------------------------------------------------------------
        # Resolve cache position
        # --------------------------------------------------------------

        if use_cache:
            if cache_position is None:
                cache_position = self.transformer.kv_cache_size()
        else:
            if cache_position is None:
                cache_position = 0

        if cache_position + len(input_ids) > self.context_length:
            raise ValueError(
                "cache position exceeds context_length."
            )

        # --------------------------------------------------------------
        # Transformer
        # --------------------------------------------------------------

        logits = self.transformer.forward(
            input_ids,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        # --------------------------------------------------------------
        # Training cache
        #
        # Keep this available even when use_cache=True. The two caches
        # have different purposes:
        #
        #   self._cache                     -> backward/training
        #   transformer SelfAttention KV    -> inference
        # --------------------------------------------------------------

        self._cache.append(
            {
                "input_ids": list(input_ids),
                "logits": [
                    list(row)
                    for row in logits
                ],
            }
        )

        return logits

    # ------------------------------------------------------------------
    # Inference KV cache
    # ------------------------------------------------------------------

    def clear_kv_cache(self):
        """
        Clear all inference KV caches.

        This does not remove the training/backward cache.
        """

        self.transformer.clear_kv_cache()

    def kv_cache_size(self):
        """
        Return the current KV-cache sequence length.

        All Transformer layers are expected to have the same length.
        """

        return self.transformer.kv_cache_size()

    # ------------------------------------------------------------------
    # Training cache
    # ------------------------------------------------------------------

    def clear_training_cache(self):
        """
        Clear training/backward cache only.
        """

        self._cache.clear()
        self.transformer.clear_training_cache()

    # ------------------------------------------------------------------
    # Complete cache reset
    # ------------------------------------------------------------------

    def clear_cache(self):
        """
        Clear both inference and training caches.

        This is the compatibility method used by inference runtimes
        and batch executors.
        """

        self.clear_kv_cache()
        self.clear_training_cache()

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    @staticmethod
    def _logsumexp(values):
        if not values:
            raise ValueError(
                "Cannot compute logsumexp of empty values."
            )

        maximum = max(values)

        total = 0.0

        for value in values:
            total += math.exp(
                value - maximum
            )

        return maximum + math.log(total)

    def loss(self, input_ids):
        """
        Next-token prediction loss.

        Example:

            input = [10, 20, 30]

            predictions:
                position 0 -> token 20
                position 1 -> token 30

        The final token has no next-token target.
        """

        if len(input_ids) < 2:
            raise ValueError(
                "CausalLM loss requires at least two tokens."
            )

        logits = self.forward(
            input_ids,
            use_cache=False,
        )

        total_loss = 0.0
        count = len(input_ids) - 1

        for position in range(count):
            target = input_ids[position + 1]
            row = logits[position]

            log_z = self._logsumexp(row)

            total_loss += (
                log_z - row[target]
            )

        return total_loss / count

    # ------------------------------------------------------------------
    # Loss backward
    # ------------------------------------------------------------------

    def backward(self, gradient=None):
        """
        Backpropagate the most recent forward pass.

        When gradient is None, preserve the original causal-LM
        next-token loss behavior.

        When gradient is supplied, it must contain one vocabulary-sized
        gradient row for every input position. This mode is used by SFT
        to preserve assistant-only masking.
        """
        if not self._cache:
            raise RuntimeError(
                "No forward pass available for backward()."
            )

        cache = self._cache.pop()

        input_ids = cache["input_ids"]
        logits = cache["logits"]

        sequence_length = len(input_ids)

        if sequence_length < 2:
            raise ValueError(
                "Backward requires at least two tokens."
            )

        if gradient is None:
            gradient = [
                [0.0 for _ in range(self.vocab_size)]
                for _ in range(sequence_length)
            ]

            scale = 1.0 / (
                sequence_length - 1
            )

            for position in range(
                sequence_length - 1
            ):
                row = logits[position]

                maximum = max(row)

                probabilities = [
                    math.exp(
                        value - maximum
                    )
                    for value in row
                ]

                total = sum(probabilities)

                for token_id in range(
                    self.vocab_size
                ):
                    probabilities[token_id] /= total

                target = input_ids[position + 1]

                for token_id in range(
                    self.vocab_size
                ):
                    gradient[position][token_id] = (
                        probabilities[token_id] * scale
                    )

                gradient[position][target] -= scale

        else:
            if len(gradient) != sequence_length:
                raise ValueError(
                    "Gradient sequence length mismatch"
                )

            for row in gradient:
                if len(row) != self.vocab_size:
                    raise ValueError(
                        "Gradient vocabulary size mismatch"
                    )

        return self.transformer.backward(
            gradient
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def parameter_count(self):
        return PARAMETERS.parameter_count()

    def cache_size(self):
        """
        Training-cache size.

        Kept for backward compatibility.
        """

        return len(self._cache)


# ----------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0

    model = CausalLM(
        vocab_size=16,
        hidden_size=8,
        heads=2,
        num_layers=2,
        intermediate_size=16,
        context_length=16,
    )

    tokens = [1, 4, 7, 3]

    logits = model.forward(tokens)

    print("Neura CausalLM")
    print("==============")
    print(
        "Vocabulary size :",
        model.vocab_size,
    )
    print(
        "Sequence length :",
        len(tokens),
    )
    print(
        "Logits shape    :",
        len(logits),
        "x",
        len(logits[0]),
    )
    print(
        "Parameters      :",
        model.parameter_count(),
    )

    value = model.loss(tokens)

    print(
        "Loss            :",
        value,
    )

    gradient = model.backward()

    print(
        "Gradient shape  :",
        len(gradient),
        "x",
        len(gradient[0]),
    )

    if not math.isfinite(value):
        raise RuntimeError(
            "Loss is not finite."
        )

    print("PASS")
