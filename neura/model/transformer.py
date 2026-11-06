"""
Neura Transformer
=================

Full decoder-only Transformer:

    Token Embedding
          ↓
    Transformer Blocks
          ↓
      Final RMSNorm
          ↓
        LM Head
          ↓
        Logits

This implementation is intentionally simple and numerically testable.
"""

from __future__ import annotations

import math

from .parameters import PARAMETERS
from .rms_norm import RMSNorm
from .transformer_block import TransformerBlock


class Transformer:
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
            raise ValueError("vocab_size must be positive")

        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if heads <= 0:
            raise ValueError("heads must be positive")

        if hidden_size % heads != 0:
            raise ValueError("hidden_size must be divisible by heads")

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if context_length <= 0:
            raise ValueError("context_length must be positive")

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.heads = heads
        self.num_layers = num_layers
        self.context_length = context_length
        self.intermediate_size = (
            intermediate_size
            if intermediate_size is not None
            else hidden_size * 4
        )
        self.tie_embeddings = tie_embeddings

        self.parameter_prefix = "transformer"

        # ---------------------------------------------------------
        # Token Embedding
        # ---------------------------------------------------------

        self.embedding_name = f"{self.parameter_prefix}.embedding.weight"

        embedding_scale = 0.02

        embedding = [
            [
                (2.0 * (i + j + 1) % 1009) / 1009.0 * embedding_scale
                - embedding_scale
                for j in range(hidden_size)
            ]
            for i in range(vocab_size)
        ]

        PARAMETERS.add(
            self.embedding_name,
            embedding,
        )

        # ---------------------------------------------------------
        # Transformer Blocks
        # ---------------------------------------------------------

        self.blocks = [
            TransformerBlock(
                hidden_size=hidden_size,
                heads=heads,
                layer_id=layer_id,
                intermediate_size=self.intermediate_size,
                rms_epsilon=rms_epsilon,
            )
            for layer_id in range(num_layers)
        ]

        # ---------------------------------------------------------
        # Final RMSNorm
        # ---------------------------------------------------------

        self.final_norm_name = (
            f"{self.parameter_prefix}.final_norm.weight"
        )

        self.final_norm = RMSNorm(
            hidden_size,
            epsilon=rms_epsilon,
            parameter_name=self.final_norm_name,
        )

        # ---------------------------------------------------------
        # LM Head
        # ---------------------------------------------------------

        self.lm_head_name = f"{self.parameter_prefix}.lm_head.weight"

        if tie_embeddings:
            self.lm_head = None
        else:
            lm_scale = 0.02

            lm_head = [
                [
                    (
                        ((i * 17 + j * 31 + 7) % 1009)
                        / 1009.0
                        * 2.0
                        - 1.0
                    ) * lm_scale
                    for j in range(hidden_size)
                ]
                for i in range(vocab_size)
            ]

            PARAMETERS.add(
                self.lm_head_name,
                lm_head,
            )

        self._cache = []

    # ============================================================
    # Helpers
    # ============================================================

    def _validate_tokens(self, token_ids):
        if len(token_ids) > self.context_length:
            raise ValueError(
                "Sequence length exceeds context_length"
            )

        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError("token_ids must contain integers")

            if token_id < 0 or token_id >= self.vocab_size:
                raise ValueError(
                    f"Token id {token_id} outside vocabulary"
                )

    def _matvec(self, vector, matrix):
        output = []

        for row in matrix:
            total = 0.0

            for i in range(len(vector)):
                total += vector[i] * row[i]

            output.append(total)

        return output

    def _transpose_matvec(self, vector, matrix):
        if not matrix:
            return []

        input_size = len(matrix[0])

        output = [0.0 for _ in range(input_size)]

        for i in range(len(matrix)):
            value = vector[i]

            for j in range(input_size):
                output[j] += value * matrix[i][j]

        return output

    def _outer_add(self, matrix, vector_a, vector_b):
        for i in range(len(vector_b)):
            for j in range(len(vector_a)):
                matrix[i][j] += vector_b[i] * vector_a[j]

    def _zeros(self, rows, cols):
        return [
            [0.0 for _ in range(cols)]
            for _ in range(rows)
        ]

    # ============================================================
    # Embedding
    # ============================================================

    def embed(self, token_ids):
        embedding = PARAMETERS.require(
            self.embedding_name
        )

        return [
            list(embedding[token_id])
            for token_id in token_ids
        ]

    # ============================================================
    # LM Head
    # ============================================================

    def project_logits(self, hidden):
        if self.tie_embeddings:
            weight = PARAMETERS.require(
                self.embedding_name
            )
        else:
            weight = PARAMETERS.require(
                self.lm_head_name
            )

        return [
            self._matvec(vector, weight)
            for vector in hidden
        ]

    # ============================================================

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        token_ids,
        use_cache=False,
        cache_position=0,
    ):
        """
        Forward pass through the decoder-only Transformer.

        Args:
            token_ids:
                Input token IDs.

            use_cache:
                Enable inference KV cache.

            cache_position:
                Absolute position of the first input token.

        Returns:
            list[list[float]]
        """

        self._validate_tokens(token_ids)

        if len(token_ids) == 0:
            raise ValueError("token_ids cannot be empty")

        if not isinstance(use_cache, bool):
            raise TypeError("use_cache must be bool")

        if not isinstance(cache_position, int):
            raise TypeError("cache_position must be int")

        if cache_position < 0:
            raise ValueError(
                "cache_position must be non-negative"
            )

        if cache_position + len(token_ids) > self.context_length:
            raise ValueError(
                "cache position exceeds context_length"
            )

        hidden = self.embed(token_ids)

        for block in self.blocks:
            hidden = block.forward(
                hidden,
                use_cache=use_cache,
                cache_position=cache_position,
            )

        normalized = [
            self.final_norm.forward(vector)
            for vector in hidden
        ]

        logits = self.project_logits(normalized)

        self._cache.append(
            {
                "token_ids": list(token_ids),
                "hidden": [list(x) for x in hidden],
                "normalized": [list(x) for x in normalized],
                "logits": [list(x) for x in logits],
            }
        )

        return logits

    # ============================================================
    # KV Cache
    # ============================================================

    def clear_kv_cache(self):
        """
        Clear inference KV caches from every Transformer layer.

        Training/backward caches are intentionally preserved.
        """

        for block in self.blocks:
            block.attention.clear_kv_cache()

    def kv_cache_size(self):
        """
        Return the current KV-cache sequence length.

        Every Transformer layer must have the same cache length.
        """

        if not self.blocks:
            return 0

        sizes = [
            block.attention.kv_cache_size()
            for block in self.blocks
        ]

        first = sizes[0]

        for size in sizes[1:]:
            if size != first:
                raise RuntimeError(
                    "Transformer KV cache layers are out of sync."
                )

        return first

    # ============================================================
    # Training Cache
    # ============================================================

    def clear_training_cache(self):
        """
        Clear forward caches required by backward().
        """

        self._cache.clear()

        for block in self.blocks:
            block.clear_cache()

    # ============================================================
    # Complete Cache Reset
    # ============================================================

    def clear_cache(self):
        """
        Clear both training and inference caches.
        """

        self.clear_training_cache()
        self.clear_kv_cache()

    # ============================================================
    # Backward
    # ============================================================

    def backward(self, gradient):
        if not self._cache:
            raise RuntimeError(
                "Transformer.backward() called without forward()"
            )

        cache = self._cache.pop()

        token_ids = cache["token_ids"]
        normalized = cache["normalized"]

        seq_len = len(token_ids)

        if len(gradient) != seq_len:
            raise ValueError(
                "Gradient sequence length mismatch"
            )

        if self.tie_embeddings:
            lm_weight = PARAMETERS.require(
                self.embedding_name
            )
        else:
            lm_weight = PARAMETERS.require(
                self.lm_head_name
            )

        grad_normalized = [
            [0.0 for _ in range(self.hidden_size)]
            for _ in range(seq_len)
        ]

        if not self.tie_embeddings:
            grad_lm_head = self._zeros(
                self.vocab_size,
                self.hidden_size,
            )

        for t in range(seq_len):
            grad_logits = gradient[t]

            if len(grad_logits) != self.vocab_size:
                raise ValueError(
                    "Gradient vocabulary size mismatch"
                )

            grad_normalized[t] = self._transpose_matvec(
                grad_logits,
                lm_weight,
            )

            if not self.tie_embeddings:
                self._outer_add(
                    grad_lm_head,
                    normalized[t],
                    grad_logits,
                )

        if not self.tie_embeddings:
            PARAMETERS.add_gradient(
                self.lm_head_name,
                grad_lm_head,
            )

        # Final RMSNorm.
        #
        # RMSNorm keeps a LIFO cache, therefore backward
        # must traverse the sequence in reverse order.
        grad_hidden = [
            [0.0 for _ in range(self.hidden_size)]
            for _ in range(seq_len)
        ]

        for t in range(seq_len - 1, -1, -1):
            grad_hidden[t] = self.final_norm.backward(
                grad_normalized[t]
            )

        # Transformer blocks are also traversed in reverse.
        for block in reversed(self.blocks):
            grad_hidden = block.backward(
                grad_hidden
            )

        # Embedding gradient.
        grad_embedding = self._zeros(
            self.vocab_size,
            self.hidden_size,
        )

        for t, token_id in enumerate(token_ids):
            for d in range(self.hidden_size):
                grad_embedding[token_id][d] += grad_hidden[t][d]

        PARAMETERS.add_gradient(
            self.embedding_name,
            grad_embedding,
        )

        return grad_hidden

    # ============================================================
    # Information
    # ============================================================

    def parameter_names(self):
        names = [
            self.embedding_name,
        ]

        for block in self.blocks:
            names.extend(block.parameter_names())

        names.append(self.final_norm_name)

        if not self.tie_embeddings:
            names.append(self.lm_head_name)

        return names

    def parameter_count(self):
        total = 0

        for name in self.parameter_names():
            value = PARAMETERS.require(name)

            if isinstance(value, list):
                if value and isinstance(value[0], list):
                    total += len(value) * len(value[0])
                else:
                    total += len(value)

        return total


# ================================================================
# Self Test
# ================================================================

if __name__ == "__main__":
    print("Neura Transformer")
    print("=================")

    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0

    model = Transformer(
        vocab_size=32,
        hidden_size=8,
        heads=2,
        num_layers=2,
        intermediate_size=16,
        context_length=16,
    )

    tokens = [3, 7, 2]

    logits = model.forward(tokens)

    print(f"Vocabulary size : {model.vocab_size}")
    print(f"Sequence length : {len(tokens)}")
    print(f"Hidden size     : {model.hidden_size}")
    print(f"Layers          : {model.num_layers}")
    print(
        f"Logits shape    : "
        f"{len(logits)} x {len(logits[0])}"
    )

    gradient = [
        [1.0 for _ in row]
        for row in logits
    ]

    input_gradient = model.backward(gradient)

    print(
        f"Gradient shape  : "
        f"{len(input_gradient)} x "
        f"{len(input_gradient[0])}"
    )

    print(f"Parameters      : {model.parameter_count()}")

    if len(logits) != len(tokens):
        raise RuntimeError(
            "Invalid logits sequence length"
        )

    if len(logits[0]) != model.vocab_size:
        raise RuntimeError(
            "Invalid vocabulary projection"
        )

    if len(input_gradient) != len(tokens):
        raise RuntimeError(
            "Invalid input gradient length"
        )

    print("PASS")
