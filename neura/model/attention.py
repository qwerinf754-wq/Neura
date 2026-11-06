"""
Neura Multi-Head Self-Attention
===============================

Reference implementation of causal multi-head self-attention.

Pipeline:

    Input
      -> Q/K/V projections
      -> split heads
      -> scaled dot-product attention
      -> causal mask
      -> softmax
      -> attention @ V
      -> merge heads
      -> output projection

This is a correctness-first reference backend.
"""

from __future__ import annotations

import math
import random
from .parameters import PARAMETERS
from .rope import RoPE


class SelfAttention:
    """Causal multi-head self-attention."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        layer_id: int = 0,
    ):
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")

        if heads <= 0:
            raise ValueError("heads must be positive.")

        if hidden_size % heads != 0:
            raise ValueError(
                "hidden_size must be divisible by heads."
            )

        self.hidden_size = hidden_size
        self.heads = heads
        self.layer_id = layer_id
        self.head_dim = hidden_size // heads

        # Rotary Position Embedding.
        # RoPE is applied to Q and K only.
        self.rope = RoPE(
            head_dim=self.head_dim,
            max_length=4096,
        )

        self.parameter_prefix = (
            f"layers.{layer_id}.attention"
        )

        self.Wq = self._random_matrix(
            hidden_size,
            hidden_size,
        )
        self.Wk = self._random_matrix(
            hidden_size,
            hidden_size,
        )
        self.Wv = self._random_matrix(
            hidden_size,
            hidden_size,
        )
        self.Wo = self._random_matrix(
            hidden_size,
            hidden_size,
        )

        PARAMETERS.add(
            f"{self.parameter_prefix}.Wq",
            self.Wq,
        )
        PARAMETERS.add(
            f"{self.parameter_prefix}.Wk",
            self.Wk,
        )
        PARAMETERS.add(
            f"{self.parameter_prefix}.Wv",
            self.Wv,
        )
        PARAMETERS.add(
            f"{self.parameter_prefix}.Wo",
            self.Wo,
        )

        # Training cache used exclusively by backward().
        self._cache = []

        # Inference KV cache.
        # Shape:
        #   keys[head][position][dimension]
        #   values[head][position][dimension]
        self._kv_cache = {
            "keys": [],
            "values": [],
        }

    # ---------------------------------------------------------
    # Initialization
    # ---------------------------------------------------------

    @staticmethod
    def _random_matrix(rows: int, cols: int):
        scale = 0.02

        return [
            [
                random.uniform(-scale, scale)
                for _ in range(cols)
            ]
            for _ in range(rows)
        ]

    # ---------------------------------------------------------
    # Basic matrix operations
    # ---------------------------------------------------------

    @staticmethod
    def _matvec(
        vector,
        matrix,
    ):
        if len(vector) != len(matrix):
            raise ValueError(
                "Vector/matrix dimension mismatch."
            )

        output = []

        for column in range(len(matrix[0])):
            value = 0.0

            for row in range(len(vector)):
                value += vector[row] * matrix[row][column]

            output.append(value)

        return output

    @staticmethod
    def _matvec_transpose(
        vector,
        matrix,
    ):
        output = []

        for row in range(len(matrix)):
            value = 0.0

            for column in range(len(vector)):
                value += vector[column] * matrix[row][column]

            output.append(value)

        return output

    # ---------------------------------------------------------
    # Head operations
    # ---------------------------------------------------------

    def split_heads(self, sequence):
        """
        [sequence, hidden]
            ->
        [heads, sequence, head_dim]
        """

        sequence_length = len(sequence)

        heads = [
            [
                [0.0] * self.head_dim
                for _ in range(sequence_length)
            ]
            for _ in range(self.heads)
        ]

        for position, vector in enumerate(sequence):
            if len(vector) != self.hidden_size:
                raise ValueError(
                    "Input vector has invalid hidden size."
                )

            for head in range(self.heads):
                start = head * self.head_dim
                end = start + self.head_dim

                heads[head][position] = list(
                    vector[start:end]
                )

        return heads

    def merge_heads(self, heads):
        """
        [heads, sequence, head_dim]
            ->
        [sequence, hidden]
        """

        if len(heads) != self.heads:
            raise ValueError(
                "Invalid number of attention heads."
            )

        if not heads:
            return []

        sequence_length = len(heads[0])

        output = []

        for position in range(sequence_length):
            vector = []

            for head in range(self.heads):
                vector.extend(
                    heads[head][position]
                )

            if len(vector) != self.hidden_size:
                raise ValueError(
                    "Merged vector has invalid hidden size."
                )

            output.append(vector)

        return output

    def apply_rope_to_heads(self, heads, positions=None):
        """Apply RoPE using explicit absolute positions."""

        if not heads:
            return []

        sequence_length = len(heads[0])

        if positions is None:
            positions = list(range(sequence_length))

        if len(positions) != sequence_length:
            raise ValueError(
                "RoPE position count must match sequence length."
            )

        output = []

        for head in range(self.heads):
            rotated_head = []

            for index, vector in enumerate(heads[head]):
                rotated_head.append(
                    self.rope.apply(
                        vector,
                        positions[index],
                    )
                )

            output.append(rotated_head)

        return output

    # ---------------------------------------------------------
    # Softmax
    # ---------------------------------------------------------

    @staticmethod
    def softmax(values):
        if not values:
            return []

        maximum = max(values)

        exponentials = [
            math.exp(value - maximum)
            for value in values
        ]

        total = sum(exponentials)

        if total == 0.0:
            raise FloatingPointError(
                "Softmax normalization produced zero."
            )

        return [
            value / total
            for value in exponentials
        ]

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(
        self,
        sequence,
        use_cache=False,
        cache_position=0,
    ):
        """
        Forward causal self-attention.

        use_cache=False:
            Normal full-sequence attention.

        use_cache=True, cache_position=0:
            Prefill. K/V are stored.

        use_cache=True, cache_position=N:
            Incremental decode. Existing K/V are reused.
        """

        if not sequence:
            return []

        if not isinstance(use_cache, bool):
            raise TypeError("use_cache must be bool")

        if not isinstance(cache_position, int):
            raise TypeError("cache_position must be int")

        if cache_position < 0:
            raise ValueError("cache_position must be non-negative")

        sequence_length = len(sequence)

        for vector in sequence:
            if len(vector) != self.hidden_size:
                raise ValueError(
                    "Input vector has invalid hidden size."
                )

        if cache_position + sequence_length > self.rope.max_length:
            raise ValueError(
                "Sequence exceeds RoPE maximum length."
            )

        # -----------------------------------------------------
        # Q / K / V projections
        # -----------------------------------------------------

        queries = [
            self._matvec(vector, self.Wq)
            for vector in sequence
        ]

        keys = [
            self._matvec(vector, self.Wk)
            for vector in sequence
        ]

        values = [
            self._matvec(vector, self.Wv)
            for vector in sequence
        ]

        # -----------------------------------------------------
        # Split heads
        # -----------------------------------------------------

        query_heads = self.split_heads(queries)
        key_heads = self.split_heads(keys)
        value_heads = self.split_heads(values)

        # -----------------------------------------------------
        # RoPE
        # -----------------------------------------------------

        positions = [
            cache_position + i
            for i in range(sequence_length)
        ]

        query_heads = self.apply_rope_to_heads(
            query_heads,
            positions,
        )

        key_heads = self.apply_rope_to_heads(
            key_heads,
            positions,
        )

        # -----------------------------------------------------
        # KV cache
        # -----------------------------------------------------

        if use_cache:

            if cache_position == 0:

                self._kv_cache = {
                    "keys": [
                        [
                            list(vector)
                            for vector in key_heads[h]
                        ]
                        for h in range(self.heads)
                    ],
                    "values": [
                        [
                            list(vector)
                            for vector in value_heads[h]
                        ]
                        for h in range(self.heads)
                    ],
                }

            else:

                if not self._kv_cache["keys"]:
                    raise RuntimeError(
                        "Cannot decode without an existing KV cache."
                    )

                current_size = len(
                    self._kv_cache["keys"][0]
                )

                if cache_position != current_size:
                    raise ValueError(
                        "cache_position does not match "
                        "the current KV cache length."
                    )

                for h in range(self.heads):

                    self._kv_cache["keys"][h].extend(
                        [
                            list(vector)
                            for vector in key_heads[h]
                        ]
                    )

                    self._kv_cache["values"][h].extend(
                        [
                            list(vector)
                            for vector in value_heads[h]
                        ]
                    )

        # -----------------------------------------------------
        # Attention
        # -----------------------------------------------------

        attention_heads = []
        all_scores = []
        all_weights = []

        scale = math.sqrt(self.head_dim)

        for head in range(self.heads):

            head_output = []
            head_scores = []
            head_weights = []

            if use_cache:
                available_keys = self._kv_cache["keys"][head]
                available_values = self._kv_cache["values"][head]
            else:
                available_keys = key_heads[head]
                available_values = value_heads[head]

            for local_position in range(sequence_length):

                query = query_heads[head][local_position]

                absolute_position = (
                    cache_position + local_position
                )

                scores = []

                # Only positions up to the current token
                # are visible.
                visible_count = absolute_position + 1

                for key_position in range(visible_count):

                    key = available_keys[key_position]

                    score = sum(
                        q * k
                        for q, k in zip(query, key)
                    )

                    score /= scale

                    scores.append(score)

                weights = self.softmax(scores)

                output = [0.0] * self.head_dim

                for key_position in range(visible_count):

                    value = available_values[key_position]

                    weight = weights[key_position]

                    for dimension in range(self.head_dim):

                        output[dimension] += (
                            weight
                            * value[dimension]
                        )

                head_scores.append(scores)
                head_weights.append(weights)
                head_output.append(output)

            attention_heads.append(head_output)
            all_scores.append(head_scores)
            all_weights.append(head_weights)

        # -----------------------------------------------------
        # Merge heads
        # -----------------------------------------------------

        merged = self.merge_heads(
            attention_heads
        )

        # -----------------------------------------------------
        # Output projection
        # -----------------------------------------------------

        output = [
            self._matvec(vector, self.Wo)
            for vector in merged
        ]

        # -----------------------------------------------------
        # Training cache
        #
        # IMPORTANT:
        # This remains separate from the inference KV cache.
        # backward() continues to consume this cache.
        # -----------------------------------------------------

        self._cache.append(
            {
                "input": [
                    list(vector)
                    for vector in sequence
                ],
                "queries": queries,
                "keys": keys,
                "values": values,
                "query_heads": query_heads,
                "key_heads": key_heads,
                "value_heads": value_heads,
                "attention_heads": attention_heads,
                "merged": merged,
                "scores": all_scores,
                "weights": all_weights,
            }
        )

        return output

    # ---------------------------------------------------------
    # Cache utilities
    # ---------------------------------------------------------

    def clear_cache(self):
        self._cache.clear()

    def cache_size(self):
        return len(self._cache)

    def clear_kv_cache(self):
        """Clear inference key/value cache."""
        self._kv_cache = {
            "keys": [],
            "values": [],
        }

    def kv_cache_size(self):
        """Return the number of cached sequence positions."""
        if not self._kv_cache["keys"]:
            return 0

        return len(self._kv_cache["keys"][0])

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    @staticmethod
    def _zeros(rows, cols):
        return [[0.0 for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def _transpose_matvec(vector, matrix):
        if not matrix:
            return []

        if len(vector) != len(matrix[0]):
            raise ValueError(
                "Vector/matrix dimension mismatch."
            )

        output = [0.0] * len(matrix)

        for row in range(len(matrix)):
            value = 0.0

            for column in range(len(vector)):
                value += (
                    vector[column]
                    * matrix[row][column]
                )

            output[row] = value

        return output

    @staticmethod
    def _outer_add(target, left, right):
        for i in range(len(left)):
            for j in range(len(right)):
                target[i][j] += left[i] * right[j]

    @staticmethod
    def _softmax_backward(gradient, weights):
        dot = sum(
            g * w
            for g, w in zip(gradient, weights)
        )

        return [
            w * (g - dot)
            for g, w in zip(gradient, weights)
        ]

    def inverse_rope_to_heads(self, heads):
        """Apply the transpose/inverse RoPE transformation to gradients."""
        output = []

        if not self.rope._cache:
            self.rope.build_cache()

        for head in range(self.heads):
            restored_head = []

            for position, vector in enumerate(heads[head]):
                cos_row = self.rope._cache["cos"][position]
                sin_row = self.rope._cache["sin"][position]

                restored = [0.0] * self.head_dim

                for i in range(0, self.head_dim, 2):
                    pair_index = i // 2

                    cos = cos_row[pair_index]
                    sin = sin_row[pair_index]

                    g1 = vector[i]
                    g2 = vector[i + 1]

                    # Transpose of:
                    # [ cos -sin ]
                    # [ sin  cos ]
                    restored[i] = g1 * cos + g2 * sin
                    restored[i + 1] = -g1 * sin + g2 * cos

                restored_head.append(restored)

            output.append(restored_head)

        return output

    def backward(self, gradient):
        if not self._cache:
            raise RuntimeError(
                "SelfAttention.backward() called before forward()."
            )

        cache = self._cache.pop()

        sequence = cache["input"]
        query_heads = cache["query_heads"]
        key_heads = cache["key_heads"]
        value_heads = cache["value_heads"]
        attention_heads = cache["attention_heads"]
        weights = cache["weights"]
        merged = cache["merged"]

        seq_len = len(sequence)
        hidden_size = self.hidden_size
        heads = self.heads
        head_dim = self.head_dim

        if len(gradient) != seq_len:
            raise ValueError(
                "Gradient sequence length mismatch."
            )

        for row in gradient:
            if len(row) != hidden_size:
                raise ValueError(
                    "Gradient hidden size mismatch."
                )

        # Output projection
        grad_merged = self._zeros(
            seq_len,
            hidden_size,
        )

        grad_Wo = self._zeros(
            hidden_size,
            hidden_size,
        )

        for t in range(seq_len):
            self._outer_add(
                grad_Wo,
                merged[t],
                gradient[t],
            )

            grad_merged[t] = self._transpose_matvec(
                gradient[t],
                self.Wo,
            )

        PARAMETERS.add_gradient(
            f"{self.parameter_prefix}.Wo",
            grad_Wo,
        )

        # Split gradient into heads
        grad_attention_heads = [
            [
                [0.0 for _ in range(head_dim)]
                for _ in range(seq_len)
            ]
            for _ in range(heads)
        ]

        for t in range(seq_len):
            offset = 0

            for h in range(heads):
                grad_attention_heads[h][t] = (
                    grad_merged[t][offset:offset + head_dim]
                )

                offset += head_dim

        # Head gradients
        grad_query_heads = [
            [
                [0.0 for _ in range(head_dim)]
                for _ in range(seq_len)
            ]
            for _ in range(heads)
        ]

        grad_key_heads = [
            [
                [0.0 for _ in range(head_dim)]
                for _ in range(seq_len)
            ]
            for _ in range(heads)
        ]

        grad_value_heads = [
            [
                [0.0 for _ in range(head_dim)]
                for _ in range(seq_len)
            ]
            for _ in range(heads)
        ]

        scale = math.sqrt(head_dim)

        for h in range(heads):
            for i in range(seq_len):
                raw_grad_weights = [0.0] * (i + 1)

                # dV and dWeights
                for j in range(i + 1):
                    raw_grad_weights[j] = sum(
                        grad_attention_heads[h][i][d]
                        * value_heads[h][j][d]
                        for d in range(head_dim)
                    )

                    coefficient = weights[h][i][j]

                    for d in range(head_dim):
                        grad_value_heads[h][j][d] += (
                            coefficient
                            * grad_attention_heads[h][i][d]
                        )

                # Softmax backward
                softmax_gradient = self._softmax_backward(
                    raw_grad_weights,
                    weights[h][i][:i + 1],
                )

                # Attention score backward
                for j in range(i + 1):
                    score_gradient = (
                        softmax_gradient[j] / scale
                    )

                    for d in range(head_dim):
                        grad_query_heads[h][i][d] += (
                            score_gradient
                            * key_heads[h][j][d]
                        )

                        grad_key_heads[h][j][d] += (
                            score_gradient
                            * query_heads[h][i][d]
                        )

        # Undo RoPE on Q/K gradients before merging heads.
        #
        # Forward:
        #     Q_rot = RoPE(Q)
        #     K_rot = RoPE(K)
        #
        # Backward:
        #     dQ = RoPE^T(dQ_rot)
        #     dK = RoPE^T(dK_rot)
        grad_query_heads = self.inverse_rope_to_heads(
            grad_query_heads
        )

        grad_key_heads = self.inverse_rope_to_heads(
            grad_key_heads
        )

        # Merge head gradients
        grad_queries = [
            [0.0 for _ in range(hidden_size)]
            for _ in range(seq_len)
        ]

        grad_keys = [
            [0.0 for _ in range(hidden_size)]
            for _ in range(seq_len)
        ]

        grad_values = [
            [0.0 for _ in range(hidden_size)]
            for _ in range(seq_len)
        ]

        for t in range(seq_len):
            offset = 0

            for h in range(heads):
                for d in range(head_dim):
                    grad_queries[t][offset + d] = (
                        grad_query_heads[h][t][d]
                    )

                    grad_keys[t][offset + d] = (
                        grad_key_heads[h][t][d]
                    )

                    grad_values[t][offset + d] = (
                        grad_value_heads[h][t][d]
                    )

                offset += head_dim

        # Projection gradients
        grad_Wq = self._zeros(
            hidden_size,
            hidden_size,
        )

        grad_Wk = self._zeros(
            hidden_size,
            hidden_size,
        )

        grad_Wv = self._zeros(
            hidden_size,
            hidden_size,
        )

        for t in range(seq_len):
            self._outer_add(
                grad_Wq,
                sequence[t],
                grad_queries[t],
            )

            self._outer_add(
                grad_Wk,
                sequence[t],
                grad_keys[t],
            )

            self._outer_add(
                grad_Wv,
                sequence[t],
                grad_values[t],
            )

        PARAMETERS.add_gradient(
            f"{self.parameter_prefix}.Wq",
            grad_Wq,
        )

        PARAMETERS.add_gradient(
            f"{self.parameter_prefix}.Wk",
            grad_Wk,
        )

        PARAMETERS.add_gradient(
            f"{self.parameter_prefix}.Wv",
            grad_Wv,
        )

        # Input gradient
        grad_input = [
            [0.0 for _ in range(hidden_size)]
            for _ in range(seq_len)
        ]

        for t in range(seq_len):
            q_grad = self._transpose_matvec(
                grad_queries[t],
                self.Wq,
            )

            k_grad = self._transpose_matvec(
                grad_keys[t],
                self.Wk,
            )

            v_grad = self._transpose_matvec(
                grad_values[t],
                self.Wv,
            )

            for d in range(hidden_size):
                grad_input[t][d] = (
                    q_grad[d]
                    + k_grad[d]
                    + v_grad[d]
                )

        return grad_input

if __name__ == "__main__":
    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0

    attention = SelfAttention(
        hidden_size=8,
        heads=2,
        layer_id=0,
    )

    x = [
        [0.1] * 8,
        [0.2] * 8,
        [0.3] * 8,
    ]

    y = attention.forward(x)

    print("SelfAttention OK")
    print("Sequence length:", len(y))
    print("Hidden size:", len(y[0]))
    print("Heads:", attention.heads)
    print("Head dimension:", attention.head_dim)
    print("Cache size:", attention.cache_size())

    weights = attention._cache[-1]["weights"]

    print()
    print("Causal mask check:")

    for position in range(len(x)):
        print(
            f"Position {position}:",
            weights[0][position],
        )
