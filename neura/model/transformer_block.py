"""
Neura Transformer Block
=======================

Pre-Norm Transformer block:

    x
    │
    ├── RMSNorm
    │
    ├── Self-Attention + RoPE
    │
    └── + Residual
            │
            ├── RMSNorm
            │
            ├── SwiGLU
            │
            └── + Residual
                    │
                    ▼
                  output
"""

from __future__ import annotations

from .attention import SelfAttention
from .rms_norm import RMSNorm
from .swiglu import SwiGLU


class TransformerBlock:
    """Reference pre-norm Transformer block."""

    def __init__(
        self,
        hidden_size: int,
        heads: int,
        layer_id: int = 0,
        intermediate_size: int | None = None,
        rms_epsilon: float = 1e-6,
    ):
        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive."
            )

        if heads <= 0:
            raise ValueError(
                "heads must be positive."
            )

        if hidden_size % heads != 0:
            raise ValueError(
                "hidden_size must be divisible by heads."
            )

        self.hidden_size = hidden_size
        self.heads = heads
        self.layer_id = layer_id

        self.parameter_prefix = (
            f"layers.{layer_id}"
        )

        # -----------------------------------------------------
        # Attention
        # -----------------------------------------------------

        self.attention = SelfAttention(
            hidden_size=hidden_size,
            heads=heads,
            layer_id=layer_id,
        )

        # -----------------------------------------------------
        # Normalization
        # -----------------------------------------------------

        self.norm1 = RMSNorm(
            hidden_size=hidden_size,
            epsilon=rms_epsilon,
            parameter_name=(
                f"{self.parameter_prefix}.norm1.weight"
            ),
        )

        self.norm2 = RMSNorm(
            hidden_size=hidden_size,
            epsilon=rms_epsilon,
            parameter_name=(
                f"{self.parameter_prefix}.norm2.weight"
            ),
        )

        # -----------------------------------------------------
        # Feed Forward
        # -----------------------------------------------------

        self.feed_forward = SwiGLU(
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            layer_id=layer_id,
        )

        # -----------------------------------------------------
        # Forward cache
        # -----------------------------------------------------

        self._cache = []

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    def _validate_sequence(self, sequence):
        if not isinstance(sequence, list):
            raise TypeError(
                "sequence must be a list."
            )

        for vector in sequence:
            if len(vector) != self.hidden_size:
                raise ValueError(
                    "Input vector has invalid hidden size."
                )

    # ---------------------------------------------------------
    # Residual
    # ---------------------------------------------------------

    @staticmethod
    def residual_add(x, y):
        if len(x) != len(y):
            raise ValueError(
                "Residual sequence length mismatch."
            )

        return [
            [
                a + b
                for a, b in zip(
                    x_vector,
                    y_vector,
                )
            ]
            for x_vector, y_vector in zip(
                x,
                y,
            )
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
        Forward pass.

        Args:
            sequence:
                list[list[float]]

            use_cache:
                Enable inference KV cache.

            cache_position:
                Absolute position of the first token.

        Returns:
            list[list[float]]
        """

        if not sequence:
            return []

        self._validate_sequence(sequence)

        # -----------------------------------------------------
        # Attention branch
        # -----------------------------------------------------

        normalized1 = [
            self.norm1.forward(vector)
            for vector in sequence
        ]

        attention_output = self.attention.forward(
            normalized1,
            use_cache=use_cache,
            cache_position=cache_position,
        )

        residual1 = self.residual_add(
            sequence,
            attention_output,
        )

        # -----------------------------------------------------
        # Feed-forward branch
        # -----------------------------------------------------

        normalized2 = [
            self.norm2.forward(vector)
            for vector in residual1
        ]

        ff_output = [
            self.feed_forward.forward(vector)
            for vector in normalized2
        ]

        output = self.residual_add(
            residual1,
            ff_output,
        )

        self._cache.append(
            {
                "input": [
                    list(vector)
                    for vector in sequence
                ],
                "normalized1": [
                    list(vector)
                    for vector in normalized1
                ],
                "attention_output": [
                    list(vector)
                    for vector in attention_output
                ],
                "residual1": [
                    list(vector)
                    for vector in residual1
                ],
                "normalized2": [
                    list(vector)
                    for vector in normalized2
                ],
                "ff_output": [
                    list(vector)
                    for vector in ff_output
                ],
            }
        )

        return output

    # ---------------------------------------------------------
    # Cache utilities
    # ---------------------------------------------------------

    def clear_cache(self):
        """
        Clear training/backward caches for this block.

        The attention KV cache is intentionally separate and is
        cleared through clear_kv_cache().
        """

        self._cache.clear()
        self.feed_forward.clear_cache()
        self.attention.clear_cache()

    def clear_kv_cache(self):
        """
        Clear only the inference KV cache.
        """

        self.attention.clear_kv_cache()

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    def backward(self, gradient):
        """
        Backward pass through the complete block.

        The residual paths are explicitly accumulated.
        """

        if not self._cache:
            raise RuntimeError(
                "TransformerBlock.backward() called "
                "before forward()."
            )

        cache = self._cache.pop()

        sequence = cache["input"]
        residual1 = cache["residual1"]

        seq_len = len(sequence)

        if len(gradient) != seq_len:
            raise ValueError(
                "Gradient sequence length mismatch."
            )

        for vector in gradient:
            if len(vector) != self.hidden_size:
                raise ValueError(
                    "Gradient hidden size mismatch."
                )

        # =====================================================
        # Output residual:
        #
        # output = residual1 + ff_output
        #
        # Gradient reaches BOTH paths.
        # =====================================================

        grad_ff = [
            list(vector)
            for vector in gradient
        ]

        grad_residual1_direct = [
            list(vector)
            for vector in gradient
        ]

        # =====================================================
        # SwiGLU backward
        #
        # Forward called vectors in order.
        # Cache is LIFO, so backward must run reversed.
        # =====================================================

        grad_normalized2_reverse = []

        for grad in reversed(grad_ff):
            grad_normalized2_reverse.append(
                self.feed_forward.backward(grad)
            )

        grad_normalized2 = list(
            reversed(grad_normalized2_reverse)
        )

        # =====================================================
        # RMSNorm 2 backward
        # =====================================================

        grad_residual1_from_norm_reverse = []

        for grad in reversed(grad_normalized2):
            grad_residual1_from_norm_reverse.append(
                self.norm2.backward(grad)
            )

        grad_residual1_from_norm = list(
            reversed(
                grad_residual1_from_norm_reverse
            )
        )

        # =====================================================
        # Merge residual gradients
        # =====================================================

        grad_residual1 = []

        for direct, norm_grad in zip(
            grad_residual1_direct,
            grad_residual1_from_norm,
        ):
            grad_residual1.append([
                a + b
                for a, b in zip(
                    direct,
                    norm_grad,
                )
            ])

        # =====================================================
        # First residual:
        #
        # residual1 = sequence + attention_output
        #
        # gradient goes directly to sequence AND through
        # Attention.
        # =====================================================

        grad_attention = [
            list(vector)
            for vector in grad_residual1
        ]

        grad_sequence_direct = [
            list(vector)
            for vector in grad_residual1
        ]

        # =====================================================
        # Self-Attention backward
        # =====================================================

        grad_normalized1 = self.attention.backward(
            grad_attention
        )

        # =====================================================
        # RMSNorm 1 backward
        #
        # Again, reverse order because its cache is LIFO.
        # =====================================================

        grad_sequence_from_norm_reverse = []

        for grad in reversed(grad_normalized1):
            grad_sequence_from_norm_reverse.append(
                self.norm1.backward(grad)
            )

        grad_sequence_from_norm = list(
            reversed(
                grad_sequence_from_norm_reverse
            )
        )

        # =====================================================
        # Final input gradient
        # =====================================================

        grad_input = []

        for direct, norm_grad in zip(
            grad_sequence_direct,
            grad_sequence_from_norm,
        ):
            grad_input.append([
                a + b
                for a, b in zip(
                    direct,
                    norm_grad,
                )
            ])

        return grad_input

    # ---------------------------------------------------------
    # Utilities
    # ---------------------------------------------------------

    def parameter_names(self):
        return [
            f"{self.parameter_prefix}.attention.Wq",
            f"{self.parameter_prefix}.attention.Wk",
            f"{self.parameter_prefix}.attention.Wv",
            f"{self.parameter_prefix}.attention.Wo",
            f"{self.parameter_prefix}.norm1.weight",
            f"{self.parameter_prefix}.norm2.weight",
            f"{self.parameter_prefix}.feed_forward.W_gate",
            f"{self.parameter_prefix}.feed_forward.W_up",
            f"{self.parameter_prefix}.feed_forward.W_down",
        ]

    def parameter_count(self):
        total = 0

        for name in self.parameter_names():
            parameter = None

            if hasattr(self, "attention"):
                if name.endswith(".Wq"):
                    parameter = self.attention.Wq
                elif name.endswith(".Wk"):
                    parameter = self.attention.Wk
                elif name.endswith(".Wv"):
                    parameter = self.attention.Wv
                elif name.endswith(".Wo"):
                    parameter = self.attention.Wo

            if name.endswith(".norm1.weight"):
                parameter = self.norm1.weight

            elif name.endswith(".norm2.weight"):
                parameter = self.norm2.weight

            elif name.endswith(".W_gate"):
                parameter = self.feed_forward.W_gate

            elif name.endswith(".W_up"):
                parameter = self.feed_forward.W_up

            elif name.endswith(".W_down"):
                parameter = self.feed_forward.W_down

            if parameter is not None:
                if isinstance(parameter, list):
                    def count(value):
                        if isinstance(value, list):
                            return sum(
                                count(v)
                                for v in value
                            )
                        return 1

                    total += count(parameter)
                else:
                    total += 1

        return total


if __name__ == "__main__":
    from .parameters import PARAMETERS

    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0

    block = TransformerBlock(
        hidden_size=8,
        heads=2,
        layer_id=0,
        intermediate_size=16,
    )

    x = [
        [
            0.1 * (i + 1)
            for i in range(8)
        ],
        [
            0.2 * (i + 1)
            for i in range(8)
        ],
        [
            0.3 * (i + 1)
            for i in range(8)
        ],
    ]

    y = block.forward(x)

    gradient = [
        [1.0] * 8
        for _ in range(3)
    ]

    dx = block.backward(gradient)

    print("TransformerBlock")
    print("================")
    print("Input length :", len(x))
    print("Output length:", len(y))
    print("Hidden size  :", len(y[0]))
    print("Gradient len :", len(dx))
    print("Parameters   :", block.parameter_count())
    print("PASS")
