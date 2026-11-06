"""
neura.model.swiglu

Reference implementation of the SwiGLU feed-forward network.
"""

from __future__ import annotations

import math
import random

from .parameters import PARAMETERS


class SwiGLU:
    """
    SwiGLU feed-forward network.

    y = (SiLU(x W_gate) * (x W_up)) W_down
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int | None = None,
        layer_id: int = 0,
    ):
        if hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive."
            )

        if intermediate_size is None:
            intermediate_size = 4 * hidden_size

        if intermediate_size <= 0:
            raise ValueError(
                "intermediate_size must be positive."
            )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.layer_id = layer_id

        self.parameter_prefix = (
            f"layers.{layer_id}.feed_forward"
        )

        self.W_gate = self._random_matrix(
            hidden_size,
            intermediate_size,
        )

        self.W_up = self._random_matrix(
            hidden_size,
            intermediate_size,
        )

        self.W_down = self._random_matrix(
            intermediate_size,
            hidden_size,
        )

        PARAMETERS.add(
            f"{self.parameter_prefix}.W_gate",
            self.W_gate,
        )

        PARAMETERS.add(
            f"{self.parameter_prefix}.W_up",
            self.W_up,
        )

        PARAMETERS.add(
            f"{self.parameter_prefix}.W_down",
            self.W_down,
        )

        self._cache = []

    @staticmethod
    def _random_matrix(rows: int, columns: int):
        scale = 0.02

        return [
            [
                random.uniform(
                    -scale,
                    scale,
                )
                for _ in range(columns)
            ]
            for _ in range(rows)
        ]

    @staticmethod
    def silu(x: float) -> float:
        """
        SiLU(x) = x * sigmoid(x)
        """
        if x >= 0.0:
            sigmoid = 1.0 / (
                1.0 + math.exp(-x)
            )
        else:
            exp_x = math.exp(x)
            sigmoid = exp_x / (
                1.0 + exp_x
            )

        return x * sigmoid

    @staticmethod
    def silu_derivative(x: float) -> float:
        """
        Derivative of SiLU.
        """
        if x >= 0.0:
            sigmoid = 1.0 / (
                1.0 + math.exp(-x)
            )
        else:
            exp_x = math.exp(x)
            sigmoid = exp_x / (
                1.0 + exp_x
            )

        return (
            sigmoid
            * (
                1.0
                + x * (1.0 - sigmoid)
            )
        )

    @staticmethod
    def _matvec(vector, matrix):
        if len(vector) != len(matrix):
            raise ValueError(
                "Vector/matrix dimension mismatch."
            )

        if not matrix:
            return []

        output = [
            0.0
            for _ in range(len(matrix[0]))
        ]

        for row, value in enumerate(vector):
            for column in range(
                len(matrix[0])
            ):
                output[column] += (
                    value
                    * matrix[row][column]
                )

        return output

    @staticmethod
    def _transpose_matvec(
        vector,
        matrix,
    ):
        if not matrix:
            return []

        if len(vector) != len(matrix[0]):
            raise ValueError(
                "Vector/matrix dimension mismatch."
            )

        output = [
            0.0
            for _ in range(len(matrix))
        ]

        for row in range(len(matrix)):
            total = 0.0

            for column in range(
                len(vector)
            ):
                total += (
                    vector[column]
                    * matrix[row][column]
                )

            output[row] = total

        return output

    @staticmethod
    def _outer_add(
        target,
        vector,
        gradient,
    ):
        for row in range(len(vector)):
            for column in range(
                len(gradient)
            ):
                target[row][column] += (
                    vector[row]
                    * gradient[column]
                )

    @staticmethod
    def _zeros(rows, columns):
        return [
            [0.0 for _ in range(columns)]
            for _ in range(rows)
        ]

    def forward(self, vector):
        if len(vector) != self.hidden_size:
            raise ValueError(
                "Input vector has invalid hidden size."
            )

        gate_pre = self._matvec(
            vector,
            self.W_gate,
        )

        up = self._matvec(
            vector,
            self.W_up,
        )

        gate = [
            self.silu(value)
            for value in gate_pre
        ]

        activated = [
            gate[i] * up[i]
            for i in range(
                self.intermediate_size
            )
        ]

        output = self._matvec(
            activated,
            self.W_down,
        )

        self._cache.append(
            {
                "input": list(vector),
                "gate_pre": gate_pre,
                "gate": gate,
                "up": up,
                "activated": activated,
            }
        )

        return output

    def clear_cache(self):
        """
        Clear forward caches used by backward().
        """
        self._cache.clear()

    def cache_size(self):
        """
        Return the number of cached forward passes.
        """
        return len(self._cache)

    def backward(self, gradient):
        if not self._cache:
            raise RuntimeError(
                "SwiGLU.backward() called "
                "before forward()."
            )

        if len(gradient) != self.hidden_size:
            raise ValueError(
                "Gradient hidden size mismatch."
            )

        cache = self._cache.pop()

        vector = cache["input"]
        gate_pre = cache["gate_pre"]
        gate = cache["gate"]
        up = cache["up"]
        activated = cache["activated"]

        # ---------------------------------------------
        # W_down
        # ---------------------------------------------

        grad_W_down = self._zeros(
            self.intermediate_size,
            self.hidden_size,
        )

        self._outer_add(
            grad_W_down,
            activated,
            gradient,
        )

        PARAMETERS.add_gradient(
            f"{self.parameter_prefix}.W_down",
            grad_W_down,
        )

        # ---------------------------------------------
        # Gradient through W_down
        # ---------------------------------------------

        grad_activated = (
            self._transpose_matvec(
                gradient,
                self.W_down,
            )
        )

        # activated = gate * up
        grad_gate = [
            grad_activated[i] * up[i]
            for i in range(
                self.intermediate_size
            )
        ]

        grad_up = [
            grad_activated[i] * gate[i]
            for i in range(
                self.intermediate_size
            )
        ]

        # ---------------------------------------------
        # SiLU backward
        # ---------------------------------------------

        grad_gate_pre = [
            grad_gate[i]
            * self.silu_derivative(
                gate_pre[i]
            )
            for i in range(
                self.intermediate_size
            )
        ]

        # ---------------------------------------------
        # W_gate / W_up
        # ---------------------------------------------

        grad_W_gate = self._zeros(
            self.hidden_size,
            self.intermediate_size,
        )

        grad_W_up = self._zeros(
            self.hidden_size,
            self.intermediate_size,
        )

        self._outer_add(
            grad_W_gate,
            vector,
            grad_gate_pre,
        )

        self._outer_add(
            grad_W_up,
            vector,
            grad_up,
        )

        PARAMETERS.add_gradient(
            f"{self.parameter_prefix}.W_gate",
            grad_W_gate,
        )

        PARAMETERS.add_gradient(
            f"{self.parameter_prefix}.W_up",
            grad_W_up,
        )

        # ---------------------------------------------
        # Input gradient
        # ---------------------------------------------

        grad_input_gate = (
            self._transpose_matvec(
                grad_gate_pre,
                self.W_gate,
            )
        )

        grad_input_up = (
            self._transpose_matvec(
                grad_up,
                self.W_up,
            )
        )

        grad_input = [
            grad_input_gate[i]
            + grad_input_up[i]
            for i in range(self.hidden_size)
        ]

        return grad_input

    def parameter_names(self):
        return [
            f"{self.parameter_prefix}.W_gate",
            f"{self.parameter_prefix}.W_up",
            f"{self.parameter_prefix}.W_down",
        ]

    def parameter_count(self):
        return (
            self.hidden_size
            * self.intermediate_size
            * 2
            +
            self.intermediate_size
            * self.hidden_size
        )


if __name__ == "__main__":
    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0

    layer = SwiGLU(
        hidden_size=4,
        intermediate_size=8,
        layer_id=0,
    )

    x = [
        0.1,
        0.2,
        0.3,
        0.4,
    ]

    y = layer.forward(x)

    dx = layer.backward(
        [1.0, 1.0, 1.0, 1.0]
    )

    print("SwiGLU")
    print("=======")
    print("Output:", y)
    print("dX   :", dx)
    print()
    print(
        "Parameters:",
        layer.parameter_count(),
    )
    print("PASS")
