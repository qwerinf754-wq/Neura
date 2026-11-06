"""
Neura RMS Normalization
=======================

Trainable RMSNorm implementation for the Neura reference backend.
"""

from __future__ import annotations

import math

from .parameters import PARAMETERS


class RMSNorm:
    """Root Mean Square Layer Normalization."""

    def __init__(
        self,
        hidden_size: int | None = None,
        epsilon: float = 1e-6,
        parameter_name: str = "rms_norm.weight",
    ):
        if epsilon <= 0:
            raise ValueError("epsilon must be positive.")

        self.hidden_size = hidden_size
        self.epsilon = epsilon
        self.parameter_name = parameter_name

        self._forward_cache = []

        if hidden_size is not None:
            if hidden_size <= 0:
                raise ValueError(
                    "hidden_size must be positive."
                )

            self.weight = [1.0] * hidden_size

            PARAMETERS.add(
                self.parameter_name,
                self.weight,
            )
        else:
            self.weight = None

    # -----------------------------------------------------------------
    # Parameter initialization
    # -----------------------------------------------------------------

    def _ensure_weight(self, size: int) -> None:
        """Create the learnable scale when size is first known."""

        if self.weight is not None:
            if len(self.weight) != size:
                raise ValueError(
                    "Input size does not match RMSNorm weight."
                )
            return

        if size <= 0:
            raise ValueError(
                "RMSNorm input size must be positive."
            )

        self.hidden_size = size
        self.weight = [1.0] * size

        PARAMETERS.add(
            self.parameter_name,
            self.weight,
        )

    # -----------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------

    def forward(self, vector):
        if not vector:
            return []

        self._ensure_weight(len(vector))

        weight = PARAMETERS.require(
            self.parameter_name
        )

        mean_square = (
            sum(x * x for x in vector)
            / len(vector)
        )

        rms = math.sqrt(
            mean_square + self.epsilon
        )

        normalized = [
            x / rms
            for x in vector
        ]

        output = [
            x * gamma
            for x, gamma in zip(
                normalized,
                weight,
            )
        ]

        self._forward_cache.append(
            {
                "input": list(vector),
                "normalized": list(normalized),
                "rms": rms,
                "weight": list(weight),
            }
        )

        return output

    # -----------------------------------------------------------------
    # Backward
    # -----------------------------------------------------------------

    def backward(self, gradient):
        if not self._forward_cache:
            raise RuntimeError(
                "RMSNorm.backward() called before forward()."
            )

        cache = self._forward_cache.pop()

        x = cache["input"]
        normalized = cache["normalized"]
        rms = cache["rms"]
        weight = cache["weight"]

        if len(gradient) != len(x):
            raise ValueError(
                "RMSNorm gradient size mismatch."
            )

        n = len(x)

        if n == 0:
            return []

        # Gradient with respect to gamma.
        weight_gradient = [
            g * norm
            for g, norm in zip(
                gradient,
                normalized,
            )
        ]

        PARAMETERS.add_gradient(
            self.parameter_name,
            weight_gradient,
        )

        # dL/d(normalized)
        normalized_gradient = [
            g * gamma
            for g, gamma in zip(
                gradient,
                weight,
            )
        ]

        # RMSNorm derivative.
        #
        # y = x / rms
        #
        # rms = sqrt(mean(x^2) + epsilon)
        #
        # dx_i =
        #   dy_i / rms
        #   - x_i / (n * rms^3)
        #     * sum_j(dy_j * x_j)
        dot = sum(
            grad * value
            for grad, value in zip(
                normalized_gradient,
                x,
            )
        )

        rms_cubed = rms ** 3

        input_gradient = [
            grad / rms
            - value * dot / (n * rms_cubed)
            for grad, value in zip(
                normalized_gradient,
                x,
            )
        ]

        return input_gradient


if __name__ == "__main__":
    PARAMETERS.weights.clear()
    PARAMETERS.grads.clear()
    PARAMETERS.count = 0

    norm = RMSNorm(
        hidden_size=4,
        parameter_name="test.norm.weight",
    )

    x = [1.0, 2.0, 3.0, 4.0]

    y = norm.forward(x)

    gradient = [1.0, 1.0, 1.0, 1.0]

    dx = norm.backward(gradient)

    print("RMSNorm OK")
    print("Input:", x)
    print("Output:", y)
    print("Input gradient:", dx)
    print(
        "Weight:",
        PARAMETERS.get("test.norm.weight"),
    )
    print(
        "Weight gradient:",
        PARAMETERS.gradient(
            "test.norm.weight"
        ),
    )
