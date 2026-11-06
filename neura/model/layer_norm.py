"""
Neura Layer Normalization
"""

import math


class LayerNorm:

    def __init__(self, epsilon=1e-5):

        self.epsilon = epsilon

        # Stack cache
        self._forward_cache = []

    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(self, vector):

        if not vector:
            return []

        n = len(vector)

        mean = sum(vector) / n

        variance = sum(
            (x - mean) ** 2
            for x in vector
        ) / n

        std = math.sqrt(
            variance + self.epsilon
        )

        normalized = [
            (x - mean) / std
            for x in vector
        ]

        self._forward_cache.append(
            {
                "input": list(vector),
                "mean": mean,
                "variance": variance,
                "std": std,
                "normalized": list(normalized),
            }
        )

        return normalized

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    def backward(self, gradient):

        if not self._forward_cache:
            raise RuntimeError(
                "LayerNorm.backward() called before forward()."
            )

        cache = self._forward_cache.pop()

        x = cache["input"]
        mean = cache["mean"]
        variance = cache["variance"]
        std = cache["std"]
        normalized_values = cache["normalized"]

        if len(gradient) != len(x):
            raise ValueError(
                "Gradient size mismatch: "
                f"{len(gradient)} != "
                f"{len(x)}"
            )

        n = len(x)

        if n == 0:
            return []

        gradient_sum = sum(gradient)

        normalized_gradient_sum = sum(
            grad * norm
            for grad, norm in zip(
                gradient,
                normalized_values
            )
        )

        input_gradient = []

        for g, normalized in zip(
            gradient,
            normalized_values
        ):

            value = (
                n * g
                - gradient_sum
                - normalized * normalized_gradient_sum
            )

            input_gradient.append(
                value / (n * std)
            )

        return input_gradient
