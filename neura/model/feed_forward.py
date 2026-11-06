"""
Neura Feed Forward Network
"""

class FeedForward:

    def __init__(self, hidden_size):

        self.hidden_size = hidden_size

        # Stack cache for backward propagation
        self._forward_cache = []

    def gelu(self, x):
        import math
        return (
            0.5 * x *
            (
                1.0 +
                math.tanh(
                    math.sqrt(2.0 / math.pi) *
                    (
                        x +
                        0.044715 * (x ** 3)
                    )
                )
            )
        )


    # ---------------------------------------------------------
    # Forward
    # ---------------------------------------------------------

    def forward(self, vector):

        if len(vector) != self.hidden_size:
            raise ValueError(
                f"Expected vector size {self.hidden_size}, "
                f"got {len(vector)}"
            )

        # Push input onto stack
        self._forward_cache.append(
            list(vector)
        )

        return [
            max(0.0, x)
            for x in vector
        ]

    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    def backward(self, gradient):

        if not self._forward_cache:
            raise RuntimeError(
                "FeedForward.backward() called before forward()."
            )

        last_input = self._forward_cache.pop()

        if len(gradient) != len(last_input):
            raise ValueError(
                f"Gradient size mismatch: "
                f"{len(gradient)} != "
                f"{len(last_input)}"
            )

        input_gradient = []

        for x, grad in zip(
            last_input,
            gradient
        ):

            # ReLU derivative
            if x > 0.0:
                input_gradient.append(grad)
            else:
                input_gradient.append(0.0)

        return input_gradient
