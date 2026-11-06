"""
Neura Trainable Language Model Head
Reference implementation.

Maps:
    hidden[hidden_size]
        ->
    weight[vocab_size, hidden_size]
        ->
    logits[vocab_size]
"""

import random

from .parameters import PARAMETERS


class LMHead:

    def __init__(
        self,
        hidden_size,
        vocab_size,
        name="lm_head",
    ):
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive")

        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.name = name

        self.weight_name = f"{self.name}.weight"

        if not PARAMETERS.exists(self.weight_name):
            weight = []

            for _ in range(self.vocab_size):
                row = [
                    random.uniform(-0.02, 0.02)
                    for _ in range(self.hidden_size)
                ]
                weight.append(row)

            PARAMETERS.add(
                self.weight_name,
                weight,
            )

        self._last_hidden = None

    @staticmethod
    def _matvec(vector, matrix):
        if len(vector) != len(matrix[0]):
            raise ValueError("Vector/matrix dimension mismatch.")

        output = []

        for row in matrix:
            value = 0.0

            for i in range(len(vector)):
                value += vector[i] * row[i]

            output.append(value)

        return output

    @staticmethod
    def _transpose_matvec(vector, matrix):
        if len(vector) != len(matrix):
            raise ValueError("Vector/matrix dimension mismatch.")

        if not matrix:
            return []

        output = [0.0] * len(matrix[0])

        for row in range(len(matrix)):
            for column in range(len(matrix[row])):
                output[column] += (
                    vector[row] * matrix[row][column]
                )

        return output

    @staticmethod
    def _outer_add(matrix, vector_a, vector_b):
        for i in range(len(vector_a)):
            for j in range(len(vector_b)):
                matrix[i][j] += (
                    vector_a[i] * vector_b[j]
                )

    def forward(self, hidden):
        if len(hidden) != self.hidden_size:
            raise ValueError(
                "Hidden dimension does not match LM head."
            )

        weight = PARAMETERS.get(
            self.weight_name
        )

        logits = self._matvec(
            hidden,
            weight,
        )

        self._last_hidden = list(hidden)

        return logits

    def backward(
        self,
        score_gradients,
        hidden=None,
    ):
        if hidden is None:
            hidden = self._last_hidden

        if hidden is None:
            raise RuntimeError(
                "LMHead.backward() called before forward()."
            )

        if len(hidden) != self.hidden_size:
            raise ValueError(
                "Hidden dimension does not match LM head."
            )

        if len(score_gradients) != self.vocab_size:
            raise ValueError(
                "Gradient dimension does not match vocabulary size."
            )

        weight = PARAMETERS.get(
            self.weight_name
        )

        weight_gradient = [
            [0.0] * self.hidden_size
            for _ in range(self.vocab_size)
        ]

        for vocab_index in range(self.vocab_size):
            gradient = score_gradients[vocab_index]

            for hidden_index in range(self.hidden_size):
                weight_gradient[vocab_index][hidden_index] = (
                    gradient * hidden[hidden_index]
                )

        PARAMETERS.set_gradient(
            self.weight_name,
            weight_gradient,
        )

        hidden_gradient = self._transpose_matvec(
            score_gradients,
            weight,
        )

        return hidden_gradient

    def forward_sequence(self, hidden_sequence):
        return [
            self.forward(hidden)
            for hidden in hidden_sequence
        ]

    def parameter_names(self):
        return [
            self.weight_name
        ]

    def parameter_count(self):
        return (
            self.vocab_size
            * self.hidden_size
        )

    def predict(self, hidden):
        scores = self.forward(hidden)

        if not scores:
            return None

        return max(
            range(len(scores)),
            key=lambda i: scores[i],
        )


if __name__ == "__main__":
    PARAMETERS.clear()

    head = LMHead(
        hidden_size=8,
        vocab_size=16,
        name="lm_head",
    )

    hidden = [
        0.10,
        -0.20,
        0.30,
        -0.40,
        0.50,
        -0.60,
        0.70,
        -0.80,
    ]

    logits = head.forward(hidden)

    gradient = [
        1.0
        for _ in logits
    ]

    hidden_gradient = head.backward(
        gradient,
        hidden,
    )

    print("Neura LM Head")
    print("=============")
    print("Vocabulary size :", head.vocab_size)
    print("Hidden size     :", head.hidden_size)
    print("Logits shape    :", len(logits))
    print("Gradient shape  :", len(hidden_gradient))
    print("Parameters      :", head.parameter_count())
    print("PASS")
