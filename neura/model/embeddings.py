"""
Neura Trainable Embedding Engine
"""

import random

from .parameters import PARAMETERS


class Embedding:

    def __init__(
        self,
        vocab_size,
        hidden_size,
        name="embedding"
    ):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.name = name

        self.table = {}

        # Backward cache
        self._last_tokens = []


    def _create_vector(self):

        return [
            random.uniform(-0.02, 0.02)
            for _ in range(self.hidden_size)
        ]


    def _register(self, token_id):

        name = f"{self.name}.{token_id}"

        vector = self._create_vector()

        self.table[token_id] = vector

        PARAMETERS.add(
            name,
            vector
        )

        return vector


    def forward(self, token_ids):

        self._last_tokens = list(token_ids)

        output = []

        for token_id in token_ids:

            if token_id not in self.table:

                vector = self._register(
                    token_id
                )

            else:

                vector = PARAMETERS.get(
                    f"{self.name}.{token_id}"
                )

                self.table[token_id] = vector

            output.append(vector)

        return output


    def parameters(self):

        return {
            name: PARAMETERS.get(name)
            for name in PARAMETERS.names()
            if name.startswith(
                self.name + "."
            )
        }


    # ---------------------------------------------------------
    # Backward
    # ---------------------------------------------------------

    def backward(self, gradients):

        if len(gradients) != len(self._last_tokens):
            raise ValueError(
                "Embedding gradient size mismatch."
            )

        for token_id, gradient in zip(
            self._last_tokens,
            gradients
        ):

            name = (
                f"{self.name}.{token_id}"
            )

            PARAMETERS.set_gradient(
                name,
                gradient
            )

        return None
