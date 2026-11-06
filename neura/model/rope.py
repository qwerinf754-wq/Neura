"""
Neura Rotary Position Embedding (RoPE)
"""

import math


class RoPE:

    def __init__(
        self,
        head_dim,
        max_length=4096
    ):

        self.head_dim = head_dim
        self.max_length = max_length

        self._cache = {}


    def build_cache(self):

        cos_cache = []
        sin_cache = []

        for position in range(
            self.max_length
        ):

            cos_row = []
            sin_row = []

            for i in range(
                0,
                self.head_dim,
                2
            ):

                frequency = (
                    1.0 /
                    (
                        10000
                        ** (
                            i /
                            self.head_dim
                        )
                    )
                )

                angle = (
                    position *
                    frequency
                )

                cos_row.append(
                    math.cos(angle)
                )

                sin_row.append(
                    math.sin(angle)
                )

            cos_cache.append(cos_row)
            sin_cache.append(sin_row)

        self._cache = {
            "cos": cos_cache,
            "sin": sin_cache
        }

        return self._cache


    def apply(
        self,
        vector,
        position
    ):
        if not self._cache:
            self.build_cache()

        dim = len(vector)

        if dim % 2 != 0:
            raise ValueError(
                "RoPE requires even vector dimension"
            )

        cos_cache = self._cache["cos"][position]
        sin_cache = self._cache["sin"][position]

        pairs_per_head = self.head_dim // 2

        if pairs_per_head <= 0:
            raise ValueError(
                "head_dim must be at least 2"
            )

        output = []

        for i in range(0, dim, 2):
            pair_index = i // 2

            # Repeat the head-frequency pattern across
            # the complete hidden dimension.
            frequency_index = pair_index % pairs_per_head

            cos = cos_cache[frequency_index]
            sin = sin_cache[frequency_index]

            x1 = vector[i]
            x2 = vector[i + 1]

            output.append(
                x1 * cos - x2 * sin
            )

            output.append(
                x1 * sin + x2 * cos
            )

        return output

