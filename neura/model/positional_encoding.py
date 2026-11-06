"""
Neura Positional Encoding
"""


class PositionalEncoding:

    def __init__(self, context_length):

        self.context_length = context_length



    def apply(self, embeddings):

        result = []

        for position, vector in enumerate(embeddings):

            result.append(
                {
                    "position": position,
                    "vector": vector,
                }
            )

        return result
