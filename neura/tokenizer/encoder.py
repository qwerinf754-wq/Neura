"""
Neura Encoder
"""

from .vocabulary import VOCAB


class Encoder:

    def encode(self, text):

        ids = []

        for word in text.split():

            if word not in VOCAB.token_to_id:

                VOCAB.add(word)

            ids.append(
                VOCAB.encode(word)
            )

        return ids


ENCODER = Encoder()
