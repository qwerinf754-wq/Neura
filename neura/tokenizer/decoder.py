"""
Neura Decoder
"""

from .vocabulary import VOCAB


class Decoder:

    def decode(self, ids):

        return " ".join(
            VOCAB.decode(i)
            for i in ids
        )


DECODER = Decoder()
