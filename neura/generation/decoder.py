"""
Neura Text Decoder
"""


class Decoder:

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer


    def decode(self, tokens):

        return self.tokenizer.decode(
            tokens
        )
