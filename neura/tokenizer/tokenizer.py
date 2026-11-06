"""
Neura Tokenizer Core
"""

from .vocabulary import VOCAB


class Tokenizer:

    def __init__(
        self,
        vocabulary=None,
    ):
        self.vocab = (
            vocabulary
            if vocabulary is not None
            else VOCAB
        )

        self.frozen = False

    @property
    def vocab_size(self):

        if self.vocab.capacity is not None:
            return self.vocab.capacity

        return len(
            self.vocab.token_to_id
        )

    def freeze(self):
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def encode(self, text):

        if not isinstance(
            text,
            str,
        ):
            raise TypeError(
                "text must be a string"
            )

        ids = []

        for word in text.split():

            if word not in self.vocab.token_to_id:

                if self.frozen:
                    ids.append(
                        self.vocab.encode(
                            "<UNK>"
                        )
                    )
                    continue

                self.vocab.add(word)

            ids.append(
                self.vocab.encode(word)
            )

        return ids

    def decode(self, ids):

        if not isinstance(
            ids,
            (list, tuple),
        ):
            raise TypeError(
                "ids must be a list or tuple"
            )

        words = []

        for idx in ids:

            words.append(
                self.vocab.decode(idx)
            )

        return " ".join(words)


TOKENIZER = Tokenizer()
