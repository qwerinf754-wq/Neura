"""
Neura Vocabulary Engine
"""

import json


class Vocabulary:

    def __init__(self):

        self.token_to_id = {}
        self.id_to_token = {}

        self._next = 0

        # Number of token IDs the model can address.
        # This is NOT the number of assigned tokens.
        self.capacity = None

        self.special_tokens = [
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<EOS>",
        ]

        for token in self.special_tokens:
            self.add(token)


    # ---------------------------------------------------------
    # Vocabulary mutation
    # ---------------------------------------------------------

    def add(self, token):

        if not isinstance(token, str):
            raise TypeError(
                "token must be a string"
            )

        if token in self.token_to_id:
            return self.token_to_id[token]

        if (
            self.capacity is not None
            and self._next >= self.capacity
        ):
            raise ValueError(
                "vocabulary capacity exceeded"
            )

        idx = self._next

        self._next += 1

        self.token_to_id[token] = idx
        self.id_to_token[idx] = token

        return idx


    def add_many(self, tokens):

        return [
            self.add(token)
            for token in tokens
        ]


    # ---------------------------------------------------------
    # Model capacity
    # ---------------------------------------------------------

    def ensure_size(self, size):

        if not isinstance(size, int):
            raise TypeError(
                "size must be an integer"
            )

        if size <= 0:
            raise ValueError(
                "size must be positive"
            )

        if size < self._next:
            raise ValueError(
                "size cannot be smaller than current vocabulary"
            )

        self.capacity = size

        return self.capacity


    # ---------------------------------------------------------
    # Token lookup
    # ---------------------------------------------------------

    def encode(self, token):

        return self.token_to_id.get(
            token,
            self.token_to_id["<UNK>"],
        )


    def decode(self, idx):

        return self.id_to_token.get(
            idx,
            "<UNK>",
        )


    def contains(self, token):

        return token in self.token_to_id


    def size(self):

        return self._next


    # ---------------------------------------------------------
    # Reserved / blocked IDs
    # ---------------------------------------------------------

    def blocked_token_ids(self):

        """
        Return IDs inside model capacity that are not assigned.

        These IDs are valid model-space IDs but do not correspond
        to actual vocabulary entries, so inference must never
        generate them.
        """

        if self.capacity is None:
            return []

        return [
            idx
            for idx in range(self.capacity)
            if idx not in self.id_to_token
        ]


    # ---------------------------------------------------------
    # Serialization
    # ---------------------------------------------------------

    def state_dict(self):

        return {
            "token_to_id": dict(
                self.token_to_id
            ),
            "id_to_token": dict(
                self.id_to_token
            ),
            "size": self._next,
            "capacity": self.capacity,
            "special_tokens": list(
                self.special_tokens
            ),
        }


    def save(self, path):

        with open(
            path,
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                self.state_dict(),
                f,
                ensure_ascii=False,
                indent=2,
            )


    def load(self, path):

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        self.token_to_id = {
            str(k): int(v)
            for k, v in data["token_to_id"].items()
        }

        self.id_to_token = {
            int(k): v
            for k, v in data["id_to_token"].items()
        }

        self._next = int(
            data["size"]
        )

        capacity = data.get(
            "capacity"
        )

        if capacity is not None:
            capacity = int(capacity)

            if capacity < self._next:
                raise ValueError(
                    "Loaded capacity is smaller than vocabulary size"
                )

        self.capacity = capacity

        loaded_specials = data.get(
            "special_tokens"
        )

        if loaded_specials is not None:
            self.special_tokens = list(
                loaded_specials
            )


    # ---------------------------------------------------------
    # Python protocol
    # ---------------------------------------------------------

    def __len__(self):

        return self._next


VOCAB = Vocabulary()
