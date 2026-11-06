"""
Neura Byte-Level BPE Decoder.
"""

from __future__ import annotations


class BPEDecoder:

    def __init__(
        self,
        id_to_token: dict[int, str],
    ):
        self.id_to_token = {
            int(k): v
            for k, v in id_to_token.items()
        }

        self.token_bytes: dict[int, bytes] = {}
        self.merge_children: dict[int, tuple[int, int]] = {}

        for token_id, token in self.id_to_token.items():

            # Base byte token.
            if (
                token.startswith("<0x")
                and token.endswith(">")
            ):
                try:
                    value = int(
                        token[3:-1],
                        16,
                    )

                    if 0 <= value <= 255:
                        self.token_bytes[token_id] = bytes(
                            [value]
                        )

                except ValueError:
                    pass

            # Learned merge token.
            elif token.startswith("<MERGE:"):

                try:
                    body = token[
                        len("<MERGE:"):-1
                    ]

                    left, right = body.split(":")

                    self.merge_children[token_id] = (
                        int(left),
                        int(right),
                    )

                except (ValueError, IndexError):
                    raise ValueError(
                        f"Invalid merge token: {token!r}"
                    )

    def _expand_token(
        self,
        token_id: int,
        cache: dict[int, bytes],
    ) -> bytes:

        if token_id in cache:
            return cache[token_id]

        if token_id in self.token_bytes:

            value = self.token_bytes[token_id]

            cache[token_id] = value

            return value

        children = self.merge_children.get(
            token_id
        )

        if children is None:
            raise ValueError(
                f"Unknown or non-byte token ID: {token_id}"
            )

        left, right = children

        value = (
            self._expand_token(left, cache)
            + self._expand_token(right, cache)
        )

        cache[token_id] = value

        return value

    def decode_bytes(
        self,
        token_ids,
    ) -> bytes:

        if not isinstance(
            token_ids,
            (list, tuple),
        ):
            raise TypeError(
                "token_ids must be a list or tuple"
            )

        cache: dict[int, bytes] = {}

        output = bytearray()

        for token_id in token_ids:

            token_id = int(token_id)

            # Special tokens are not part of text bytes.
            if token_id < 4:
                continue

            output.extend(
                self._expand_token(
                    token_id,
                    cache,
                )
            )

        return bytes(output)

    def decode(
        self,
        token_ids,
    ) -> str:

        data = self.decode_bytes(
            token_ids
        )

        try:
            return data.decode(
                "utf-8",
                errors="strict",
            )

        except UnicodeDecodeError as exc:
            raise RuntimeError(
                "Decoded byte sequence is not valid UTF-8"
            ) from exc
