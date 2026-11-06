"""
Neura Byte-Level BPE Encoder / Decoder.

The encoder and trainer share exactly the same byte-ID contract:

    byte 0   -> token 4
    byte 255 -> token 259

All whitespace and punctuation are preserved because encoding
operates on the complete UTF-8 string.
"""

from __future__ import annotations


class BPEEncoder:

    BYTE_OFFSET = 4

    def __init__(
        self,
        vocab: dict[str, int],
        merges: list[tuple[int, int]],
    ):
        self.vocab = {
            str(token): int(token_id)
            for token, token_id in vocab.items()
        }

        self.token_to_id = self.vocab

        self.merge_ranks = {
            (
                int(pair[0]),
                int(pair[1]),
            ): rank
            for rank, pair in enumerate(merges)
        }

        self.merges = [
            (
                int(pair[0]),
                int(pair[1]),
            )
            for pair in merges
        ]

        self.id_to_token = {
            token_id: token
            for token, token_id in self.vocab.items()
        }

        self._merge_token_ids: dict[
            tuple[int, int],
            int,
        ] = {}

        for token, token_id in self.vocab.items():

            if not (
                token.startswith("<MERGE:")
                and token.endswith(">")
            ):
                continue

            try:
                body = token[len("<MERGE:"):-1]
                left, right = body.split(":")

                self._merge_token_ids[
                    (
                        int(left),
                        int(right),
                    )
                ] = token_id

            except (
                ValueError,
                IndexError,
            ):
                continue

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    @classmethod
    def _bytes_to_base_ids(
        cls,
        text: str,
    ) -> list[int]:

        return [
            cls.BYTE_OFFSET + byte
            for byte in text.encode("utf-8")
        ]

    def _apply_merges(
        self,
        ids: list[int],
    ) -> list[int]:

        ids = list(ids)

        while len(ids) >= 2:

            best_rank = None
            best_position = None
            best_pair = None

            for i in range(len(ids) - 1):

                pair = (
                    ids[i],
                    ids[i + 1],
                )

                rank = self.merge_ranks.get(pair)

                if rank is None:
                    continue

                if (
                    best_rank is None
                    or rank < best_rank
                ):
                    best_rank = rank
                    best_position = i
                    best_pair = pair

            if best_pair is None:
                break

            new_id = self._merge_token_ids.get(
                best_pair
            )

            if new_id is None:
                raise RuntimeError(
                    "Merge pair exists in merge table but "
                    f"has no vocabulary token: {best_pair}"
                )

            i = best_position

            ids[i:i + 2] = [new_id]

        return ids

    def encode(
        self,
        text: str,
    ) -> list[int]:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        base_ids = self._bytes_to_base_ids(text)

        return self._apply_merges(base_ids)

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def _expand_token(
        self,
        token_id: int,
        output: bytearray,
    ) -> None:

        token = self.id_to_token.get(token_id)

        if token is None:
            raise ValueError(
                f"Unknown token ID: {token_id}"
            )

        # Raw byte token.
        if (
            token.startswith("<0x")
            and token.endswith(">")
        ):
            value = int(
                token[3:-1],
                16,
            )

            if not 0 <= value <= 255:
                raise ValueError(
                    f"Invalid byte token: {token}"
                )

            output.append(value)
            return

        # Backward-compatible BYTE token.
        if (
            token.startswith("<BYTE:")
            and token.endswith(">")
        ):
            value = int(
                token[len("<BYTE:"):-1]
            )

            if not 0 <= value <= 255:
                raise ValueError(
                    f"Invalid BYTE token: {token}"
                )

            output.append(value)
            return

        # Recursive BPE merge.
        if (
            token.startswith("<MERGE:")
            and token.endswith(">")
        ):
            body = token[len("<MERGE:"):-1]
            parts = body.split(":")

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid merge token: {token}"
                )

            self._expand_token(
                int(parts[0]),
                output,
            )

            self._expand_token(
                int(parts[1]),
                output,
            )

            return

        raise ValueError(
            f"Token cannot be decoded to bytes: {token}"
        )

    def decode(
        self,
        ids,
    ) -> str:

        if not isinstance(ids, (list, tuple)):
            raise TypeError(
                "ids must be a list or tuple"
            )

        output = bytearray()

        for token_id in ids:
            self._expand_token(
                int(token_id),
                output,
            )

        return bytes(output).decode(
            "utf-8",
            errors="strict",
        )

    # ------------------------------------------------------------------
    # Round-trip helper
    # ------------------------------------------------------------------

    def round_trip(
        self,
        text: str,
    ) -> bool:

        return self.decode(
            self.encode(text)
        ) == text
