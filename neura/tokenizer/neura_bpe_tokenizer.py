"""
Neura Production Byte-Level BPE Tokenizer.

Adapter around the canonical Neura BPE encoder/decoder.

Artifact format:
    {
        "vocab": {...},
        "merges": [...],
        ...
    }

ID contract:
    0       <PAD>
    1       <UNK>
    2       <BOS>
    3       <EOS>
    4..259  raw byte tokens
    260+    learned merge tokens
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from neura.tokenizer.bpe_encoder import BPEEncoder
from neura.tokenizer.bpe_decoder import BPEDecoder


class NeuraBPETokenizer:
    """
    Production-facing adapter around Neura byte-level BPE.

    Public API:

        encode(text, add_bos=False, add_eos=False)
        decode(ids, skip_special_tokens=True)

        vocab_size
        freeze()
        unfreeze()

        round_trip(text)
    """

    PAD_ID = 0
    UNK_ID = 1
    BOS_ID = 2
    EOS_ID = 3

    MIN_VOCAB_SIZE = 260

    def __init__(
        self,
        artifact_path: str | Path,
        *,
        frozen: bool = True,
    ) -> None:

        self.artifact_path = Path(artifact_path)

        if not self.artifact_path.exists():
            raise FileNotFoundError(
                f"BPE artifact not found: {self.artifact_path}"
            )

        if not self.artifact_path.is_file():
            raise ValueError(
                f"BPE artifact is not a file: {self.artifact_path}"
            )

        self.frozen = frozen

        self._load_artifact()
        self._validate_artifact()

    # ------------------------------------------------------------------
    # Artifact loading
    # ------------------------------------------------------------------

    def _load_artifact(self) -> None:
        with self.artifact_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            data = json.load(handle)

        if not isinstance(data, dict):
            raise ValueError(
                "BPE artifact root must be a JSON object"
            )

        self.artifact = data

        vocab = data.get("vocab")
        merges = data.get("merges")

        if not isinstance(vocab, dict):
            raise ValueError(
                "BPE artifact is missing a valid 'vocab' dictionary"
            )

        if not isinstance(merges, list):
            raise ValueError(
                "BPE artifact is missing a valid 'merges' list"
            )

        normalized_vocab: dict[str, int] = {}

        for token, token_id in vocab.items():
            normalized_vocab[str(token)] = int(token_id)

        normalized_merges: list[tuple[int, int]] = []

        for pair in merges:
            if not isinstance(pair, (list, tuple)):
                raise ValueError(
                    f"Invalid BPE merge entry: {pair!r}"
                )

            if len(pair) != 2:
                raise ValueError(
                    f"Invalid BPE merge pair: {pair!r}"
                )

            normalized_merges.append(
                (
                    int(pair[0]),
                    int(pair[1]),
                )
            )

        self.vocab = normalized_vocab
        self.merges = normalized_merges

        self.id_to_token = {
            token_id: token
            for token, token_id in self.vocab.items()
        }

        self.encoder = BPEEncoder(
            self.vocab,
            self.merges,
        )

        self.decoder = BPEDecoder(
            self.id_to_token,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_artifact(self) -> None:
        vocab_size = self.vocab_size

        if vocab_size < self.MIN_VOCAB_SIZE:
            raise ValueError(
                f"Invalid BPE vocabulary size: {vocab_size}. "
                f"Byte-level BPE requires at least "
                f"{self.MIN_VOCAB_SIZE} IDs."
            )

        required = {
            "<PAD>": self.PAD_ID,
            "<UNK>": self.UNK_ID,
            "<BOS>": self.BOS_ID,
            "<EOS>": self.EOS_ID,
        }

        for token, expected_id in required.items():
            actual_id = self.vocab.get(token)

            if actual_id != expected_id:
                raise ValueError(
                    f"Special token {token!r}: "
                    f"expected ID {expected_id}, "
                    f"got {actual_id}"
                )

        expected_size = self.artifact.get("vocab_size")

        if expected_size is not None:
            expected_size = int(expected_size)

            if expected_size != vocab_size:
                raise ValueError(
                    "Artifact vocab_size mismatch: "
                    f"metadata={expected_size}, "
                    f"actual={vocab_size}"
                )

        # The raw byte vocabulary must be present.
        for value in range(256):
            token = f"<0x{value:02X}>"
            expected_id = self.encoder.BYTE_OFFSET + value

            actual_id = self.vocab.get(token)

            if actual_id != expected_id:
                raise ValueError(
                    f"Invalid byte token {token!r}: "
                    f"expected ID {expected_id}, "
                    f"got {actual_id}"
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        ids = list(
            self.encoder.encode(text)
        )

        if add_bos:
            ids.insert(
                0,
                self.BOS_ID,
            )

        if add_eos:
            ids.append(
                self.EOS_ID,
            )

        return ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(
        self,
        ids: Iterable[int],
        *,
        skip_special_tokens: bool = True,
    ) -> str:

        values = [
            int(token_id)
            for token_id in ids
        ]

        if skip_special_tokens:
            values = [
                token_id
                for token_id in values
                if token_id not in {
                    self.PAD_ID,
                    self.UNK_ID,
                    self.BOS_ID,
                    self.EOS_ID,
                }
            ]

        return self.decoder.decode(
            values
        )

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def round_trip(
        self,
        text: str,
    ) -> bool:

        return (
            self.decode(
                self.encode(text)
            )
            == text
        )

    # ------------------------------------------------------------------
    # Freeze state
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        self.frozen = True

    def unfreeze(self) -> None:
        self.frozen = False

    # ------------------------------------------------------------------
    # Artifact persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        The BPE artifact is immutable after training.

        Retraining BPE is required to create a new artifact.
        """

        raise RuntimeError(
            "NeuraBPETokenizer is backed by an immutable "
            "BPE artifact. Retrain BPE to create a new "
            "tokenizer artifact."
        )


__all__ = [
    "NeuraBPETokenizer",
]
