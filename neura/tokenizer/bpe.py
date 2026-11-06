"""
Neura Byte-Level BPE Trainer v4.

Canonical byte-level BPE implementation.

ID contract:
    0..3       special tokens
    4..259     raw UTF-8 byte tokens
    260+       learned merge tokens

Important:
    - Complete UTF-8 text is preserved.
    - Spaces are preserved.
    - Tabs are preserved.
    - Newlines are preserved.
    - Punctuation is preserved.
    - Training uses sequence IDs instead of storing full
      sequence tuples inside every pair index.

The v4 trainer is designed to reduce memory pressure and
Python object overhead compared with the previous v3 index.
"""

from __future__ import annotations

import heapq
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path



class _SequenceBitSet:
    """
    Compact set of non-negative integer sequence IDs.

    Each sequence ID is represented by one bit instead of a
    Python int object + hash-table slot.

    This is specifically used by the BPE pair -> sequence
    inverted index.  It preserves the required set semantics:
    add(), discard(), membership, truth testing and iteration.
    """

    __slots__ = ("_data",)

    def __init__(self) -> None:
        self._data = bytearray()

    def _ensure(self, sequence_id: int) -> None:
        if sequence_id < 0:
            raise ValueError(
                "sequence_id must be non-negative."
            )

        byte_index = sequence_id >> 3
        required = byte_index + 1

        if required > len(self._data):
            self._data.extend(
                b"\x00" * (
                    required - len(self._data)
                )
            )

    def add(self, sequence_id: int) -> None:
        self._ensure(sequence_id)

        byte_index = sequence_id >> 3
        bit_index = sequence_id & 7

        self._data[byte_index] |= (
            1 << bit_index
        )

    def discard(self, sequence_id: int) -> None:
        if sequence_id < 0:
            return

        byte_index = sequence_id >> 3

        if byte_index >= len(self._data):
            return

        bit_index = sequence_id & 7

        self._data[byte_index] &= (
            ~(1 << bit_index)
        )

    def __contains__(
        self,
        sequence_id: object,
    ) -> bool:
        if not isinstance(
            sequence_id,
            int,
        ):
            return False

        if sequence_id < 0:
            return False

        byte_index = sequence_id >> 3

        if byte_index >= len(self._data):
            return False

        bit_index = sequence_id & 7

        return bool(
            self._data[byte_index]
            & (1 << bit_index)
        )

    def __bool__(self) -> bool:
        return any(self._data)

    def __iter__(self):
        """
        Yield active sequence IDs in ascending order.
        """

        for byte_index, value in enumerate(
            self._data
        ):
            while value:
                lowest_bit = (
                    value & -value
                )

                bit_index = (
                    lowest_bit.bit_length()
                    - 1
                )

                yield (
                    (byte_index << 3)
                    + bit_index
                )

                value ^= lowest_bit

    def __len__(self) -> int:
        return sum(
            byte.bit_count()
            for byte in self._data
        )


class BPETrainer:
    SPECIAL_TOKENS = [
        "<PAD>",
        "<UNK>",
        "<BOS>",
        "<EOS>",
    ]

    BYTE_OFFSET = 4
    BYTE_VOCAB_SIZE = 256
    BASE_VOCAB_SIZE = BYTE_OFFSET + BYTE_VOCAB_SIZE

    FORMAT = "neura-byte-bpe-v4"

    def __init__(
        self,
        vocab_size: int = 8192,
        min_frequency: int = 2,
        checkpoint_path: str | Path | None = None,
    ):
        if not isinstance(vocab_size, int):
            raise TypeError(
                "vocab_size must be an integer"
            )

        if vocab_size < self.BASE_VOCAB_SIZE:
            raise ValueError(
                f"vocab_size must be >= "
                f"{self.BASE_VOCAB_SIZE}"
            )

        if not isinstance(min_frequency, int):
            raise TypeError(
                "min_frequency must be an integer"
            )

        if min_frequency < 1:
            raise ValueError(
                "min_frequency must be >= 1"
            )

        self.vocab_size = vocab_size
        self.min_frequency = min_frequency

        self.vocab: dict[str, int] = {}
        self.merges: list[tuple[int, int]] = []

        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else None
        )

        self._initialize_base_vocab()

    # ================================================================
    # Vocabulary
    # ================================================================

    def _initialize_base_vocab(self) -> None:
        self.vocab.clear()

        for idx, token in enumerate(
            self.SPECIAL_TOKENS
        ):
            self.vocab[token] = idx

        for value in range(
            self.BYTE_VOCAB_SIZE
        ):
            self.vocab[
                self._byte_token(value)
            ] = (
                self.BYTE_OFFSET + value
            )

    @staticmethod
    def _byte_token(
        value: int,
    ) -> str:
        return f"<0x{value:02X}>"

    @staticmethod
    def _merge_token(
        left: int,
        right: int,
    ) -> str:
        return (
            f"<MERGE:{left}:{right}>"
        )

    # ================================================================
    # UTF-8 -> base IDs
    # ================================================================

    @classmethod
    def _encode_bytes(
        cls,
        text: str,
    ) -> tuple[int, ...]:

        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        return tuple(
            cls.BYTE_OFFSET + byte
            for byte in text.encode("utf-8")
        )

    # ================================================================
    # Pair helpers
    # ================================================================

    @staticmethod
    def _sequence_pairs(
        sequence: tuple[int, ...],
    ):
        for index in range(
            len(sequence) - 1
        ):
            yield (
                sequence[index],
                sequence[index + 1],
            )

    @staticmethod
    def _unique_pairs(
        sequence: tuple[int, ...],
    ) -> set[tuple[int, int]]:

        return set(
            BPETrainer._sequence_pairs(
                sequence
            )
        )

    # ================================================================
    # Merge one sequence
    # ================================================================

    @staticmethod
    def _merge_sequence(
        sequence: tuple[int, ...],
        pair: tuple[int, int],
        new_id: int,
    ) -> tuple[int, ...]:

        left, right = pair

        if len(sequence) < 2:
            return sequence

        output: list[int] = []

        index = 0
        length = len(sequence)

        while index < length:

            if (
                index + 1 < length
                and sequence[index] == left
                and sequence[index + 1] == right
            ):
                output.append(new_id)
                index += 2
            else:
                output.append(
                    sequence[index]
                )
                index += 1

        return tuple(output)

    # ================================================================
    # Sequence-ID index
    # ================================================================

    def _build_sequence_store(
        self,
        sequences: Counter[
            tuple[int, ...]
        ],
    ):
        """
        Convert Counter[sequence] into:

            sequences_by_id:
                sequence_id -> tuple

            frequencies:
                sequence_id -> frequency

        Sequence IDs are stable for the lifetime of
        one training run.
        """

        sequences_by_id: dict[
            int,
            tuple[int, ...],
        ] = {}

        frequencies: dict[
            int,
            int,
        ] = {}

        sequence_to_id: dict[
            tuple[int, ...],
            int,
        ] = {}

        next_id = 0

        for sequence, frequency in (
            sequences.items()
        ):
            if not sequence:
                continue

            if frequency <= 0:
                continue

            sequence_id = next_id
            next_id += 1

            sequences_by_id[
                sequence_id
            ] = sequence

            frequencies[
                sequence_id
            ] = frequency

            sequence_to_id[
                sequence
            ] = sequence_id

        return (
            sequences_by_id,
            frequencies,
            sequence_to_id,
        )

    def _build_pair_index(
        self,
        sequences: Counter[
            tuple[int, ...]
        ],
    ):
        """
        Build an index using integer sequence IDs.

        Returns:

            pair_counts:
                pair -> weighted frequency

            pair_sequences:
                pair -> set(sequence_id)

            sequences_by_id:
                sequence_id -> sequence

            frequencies:
                sequence_id -> frequency

            sequence_to_id:
                sequence -> sequence_id
        """

        (
            sequences_by_id,
            frequencies,
            sequence_to_id,
        ) = self._build_sequence_store(
            sequences
        )

        # The original Counter is no longer needed after
        # its entries have been transferred into the indexed
        # sequence stores above.  Keeping it alive duplicates
        # a large dictionary structure throughout BPE training.
        #
        # The tuples themselves remain alive because they are
        # referenced by sequences_by_id / sequence_to_id.
        sequences.clear()

        pair_counts: Counter[
            tuple[int, int]
        ] = Counter()

        pair_sequences: dict[
            tuple[int, int],
            _SequenceBitSet,
        ] = defaultdict(
            _SequenceBitSet
        )

        for sequence_id, sequence in (
            sequences_by_id.items()
        ):
            frequency = frequencies[
                sequence_id
            ]

            for pair in self._unique_pairs(
                sequence
            ):
                pair_counts[pair] += (
                    frequency
                )

                pair_sequences[
                    pair
                ].add(sequence_id)

        return (
            pair_counts,
            pair_sequences,
            sequences_by_id,
            frequencies,
            sequence_to_id,
        )

    # ================================================================
    # Index removal
    # ================================================================

    @staticmethod
    def _remove_sequence_from_index(
        sequence_id: int,
        sequence: tuple[int, ...],
        frequency: int,
        pair_counts: Counter[
            tuple[int, int]
        ],
        pair_sequences: dict[
            tuple[int, int],
            _SequenceBitSet,
        ],
    ) -> None:

        if frequency <= 0:
            return

        for pair in BPETrainer._unique_pairs(
            sequence
        ):
            current = pair_counts.get(
                pair,
                0,
            )

            updated = (
                current - frequency
            )

            if updated > 0:
                pair_counts[pair] = (
                    updated
                )
            else:
                pair_counts.pop(
                    pair,
                    None,
                )

            members = pair_sequences.get(
                pair
            )

            if members is not None:
                members.discard(
                    sequence_id
                )

                if not members:
                    pair_sequences.pop(
                        pair,
                        None,
                    )

    # ================================================================
    # Index addition
    # ================================================================

    @staticmethod
    def _add_sequence_to_index(
        sequence_id: int,
        sequence: tuple[int, ...],
        frequency: int,
        pair_counts: Counter[
            tuple[int, int]
        ],
        pair_sequences: dict[
            tuple[int, int],
            _SequenceBitSet,
        ],
    ) -> None:

        if frequency <= 0:
            return

        for pair in BPETrainer._unique_pairs(
            sequence
        ):
            pair_counts[pair] += (
                frequency
            )

            pair_sequences[
                pair
            ].add(sequence_id)

    # ================================================================
    # Checkpoint
    # ================================================================

    def _checkpoint_state(self) -> dict:
        return {
            "format": self.FORMAT,

            "vocab_size":
                self.vocab_size,

            "min_frequency":
                self.min_frequency,

            "vocab":
                self.vocab,

            "merges": [
                list(pair)
                for pair in self.merges
            ],
        }

    def save_checkpoint(self) -> None:
        if self.checkpoint_path is None:
            return

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = (
            self._checkpoint_state()
        )

        temporary = (
            self.checkpoint_path.with_suffix(
                self.checkpoint_path.suffix
                + ".tmp"
            )
        )

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
            )

            handle.write("\n")

        os.replace(
            temporary,
            self.checkpoint_path,
        )

    def load_checkpoint(
        self,
        path: str | Path | None = None,
    ) -> dict:

        checkpoint_path = (
            Path(path)
            if path is not None
            else self.checkpoint_path
        )

        if checkpoint_path is None:
            raise ValueError(
                "No checkpoint path provided."
            )

        with checkpoint_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            payload = json.load(handle)

        fmt = payload.get(
            "format"
        )

        if fmt not in {
            "neura-byte-bpe-v2",
            "neura-byte-bpe-v3",
            "neura-byte-bpe-v4",
        }:
            raise ValueError(
                "Unsupported BPE checkpoint format: "
                f"{fmt!r}"
            )

        checkpoint_vocab_size = int(
            payload["vocab_size"]
        )

        checkpoint_min_frequency = int(
            payload["min_frequency"]
        )

        # Keep the target requested by the current trainer.
        #
        # Older checkpoints store "vocab_size" as the training
        # target, not necessarily the current number of learned
        # tokens. Therefore it must NOT blindly overwrite the
        # target of a new trainer.
        #
        # The loaded vocabulary itself is authoritative for the
        # current checkpoint state.
        self.min_frequency = (
            checkpoint_min_frequency
        )

        self.vocab = {
            str(token): int(token_id)
            for token, token_id
            in payload["vocab"].items()
        }

        self.merges = [
            (
                int(pair[0]),
                int(pair[1]),
            )
            for pair in payload.get(
                "merges",
                [],
            )
        ]

        # A checkpoint must never load into a trainer whose target
        # is smaller than the already learned vocabulary.
        loaded_vocab_size = len(self.vocab)

        if self.vocab_size < loaded_vocab_size:
            self.vocab_size = loaded_vocab_size

        return payload

    # ================================================================
    # Public training API
    # ================================================================

    def train_texts(
        self,
        texts,
        checkpoint_interval: int = 100,
    ) -> dict[str, int]:
        """
        Train on complete text sequences.

        No split().
        No strip().

        Therefore spaces, tabs, newlines and
        punctuation remain part of the byte sequence.
        """

        sequences = self._build_raw_sequences(
            texts
        )

        # If this trainer was restored from a checkpoint,
        # reconstruct the sequence state represented by the
        # already-learned merges before continuing training.
        if self.merges:
            sequences = self._replay_saved_merges(
                sequences
            )

        return self._train_sequences(
            sequences,
            checkpoint_interval=(
                checkpoint_interval
            ),
        )

    # ================================================================
    # Resume reconstruction
    # ================================================================

    def _build_raw_sequences(self, texts) -> Counter[tuple[int, ...]]:
        """
        Build the raw byte-level sequence Counter from training texts.

        This intentionally mirrors train_texts() preprocessing so a
        checkpoint can reconstruct the exact sequence state needed by
        the incremental trainer.
        """
        sequences: Counter[tuple[int, ...]] = Counter()

        for text in texts:
            if not isinstance(text, str):
                continue

            if not text:
                continue

            encoded = self._encode_bytes(text)

            if encoded:
                sequences[encoded] += 1

        if not sequences:
            raise RuntimeError(
                "No training text was found."
            )

        return sequences

    def _replay_saved_merges(
        self,
        sequences: Counter[tuple[int, ...]],
    ) -> Counter[tuple[int, ...]]:
        """
        Reconstruct the sequence state represented by self.merges.

        Each saved merge is replayed in its original order using the
        exact token ID assigned when that merge was learned.
        """
        if not self.merges:
            return sequences

        rebuilt = Counter(sequences)

        for merge_index, pair in enumerate(self.merges):
            new_id = self.BASE_VOCAB_SIZE + merge_index

            updated = Counter()

            for sequence, frequency in rebuilt.items():
                new_sequence = self._merge_sequence(
                    sequence,
                    pair,
                    new_id,
                )

                updated[new_sequence] += frequency

            rebuilt = updated

        return rebuilt

    # ================================================================
    # Main incremental trainer
    # ================================================================

    def _train_sequences(
        self,
        sequences: Counter[
            tuple[int, ...]
        ],
        checkpoint_interval: int = 100,
    ) -> dict[str, int]:

        if len(self.vocab) > self.vocab_size:
            raise RuntimeError(
                "Current vocabulary exceeds target vocabulary size."
            )

        if len(self.vocab) >= self.vocab_size:
            return {
                "vocab_size": len(self.vocab),
                "merges": len(self.merges),
                "new_merges": 0,
                "sequence_types": len(sequences),
            }

        if checkpoint_interval < 0:
            raise ValueError(
                "checkpoint_interval must be >= 0"
            )

        (
            pair_counts,
            pair_sequences,
            sequences_by_id,
            frequencies,
            sequence_to_id,
        ) = self._build_pair_index(sequences)

        heap = [
            (
                -frequency,
                pair[0],
                pair[1],
            )
            for pair, frequency in pair_counts.items()
            if frequency >= self.min_frequency
        ]

        heapq.heapify(heap)

        new_merges = 0

        # Keep this small on memory-constrained devices.
        REPLACEMENT_BATCH_SIZE = 256

        while (
            len(self.vocab) < self.vocab_size
            and heap
        ):

            (
                neg_frequency,
                left,
                right,
            ) = heapq.heappop(heap)

            pair = (left, right)

            current_frequency = pair_counts.get(
                pair,
                0,
            )

            if current_frequency != -neg_frequency:
                continue

            if current_frequency < self.min_frequency:
                break

            new_id = len(self.vocab)

            token = self._merge_token(
                left,
                right,
            )

            self.vocab[token] = new_id
            self.merges.append(pair)
            new_merges += 1

            affected_ids = tuple(
                pair_sequences.get(pair, ())
            )

            # --------------------------------------------------------
            # Process affected sequences in bounded batches.
            #
            # This preserves aggregation semantics while preventing
            # thousands of newly-created tuples from staying alive at
            # the same time.
            # --------------------------------------------------------

            touched_pairs: set[
                tuple[int, int]
            ] = set()

            replacements: dict[
                tuple[int, ...],
                int,
            ] = {}

            def flush_replacements() -> None:
                if not replacements:
                    return

                for (
                    new_sequence,
                    frequency,
                ) in replacements.items():

                    existing_id = sequence_to_id.get(
                        new_sequence
                    )

                    if existing_id is not None:
                        frequencies[existing_id] += frequency

                        self._add_sequence_to_index(
                            existing_id,
                            new_sequence,
                            frequency,
                            pair_counts,
                            pair_sequences,
                        )

                        touched_pairs.update(
                            self._unique_pairs(
                                new_sequence
                            )
                        )

                        continue

                    sequence_id = self._next_sequence_id(
                        sequences_by_id
                    )

                    sequences_by_id[
                        sequence_id
                    ] = new_sequence

                    frequencies[
                        sequence_id
                    ] = frequency

                    sequence_to_id[
                        new_sequence
                    ] = sequence_id

                    self._add_sequence_to_index(
                        sequence_id,
                        new_sequence,
                        frequency,
                        pair_counts,
                        pair_sequences,
                    )

                    touched_pairs.update(
                        self._unique_pairs(
                            new_sequence
                        )
                    )

                replacements.clear()

            for sequence_id in affected_ids:

                old_sequence = sequences_by_id.get(
                    sequence_id
                )

                if old_sequence is None:
                    continue

                frequency = frequencies.get(
                    sequence_id,
                    0,
                )

                if frequency <= 0:
                    continue

                new_sequence = self._merge_sequence(
                    old_sequence,
                    pair,
                    new_id,
                )

                if new_sequence == old_sequence:
                    continue

                self._remove_sequence_from_index(
                    sequence_id,
                    old_sequence,
                    frequency,
                    pair_counts,
                    pair_sequences,
                )

                sequences_by_id.pop(
                    sequence_id,
                    None,
                )

                frequencies.pop(
                    sequence_id,
                    None,
                )

                sequence_to_id.pop(
                    old_sequence,
                    None,
                )

                replacements[
                    new_sequence
                ] = (
                    replacements.get(
                        new_sequence,
                        0,
                    )
                    + frequency
                )

                if len(replacements) >= REPLACEMENT_BATCH_SIZE:
                    flush_replacements()

            flush_replacements()

            # --------------------------------------------------------
            # Selected pair is consumed.
            # --------------------------------------------------------

            pair_counts.pop(
                pair,
                None,
            )

            pair_sequences.pop(
                pair,
                None,
            )

            # --------------------------------------------------------
            # Push touched pairs.
            # --------------------------------------------------------

            for touched_pair in touched_pairs:

                frequency = pair_counts.get(
                    touched_pair,
                    0,
                )

                if frequency >= self.min_frequency:
                    heapq.heappush(
                        heap,
                        (
                            -frequency,
                            touched_pair[0],
                            touched_pair[1],
                        ),
                    )

            if (
                checkpoint_interval > 0
                and new_merges % checkpoint_interval == 0
            ):
                self.save_checkpoint()

                print(
                    "[BPE CHECKPOINT]",
                    "merge=",
                    new_merges,
                    "vocab=",
                    len(self.vocab),
                )

        self.save_checkpoint()

        return {
            "vocab_size": len(self.vocab),
            "merges": len(self.merges),
            "new_merges": new_merges,
            "sequence_types": len(sequences_by_id),
        }

    @staticmethod
    def _next_sequence_id(
        sequences_by_id: dict[
            int,
            tuple[int, ...],
        ],
    ) -> int:

        if not sequences_by_id:
            return 0

        return (
            max(
                sequences_by_id
            )
            + 1
        )

    # ================================================================
    # Compatibility API
    # ================================================================

    def _train_words(
        self,
        sequences: Counter[
            tuple[int, ...]
        ],
        checkpoint_interval: int = 100,
    ) -> dict[str, int]:

        return self._train_sequences(
            sequences,
            checkpoint_interval=(
                checkpoint_interval
            ),
        )

    # ================================================================
    # Artifact helpers
    # ================================================================

    def save_artifact(
        self,
        path: str | Path,
        metadata: dict | None = None,
    ) -> None:

        path = Path(path)

        path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "format": self.FORMAT,

            "vocab_size":
                len(self.vocab),

            "target_vocab_size":
                self.vocab_size,

            "min_frequency":
                self.min_frequency,

            "special_tokens": {
                token: index
                for index, token
                in enumerate(
                    self.SPECIAL_TOKENS
                )
            },

            "vocab":
                self.vocab,

            "merges": [
                list(pair)
                for pair in self.merges
            ],
        }

        if metadata is not None:
            payload[
                "metadata"
            ] = metadata

        temporary = (
            path.with_suffix(
                path.suffix + ".tmp"
            )
        )

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
            )

            handle.write("\n")

        os.replace(
            temporary,
            path
        )

    # ================================================================
    # Statistics
    # ================================================================

    def info(self) -> dict[str, int]:
        return {
            "vocab_size":
                len(self.vocab),

            "target_vocab_size":
                self.vocab_size,

            "merges":
                len(self.merges),

            "base_vocab_size":
                self.BASE_VOCAB_SIZE,
        }
