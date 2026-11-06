"""
Neura Causal Language Modeling Dataset
======================================

Creates fixed-length next-token prediction samples.

For tokens:

    [A, B, C, D, E]

with sequence_length=4:

    input_ids  = [A, B, C, D]
    target_ids = [B, C, D, E]
"""

from __future__ import annotations


class CausalLMDataset:
    """Fixed-length dataset for autoregressive language modeling."""

    def __init__(
        self,
        token_ids,
        sequence_length: int,
        stride: int | None = None,
    ):
        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive."
            )

        if not token_ids:
            raise ValueError(
                "token_ids must not be empty."
            )

        for token_id in token_ids:
            if not isinstance(token_id, int):
                raise TypeError(
                    "token_ids must contain integers."
                )

        self.token_ids = list(token_ids)
        self.sequence_length = sequence_length

        if stride is None:
            stride = sequence_length

        if stride <= 0:
            raise ValueError(
                "stride must be positive."
            )

        self.stride = stride

        self._samples = self._build_samples()

    def _build_samples(self):
        samples = []

        # Need sequence_length + 1 tokens because:
        #
        # input  = tokens[i : i+L]
        # target = tokens[i+1 : i+L+1]
        #
        maximum_start = (
            len(self.token_ids)
            - self.sequence_length
        )

        for start in range(
            0,
            maximum_start,
            self.stride,
        ):
            end = start + self.sequence_length + 1

            chunk = self.token_ids[start:end]

            if len(chunk) != self.sequence_length + 1:
                continue

            input_ids = chunk[:-1]
            target_ids = chunk[1:]

            samples.append(
                {
                    "input_ids": input_ids,
                    "target_ids": target_ids,
                }
            )

        return samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, index):
        if index < 0:
            index += len(self._samples)

        if index < 0 or index >= len(self._samples):
            raise IndexError(
                "Dataset index out of range."
            )

        sample = self._samples[index]

        return {
            "input_ids": list(sample["input_ids"]),
            "target_ids": list(sample["target_ids"]),
        }

    def samples(self):
        """Return a copy of all dataset samples."""
        return [
            {
                "input_ids": list(sample["input_ids"]),
                "target_ids": list(sample["target_ids"]),
            }
            for sample in self._samples
        ]

    def token_count(self):
        return len(self.token_ids)


if __name__ == "__main__":
    tokens = list(range(12))

    dataset = CausalLMDataset(
        tokens,
        sequence_length=4,
        stride=4,
    )

    print("Neura CausalLM Dataset")
    print("======================")
    print("Token count     :", dataset.token_count())
    print("Sequence length :", dataset.sequence_length)
    print("Stride          :", dataset.stride)
    print("Samples         :", len(dataset))
    print()

    for index in range(len(dataset)):
        sample = dataset[index]

        print(
            f"Sample {index}:"
        )
        print(
            "  input :",
            sample["input_ids"],
        )
        print(
            "  target:",
            sample["target_ids"],
        )

    assert len(dataset) == 2

    assert dataset[0]["input_ids"] == [
        0, 1, 2, 3
    ]

    assert dataset[0]["target_ids"] == [
        1, 2, 3, 4
    ]

    assert dataset[1]["input_ids"] == [
        4, 5, 6, 7
    ]

    assert dataset[1]["target_ids"] == [
        5, 6, 7, 8
    ]

    print()
    print("DATASET TEST: PASS")
