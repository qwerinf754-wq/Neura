"""
Neura CausalLM DataLoader
=========================

Batches samples produced by CausalLMDataset.
"""

from __future__ import annotations

import random


class CausalLMDataLoader:
    """Simple deterministic DataLoader for CausalLM samples."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        seed: int | None = None,
    ):
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive."
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

        self._epoch = 0

    def __len__(self):
        size = len(self.dataset)

        if self.drop_last:
            return size // self.batch_size

        return (
            size + self.batch_size - 1
        ) // self.batch_size

    def __iter__(self):
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            rng = random.Random(
                None
                if self.seed is None
                else self.seed + self._epoch
            )
            rng.shuffle(indices)

        self._epoch += 1

        batch = []

        for index in indices:
            batch.append(self.dataset[index])

            if len(batch) == self.batch_size:
                yield self._make_batch(batch)
                batch = []

        if batch and not self.drop_last:
            yield self._make_batch(batch)

    @staticmethod
    def _make_batch(samples):
        return {
            "input_ids": [
                sample["input_ids"]
                for sample in samples
            ],
            "target_ids": [
                sample["target_ids"]
                for sample in samples
            ],
        }


if __name__ == "__main__":
    from .dataset import CausalLMDataset

    tokens = list(range(20))

    dataset = CausalLMDataset(
        tokens,
        sequence_length=4,
        stride=4,
    )

    loader = CausalLMDataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    print("Neura CausalLM DataLoader")
    print("=========================")
    print("Dataset samples :", len(dataset))
    print("Batch size      :", loader.batch_size)
    print("Batches         :", len(loader))
    print()

    batches = list(loader)

    for index, batch in enumerate(batches):
        print(f"Batch {index}:")
        print("  input :", batch["input_ids"])
        print("  target:", batch["target_ids"])

    assert len(batches) == 2
    assert len(batches[0]["input_ids"]) == 2
    assert len(batches[0]["target_ids"]) == 2

    print()
    print("DATALOADER TEST: PASS")
