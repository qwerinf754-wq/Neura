"""
Neura SFT Dataset / Collator

Builds training batches for supervised fine-tuning.
Only assistant tokens contribute to the loss.
"""

from dataclasses import dataclass


@dataclass(slots=True)
class SFTExample:
    input_ids: list[int]
    labels: list[int]
    loss_mask: list[int]


class SFTCollator:

    def __init__(
        self,
        pad_token_id=0,
        ignore_index=-100,
    ):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def collate(self, examples):
        if not examples:
            raise ValueError("examples cannot be empty")

        max_length = max(
            len(example.input_ids)
            for example in examples
        )

        batch_input_ids = []
        batch_labels = []
        batch_loss_mask = []

        for example in examples:
            length = len(example.input_ids)
            padding = max_length - length

            input_ids = list(example.input_ids)
            labels = list(example.labels)
            mask = list(example.loss_mask)

            labels += [
                self.ignore_index
            ] * padding

            mask += [0] * padding

            input_ids += [
                self.pad_token_id
            ] * padding

            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            batch_loss_mask.append(mask)

        return {
            "input_ids": batch_input_ids,
            "labels": batch_labels,
            "loss_mask": batch_loss_mask,
        }


__all__ = [
    "SFTExample",
    "SFTCollator",
]
